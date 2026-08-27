# SPDX-License-Identifier: Apache-2.0
"""Excess-entropy commit policy shared by Mk1 and ChatDLM1 inference.

Semantics match ``dlm/chatdlm1/commit_policy.py`` and
``Modilify-Mk1-MLX/modilify_mlx/commit_policy.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import mlx.core as mx

from vllm_metal.modilify.config import JUMP_FAILURE_BUDGET
from vllm_metal.modilify.latent_deliberation import (
    advance_trajectory_clocks,
    should_force_trajectory_jump,
)

FUSED_EPS = 1e-6


def fused_commit_confidence(
    proposal_confidence: mx.array,
    token_entropy: mx.array,
    *,
    vocab_size: int = 256000,
    eps: float = FUSED_EPS,
) -> mx.array:
    """``sigmoid(logit(p) - max(H - h2, 0)) ** 2``."""
    del vocab_size
    p = mx.clip(proposal_confidence.astype(mx.float32), eps, 1.0 - eps)
    entropy = mx.maximum(token_entropy.astype(mx.float32), 0.0)
    binary_entropy = -p * mx.log(p) - (1.0 - p) * mx.log1p(-p)
    excess = mx.maximum(entropy - binary_entropy, 0.0)
    logit_p = mx.log(p) - mx.log1p(-p)
    fused = mx.square(mx.sigmoid(logit_p - excess))
    return mx.clip(fused, eps, 1.0 - eps)


def fused_commit_failure_rate(
    proposal_confidence: mx.array,
    token_entropy: mx.array,
    **kwargs: object,
) -> mx.array:
    return 1.0 - fused_commit_confidence(
        proposal_confidence, token_entropy, **kwargs
    )


@dataclass(frozen=True)
class CommitPolicyDecision:
    """One inference transition from proposal to committed prefix."""

    normal_lengths: mx.array
    commit_lengths: mx.array
    commit_token_ids: mx.array
    jump_rows: mx.array
    ponder_steps: mx.array
    stagnation_steps: mx.array


def prefix_failure_commit_lengths(
    failure_rate: mx.array,
    *,
    failure_budget: float,
    valid_mask: mx.array | None = None,
) -> mx.array:
    """Longest prefix with ``cumsum(failure_rate) < budget``."""
    if failure_rate.ndim != 2:
        raise ValueError("Failure rate must have shape [batch, canvas].")
    if not math.isfinite(failure_budget) or failure_budget <= 0:
        raise ValueError("Commit failure budget must be finite and positive.")
    if valid_mask is None:
        valid_mask = mx.ones(failure_rate.shape, dtype=mx.bool_)
    if valid_mask.shape != failure_rate.shape:
        raise ValueError("Commit validity mask must match failure rate.")

    risk = mx.clip(failure_rate.astype(mx.float32), 0.0, 1.0) * valid_mask.astype(
        mx.float32
    )
    cumulative_risk = mx.cumsum(risk, axis=-1)
    contiguous_valid = mx.cumprod(valid_mask.astype(mx.int32), axis=-1).astype(
        mx.bool_
    )
    allowed = (cumulative_risk < float(failure_budget)) & contiguous_valid
    return mx.sum(mx.cumprod(allowed.astype(mx.int32), axis=-1), axis=-1)


def first_committed_token_lengths(
    proposal: mx.array,
    commit_lengths: mx.array,
    token_id: int | Sequence[int],
) -> mx.array:
    """Clip each prefix immediately after its first stop token."""
    if proposal.ndim != 2 or commit_lengths.shape != proposal.shape[:1]:
        raise ValueError("Proposal and commit lengths must share a batch dimension.")
    canvas = proposal.shape[1]
    positions = mx.arange(canvas)[None, :]
    committed = positions < commit_lengths[:, None]
    if isinstance(token_id, int):
        stop_token_ids = (int(token_id),)
    else:
        stop_token_ids = tuple(dict.fromkeys(int(value) for value in token_id))
    if not stop_token_ids:
        raise ValueError("At least one stop token ID is required.")
    matches = proposal == stop_token_ids[0]
    for value in stop_token_ids[1:]:
        matches = matches | (proposal == value)
    matches = matches & committed
    sentinel = mx.full(positions.shape, canvas, dtype=positions.dtype)
    first = mx.min(mx.where(matches, positions, sentinel), axis=-1)
    clipped = mx.where(first < canvas, first + 1, commit_lengths)
    return mx.minimum(clipped, commit_lengths)


def bounded_prefix_failure_commit_lengths(
    committed_token_ids: mx.array,
    failure_rate: mx.array,
    *,
    failure_budget: float,
    remaining_lengths: mx.array,
    stop_token_id: int | Sequence[int],
    valid_mask: mx.array | None = None,
) -> mx.array:
    if committed_token_ids.shape != failure_rate.shape:
        raise ValueError(
            "Committed token IDs and failure rate must share [batch, canvas]."
        )
    if remaining_lengths.shape != committed_token_ids.shape[:1]:
        raise ValueError("Remaining lengths must have shape [batch].")
    commit_lengths = prefix_failure_commit_lengths(
        failure_rate,
        failure_budget=failure_budget,
        valid_mask=valid_mask,
    )
    commit_lengths = mx.minimum(commit_lengths, mx.maximum(remaining_lengths, 0))
    return first_committed_token_lengths(
        committed_token_ids,
        commit_lengths,
        stop_token_id,
    )


def select_commit_lengths(
    sampled_token_ids: mx.array,
    normal_failure_rate: mx.array,
    previous_failure_rate: mx.array,
    greedy_token_ids: mx.array,
    jump_failure_rate: mx.array,
    *,
    ponder_steps: mx.array,
    stagnation_steps: mx.array,
    active_rows: mx.array,
    remaining_lengths: mx.array,
    failure_budget: float,
    stop_token_id: int | Sequence[int],
    max_ponder_steps: int,
    stagnation_threshold: int,
    min_progress: float,
    jump_failure_budget: float = JUMP_FAILURE_BUDGET,
    valid_mask: mx.array | None = None,
) -> CommitPolicyDecision:
    """Sampled prefix, or a greedy jump after ponder/stagnation exhaustion."""
    if not (
        sampled_token_ids.shape
        == normal_failure_rate.shape
        == previous_failure_rate.shape
        == greedy_token_ids.shape
        == jump_failure_rate.shape
    ):
        raise ValueError("Sampled and greedy statistics must share [batch, canvas].")

    normal = bounded_prefix_failure_commit_lengths(
        sampled_token_ids,
        normal_failure_rate,
        failure_budget=failure_budget,
        remaining_lengths=remaining_lengths,
        stop_token_id=stop_token_id,
        valid_mask=valid_mask,
    )
    canvas_length = normal_failure_rate.shape[1]
    previous_prefix_length = prefix_failure_commit_lengths(
        previous_failure_rate,
        failure_budget=failure_budget,
        valid_mask=valid_mask,
    )
    frontier_length = mx.maximum(previous_prefix_length, normal) + 1
    if valid_mask is not None:
        valid_lengths = mx.sum(valid_mask.astype(mx.int32), axis=-1)
    else:
        valid_lengths = mx.full(frontier_length.shape, canvas_length)
    frontier_length = mx.minimum(frontier_length, valid_lengths)
    positions = mx.arange(canvas_length)[None, :]
    progress_mask = positions < frontier_length[:, None]
    if valid_mask is not None:
        progress_mask = progress_mask & valid_mask
    progress_mask = progress_mask & active_rows[:, None]
    signed_improvement = previous_failure_rate.astype(mx.float32) - (
        normal_failure_rate.astype(mx.float32)
    )
    weights = progress_mask.astype(mx.float32)
    progress = mx.sum(signed_improvement * weights, axis=-1) / mx.maximum(
        mx.sum(weights, axis=-1), 1.0
    )
    next_ponder, next_stagnation = advance_trajectory_clocks(
        ponder_steps,
        stagnation_steps,
        commit_lengths=normal,
        active_rows=active_rows,
        progress_scores=progress,
        min_progress=min_progress,
    )
    jump_rows = (
        (normal == 0)
        & active_rows
        & should_force_trajectory_jump(
            next_ponder,
            next_stagnation,
            max_ponder_steps=max_ponder_steps,
            stagnation_threshold=stagnation_threshold,
        )
    )
    jump_commit = bounded_prefix_failure_commit_lengths(
        greedy_token_ids,
        jump_failure_rate,
        failure_budget=jump_failure_budget,
        remaining_lengths=remaining_lengths,
        stop_token_id=stop_token_id,
        valid_mask=valid_mask,
    )
    committed = mx.where(jump_rows, jump_commit, normal)
    commit_token_ids = mx.where(
        jump_rows[:, None],
        greedy_token_ids,
        sampled_token_ids,
    )
    committed = first_committed_token_lengths(
        commit_token_ids,
        committed,
        stop_token_id,
    )
    committed = mx.where(active_rows, committed, 0)
    jump_rows = jump_rows & (committed > 0)
    next_ponder = mx.where(committed > 0, 0, next_ponder).astype(mx.int32)
    next_stagnation = mx.where(committed > 0, 0, next_stagnation).astype(mx.int32)
    return CommitPolicyDecision(
        normal_lengths=normal,
        commit_lengths=committed,
        commit_token_ids=commit_token_ids,
        jump_rows=jump_rows,
        ponder_steps=next_ponder,
        stagnation_steps=next_stagnation,
    )
