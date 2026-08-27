# SPDX-License-Identifier: Apache-2.0
"""High-batch denoise, chunked prefill, and prefix-cache packing.

Persistent encoder KV stays per-request and unpadded. Each heavy denoise
builds a temporary left-padded view; padding never writes back.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import mlx.core as mx

from vllm_metal.modilify.prefix_cache import PromptPrefixCache, clone_encoder_cache
from vllm_metal.modilify.latent_deliberation import LatentDeliberationState


class PackedPrefixCache:
    """Read-only batched prefix view for one attention layer."""

    def __init__(self, keys: mx.array, values: mx.array, offset: int) -> None:
        self.keys = keys
        self.values = values
        self.offset = int(offset)
        self.max_size = int(keys.shape[2])

    @property
    def decoder_state(self) -> tuple[mx.array, mx.array]:
        return self.keys, self.values


def left_pad_prefix(
    keys: mx.array, values: mx.array, logical_length: int, max_length: int
) -> tuple[mx.array, mx.array]:
    """Left-pad one request's populated prefix to *max_length*."""
    populated = keys[..., :logical_length, :]
    populated_v = values[..., :logical_length, :]
    pad = max_length - logical_length
    if pad < 0:
        raise ValueError("logical_length exceeds max_length")
    if pad == 0:
        return populated, populated_v
    zeros_k = mx.zeros(
        (*populated.shape[:2], pad, populated.shape[-1]), dtype=populated.dtype
    )
    zeros_v = mx.zeros(
        (*populated_v.shape[:2], pad, populated_v.shape[-1]), dtype=populated_v.dtype
    )
    return (
        mx.concatenate([zeros_k, populated], axis=2),
        mx.concatenate([zeros_v, populated_v], axis=2),
    )


def pack_encoder_caches(
    caches: Sequence[list[Any]],
    logical_lengths: Sequence[int],
) -> tuple[list[PackedPrefixCache], int]:
    """Stack per-request caches into a left-padded batched view."""
    if not caches:
        raise ValueError("caches must be non-empty")
    if len(caches) != len(logical_lengths):
        raise ValueError("caches and logical_lengths must align")
    max_length = max(int(length) for length in logical_lengths)
    if max_length <= 0:
        raise ValueError("packed prefix length must be positive")
    uniform = all(int(length) == max_length for length in logical_lengths)
    n_layers = len(caches[0])
    packed: list[PackedPrefixCache] = []
    for layer_idx in range(n_layers):
        keys_rows = []
        values_rows = []
        for cache, length in zip(caches, logical_lengths):
            layer = cache[layer_idx]
            keys, values = layer.decoder_state
            if keys is None or values is None:
                raise RuntimeError("Cannot pack an empty encoder cache layer")
            if uniform:
                padded_k = keys[..., :max_length, :]
                padded_v = values[..., :max_length, :]
            else:
                padded_k, padded_v = left_pad_prefix(
                    keys, values, int(length), max_length
                )
            keys_rows.append(padded_k)
            values_rows.append(padded_v)
        packed.append(
            PackedPrefixCache(
                mx.concatenate(keys_rows, axis=0),
                mx.concatenate(values_rows, axis=0),
                max_length,
            )
        )
    return packed, max_length


def prefix_valid_mask(logical_lengths: mx.array, max_length: int) -> mx.array:
    """Boolean [batch, max_length]: True on real (right-aligned) prefix tokens."""
    positions = mx.arange(max_length)
    starts = max_length - logical_lengths
    return positions[None, :] >= starts[:, None]


def chunked_prefill(
    model: Any,
    input_ids: mx.array,
    *,
    chunk_size: int | None,
    cache=None,
    attention_mask: mx.array | None = None,
):
    """Encode *input_ids* in chunks, appending to *cache*."""
    seq_len = int(input_ids.shape[1])
    if chunk_size is None or chunk_size <= 0 or seq_len <= chunk_size:
        return model.prefill(input_ids, attention_mask=attention_mask, cache=cache)
    if cache is None:
        cache = model.make_cache(max_size=seq_len)
    for start in range(0, seq_len, int(chunk_size)):
        stop = min(start + int(chunk_size), seq_len)
        chunk = input_ids[:, start:stop]
        chunk_mask = None if attention_mask is None else attention_mask[:, start:stop]
        cache = model.prefill(chunk, attention_mask=chunk_mask, cache=cache)
    return cache


def prefill_with_prefix_cache(
    model: Any,
    token_ids: list[int],
    *,
    prefix_cache: PromptPrefixCache | None,
    chunk_size: int | None,
    max_size: int,
):
    """Prefill one prompt, reusing cached prefix blocks when present."""
    hit_n = 0
    cache = None
    if prefix_cache is not None:
        hit_n, cache = prefix_cache.lookup(token_ids)
    remainder = token_ids[hit_n:]
    consumed = hit_n
    if remainder:
        if cache is None:
            cache = model.make_cache(max_size=max_size)
        step = int(chunk_size) if chunk_size and chunk_size > 0 else len(remainder)
        for start in range(0, len(remainder), step):
            stop = min(start + step, len(remainder))
            chunk = mx.array([remainder[start:stop]], dtype=mx.int32)
            cache = model.prefill(chunk, cache=cache)
            consumed += stop - start
            if prefix_cache is not None and consumed % prefix_cache.block_size == 0:
                prefix_cache.store(token_ids[:consumed], cache)
    elif cache is None:
        cache = model.make_cache(max_size=max_size)
    if prefix_cache is not None and token_ids:
        prefix_cache.store(token_ids, cache)
    return cache, len(token_ids)


def pack_latents(states: Sequence[LatentDeliberationState]) -> LatentDeliberationState:
    def cat(name: str) -> mx.array:
        return mx.concatenate([getattr(state, name) for state in states], axis=0)

    return LatentDeliberationState(
        token_latents=cat("token_latents"),
        memory_slots=cat("memory_slots"),
        confidence=cat("confidence"),
        entropy=cat("entropy"),
        age=cat("age"),
        token_changed=cat("token_changed"),
        confidence_delta=cat("confidence_delta"),
        entropy_delta=cat("entropy_delta"),
        ponder_steps=cat("ponder_steps"),
        stagnation_steps=cat("stagnation_steps"),
    )


def slice_latent(state: LatentDeliberationState, row: int) -> LatentDeliberationState:
    def take(name: str) -> mx.array:
        return getattr(state, name)[row : row + 1]

    return LatentDeliberationState(
        token_latents=take("token_latents"),
        memory_slots=take("memory_slots"),
        confidence=take("confidence"),
        entropy=take("entropy"),
        age=take("age"),
        token_changed=take("token_changed"),
        confidence_delta=take("confidence_delta"),
        entropy_delta=take("entropy_delta"),
        ponder_steps=take("ponder_steps"),
        stagnation_steps=take("stagnation_steps"),
    )


@dataclass
class RollingRow:
    canvas: mx.array
    confidence: mx.array
    entropy: mx.array
    age: mx.array
    latent_state: LatentDeliberationState
    history_hidden_state: mx.array | None
    cache: list[Any]
    logical_length: int
    generated: list[int]
    max_new_tokens: int
    active: bool = True
    stop_reason: str = "episode_watchdog"
    denoise_steps: int = 0
    jumps: int = 0


def shift_row_canvas(
    tensor: mx.array, committed: int, fill_value: float | int
) -> mx.array:
    """Left-shift one or more rows by a uniform committed count."""
    if committed <= 0:
        return tensor
    canvas = tensor.shape[1]
    if committed >= canvas:
        return mx.full(tensor.shape, fill_value, dtype=tensor.dtype)
    kept = tensor[:, committed:]
    fill = mx.full(
        (tensor.shape[0], committed, *tensor.shape[2:]),
        fill_value,
        dtype=tensor.dtype,
    )
    return mx.concatenate([kept, fill], axis=1)


def split_batched_cache(cache: list[Any], batch_size: int) -> list[list[Any]]:
    """Split a batched encoder cache into B independent unpadded copies."""
    rows: list[list[Any]] = [[] for _ in range(batch_size)]
    for layer in cache:
        offset = int(getattr(layer, "offset", 0) or 0)
        keys = layer.keys
        values = layer.values
        for row in range(batch_size):
            replica = type(layer)(
                int(layer.max_size), int(getattr(layer, "step", 256))
            )
            if hasattr(replica, "read_only"):
                replica.read_only = False
            if keys is not None:
                replica.keys = keys[row : row + 1] + 0
                replica.values = values[row : row + 1] + 0
            replica.offset = offset
            rows[row].append(replica)
    return rows


def generate_batched(
    model: Any,
    prompt_rows: Sequence[list[int]],
    *,
    max_new_tokens: int,
    temperature: float,
    chunk_size: int | None,
    prefix_cache: PromptPrefixCache | None,
    unknown_entropy: float,
    vocab_size: int,
    canvas_length: int,
    stop_token_ids: tuple[int, ...],
    turn_end: int,
    denoise_temperature: float,
    config: Any,
) -> tuple[list[list[int]], int, int, list[str], float, float, float]:
    """Prefill (chunked, cached) then denoise all rows in one packed forward per step."""
    from vllm_metal.modilify.commit_policy import (
        fused_commit_failure_rate,
        select_commit_lengths,
    )
    import time as _time

    dtype = model.model.decoder.embed_tokens.weight.dtype
    max_prompt = max((len(row) for row in prompt_rows), default=0)
    max_size = max(max_prompt, 1) + int(max_new_tokens)
    rows: list[RollingRow] = []
    t0 = _time.perf_counter()
    same_len = len({len(row) for row in prompt_rows}) == 1 and max_prompt > 0
    caches: list[list[Any]]
    shared_cache: list[Any] | None = None
    if same_len:
        stacked = mx.array(list(prompt_rows), dtype=mx.int32)
        shared_cache = model.make_cache(max_size=max_size)
        shared_cache = chunked_prefill(
            model, stacked, chunk_size=chunk_size, cache=shared_cache
        )
        caches = [shared_cache] * len(prompt_rows)
        if prefix_cache is not None:
            for tokens, cache in zip(
                prompt_rows, split_batched_cache(shared_cache, len(prompt_rows))
            ):
                prefix_cache.store(list(tokens), cache)
    else:
        caches = []
        for tokens in prompt_rows:
            cache, _logical = prefill_with_prefix_cache(
                model,
                list(tokens),
                prefix_cache=prefix_cache,
                chunk_size=chunk_size,
                max_size=max_size,
            )
            caches.append(cache)
    for tokens, cache in zip(prompt_rows, caches):
        latent = LatentDeliberationState.empty(
            batch_size=1,
            canvas_length=canvas_length,
            latent_dim=config.latent_dim,
            memory_slots=config.latent_memory_slots,
            dtype=dtype,
        )
        rows.append(
            RollingRow(
                canvas=mx.random.randint(0, vocab_size, (1, canvas_length)),
                confidence=mx.zeros((1, canvas_length), dtype=mx.float32),
                entropy=mx.full(
                    (1, canvas_length), unknown_entropy, dtype=mx.float32
                ),
                age=mx.zeros((1, canvas_length), dtype=mx.int32),
                latent_state=latent,
                history_hidden_state=None,
                cache=cache,
                logical_length=len(tokens),
                generated=[],
                max_new_tokens=int(max_new_tokens),
            )
        )
    prefill_seconds = _time.perf_counter() - t0

    denoise_steps = 0
    jumps = 0
    max_iterations = max(1, int(max_new_tokens) * int(config.max_ponder_steps))
    generate_started = _time.perf_counter()
    first_denoise_seconds = 0.0

    while denoise_steps < max_iterations and any(row.active for row in rows):
        step_started = _time.perf_counter()
        active_idx = [i for i, row in enumerate(rows) if row.active]
        active_rows_state = [rows[i] for i in active_idx]
        all_active = len(active_idx) == len(rows)
        length_set = {row.logical_length for row in active_rows_state}
        sharing = (
            shared_cache is not None and all_active and len(length_set) == 1
        )
        if shared_cache is not None and not sharing:
            split = split_batched_cache(shared_cache, len(rows))
            for i, row in enumerate(rows):
                rows[i] = replace(row, cache=split[i])
            shared_cache = None
            active_rows_state = [rows[i] for i in active_idx]
        if sharing:
            packed = shared_cache
            max_prefix = next(iter(length_set))
            prefix_arg: int | mx.array = int(max_prefix)
        else:
            packed, max_prefix = pack_encoder_caches(
                [row.cache for row in active_rows_state],
                [row.logical_length for row in active_rows_state],
            )
            prefix_arg = mx.array(
                [row.logical_length for row in active_rows_state], dtype=mx.int32
            )
        canvas = mx.concatenate([row.canvas for row in active_rows_state], axis=0)
        packed_latent = pack_latents(
            [row.latent_state for row in active_rows_state]
        )
        history = None
        if any(row.history_hidden_state is not None for row in active_rows_state):
            hidden_dim = config.hidden_size
            parts = []
            for row in active_rows_state:
                if row.history_hidden_state is None:
                    parts.append(
                        mx.zeros(
                            (1, canvas_length, hidden_dim),
                            dtype=dtype,
                        )
                    )
                else:
                    parts.append(row.history_hidden_state)
            history = mx.concatenate(parts, axis=0)
        remaining = mx.array(
            [
                max(row.max_new_tokens - len(row.generated), 0)
                for row in active_rows_state
            ],
            dtype=mx.int32,
        )
        confidence = mx.concatenate(
            [row.confidence for row in active_rows_state], axis=0
        )
        entropy = mx.concatenate([row.entropy for row in active_rows_state], axis=0)
        age = mx.concatenate([row.age for row in active_rows_state], axis=0)
        output = model(
            decoder_input_ids=canvas,
            cache=packed,
            previous_confidence=confidence,
            previous_entropy=entropy,
            token_age=age,
            latent_state=packed_latent,
            history_hidden_state=history,
            denoise_temperature=denoise_temperature,
            prefix_len=prefix_arg,
            cache_capacity=max_prefix,
        )
        denoise_steps += 1
        for local, row in enumerate(active_rows_state):
            row.denoise_steps += 1
        policy = select_commit_lengths(
            sampled_token_ids=output.proposal,
            normal_failure_rate=fused_commit_failure_rate(
                output.proposal_confidence, output.token_entropy, vocab_size=vocab_size
            ),
            previous_failure_rate=fused_commit_failure_rate(
                confidence, entropy, vocab_size=vocab_size
            ),
            greedy_token_ids=output.greedy_proposal,
            jump_failure_rate=fused_commit_failure_rate(
                output.greedy_confidence, output.token_entropy, vocab_size=vocab_size
            ),
            ponder_steps=packed_latent.ponder_steps,
            stagnation_steps=packed_latent.stagnation_steps,
            active_rows=mx.array([True] * len(active_rows_state)),
            remaining_lengths=remaining,
            failure_budget=float(config.commit_failure_budget),
            jump_failure_budget=float(config.jump_failure_budget),
            stop_token_id=stop_token_ids,
            max_ponder_steps=int(config.max_ponder_steps),
            stagnation_threshold=int(config.jump_on_no_progress_after),
            min_progress=float(config.min_trajectory_progress),
        )
        mx.eval(policy.commit_lengths, policy.jump_rows, policy.commit_token_ids)
        if denoise_steps == 1:
            first_denoise_seconds = _time.perf_counter() - step_started
            generate_started = _time.perf_counter()
        commit_lens = [int(v) for v in policy.commit_lengths.tolist()]
        jump_flags = [bool(v) for v in policy.jump_rows.tolist()]
        next_latent = output.next_latent_state
        shared_commit = sharing and len(set(commit_lens)) == 1
        if sharing and not shared_commit:
            split = split_batched_cache(shared_cache, len(rows))
            for i, row in enumerate(rows):
                rows[i] = replace(row, cache=split[i])
            shared_cache = None
        if shared_commit:
            commit_len = commit_lens[0]
            if commit_len > 0:
                committed = policy.commit_token_ids[:, :commit_len]
                shared_cache = model.update_cache(committed, cache=shared_cache)
        for local, row_i in enumerate(active_idx):
            row = rows[row_i]
            commit_len = commit_lens[local]
            next_canvas = output.proposal[local : local + 1]
            if jump_flags[local]:
                next_canvas = policy.commit_token_ids[local : local + 1]
                row.jumps += 1
                jumps += 1
            next_conf = output.proposal_confidence[local : local + 1].astype(mx.float32)
            next_ent = output.token_entropy[local : local + 1].astype(mx.float32)
            sliced = slice_latent(next_latent, local)
            sliced = replace(
                sliced,
                confidence=next_conf,
                entropy=next_ent,
                age=row.age + 1,
                token_changed=(next_canvas != row.canvas).astype(mx.float32),
                confidence_delta=next_conf - row.confidence,
                entropy_delta=next_ent - row.entropy,
                ponder_steps=policy.ponder_steps[local : local + 1],
                stagnation_steps=policy.stagnation_steps[local : local + 1],
            )
            history_row = (
                None
                if output.heavy_hidden_state is None
                else output.heavy_hidden_state[local : local + 1]
            )
            row = replace(
                row,
                canvas=next_canvas,
                confidence=next_conf,
                entropy=next_ent,
                age=row.age + 1,
                latent_state=sliced,
                history_hidden_state=history_row,
                cache=shared_cache if shared_commit else row.cache,
            )
            if commit_len > 0:
                tokens = [
                    int(t)
                    for t in policy.commit_token_ids[local, :commit_len].tolist()
                ]
                row.generated.extend(tokens)
                if not shared_commit:
                    committed = policy.commit_token_ids[local : local + 1, :commit_len]
                    row.cache = model.update_cache(committed, cache=row.cache)
                row.logical_length += commit_len
                if turn_end in tokens:
                    row.stop_reason = "turn_end"
                    row.active = False
                elif any(tok in stop_token_ids and tok != turn_end for tok in tokens):
                    row.stop_reason = "eos"
                    row.active = False
                elif len(row.generated) >= row.max_new_tokens:
                    row.stop_reason = "max_new_tokens"
                    row.active = False
            row = shift_row_state(
                row,
                commit_len,
                vocab_size=vocab_size,
                canvas_length=canvas_length,
                unknown_entropy=unknown_entropy,
            )
            rows[row_i] = row

    generate_seconds = _time.perf_counter() - generate_started
    return (
        [row.generated for row in rows],
        denoise_steps,
        jumps,
        [row.stop_reason for row in rows],
        prefill_seconds,
        first_denoise_seconds,
        generate_seconds,
    )


def shift_row_state(
    row: RollingRow,
    committed: int,
    *,
    vocab_size: int,
    canvas_length: int,
    unknown_entropy: float,
) -> RollingRow:
    if committed <= 0:
        return row
    tail = mx.random.randint(0, vocab_size, (1, committed))
    kept = row.canvas[:, committed:] if committed < canvas_length else tail[:, :0]
    canvas = mx.concatenate([kept, tail], axis=1) if committed < canvas_length else tail
    latent = row.latent_state
    history = row.history_hidden_state
    if history is not None:
        history = shift_row_canvas(history, committed, 0)
    return replace(
        row,
        canvas=canvas,
        confidence=shift_row_canvas(row.confidence, committed, 0),
        entropy=shift_row_canvas(row.entropy, committed, unknown_entropy),
        age=shift_row_canvas(row.age, committed, 0),
        latent_state=LatentDeliberationState(
            token_latents=shift_row_canvas(latent.token_latents, committed, 0),
            memory_slots=latent.memory_slots,
            confidence=shift_row_canvas(latent.confidence, committed, 0),
            entropy=shift_row_canvas(latent.entropy, committed, unknown_entropy),
            age=shift_row_canvas(latent.age, committed, 0),
            token_changed=shift_row_canvas(latent.token_changed, committed, 0),
            confidence_delta=shift_row_canvas(latent.confidence_delta, committed, 0),
            entropy_delta=shift_row_canvas(latent.entropy_delta, committed, 0),
            ponder_steps=mx.zeros_like(latent.ponder_steps),
            stagnation_steps=mx.zeros_like(latent.stagnation_steps),
        ),
        history_hidden_state=history,
    )
