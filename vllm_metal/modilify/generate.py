# SPDX-License-Identifier: Apache-2.0
"""Rolling-canvas generation for Modilify."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time
from typing import Any

import mlx.core as mx

from vllm_metal.modilify.commit_policy import (
    fused_commit_failure_rate,
    select_commit_lengths,
)
from vllm_metal.modilify.latent_deliberation import LatentDeliberationState
from vllm_metal.modilify.modeling import ModilifyForBlockDiffusion


@dataclass
class ModilifyGenerationOutput:
    sequences: list[list[int]]
    generated_ids: list[list[int]]
    denoise_steps: int
    jump_count: int
    stop_reasons: list[str]
    prefill_seconds: float = 0.0
    first_denoise_seconds: float = 0.0
    generate_seconds: float = 0.0
    compile_seconds: float = 0.0
    tokens_per_second: float = 0.0
    heavy_denoise_per_second: float = 0.0


@dataclass
class _RollingState:
    canvas: mx.array
    confidence: mx.array
    entropy: mx.array
    age: mx.array
    latent_state: LatentDeliberationState
    history_hidden_state: mx.array | None


def _shift_prefix(tensor: mx.array, committed: int, fill_value: float | int) -> mx.array:
    """Left-shift a batched canvas tensor by a uniform committed length."""
    if committed <= 0:
        return tensor
    canvas = tensor.shape[1]
    if committed >= canvas:
        return mx.full(tensor.shape, fill_value, dtype=tensor.dtype)
    kept = tensor[:, committed:]
    fill_shape = (tensor.shape[0], committed, *tensor.shape[2:])
    fill = mx.full(fill_shape, fill_value, dtype=tensor.dtype)
    return mx.concatenate([kept, fill], axis=1)


def _shift_state(
    state: _RollingState,
    committed: int,
    *,
    vocab_size: int,
    canvas_length: int,
    unknown_entropy: float,
) -> _RollingState:
    if committed <= 0:
        return state
    tail = mx.random.randint(0, vocab_size, (state.canvas.shape[0], committed))
    kept = state.canvas[:, committed:] if committed < canvas_length else tail[:, :0]
    canvas = mx.concatenate([kept, tail], axis=1) if committed < canvas_length else tail
    if canvas.shape[1] != canvas_length:
        raise RuntimeError("Canvas shift produced an unexpected length.")
    latent = state.latent_state
    shifted_latent = LatentDeliberationState(
        token_latents=_shift_prefix(latent.token_latents, committed, 0),
        memory_slots=latent.memory_slots,
        confidence=_shift_prefix(latent.confidence, committed, 0),
        entropy=_shift_prefix(latent.entropy, committed, unknown_entropy),
        age=_shift_prefix(latent.age, committed, 0),
        token_changed=_shift_prefix(latent.token_changed, committed, 0),
        confidence_delta=_shift_prefix(latent.confidence_delta, committed, 0),
        entropy_delta=_shift_prefix(latent.entropy_delta, committed, 0),
        ponder_steps=mx.zeros_like(latent.ponder_steps),
        stagnation_steps=mx.zeros_like(latent.stagnation_steps),
    )
    history = state.history_hidden_state
    if history is not None:
        history = _shift_prefix(history, committed, 0)
    return _RollingState(
        canvas=canvas,
        confidence=_shift_prefix(state.confidence, committed, 0),
        entropy=_shift_prefix(state.entropy, committed, unknown_entropy),
        age=_shift_prefix(state.age, committed, 0),
        latent_state=shifted_latent,
        history_hidden_state=history,
    )


def generate(
    model: ModilifyForBlockDiffusion,
    input_ids: mx.array,
    *,
    max_new_tokens: int = 256,
    temperature: float | None = None,
    attention_mask: mx.array | None = None,
    seed: int | None = None,
    chunk_size: int | None = None,
    prefix_cache: Any | None = None,
) -> ModilifyGenerationOutput:
    """Batched rolling block diffusion. Rows stop independently."""
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence].")
    if max_new_tokens <= 0:
        raise ValueError("`max_new_tokens` must be positive.")
    if seed is not None:
        mx.random.seed(int(seed))
    if input_ids.shape[0] > 1 or chunk_size or prefix_cache is not None:
        return _generate_packed(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            attention_mask=attention_mask,
            chunk_size=chunk_size,
            prefix_cache=prefix_cache,
        )

    config = model.config.with_denoise_temperature(temperature)
    canvas_length = int(config.canvas_length)
    vocab_size = int(config.vocab_size)
    dtype = model.model.decoder.embed_tokens.weight.dtype
    batch = int(input_ids.shape[0])
    unknown_entropy = math.log(vocab_size)
    denoise_temperature = float(config.denoise_temperature)

    prefill_started = time.perf_counter()
    cache = model.make_cache(max_size=int(input_ids.shape[1]) + max_new_tokens)
    # Fully-valid prompts omit attention_mask so the encoder uses the causal
    # SDPA shortcut instead of a dense boolean tensor.
    cache = model.prefill(input_ids, attention_mask=attention_mask, cache=cache)
    mx.eval([item for block in cache for item in getattr(block, "state", ())])
    prefill_seconds = time.perf_counter() - prefill_started
    prefix_len = int(input_ids.shape[1])
    compile_started = time.perf_counter()
    model.compile_attention(cache)
    compile_seconds = time.perf_counter() - compile_started

    latent = LatentDeliberationState.empty(
        batch_size=batch,
        canvas_length=canvas_length,
        latent_dim=config.latent_dim,
        memory_slots=config.latent_memory_slots,
        dtype=dtype,
    )
    state = _RollingState(
        canvas=mx.random.randint(0, vocab_size, (batch, canvas_length)),
        confidence=mx.zeros((batch, canvas_length), dtype=mx.float32),
        entropy=mx.full((batch, canvas_length), unknown_entropy, dtype=mx.float32),
        age=mx.zeros((batch, canvas_length), dtype=mx.int32),
        latent_state=latent,
        history_hidden_state=None,
    )
    stop_token_ids = config.stop_token_ids
    turn_end = int(config.turn_end_token_id)
    generated: list[list[int]] = [[] for _ in range(batch)]
    stop_reasons = ["episode_watchdog"] * batch
    active = [True] * batch
    denoise_steps = 0
    jumps = 0
    max_iterations = max(1, max_new_tokens * int(config.max_ponder_steps))
    first_denoise_seconds = 0.0
    generate_started = time.perf_counter()

    while denoise_steps < max_iterations and any(active):
        step_started = time.perf_counter()
        remaining = mx.array(
            [
                max(max_new_tokens - len(generated[row]), 0) if active[row] else 0
                for row in range(batch)
            ],
            dtype=mx.int32,
        )
        active_rows = mx.array(active)
        output = model(
            decoder_input_ids=state.canvas,
            cache=cache,
            previous_confidence=state.confidence,
            previous_entropy=state.entropy,
            token_age=state.age,
            latent_state=state.latent_state,
            history_hidden_state=state.history_hidden_state,
            denoise_temperature=denoise_temperature,
            prefix_len=prefix_len,
        )
        denoise_steps += 1
        proposal = output.proposal
        proposal_confidence = output.proposal_confidence
        token_entropy = output.token_entropy
        next_canvas = proposal
        next_confidence = proposal_confidence.astype(mx.float32)
        next_entropy = token_entropy.astype(mx.float32)
        next_latent = replace(
            output.next_latent_state,
            confidence=next_confidence,
            entropy=next_entropy,
            age=state.age + 1,
            token_changed=(next_canvas != state.canvas).astype(mx.float32),
            confidence_delta=next_confidence - state.confidence,
            entropy_delta=next_entropy - state.entropy,
        )
        next_state = _RollingState(
            canvas=next_canvas,
            confidence=next_confidence,
            entropy=next_entropy,
            age=state.age + 1,
            latent_state=next_latent,
            history_hidden_state=output.heavy_hidden_state,
        )
        policy = select_commit_lengths(
            sampled_token_ids=proposal,
            normal_failure_rate=fused_commit_failure_rate(
                proposal_confidence, token_entropy, vocab_size=vocab_size
            ),
            previous_failure_rate=fused_commit_failure_rate(
                state.confidence, state.entropy, vocab_size=vocab_size
            ),
            greedy_token_ids=output.greedy_proposal,
            jump_failure_rate=fused_commit_failure_rate(
                output.greedy_confidence, token_entropy, vocab_size=vocab_size
            ),
            ponder_steps=state.latent_state.ponder_steps,
            stagnation_steps=state.latent_state.stagnation_steps,
            active_rows=active_rows,
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
            first_denoise_seconds = time.perf_counter() - step_started
            generate_started = time.perf_counter()
        next_state = replace(
            next_state,
            latent_state=replace(
                next_state.latent_state,
                ponder_steps=policy.ponder_steps,
                stagnation_steps=policy.stagnation_steps,
            ),
        )
        jump_flags = [bool(v) for v in policy.jump_rows.tolist()]
        if any(jump_flags):
            next_state = replace(next_state, canvas=policy.commit_token_ids)
            jumps += sum(jump_flags)

        commit_lens = [int(v) for v in policy.commit_lengths.tolist()]
        # Batch-1 is the serving path; ragged multi-row commits are sequential
        # at generate() entry so persistent KV is never padded.
        commit_len = commit_lens[0]
        if active[0] and commit_len > 0:
            tokens = [
                int(t) for t in policy.commit_token_ids[0, :commit_len].tolist()
            ]
            generated[0].extend(tokens)
            committed_block = policy.commit_token_ids[:, :commit_len]
            cache = model.update_cache(committed_block, cache=cache)
            prefix_len += commit_len
            if turn_end in tokens:
                stop_reasons[0] = "turn_end"
                active[0] = False
            elif any(token in stop_token_ids and token != turn_end for token in tokens):
                stop_reasons[0] = "eos"
                active[0] = False
            elif len(generated[0]) >= max_new_tokens:
                stop_reasons[0] = "max_new_tokens"
                active[0] = False

        state = _shift_state(
            next_state,
            commit_len,
            vocab_size=vocab_size,
            canvas_length=canvas_length,
            unknown_entropy=unknown_entropy,
        )

    generate_seconds = time.perf_counter() - generate_started
    n_tokens = sum(len(row) for row in generated)
    prompt = [list(map(int, row)) for row in input_ids.tolist()]
    sequences = [p + g for p, g in zip(prompt, generated)]
    return ModilifyGenerationOutput(
        sequences=sequences,
        generated_ids=generated,
        denoise_steps=denoise_steps,
        jump_count=jumps,
        stop_reasons=stop_reasons,
        prefill_seconds=prefill_seconds,
        first_denoise_seconds=first_denoise_seconds,
        generate_seconds=generate_seconds,
        compile_seconds=compile_seconds,
        tokens_per_second=(
            n_tokens / generate_seconds if generate_seconds > 0 else 0.0
        ),
        heavy_denoise_per_second=(
            denoise_steps / (first_denoise_seconds + generate_seconds)
            if (first_denoise_seconds + generate_seconds) > 0
            else 0.0
        ),
    )


def _prompt_rows(
    input_ids: mx.array, attention_mask: mx.array | None
) -> list[list[int]]:
    rows = [list(map(int, row)) for row in input_ids.tolist()]
    if attention_mask is None:
        return rows
    mask = [list(map(int, row)) for row in attention_mask.tolist()]
    trimmed: list[list[int]] = []
    for tokens, flags in zip(rows, mask):
        kept = [tok for tok, flag in zip(tokens, flags) if flag]
        trimmed.append(kept or tokens)
    return trimmed


def _generate_packed(
    model: ModilifyForBlockDiffusion,
    input_ids: mx.array,
    *,
    max_new_tokens: int,
    temperature: float | None,
    attention_mask: mx.array | None,
    chunk_size: int | None,
    prefix_cache: Any | None,
) -> ModilifyGenerationOutput:
    from vllm_metal.modilify.continuous_batch import generate_batched

    config = model.config.with_denoise_temperature(temperature)
    prompts = _prompt_rows(input_ids, attention_mask)
    generated, denoise_steps, jumps, stop_reasons, prefill_s, first_s, gen_s = (
        generate_batched(
            model,
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=float(config.denoise_temperature),
            chunk_size=chunk_size,
            prefix_cache=prefix_cache,
            unknown_entropy=math.log(int(config.vocab_size)),
            vocab_size=int(config.vocab_size),
            canvas_length=int(config.canvas_length),
            stop_token_ids=config.stop_token_ids,
            turn_end=int(config.turn_end_token_id),
            denoise_temperature=float(config.denoise_temperature),
            config=config,
        )
    )
    n_tokens = sum(len(row) for row in generated)
    sequences = [p + g for p, g in zip(prompts, generated)]
    return ModilifyGenerationOutput(
        sequences=sequences,
        generated_ids=generated,
        denoise_steps=denoise_steps,
        jump_count=jumps,
        stop_reasons=stop_reasons,
        prefill_seconds=prefill_s,
        first_denoise_seconds=first_s,
        generate_seconds=gen_s,
        tokens_per_second=(n_tokens / gen_s if gen_s > 0 else 0.0),
        heavy_denoise_per_second=(
            denoise_steps / (first_s + gen_s) if (first_s + gen_s) > 0 else 0.0
        ),
    )
