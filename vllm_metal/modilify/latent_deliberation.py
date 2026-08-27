# SPDX-License-Identifier: Apache-2.0
"""Fixed-shape latent deliberation for Modilify decoding.

Ported from ``Modilify-Mk1-MLX/modilify_mlx/latent_deliberation.py``.
Memory slots do not shift on canvas commit; slot identity is addressing-only.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import mlx.core as mx
import mlx.nn as nn


@dataclass
class LatentDeliberationState:
    """Persistent, fixed-size state for one or more canvas episodes."""

    token_latents: mx.array
    memory_slots: mx.array
    confidence: mx.array
    entropy: mx.array
    age: mx.array
    token_changed: mx.array
    confidence_delta: mx.array
    entropy_delta: mx.array
    ponder_steps: mx.array
    stagnation_steps: mx.array

    @classmethod
    def empty(
        cls,
        *,
        batch_size: int,
        canvas_length: int,
        latent_dim: int,
        memory_slots: int,
        dtype: mx.Dtype,
    ) -> LatentDeliberationState:
        return cls(
            token_latents=mx.zeros((batch_size, canvas_length, latent_dim), dtype=dtype),
            memory_slots=mx.zeros((batch_size, memory_slots, latent_dim), dtype=dtype),
            confidence=mx.zeros((batch_size, canvas_length), dtype=mx.float32),
            entropy=mx.zeros((batch_size, canvas_length), dtype=mx.float32),
            age=mx.zeros((batch_size, canvas_length), dtype=mx.int32),
            token_changed=mx.zeros((batch_size, canvas_length), dtype=mx.float32),
            confidence_delta=mx.zeros((batch_size, canvas_length), dtype=mx.float32),
            entropy_delta=mx.zeros((batch_size, canvas_length), dtype=mx.float32),
            ponder_steps=mx.zeros((batch_size,), dtype=mx.int32),
            stagnation_steps=mx.zeros((batch_size,), dtype=mx.int32),
        )

    def shift(
        self, committed: int, *, entropy_fill_value: float = 0.0
    ) -> LatentDeliberationState:
        """Drop committed canvas positions without moving long-term memory."""
        canvas_length = self.token_latents.shape[1]
        if not 0 <= committed <= canvas_length:
            raise ValueError("`committed` must be in [0, canvas_length].")
        if committed == 0:
            return self

        def shifted(tensor: mx.array, fill_value: float | int = 0) -> mx.array:
            if committed >= canvas_length:
                return mx.full(tensor.shape, fill_value, dtype=tensor.dtype)
            kept = tensor[:, committed:]
            fill_shape = (tensor.shape[0], committed, *tensor.shape[2:])
            fill = mx.full(fill_shape, fill_value, dtype=tensor.dtype)
            return mx.concatenate([kept, fill], axis=1)

        return LatentDeliberationState(
            token_latents=shifted(self.token_latents),
            memory_slots=self.memory_slots,
            confidence=shifted(self.confidence),
            entropy=shifted(self.entropy, entropy_fill_value),
            age=shifted(self.age),
            token_changed=shifted(self.token_changed),
            confidence_delta=shifted(self.confidence_delta),
            entropy_delta=shifted(self.entropy_delta),
            ponder_steps=mx.zeros_like(self.ponder_steps),
            stagnation_steps=mx.zeros_like(self.stagnation_steps),
        )


def advance_trajectory_clocks(
    ponder_steps: mx.array,
    stagnation_steps: mx.array,
    *,
    commit_lengths: mx.array,
    active_rows: mx.array,
    progress_scores: mx.array,
    min_progress: float,
) -> tuple[mx.array, mx.array]:
    if min_progress < 0:
        raise ValueError("`min_progress` must be non-negative.")
    if not (
        ponder_steps.shape
        == stagnation_steps.shape
        == commit_lengths.shape
        == active_rows.shape
        == progress_scores.shape
    ):
        raise ValueError("Trajectory clock inputs must share shape [batch].")
    committed = commit_lengths > 0
    waiting = active_rows & (~committed)
    improving = progress_scores >= min_progress
    next_ponder = mx.where(
        committed,
        mx.zeros_like(ponder_steps),
        ponder_steps + waiting.astype(mx.int32),
    )
    next_stagnation = mx.where(
        committed,
        mx.zeros_like(stagnation_steps),
        mx.where(
            waiting & improving,
            mx.zeros_like(stagnation_steps),
            stagnation_steps + waiting.astype(mx.int32),
        ),
    )
    return next_ponder.astype(mx.int32), next_stagnation.astype(mx.int32)


def should_force_trajectory_jump(
    ponder_steps: mx.array,
    stagnation_steps: mx.array,
    *,
    max_ponder_steps: int,
    stagnation_threshold: int,
) -> mx.array:
    if max_ponder_steps <= 0 or stagnation_threshold <= 0:
        raise ValueError("Trajectory jump limits must be positive.")
    return (ponder_steps >= max_ponder_steps) | (
        stagnation_steps >= stagnation_threshold
    )


class BiasedMHA(nn.Module):
    """Explicit QKV attention matching PyTorch MultiheadAttention weights."""

    def __init__(self, dims: int, num_heads: int) -> None:
        super().__init__()
        if dims % num_heads:
            raise ValueError("`dims` must be divisible by `num_heads`.")
        self.num_heads = num_heads
        self.head_dim = dims // num_heads
        self.scale = self.head_dim**-0.5
        self.query_proj = nn.Linear(dims, dims, bias=True)
        self.key_proj = nn.Linear(dims, dims, bias=True)
        self.value_proj = nn.Linear(dims, dims, bias=True)
        self.out_proj = nn.Linear(dims, dims, bias=True)

    def __call__(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        batch, query_len, dims = queries.shape
        key_len = keys.shape[1]
        queries = self.query_proj(queries).reshape(
            batch, query_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        keys = self.key_proj(keys).reshape(
            batch, key_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        values = self.value_proj(values).reshape(
            batch, key_len, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        attn_mask = None
        if mask is not None:
            attn_mask = mask.reshape(1, 1, mask.shape[-2], mask.shape[-1])
        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=attn_mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, query_len, dims)
        return self.out_proj(output)


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


class _TemporalTransformerCell(nn.Module):
    """One-step recurrent token update with fixed-slot memory attention."""

    def __init__(
        self,
        latent_dim: int,
        num_heads: int,
        local_attention_window: int,
    ) -> None:
        super().__init__()
        self.state_norm = nn.LayerNorm(latent_dim)
        self.observation_norm = nn.LayerNorm(latent_dim)
        self.memory_address_norm = nn.LayerNorm(latent_dim)
        self.memory_value_norm = nn.LayerNorm(latent_dim)
        self.temporal_update = nn.Linear(2 * latent_dim, 2 * latent_dim)
        self.local_attention = BiasedMHA(latent_dim, num_heads)
        self.local_attention_window = local_attention_window
        self.token_memory_attention = BiasedMHA(latent_dim, num_heads)
        self.memory_token_attention = BiasedMHA(latent_dim, num_heads)
        self.token_ff_norm = nn.LayerNorm(latent_dim)
        self.memory_ff_norm = nn.LayerNorm(latent_dim)
        self.stored_token_norm = nn.LayerNorm(latent_dim)
        self.stored_memory_norm = nn.LayerNorm(latent_dim)
        expansion = latent_dim * 4
        self.token_ff = nn.Sequential(
            nn.Linear(latent_dim, expansion),
            nn.SiLU(),
            nn.Linear(expansion, latent_dim),
        )
        self.memory_ff = nn.Sequential(
            nn.Linear(latent_dim, expansion),
            nn.SiLU(),
            nn.Linear(expansion, latent_dim),
        )

    def _local_mask(self, length: int, dtype: mx.Dtype) -> mx.array:
        positions = mx.arange(length)
        allowed = (
            mx.abs(positions[:, None] - positions[None, :]) < self.local_attention_window
        )
        return mx.where(
            allowed,
            mx.zeros((length, length), dtype=dtype),
            mx.full((length, length), mx.finfo(dtype).min, dtype=dtype),
        )

    def __call__(
        self,
        previous_tokens: mx.array,
        observation: mx.array,
        memory: mx.array,
        memory_slot_identity: mx.array,
    ) -> tuple[mx.array, mx.array]:
        gate_logits, candidate = mx.split(
            self.temporal_update(
                mx.concatenate(
                    (
                        self.state_norm(previous_tokens),
                        self.observation_norm(observation),
                    ),
                    axis=-1,
                )
            ),
            2,
            axis=-1,
        )
        gate = mx.sigmoid(gate_logits)
        tokens = gate * previous_tokens + (1.0 - gate) * _silu(candidate)
        local_mask = self._local_mask(tokens.shape[1], tokens.dtype)
        normed_tokens = self.state_norm(tokens)
        tokens = tokens + self.local_attention(
            normed_tokens, normed_tokens, normed_tokens, mask=local_mask
        )

        addressed_memory = self.memory_address_norm(memory + memory_slot_identity)
        memory_values = self.memory_value_norm(memory)
        normed_tokens = self.state_norm(tokens)
        tokens = tokens + self.token_memory_attention(
            normed_tokens, addressed_memory, memory_values
        )
        tokens = tokens + self.token_ff(self.token_ff_norm(tokens))

        normed_tokens = self.state_norm(tokens)
        memory = memory + self.memory_token_attention(
            addressed_memory, normed_tokens, normed_tokens
        )
        memory = memory + self.memory_ff(self.memory_ff_norm(memory))
        return (
            self.stored_token_norm(tokens),
            self.stored_memory_norm(memory) + memory_slot_identity,
        )


class LatentDeliberationTransformer(nn.Module):
    """Small recurrent Transformer that compresses repeated denoise context."""

    def __init__(
        self,
        *,
        hidden_size: int,
        latent_dim: int = 1536,
        memory_slots: int = 64,
        num_layers: int = 4,
        num_heads: int = 16,
        local_attention_window: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        del dropout
        if latent_dim % num_heads:
            raise ValueError("`latent_dim` must be divisible by `num_heads`.")
        if local_attention_window <= 0:
            raise ValueError("`local_attention_window` must be positive.")
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.memory_slots = memory_slots
        self.heavy_projection = nn.Linear(hidden_size, latent_dim, bias=False)
        self.embedding_projection = nn.Linear(hidden_size, latent_dim, bias=False)
        self.scalar_projection = nn.Linear(11, latent_dim, bias=False)
        self.blocks = [
            _TemporalTransformerCell(latent_dim, num_heads, local_attention_window)
            for _ in range(num_layers)
        ]
        self.output_norm = nn.LayerNorm(latent_dim)
        self.output_projection = nn.Linear(latent_dim, hidden_size, bias=False)
        self.memory_slot_identity = mx.zeros((memory_slots, latent_dim))

    def scaled_memory_slot_identity(
        self,
        *,
        batch_size: int,
        dtype: mx.Dtype,
    ) -> mx.array:
        identity = self.memory_slot_identity.astype(mx.float32)
        identity = identity / mx.maximum(
            mx.linalg.norm(identity, axis=-1, keepdims=True), 1.0e-12
        )
        identity = identity * math.sqrt(self.latent_dim)
        return mx.broadcast_to(
            identity.astype(dtype)[None, :, :],
            (batch_size, self.memory_slots, self.latent_dim),
        )

    def project_context(self, token_latents: mx.array) -> mx.array:
        return self.output_projection(self.output_norm(token_latents))

    def __call__(
        self,
        *,
        heavy_hidden: mx.array,
        token_embeddings: mx.array,
        confidence: mx.array,
        entropy: mx.array,
        state: LatentDeliberationState,
    ) -> tuple[mx.array, LatentDeliberationState]:
        if heavy_hidden.ndim != 3:
            raise ValueError("`heavy_hidden` must have shape [batch, canvas, hidden].")
        if heavy_hidden.shape != token_embeddings.shape:
            raise ValueError(
                "`heavy_hidden` and `token_embeddings` must have the same shape."
            )
        batch_size, canvas_length, hidden_size = heavy_hidden.shape
        if hidden_size != self.hidden_size:
            raise ValueError("Unexpected hidden size for latent deliberation.")
        expected_state = (batch_size, canvas_length, self.latent_dim)
        if state.token_latents.shape != expected_state:
            raise ValueError("State token latents do not match the current canvas.")
        if state.memory_slots.shape != (
            batch_size,
            self.memory_slots,
            self.latent_dim,
        ):
            raise ValueError("State memory slots do not match this module.")

        dtype = heavy_hidden.dtype
        position = mx.linspace(-1.0, 1.0, canvas_length, dtype=dtype)
        position = mx.broadcast_to(position[None, :], (batch_size, canvas_length))
        age = mx.minimum(state.age.astype(dtype), 32767).astype(dtype)
        entropy_delta = state.entropy_delta.astype(dtype)
        scalars = mx.stack(
            (
                confidence.astype(dtype),
                mx.log1p(entropy.astype(dtype)),
                mx.log1p(age),
                position,
                state.token_changed.astype(dtype),
                state.confidence_delta.astype(dtype),
                mx.sign(entropy_delta) * mx.log1p(mx.abs(entropy_delta)),
                mx.broadcast_to(
                    mx.log1p(state.ponder_steps.astype(dtype))[:, None],
                    (batch_size, canvas_length),
                ),
                mx.broadcast_to(
                    mx.log1p(state.stagnation_steps.astype(dtype))[:, None],
                    (batch_size, canvas_length),
                ),
                confidence.astype(dtype)
                * mx.exp(-mx.maximum(entropy.astype(dtype), 0.0)),
                mx.maximum(state.confidence_delta.astype(dtype), 0.0)
                + mx.log1p(mx.maximum(-entropy_delta, 0.0)),
            ),
            axis=-1,
        )
        observation = (
            self.heavy_projection(heavy_hidden)
            + self.embedding_projection(token_embeddings)
            + self.scalar_projection(scalars)
        )
        tokens = state.token_latents
        memory = state.memory_slots
        slot_identity = self.scaled_memory_slot_identity(
            batch_size=batch_size,
            dtype=memory.dtype,
        )
        for block in self.blocks:
            tokens, memory = block(tokens, observation, memory, slot_identity)
            observation = tokens
        next_state = LatentDeliberationState(
            token_latents=tokens,
            memory_slots=memory,
            confidence=confidence.astype(mx.float32),
            entropy=entropy.astype(mx.float32),
            age=state.age,
            token_changed=state.token_changed,
            confidence_delta=state.confidence_delta,
            entropy_delta=state.entropy_delta,
            ponder_steps=state.ponder_steps,
            stagnation_steps=state.stagnation_steps,
        )
        return self.project_context(tokens), next_state
