# SPDX-License-Identifier: Apache-2.0
"""Canvas × prefix attention with compiled residuals and unsorted MoE.

Prefix KV is read-only encoder cache. Canvas is bidirectional and ephemeral.
Sliding layers only attend to the last ``sliding_window - 1`` prefix tokens.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_vlm.models.base import scaled_dot_product_attention
from mlx_vlm.models.diffusion_gemma.language import geglu


def _cache_capacity(cache) -> int:
    first = cache[0]
    keys = getattr(first, "keys", None)
    if keys is None:
        return int(getattr(first, "max_size", 0) or 0)
    return int(keys.shape[2])


def prefix_length(cache) -> int:
    offset = getattr(cache[0], "offset", 0)
    if isinstance(offset, mx.array):
        return int(mx.max(offset).item())
    return int(offset)


def _uniform_prefix_len(offset: mx.array | int | None) -> int | None:
    """Python int when every row shares the same prefix length."""
    if offset is None:
        return None
    if isinstance(offset, int):
        return int(offset)
    return None


def _slice_prefix_kv(
    keys: mx.array,
    values: mx.array,
    prefix_len: int,
    *,
    sliding_window: int | None,
) -> tuple[mx.array, mx.array]:
    """Keep only populated (and in-window) prefix keys. No padding."""
    populated = max(min(int(prefix_len), int(keys.shape[2])), 0)
    if sliding_window is None:
        if populated == keys.shape[2]:
            return keys, values
        return keys[..., :populated, :], values[..., :populated, :]
    window = max(int(sliding_window) - 1, 1)
    slide_k = min(window, populated)
    start = populated - slide_k
    return keys[..., start:populated, :], values[..., start:populated, :]


def build_decoder_masks(
    *,
    prefix_len: int | mx.array,
    canvas_length: int,
    cache_capacity: int,
    sliding_window: int,
    batch_size: int = 1,
) -> tuple[mx.array | None, mx.array | None]:
    """Boolean SDPA masks for full and sliding layers.

    Uniform (*int*) prefixes slice KV down to valid tokens, so the mask is
    all-true and SDPA can run unmasked. Ragged ``[batch]`` lengths keep a
    left-padded prefix of width *cache_capacity* and hide the pad.
    """
    if isinstance(prefix_len, int):
        # Sliced populated KV is entirely valid; skip the dense mask.
        return None, None

    canvas_valid = mx.ones((canvas_length,), dtype=mx.bool_)
    window = max(int(sliding_window) - 1, 1)
    positions = mx.arange(cache_capacity)
    lengths = prefix_len.astype(mx.int32)
    if lengths.ndim != 1:
        raise ValueError("Batched prefix_len must have shape [batch].")
    batch_size = int(lengths.shape[0])
    starts = cache_capacity - lengths
    cache_valid = positions[None, :] >= starts[:, None]
    canvas = mx.broadcast_to(canvas_valid[None, :], (batch_size, canvas_length))
    full_row = mx.concatenate([cache_valid, canvas], axis=1)
    full = mx.broadcast_to(
        full_row[:, None, None, :],
        (batch_size, 1, canvas_length, cache_capacity + canvas_length),
    )
    slide_start = mx.maximum(starts, cache_capacity - window)
    slide_valid = positions[None, :] >= slide_start[:, None]
    slide_row = mx.concatenate([slide_valid, canvas], axis=1)
    slide = mx.broadcast_to(
        slide_row[:, None, None, :],
        (batch_size, 1, canvas_length, cache_capacity + canvas_length),
    )
    return full, slide


def decoder_attention(
    attn: Any,
    x: mx.array,
    mask: mx.array | None,
    cache,
    offset: mx.array | int,
) -> mx.array:
    """Bidirectional canvas attention against a read-only encoder prefix."""
    batch, length, _ = x.shape
    queries = attn.q_proj(x).reshape(batch, length, attn.n_heads, attn.head_dim)
    queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
    queries = attn.rope(queries, offset=offset)

    keys = attn.k_proj(x).reshape(batch, length, attn.n_kv_heads, attn.head_dim)
    values = (
        attn.v_proj(x).reshape(batch, length, attn.n_kv_heads, attn.head_dim)
        if attn.v_proj is not None
        else keys
    )
    keys = attn.k_norm(keys).transpose(0, 2, 1, 3)
    keys = attn.rope(keys, offset=offset)
    values = attn.v_norm(values).transpose(0, 2, 1, 3)

    encoder_keys, encoder_values = cache.decoder_state
    # Uniform (Python int) batches slice to the populated prefix. Ragged
    # packed prefixes pass a [batch] offset and keep left-padded KV + mask.
    prefix = _uniform_prefix_len(offset)
    if prefix is not None and encoder_keys is not None:
        encoder_keys, encoder_values = _slice_prefix_kv(
            encoder_keys,
            encoder_values,
            prefix,
            sliding_window=(
                int(attn.config.sliding_window) if attn.is_sliding else None
            ),
        )
    keys = mx.concatenate([encoder_keys, keys], axis=2)
    values = mx.concatenate([encoder_values, values], axis=2)
    output = scaled_dot_product_attention(
        queries, keys, values, cache=None, scale=attn.scale, mask=mask
    )
    output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
    return attn.o_proj(output)


def experts_unsorted(
    experts: nn.Module, x: mx.array, top_k_indices, top_k_weights
) -> mx.array:
    """Expert FFN without argsort gather, so the decoder graph can compile."""
    x = mx.expand_dims(x, (-2, -3))
    gate_up = experts.gate_up_proj(x, top_k_indices, sorted_indices=False)
    gate = gate_up[..., : experts.hidden_dims]
    up = gate_up[..., experts.hidden_dims :]
    y = experts.down_proj(geglu(gate, up), top_k_indices, sorted_indices=False)
    y = y.squeeze(-2)
    return (y * top_k_weights[..., None]).sum(axis=-2)


def attn_residual(
    layer: Any,
    x: mx.array,
    mask: mx.array | None,
    cache,
    offset: mx.array | int,
) -> mx.array:
    residual = x
    hidden = layer.input_layernorm(x)
    hidden = decoder_attention(layer.self_attn, hidden, mask, cache, offset)
    hidden = layer.post_attention_layernorm(hidden)
    return residual + hidden


def ffn_residual(layer: Any, hidden: mx.array) -> mx.array:
    residual = hidden
    shared = layer.pre_feedforward_layernorm(hidden)
    shared = layer.mlp(shared)
    shared = layer.post_feedforward_layernorm_1(shared)

    flat = residual.reshape(-1, residual.shape[-1])
    top_k_indices, top_k_weights = layer.router(flat)
    routed = layer.pre_feedforward_layernorm_2(flat)
    # 256-token canvas × top-8 is in the grouped-GEMM regime; mlx-vlm's
    # default expert path sorts and is faster here than unsorted gather.
    routed = layer.experts(routed, top_k_indices, top_k_weights)
    routed = routed.reshape(residual.shape)
    routed = layer.post_feedforward_layernorm_2(routed)

    hidden = layer.post_feedforward_layernorm(shared + routed)
    return (residual + hidden) * layer.layer_scalar


def decoder_layer(
    layer: Any,
    x: mx.array,
    mask: mx.array | None,
    cache,
    offset: mx.array | int,
) -> mx.array:
    hidden = attn_residual(layer, x, mask, cache, offset)
    return ffn_residual(layer, hidden)


def make_compiled_attn_layers(decoder: nn.Module, cache) -> list:
    """Compile attention residuals only. Expert FFNs stay eager."""
    compiled = []
    for layer, layer_cache in zip(decoder.layers, cache):

        def _fn(x, offset, mask, _layer=layer, _cache=layer_cache):
            return attn_residual(_layer, x, mask, _cache, offset)

        compiled.append(mx.compile(_fn, shapeless=True))
    return compiled


def decoder_hidden_states(
    decoder: Any,
    canvas_ids: mx.array,
    latent_context: mx.array,
    cache,
    offset: mx.array | int,
    full_mask: mx.array,
    slide_mask: mx.array,
    compiled_attn_layers=None,
) -> mx.array:
    hidden = decoder._embed_canvas(
        canvas_ids,
        self_conditioning_embeddings=latent_context,
    )
    for index, (layer, layer_cache) in enumerate(zip(decoder.layers, cache)):
        mask = slide_mask if layer.layer_type == "sliding_attention" else full_mask
        if compiled_attn_layers is not None:
            hidden = compiled_attn_layers[index](hidden, offset, mask)
            hidden = ffn_residual(layer, hidden)
        else:
            hidden = decoder_layer(layer, hidden, mask, layer_cache, offset)
    return decoder.norm(hidden)


def cache_meta(cache) -> tuple[int, int]:
    return prefix_length(cache), max(_cache_capacity(cache), 1)
