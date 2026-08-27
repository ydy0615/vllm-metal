# SPDX-License-Identifier: Apache-2.0
"""One heavy-denoise step over a rolling canvas."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from vllm_metal.modilify.attention import (
    build_decoder_masks,
    cache_meta,
    decoder_hidden_states,
    make_compiled_attn_layers,
)
from vllm_metal.modilify.config import ModilifyRuntimeConfig
from vllm_metal.modilify.language import build_modilify_backbone
from vllm_metal.modilify.latent_deliberation import (
    LatentDeliberationState,
    LatentDeliberationTransformer,
)
from vllm_metal.modilify.vocab_ops import canvas_vocab_statistics

try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


@dataclass
class ModilifyStepOutput:
    heavy_hidden_state: mx.array
    next_latent_state: LatentDeliberationState
    cache: Any
    latent_context: mx.array
    proposal: mx.array | None = None
    proposal_confidence: mx.array | None = None
    token_entropy: mx.array | None = None
    greedy_proposal: mx.array | None = None
    greedy_confidence: mx.array | None = None


class ModilifyForBlockDiffusion(nn.Module):
    """Inference-only text Modilify model on the DiffusionGemma trunk."""

    def __init__(self, config: ModilifyRuntimeConfig) -> None:
        super().__init__()
        self.config = config
        self.model = build_modilify_backbone(config)
        self.latent_deliberation = LatentDeliberationTransformer(
            hidden_size=config.hidden_size,
            latent_dim=config.latent_dim,
            memory_slots=config.latent_memory_slots,
            num_layers=config.latent_num_layers,
            num_heads=config.latent_num_heads,
            local_attention_window=config.latent_local_attention_window,
            dropout=config.latent_dropout,
        )
        self.final_logit_softcapping = float(
            config.text_config.final_logit_softcapping
        )
        self._compiled_attn_layers = None
        self._cache_capacity: int | None = None

    def make_cache(self, max_size: int | None = None):
        return self.model.encoder.make_cache(max_size=max_size)

    def embed_canvas_tokens(self, decoder_input_ids: mx.array) -> mx.array:
        return (
            self.model.decoder.embed_tokens(decoder_input_ids)
            * self.model.decoder.embed_scale
        )

    def compile_attention(self, cache) -> None:
        """Warm compiled attention residuals for the current prefix cache."""
        compiled = make_compiled_attn_layers(self.model.decoder, cache)
        prefix_len, capacity = cache_meta(cache)
        canvas = int(self.config.canvas_length)
        hidden = int(self.config.hidden_size)
        dtype = self.model.decoder.embed_tokens.weight.dtype
        dummy = mx.zeros((1, canvas, hidden), dtype=dtype)
        offset = int(prefix_len)
        full_mask, slide_mask = build_decoder_masks(
            prefix_len=prefix_len,
            canvas_length=canvas,
            cache_capacity=max(capacity, 1),
            sliding_window=int(self.config.text_config.sliding_window),
        )
        try:
            for layer, attn_fn in zip(self.model.decoder.layers, compiled):
                mask = (
                    slide_mask
                    if layer.layer_type == "sliding_attention"
                    else full_mask
                )
                dummy = attn_fn(dummy, offset, mask)
            mx.eval(dummy)
            self._compiled_attn_layers = compiled
        except (ValueError, RuntimeError) as exc:
            logger.debug("Modilify attention compile fallback: %s", exc)
            self._compiled_attn_layers = None

    def prefill(
        self,
        input_ids: mx.array,
        *,
        attention_mask: mx.array | None = None,
        cache=None,
    ):
        if cache is None:
            cache = self.make_cache()
        # Unpadded prompts use the encoder's causal-mask shortcut (string
        # "causal" / sliding window) instead of a dense boolean tensor.
        _, cache = self.model.encoder(
            input_ids,
            attention_mask=attention_mask,
            cache=cache,
        )
        self._cache_capacity = cache_meta(cache)[1]
        return cache

    def update_cache(self, input_ids: mx.array, *, cache, attention_mask=None):
        _, cache = self.model.encoder(
            input_ids,
            attention_mask=attention_mask,
            cache=cache,
        )
        return cache

    def _prepare_latent_context(
        self,
        decoder_input_ids: mx.array,
        *,
        history_hidden_state: mx.array | None,
        confidence: mx.array | None,
        entropy: mx.array | None,
        age: mx.array | None,
        latent_state: LatentDeliberationState | None,
        dtype: mx.Dtype,
    ) -> tuple[mx.array, LatentDeliberationState]:
        batch_size, canvas_length = decoder_input_ids.shape
        if latent_state is None:
            latent_state = LatentDeliberationState.empty(
                batch_size=batch_size,
                canvas_length=canvas_length,
                latent_dim=self.config.latent_dim,
                memory_slots=self.config.latent_memory_slots,
                dtype=dtype,
            )
        if confidence is None:
            confidence = latent_state.confidence
        else:
            confidence = confidence.astype(mx.float32)
            if confidence.ndim == 3:
                confidence = mx.squeeze(confidence, axis=-1)
        if entropy is None:
            entropy = latent_state.entropy
        else:
            entropy = entropy.astype(mx.float32)
            if entropy.ndim == 3:
                entropy = mx.squeeze(entropy, axis=-1)
        if age is not None:
            latent_state = replace(latent_state, age=age.astype(mx.int32))
        token_embeddings = self.embed_canvas_tokens(decoder_input_ids)
        history = (
            mx.zeros_like(token_embeddings)
            if history_hidden_state is None
            else history_hidden_state
        )
        return self.latent_deliberation(
            heavy_hidden=history,
            token_embeddings=token_embeddings,
            confidence=confidence,
            entropy=entropy,
            state=latent_state,
        )

    def __call__(
        self,
        *,
        decoder_input_ids: mx.array,
        cache,
        previous_confidence: mx.array | None = None,
        previous_entropy: mx.array | None = None,
        token_age: mx.array | None = None,
        latent_state: LatentDeliberationState | None = None,
        history_hidden_state: mx.array | None = None,
        denoise_temperature: float | None = None,
        prefix_len: int | mx.array | None = None,
        cache_capacity: int | None = None,
    ) -> ModilifyStepOutput:
        dtype = self.model.decoder.embed_tokens.weight.dtype
        latent_context, next_state = self._prepare_latent_context(
            decoder_input_ids,
            history_hidden_state=history_hidden_state,
            confidence=previous_confidence,
            entropy=previous_entropy,
            age=token_age,
            latent_state=latent_state,
            dtype=dtype,
        )
        if cache_capacity is None or prefix_len is None:
            tracked_len, allocated = cache_meta(cache)
            self._cache_capacity = allocated
            if cache_capacity is None:
                cache_capacity = allocated
            if prefix_len is None:
                prefix_len = tracked_len
        capacity = int(cache_capacity)
        canvas_length = int(decoder_input_ids.shape[1])
        batch_size = int(decoder_input_ids.shape[0])
        full_mask, slide_mask = build_decoder_masks(
            prefix_len=prefix_len,
            canvas_length=canvas_length,
            cache_capacity=capacity,
            sliding_window=int(self.config.text_config.sliding_window),
            batch_size=batch_size,
        )
        # Keep Python ints as ints so attention can slice KV without a sync.
        rope_offset = prefix_len
        compiled = (
            None
            if isinstance(prefix_len, mx.array)
            else self._compiled_attn_layers
        )
        hidden_states = decoder_hidden_states(
            self.model.decoder,
            decoder_input_ids,
            latent_context,
            cache,
            rope_offset,
            full_mask,
            slide_mask,
            compiled_attn_layers=compiled,
        )
        temperature = (
            self.config.denoise_temperature
            if denoise_temperature is None
            else float(denoise_temperature)
        )
        stats = canvas_vocab_statistics(
            hidden_states,
            self.model.decoder.embed_tokens.weight,
            temperature=temperature,
            softcap=self.final_logit_softcapping,
            chunk_size=self.config.vocab_chunk_size,
        )
        return ModilifyStepOutput(
            heavy_hidden_state=hidden_states,
            next_latent_state=next_state,
            cache=cache,
            latent_context=latent_context,
            proposal=stats.proposal,
            proposal_confidence=stats.proposal_confidence,
            token_entropy=stats.token_entropy,
            greedy_proposal=stats.greedy_proposal,
            greedy_confidence=stats.greedy_confidence,
        )
