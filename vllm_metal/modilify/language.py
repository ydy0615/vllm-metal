# SPDX-License-Identifier: Apache-2.0
"""Mk1 / ChatDLM1 wrappers around mlx-vlm DiffusionGemma trunk layers."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from vllm_metal.modilify.config import (
    LATENT_RESIDUAL_RMS_RATIO_CAP,
    ModilifyRuntimeConfig,
    ModilifyTextConfig,
)
from vllm_metal.modilify.fused_ops import geglu


def _require_diffusion_gemma():
    try:
        from mlx_vlm.models.diffusion_gemma.config import ModelConfig, TextConfig
        from mlx_vlm.models.diffusion_gemma.language import (
            DiffusionGemma4Backbone,
            Router,
        )
    except ImportError as exc:
        raise ImportError(
            "Modilify serving requires mlx-vlm with models.diffusion_gemma "
            "(mlx-vlm >= 0.6.3)."
        ) from exc
    return ModelConfig, TextConfig, DiffusionGemma4Backbone, Router


def to_trunk_text_config(text: ModilifyTextConfig):
    _, TextConfig, _, _ = _require_diffusion_gemma()
    payload = text.to_dict()
    payload["layer_types"] = list(text.layer_types) if text.layer_types else None
    return TextConfig(**{
        key: value
        for key, value in payload.items()
        if key in TextConfig.__dataclass_fields__
    })


def to_trunk_model_config(config: ModilifyRuntimeConfig):
    ModelConfig, _, _, _ = _require_diffusion_gemma()
    return ModelConfig(
        text_config=to_trunk_text_config(config.text_config),
        vision_config=None,
        model_type="diffusion_gemma",
        canvas_length=config.canvas_length,
        eos_token_id=list(config.eos_token_id),
        dtype=config.dtype,
    )


class ModilifyRouter(nn.Module):
    """Softmax over all experts, then top-k and renormalize.

    Replaces mlx-vlm's score-space top-k. Weights come from the official
    router parameters.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.config = inner.config
        self.eps = inner.eps
        self.proj = inner.proj
        self.scale = inner.scale
        self.per_expert_scale = inner.per_expert_scale
        self._root_size = inner._root_size

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        x = mx.fast.rms_norm(x, None, self.eps)
        x = x * self.scale * self._root_size
        scores = self.proj(x)
        probabilities = mx.softmax(scores, axis=-1, precise=True)
        top_k = self.config.top_k_experts
        indices = mx.argpartition(probabilities, kth=-top_k, axis=-1)[..., -top_k:]
        weights = mx.take_along_axis(probabilities, indices, axis=-1)
        weights = weights / mx.sum(weights, axis=-1, keepdims=True)
        weights = weights * self.per_expert_scale[indices]
        return indices, weights


def merge_latent_context(
    mapper: nn.Module,
    token_embeddings: mx.array,
    latent_context: mx.array | None,
    *,
    rms_ratio_cap: float = LATENT_RESIDUAL_RMS_RATIO_CAP,
) -> mx.array:
    """Native self-conditioning bridge with RMS-capped latent residual."""
    context = (
        mx.zeros_like(token_embeddings)
        if latent_context is None
        else latent_context.astype(token_embeddings.dtype)
    )
    if context.shape != token_embeddings.shape:
        raise ValueError("Latent context must match the canvas embedding shape.")
    normalized = mapper.pre_norm(context)
    mapped = mapper.down_proj(
        geglu(mapper.gate_proj(normalized), mapper.up_proj(normalized))
    )
    mapped_rms = mx.sqrt(
        mx.mean(mx.square(mapped.astype(mx.float32)), axis=-1, keepdims=True)
    )
    token_rms = mx.sqrt(
        mx.mean(
            mx.square(token_embeddings.astype(mx.float32)), axis=-1, keepdims=True
        )
    )
    cap = rms_ratio_cap * token_rms
    scale = cap / mx.sqrt(mx.square(mapped_rms) + mx.square(cap) + 1.0e-12)
    mapped = mapped * scale.astype(mapped.dtype)
    return mapper.post_norm(token_embeddings + mapped)


def _install_embed_canvas(decoder: nn.Module) -> None:
    def _embed_canvas(
        canvas_ids,
        self_conditioning_logits=None,
        self_conditioning_embeddings=None,
    ):
        if self_conditioning_logits is not None:
            raise ValueError(
                "Modilify uses latent embeddings, not logits self-conditioning."
            )
        token_embeddings = decoder.embed_tokens(canvas_ids) * decoder.embed_scale
        return merge_latent_context(
            decoder.self_conditioning,
            token_embeddings,
            self_conditioning_embeddings,
        )

    decoder._embed_canvas = _embed_canvas


def _install_routers(backbone: Any) -> None:
    for layer in backbone.decoder.layers:
        layer.router = ModilifyRouter(layer.router)


def build_modilify_backbone(config: ModilifyRuntimeConfig):
    """Construct the DiffusionGemma trunk and install Modilify forwards."""
    _, _, DiffusionGemma4Backbone, _ = _require_diffusion_gemma()
    backbone = DiffusionGemma4Backbone(to_trunk_model_config(config))
    _install_routers(backbone)
    _install_embed_canvas(backbone.decoder)
    return backbone
