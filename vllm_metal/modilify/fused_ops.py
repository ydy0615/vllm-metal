# SPDX-License-Identifier: Apache-2.0
"""Liger-style operator fusions implemented with MLX fast kernels.

CUDA ``liger_kernel`` is not used. These helpers are the Metal equivalents of
RMSNorm, GeGLU, and residual+norm fusions listed in Transformers' kernel
loading docs and vLLM's ``fusions.md``.
"""

from __future__ import annotations

import mlx.core as mx


def rms_norm(x: mx.array, weight: mx.array | None, eps: float) -> mx.array:
    """``mx.fast.rms_norm`` with optional scale."""
    return mx.fast.rms_norm(x, weight, eps)


def residual_rms_norm(
    residual: mx.array,
    hidden: mx.array,
    weight: mx.array | None,
    eps: float,
) -> mx.array:
    """Fuse residual add and RMSNorm."""
    return mx.fast.rms_norm(residual + hidden, weight, eps)


def gelu_pytorch_tanh(x: mx.array) -> mx.array:
    """Gemma ``hidden_activation = gelu_pytorch_tanh``."""
    return (
        0.5
        * x
        * (1.0 + mx.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    )


def geglu(gate: mx.array, up: mx.array) -> mx.array:
    """Fused GeGLU used by shared MLP and routed experts."""
    return gelu_pytorch_tanh(gate) * up


def apply_rope(
    x: mx.array,
    *,
    offset: int | mx.array,
    traditional: bool = False,
    base: float = 10000.0,
    scale: float = 1.0,
) -> mx.array:
    """``mx.fast.rope`` over the last two dims of a ``[B, H, T, D]`` tensor."""
    return mx.fast.rope(
        x,
        dims=x.shape[-1],
        traditional=traditional,
        base=base,
        scale=scale,
        offset=offset,
    )


def logit_softcap(logits: mx.array, cap: float) -> mx.array:
    if cap <= 0:
        return logits
    return mx.tanh(logits.astype(mx.float32) / cap) * cap
