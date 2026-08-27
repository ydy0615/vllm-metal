# SPDX-License-Identifier: Apache-2.0
"""PyTorch Mk1 safetensors → MLX parameter names.

Ported from ``Modilify-Mk1-MLX/modilify_mlx/convert_utils.py``.
"""

from __future__ import annotations

from collections.abc import Iterable

import mlx.core as mx

_ATTENTION_MODULES = (
    "local_attention",
    "token_memory_attention",
    "memory_token_attention",
)

_SKIP_SUBSTRINGS = (
    "rotary_emb",
    "lm_head.weight",
)

_CLIP_MARKERS = ("input_max", "input_min", "output_max", "output_min")


def should_keep_source_key(key: str, *, skip_vision: bool = True) -> bool:
    if any(marker in key for marker in _SKIP_SUBSTRINGS):
        return False
    if key.startswith("model.encoder.language_model.") and not key.endswith(
        ".layer_scalar"
    ):
        return False
    if key.startswith("model.encoder.vision_tower.") or key.startswith(
        "model.encoder.embed_vision."
    ):
        if skip_vision:
            return False
        if any(marker in key for marker in _CLIP_MARKERS):
            return False
    return True


def _split_qkv(prefix: str, value: mx.array) -> list[tuple[str, mx.array]]:
    if value.ndim == 1:
        width = value.shape[0]
        if width % 3:
            raise ValueError(f"Cannot split QKV bias for {prefix}: shape {value.shape}")
        head = width // 3
        pieces = (value[:head], value[head : 2 * head], value[2 * head :])
        names = ("query_proj.bias", "key_proj.bias", "value_proj.bias")
    elif value.ndim == 2:
        width = value.shape[0]
        if width % 3:
            raise ValueError(
                f"Cannot split QKV weight for {prefix}: shape {value.shape}"
            )
        head = width // 3
        pieces = (value[:head], value[head : 2 * head], value[2 * head :])
        names = ("query_proj.weight", "key_proj.weight", "value_proj.weight")
        if pieces[0].shape[0] != pieces[0].shape[1]:
            raise ValueError(
                f"Split QKV weight for {prefix} is not square: {pieces[0].shape}"
            )
    else:
        raise ValueError(f"Unexpected QKV tensor rank for {prefix}: {value.shape}")
    return [(f"{prefix}.{name}", piece) for name, piece in zip(names, pieces)]


def remap_weight(
    key: str, value: mx.array, *, skip_vision: bool = True
) -> list[tuple[str, mx.array]]:
    if not should_keep_source_key(key, skip_vision=skip_vision):
        return []

    if key.endswith(".experts.down_proj"):
        return [(key + ".weight", value)]
    if key.endswith(".experts.gate_up_proj"):
        return [(key + ".weight", value)]

    for module in _ATTENTION_MODULES:
        in_proj_weight = f".{module}.in_proj_weight"
        in_proj_bias = f".{module}.in_proj_bias"
        if key.endswith(in_proj_weight):
            prefix = key[: -len(".in_proj_weight")]
            return _split_qkv(prefix, value)
        if key.endswith(in_proj_bias):
            prefix = key[: -len(".in_proj_bias")]
            return _split_qkv(prefix, value)

    replacements = (
        (".token_ff.0.", ".token_ff.layers.0."),
        (".token_ff.2.", ".token_ff.layers.2."),
        (".memory_ff.0.", ".memory_ff.layers.0."),
        (".memory_ff.2.", ".memory_ff.layers.2."),
    )
    for old, new in replacements:
        if old in key:
            return [(key.replace(old, new), value)]
    return [(key, value)]


def remap_state_dict(
    source: Iterable[tuple[str, mx.array]],
    *,
    skip_vision: bool = True,
) -> dict[str, mx.array]:
    remapped: dict[str, mx.array] = {}
    for key, value in source:
        for new_key, new_value in remap_weight(key, value, skip_vision=skip_vision):
            if new_key in remapped:
                raise ValueError(f"Duplicate remapped key: {new_key}")
            remapped[new_key] = new_value
    return remapped
