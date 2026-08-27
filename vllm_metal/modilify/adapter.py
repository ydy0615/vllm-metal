# SPDX-License-Identifier: Apache-2.0
"""ModelAdapter hooks for Modilify (Gemma4-like per-layer KV, no YOCO)."""

from __future__ import annotations

from typing import Any

from vllm_metal.v1.model_adapter import DefaultModelAdapter


class ModilifyModelAdapter(DefaultModelAdapter):
    """Text-only Modilify serving: per-layer KV + sliding window, no YOCO."""

    def should_force_text_backbone(self, hf_config: Any) -> bool:
        return True

    def build_yoco_cache_mapping(
        self, args: dict[str, Any]
    ) -> tuple[int, dict[int, int]] | None:
        return None
