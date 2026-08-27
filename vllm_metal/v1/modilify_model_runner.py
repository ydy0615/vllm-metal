# SPDX-License-Identifier: Apache-2.0
"""Dedicated vLLM v1 runner for Modilify block diffusion."""

from __future__ import annotations

from typing import Any, Literal

import mlx.core as mx
import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput

from vllm_metal.modilify.generate import generate
from vllm_metal.modilify.loader import load_modilify, load_tokenizer
from vllm_metal.modilify.prefix_cache import PromptPrefixCache
from vllm_metal.utils import get_model_download_path

logger = init_logger(__name__)

_MODILIFY_SCHED_HEAD_SIZE = 256
_MODILIFY_SCHED_BLOCK_BYTES = 256


class ModilifyModelRunner:
    """Run rolling-canvas denoise internally; return committed tokens."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device | None = None) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device = device
        self.model: Any = None
        self.tokenizer: Any = None
        self.config: Any = None
        self._pending_output: ModelRunnerOutput | None = None
        self._prefix_cache: PromptPrefixCache | None = None
        self._prefill_chunk_size = 2048

    def load_model(self) -> None:
        model_name = get_model_download_path(self.model_config.model)
        logger.info("Loading Modilify model from %s", model_name)
        self.model, self.config = load_modilify(model_name)
        self.tokenizer = load_tokenizer(model_name)
        self._prefix_cache = PromptPrefixCache(block_size=128, max_entries=64)
        logger.info(
            "Modilify loaded model_type=%s canvas=%d",
            self.config.model_type,
            self.config.canvas_length,
        )

    def warm_up(self) -> None:
        if self.model is None:
            logger.warning("Modilify model not loaded, skipping warm-up")
            return
        logger.info("Warming up Modilify (one dummy prefill)")
        dummy = mx.array([[self.config.bos_token_id]], dtype=mx.int32)
        cache = self.model.make_cache(max_size=32)
        cache = self.model.prefill(dummy, cache=cache)
        mx.eval([item for block in cache for item in getattr(block, "state", ())])
        logger.info("Modilify warm-up complete")

    def supported_worker_tasks(self) -> tuple[SupportedTask, ...]:
        return ("generate",)

    def validate_paged_attention_support(self) -> None:
        return None

    def scheduler_memory_reporting_mode(
        self, *, paged_attention_enabled: bool
    ) -> Literal["modilify_internal"]:
        return "modilify_internal"

    def add_lora(self, lora_request: Any) -> bool:
        return False

    def remove_lora(self, lora_id: int) -> bool:
        return False

    def pin_lora(self, lora_id: int) -> bool:
        return False

    def list_loras(self) -> set[int]:
        return set()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return {
            "layers.0.self_attn": FullAttentionSpec(
                block_size=self.cache_config.block_size,
                num_kv_heads=int(self.config.text_config.num_key_value_heads)
                if self.config is not None
                else 8,
                head_size=_MODILIFY_SCHED_HEAD_SIZE,
                dtype=torch.bfloat16,
            ),
        }

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        logger.info(
            "KV cache config received: %d blocks (Modilify owns prefix KV internally)",
            kv_cache_config.num_blocks,
        )

    def get_cache_block_size_bytes(self) -> int:
        return _MODILIFY_SCHED_BLOCK_BYTES

    def reset_mm_cache(self) -> None:
        return None

    def reset_encoder_cache(self) -> None:
        return None

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return None

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | None:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return self._execute(scheduler_output)

    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> ModelRunnerOutput | None:
        output = self._pending_output
        self._pending_output = None
        return output

    def _execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput | None:
        req_ids: list[str] = []
        req_id_to_index: dict[str, int] = {}
        sampled_tokens: list[list[int]] = []

        new_reqs = list(scheduler_output.scheduled_new_reqs)
        if new_reqs:
            sampling_params = new_reqs[0].sampling_params or SamplingParams()
            max_new = int(sampling_params.max_tokens or 256)
            temperature = None
            if not self.config.temperature_locked:
                temperature = float(sampling_params.temperature)
            prompt_rows = [list(req.prompt_token_ids) for req in new_reqs]
            max_len = max(len(row) for row in prompt_rows)
            pad_id = int(self.config.pad_token_id)
            padded = [
                [pad_id] * (max_len - len(row)) + row for row in prompt_rows
            ]
            mask = [
                [0] * (max_len - len(row)) + [1] * len(row) for row in prompt_rows
            ]
            packed_kwargs: dict[str, Any] = {}
            if len(new_reqs) > 1:
                packed_kwargs["prefix_cache"] = self._prefix_cache
                packed_kwargs["chunk_size"] = self._prefill_chunk_size
            elif any(len(row) > self._prefill_chunk_size for row in prompt_rows):
                packed_kwargs["chunk_size"] = self._prefill_chunk_size
            result = generate(
                self.model,
                mx.array(padded, dtype=mx.int32),
                max_new_tokens=max_new,
                temperature=temperature,
                attention_mask=mx.array(mask, dtype=mx.int32),
                **packed_kwargs,
            )
            for req, tokens in zip(new_reqs, result.generated_ids):
                if not tokens:
                    tokens = [int(self.config.turn_end_token_id)]
                req_ids.append(req.req_id)
                req_id_to_index[req.req_id] = len(req_ids) - 1
                sampled_tokens.append(tokens)

        for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
            req_ids.append(req_id)
            req_id_to_index[req_id] = len(req_ids) - 1
            sampled_tokens.append([int(self.config.turn_end_token_id)])

        if not req_ids:
            return ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                sampled_token_ids=[],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            )

        self._pending_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled_tokens,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[None] * len(req_ids),
        )
        return None
