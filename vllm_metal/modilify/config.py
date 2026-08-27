# SPDX-License-Identifier: Apache-2.0
"""Unified inference configuration for Modilify Mk1 and ChatDLM1."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any

MK1_MODEL_TYPE = "modilify_mk1"
CHATDLM1_MODEL_TYPE = "chatdlm1"
CHATDLM1_COMPATIBLE_SCHEMAS = frozenset({19, 20})
CHATDLM1_STATE_SCHEMA_VERSION = 20
CHATDLM1_MEMORY_SCHEME = "latent_transformer"
CHATDLM1_TRAINING_SCHEME = (
    "gold_prefix_random_commit_0_canvas_x0_ce_confidence_calibration_fused_throughput_sft"
)
CHATDLM1_LEGACY_TRAINING_SCHEMES = frozenset(
    {
        "gold_prefix_random_commit_0_20_x0_ce_confidence_calibration_fused_throughput_sft"
    }
)
CHATDLM1_COMPATIBLE_TRAINING_SCHEMES = frozenset(
    {CHATDLM1_TRAINING_SCHEME, *CHATDLM1_LEGACY_TRAINING_SCHEMES}
)

COMMIT_FAILURE_BUDGET = 0.2
JUMP_FAILURE_BUDGET = 2.0
CHATDLM1_DENOISE_TEMPERATURE = 1.0
MK1_DENOISE_TEMPERATURE = 0.8
VOCAB_CHUNK_SIZE = 262_144
LATENT_RESIDUAL_RMS_RATIO_CAP = 0.5
CANVAS_LENGTH = 256


@dataclass(frozen=True, slots=True)
class ModilifyTextConfig:
    """Decoder/encoder text trunk fields needed at runtime."""

    vocab_size: int = 262_144
    hidden_size: int = 2816
    intermediate_size: int = 2112
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 256
    max_position_embeddings: int = 262_144
    sliding_window: int = 1024
    num_global_key_value_heads: int = 2
    global_head_dim: int = 512
    num_experts: int = 128
    top_k_experts: int = 8
    moe_intermediate_size: int = 704
    rms_norm_eps: float = 1e-6
    final_logit_softcapping: float = 30.0
    hidden_activation: str = "gelu_pytorch_tanh"
    tie_word_embeddings: bool = True
    layer_types: tuple[str, ...] = field(default_factory=tuple)
    rope_parameters: dict[str, Any] = field(default_factory=dict)
    attention_bias: bool = False
    use_bidirectional_attention: str | None = "vision"
    model_type: str = "modilify_mk1_text"

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> ModilifyTextConfig:
        if not payload:
            return cls()
        known = {key: value for key, value in payload.items() if key in cls.__dataclass_fields__}
        layer_types = known.get("layer_types")
        if isinstance(layer_types, list):
            known["layer_types"] = tuple(str(item) for item in layer_types)
        return cls(**known)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            key: getattr(self, key) for key in self.__dataclass_fields__
        }
        payload["layer_types"] = list(self.layer_types)
        return payload


@dataclass(frozen=True, slots=True)
class ModilifyRuntimeConfig:
    """Serving-time Modilify configuration. ChatDLM1 locks temperature at 1.0."""

    model_type: str
    text_config: ModilifyTextConfig
    canvas_length: int = CANVAS_LENGTH
    denoise_temperature: float = MK1_DENOISE_TEMPERATURE
    temperature_locked: bool = False
    commit_failure_budget: float = COMMIT_FAILURE_BUDGET
    jump_failure_budget: float = JUMP_FAILURE_BUDGET
    jump_on_no_progress_after: int = 12
    max_ponder_steps: int = 64
    min_trajectory_progress: float = 0.005
    latent_dim: int = 1536
    latent_dropout: float = 0.0
    latent_local_attention_window: int = 128
    latent_memory_slots: int = 64
    latent_num_heads: int = 16
    latent_num_layers: int = 4
    turn_end_token_id: int = 106
    eos_token_id: tuple[int, ...] = (1, 106)
    pad_token_id: int = 0
    bos_token_id: int = 2
    repetition_penalty: float = 1.0
    vocab_chunk_size: int = VOCAB_CHUNK_SIZE
    state_schema_version: int | None = None
    training_scheme: str | None = None
    memory_scheme: str | None = None
    skip_vision: bool = True
    dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if self.model_type not in {MK1_MODEL_TYPE, CHATDLM1_MODEL_TYPE}:
            raise ValueError(f"Unsupported Modilify model_type={self.model_type!r}")
        if self.canvas_length <= 0:
            raise ValueError("canvas_length must be positive")
        if self.commit_failure_budget <= 0 or self.jump_failure_budget <= 0:
            raise ValueError("Failure budgets must be positive")
        if self.denoise_temperature <= 0:
            raise ValueError("denoise_temperature must be positive")

    @property
    def vocab_size(self) -> int:
        return int(self.text_config.vocab_size)

    @property
    def hidden_size(self) -> int:
        return int(self.text_config.hidden_size)

    @property
    def stop_token_ids(self) -> tuple[int, ...]:
        ids = (int(self.turn_end_token_id), *tuple(int(v) for v in self.eos_token_id))
        return tuple(dict.fromkeys(ids))

    def with_denoise_temperature(self, temperature: float | None) -> ModilifyRuntimeConfig:
        """Apply a client temperature unless ChatDLM1 has locked it."""
        if temperature is None or self.temperature_locked:
            return self
        if temperature <= 0:
            raise ValueError("denoise_temperature must be positive")
        return replace(self, denoise_temperature=float(temperature))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModilifyRuntimeConfig:
        model_type = str(payload.get("model_type", "")).lower()
        text = ModilifyTextConfig.from_dict(payload.get("text_config"))
        eos = payload.get("eos_token_id", (1, 106))
        if isinstance(eos, int):
            eos_ids = (int(eos),)
        else:
            eos_ids = tuple(int(value) for value in eos)
        if model_type == CHATDLM1_MODEL_TYPE:
            return cls(
                model_type=CHATDLM1_MODEL_TYPE,
                text_config=text,
                canvas_length=int(payload.get("canvas_length", CANVAS_LENGTH)),
                denoise_temperature=CHATDLM1_DENOISE_TEMPERATURE,
                temperature_locked=True,
                commit_failure_budget=COMMIT_FAILURE_BUDGET,
                jump_failure_budget=JUMP_FAILURE_BUDGET,
                jump_on_no_progress_after=int(
                    payload.get("jump_on_no_progress_after", 12)
                ),
                max_ponder_steps=int(payload.get("max_ponder_steps", 64)),
                min_trajectory_progress=float(
                    payload.get("min_trajectory_progress", 0.005)
                ),
                latent_dim=int(payload.get("latent_dim", 1536)),
                latent_dropout=float(payload.get("latent_dropout", 0.0)),
                latent_local_attention_window=int(
                    payload.get("latent_local_attention_window", 128)
                ),
                latent_memory_slots=int(payload.get("latent_memory_slots", 64)),
                latent_num_heads=int(payload.get("latent_num_heads", 16)),
                latent_num_layers=int(payload.get("latent_num_layers", 4)),
                turn_end_token_id=int(payload.get("turn_end_token_id", 106)),
                eos_token_id=eos_ids,
                pad_token_id=int(payload.get("pad_token_id", 0)),
                bos_token_id=int(payload.get("bos_token_id", 2)),
                repetition_penalty=float(payload.get("repetition_penalty", 1.0)),
                vocab_chunk_size=int(payload.get("vocab_chunk_size", VOCAB_CHUNK_SIZE)),
                state_schema_version=payload.get("state_schema_version"),
                training_scheme=payload.get("training_scheme"),
                memory_scheme=payload.get("memory_scheme"),
                skip_vision=True,
                dtype=str(payload.get("dtype", "bfloat16")),
            )
        if model_type != MK1_MODEL_TYPE:
            raise ValueError(f"Expected Modilify model_type, got {model_type!r}")
        return cls(
            model_type=MK1_MODEL_TYPE,
            text_config=text,
            canvas_length=int(payload.get("canvas_length", CANVAS_LENGTH)),
            denoise_temperature=float(
                payload.get("denoise_temperature", MK1_DENOISE_TEMPERATURE)
            ),
            temperature_locked=False,
            commit_failure_budget=float(
                payload.get("commit_failure_budget", COMMIT_FAILURE_BUDGET)
            ),
            jump_failure_budget=float(
                payload.get("jump_failure_budget", JUMP_FAILURE_BUDGET)
            ),
            jump_on_no_progress_after=int(payload.get("jump_on_no_progress_after", 12)),
            max_ponder_steps=int(payload.get("max_ponder_steps", 64)),
            min_trajectory_progress=float(
                payload.get("min_trajectory_progress", 0.005)
            ),
            latent_dim=int(payload.get("latent_dim", 1536)),
            latent_dropout=float(payload.get("latent_dropout", 0.0)),
            latent_local_attention_window=int(
                payload.get("latent_local_attention_window", 128)
            ),
            latent_memory_slots=int(payload.get("latent_memory_slots", 64)),
            latent_num_heads=int(payload.get("latent_num_heads", 16)),
            latent_num_layers=int(payload.get("latent_num_layers", 4)),
            turn_end_token_id=int(payload.get("turn_end_token_id", 106)),
            eos_token_id=eos_ids,
            pad_token_id=int(payload.get("pad_token_id", 0)),
            bos_token_id=int(payload.get("bos_token_id", 2)),
            repetition_penalty=float(payload.get("repetition_penalty", 1.0)),
            vocab_chunk_size=VOCAB_CHUNK_SIZE,
            skip_vision=True,
            dtype=str(payload.get("dtype", "bfloat16")),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> ModilifyRuntimeConfig:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def require_compatible_chatdlm1_protocol(metadata: dict[str, Any]) -> None:
    """Accept schema 19/20 latent-bridge ChatDLM1 checkpoints; reject older."""
    schema = metadata.get("state_schema_version")
    training_scheme = metadata.get("training_scheme")
    memory_scheme = metadata.get("memory_scheme")
    if (
        schema not in CHATDLM1_COMPATIBLE_SCHEMAS
        or training_scheme not in CHATDLM1_COMPATIBLE_TRAINING_SCHEMES
        or memory_scheme != CHATDLM1_MEMORY_SCHEME
    ):
        raise RuntimeError(
            "ChatDLM1 checkpoint protocol does not match schema-v20 "
            "latent-bridge layout."
        )
