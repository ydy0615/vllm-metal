# SPDX-License-Identifier: Apache-2.0
"""Load official Mk1 shards or ChatDLM1 LoRA checkpoints at full precision."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from vllm_metal.modilify.config import (
    CHATDLM1_MODEL_TYPE,
    MK1_MODEL_TYPE,
    ModilifyRuntimeConfig,
    require_compatible_chatdlm1_protocol,
)
from vllm_metal.modilify.detection import is_chatdlm1_checkpoint, read_model_type
from vllm_metal.modilify.modeling import ModilifyForBlockDiffusion
from vllm_metal.modilify.remap import remap_state_dict

try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except ImportError:  # standalone benches without the vLLM plugin
    import logging

    logger = logging.getLogger(__name__)


def _load_weight_files(model_path: Path) -> dict[str, mx.array]:
    weight_files = sorted(model_path.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")
    weights: dict[str, mx.array] = {}
    for weight_file in weight_files:
        if weight_file.name == "trainable_model.safetensors":
            continue
        weights.update(mx.load(str(weight_file)))
    return weights


def _merge_lora_into_linear(
    weight: mx.array, lora_a: mx.array, lora_b: mx.array, *, scale: float
) -> mx.array:
    """W <- W + scale * (B @ A). PyTorch LoRALinear stores A [r, in], B [out, r]."""
    delta = (lora_b @ lora_a) * scale
    return weight + delta.astype(weight.dtype)


def _apply_chatdlm1_trainable(
    model: ModilifyForBlockDiffusion,
    trainable: dict[str, mx.array],
    *,
    lora_alpha: float,
    lora_r: int,
    expert_alpha: float,
    expert_r: int,
) -> None:
    """Merge LoRA into trunk linears and overlay latent-planner tensors."""
    params = dict(tree_flatten(model.parameters()))
    scale = float(lora_alpha) / max(int(lora_r), 1)
    expert_scale = float(expert_alpha) / max(int(expert_r), 1)
    pending_lora: dict[str, dict[str, mx.array]] = {}
    overlay: dict[str, mx.array] = {}

    for name, tensor in trainable.items():
        mlx_name = name
        if mlx_name.startswith("model.decoder.") or mlx_name.startswith(
            "latent_deliberation."
        ):
            pass
        if ".lora_a" in name or name.endswith(".lora_a"):
            base = name[: name.rfind(".lora_a")]
            pending_lora.setdefault(base, {})["a"] = tensor
            continue
        if ".lora_b" in name or name.endswith(".lora_b"):
            base = name[: name.rfind(".lora_b")]
            pending_lora.setdefault(base, {})["b"] = tensor
            continue
        if name.endswith("lora_gate_up_a"):
            pending_lora.setdefault(name[: -len("lora_gate_up_a")] + "experts.gate_up_proj", {})[
                "a"
            ] = tensor
            pending_lora[name[: -len("lora_gate_up_a")] + "experts.gate_up_proj"][
                "expert"
            ] = True
            continue
        if name.endswith("lora_gate_up_b"):
            pending_lora.setdefault(name[: -len("lora_gate_up_b")] + "experts.gate_up_proj", {})[
                "b"
            ] = tensor
            pending_lora[name[: -len("lora_gate_up_b")] + "experts.gate_up_proj"][
                "expert"
            ] = True
            continue
        if name.endswith("lora_down_a"):
            pending_lora.setdefault(name[: -len("lora_down_a")] + "experts.down_proj", {})[
                "a"
            ] = tensor
            pending_lora[name[: -len("lora_down_a")] + "experts.down_proj"]["expert"] = True
            continue
        if name.endswith("lora_down_b"):
            pending_lora.setdefault(name[: -len("lora_down_b")] + "experts.down_proj", {})[
                "b"
            ] = tensor
            pending_lora[name[: -len("lora_down_b")] + "experts.down_proj"]["expert"] = True
            continue
        if name.startswith("latent_deliberation."):
            overlay[name] = tensor

    updates: list[tuple[str, mx.array]] = []
    for base, pair in pending_lora.items():
        weight_key = base + ".weight"
        if weight_key not in params or "a" not in pair or "b" not in pair:
            logger.debug("Skipping unmatched LoRA tensors for %s", base)
            continue
        applied_scale = expert_scale if pair.get("expert") else scale
        updates.append(
            (
                weight_key,
                _merge_lora_into_linear(
                    params[weight_key], pair["a"], pair["b"], scale=applied_scale
                ),
            )
        )
    if overlay:
        remapped = remap_state_dict(overlay.items(), skip_vision=True)
        updates.extend(remapped.items())
    if updates:
        model.load_weights(updates, strict=False)


def load_modilify(
    model_path: str | Path,
    *,
    base_model_override: str | Path | None = None,
) -> tuple[ModilifyForBlockDiffusion, ModilifyRuntimeConfig]:
    """Load a Modilify Mk1 or ChatDLM1 checkpoint. Full precision only."""
    model_path = Path(model_path)
    model_type = read_model_type(str(model_path))
    if model_type not in {MK1_MODEL_TYPE, CHATDLM1_MODEL_TYPE}:
        raise ValueError(f"Not a Modilify checkpoint: model_type={model_type!r}")

    if is_chatdlm1_checkpoint(str(model_path)):
        return _load_chatdlm1(model_path, base_model_override=base_model_override)

    config = ModilifyRuntimeConfig.from_json(model_path / "config.json")
    model = ModilifyForBlockDiffusion(config)
    weights = _load_weight_files(model_path)
    remapped = remap_state_dict(weights.items(), skip_vision=config.skip_vision)
    del weights
    logger.info("Loading %d remapped Modilify tensors", len(remapped))
    model.load_weights(list(remapped.items()), strict=False)
    del remapped
    mx.eval(model.parameters())
    return model, config


def _load_chatdlm1(
    checkpoint: Path,
    *,
    base_model_override: str | Path | None,
) -> tuple[ModilifyForBlockDiffusion, ModilifyRuntimeConfig]:
    import torch

    metadata = torch.load(
        checkpoint / "training_state.pt", map_location="cpu", weights_only=False
    )
    require_compatible_chatdlm1_protocol(metadata)
    base_model = base_model_override or metadata.get("base_model")
    if not base_model:
        raise ValueError("ChatDLM1 checkpoint has no base_model; pass an override.")
    config = ModilifyRuntimeConfig.from_json(checkpoint / "config.json")
    if config.model_type != CHATDLM1_MODEL_TYPE:
        raise ValueError("ChatDLM1 loader requires model_type=chatdlm1")

    base_path = Path(str(base_model))
    model = ModilifyForBlockDiffusion(config)
    weights = _load_weight_files(base_path)
    remapped = remap_state_dict(weights.items(), skip_vision=True)
    del weights
    model.load_weights(list(remapped.items()), strict=False)
    del remapped

    trainable_path = checkpoint / "trainable_model.safetensors"
    if not trainable_path.is_file():
        raise FileNotFoundError(f"Missing {trainable_path}")
    trainable = mx.load(str(trainable_path))
    lora_payload = metadata.get("lora_config") or {}
    _apply_chatdlm1_trainable(
        model,
        trainable,
        lora_alpha=float(lora_payload.get("alpha", 16)),
        lora_r=int(lora_payload.get("r", 16)),
        expert_alpha=float(lora_payload.get("expert_alpha", 8)),
        expert_r=int(lora_payload.get("expert_r", 8)),
    )
    mx.eval(model.parameters())
    logger.info("Loaded ChatDLM1 checkpoint %s onto base %s", checkpoint, base_path)
    return model, config


def load_tokenizer(model_path: str | Path):
    from transformers import AutoTokenizer

    path = Path(model_path)
    processor = path / "processor"
    source = processor if processor.is_dir() else path
    return AutoTokenizer.from_pretrained(str(source), trust_remote_code=True)
