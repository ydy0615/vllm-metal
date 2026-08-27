# SPDX-License-Identifier: Apache-2.0
"""Detect Modilify Mk1 and ChatDLM1 checkpoints from config.json."""

from __future__ import annotations

import json
import logging
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:  # pragma: no cover
    hf_hub_download = None

logger = logging.getLogger(__name__)

MODILIFY_MODEL_TYPES = frozenset({"modilify_mk1", "chatdlm1"})


def _resolve_config_file(model_path: str) -> Path | None:
    path = Path(model_path)
    if path.is_dir():
        config_file = path / "config.json"
        if config_file.is_file():
            return config_file
        logger.debug("No config.json found in local model directory: %s", model_path)

    if hf_hub_download is None:
        logger.debug(
            "huggingface_hub not installed; cannot resolve remote model config: %s",
            model_path,
        )
        return None

    try:
        return Path(hf_hub_download(repo_id=model_path, filename="config.json"))
    except (OSError, ValueError) as exc:
        logger.debug("Failed to download config.json for %s: %s", model_path, exc)
        return None


def read_model_type(model_path: str) -> str | None:
    """Return ``model_type`` from ``config.json``, or None if unreadble."""
    config_file = _resolve_config_file(model_path)
    if config_file is None:
        return None
    try:
        with config_file.open() as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Failed reading model config %s: %s", config_file, exc)
        return None
    model_type = payload.get("model_type")
    if not isinstance(model_type, str):
        return None
    return model_type.lower()


def is_modilify_model(model_path: str) -> bool:
    """Return True when *model_path* is Modilify Mk1 or ChatDLM1."""
    model_type = read_model_type(model_path)
    return model_type in MODILIFY_MODEL_TYPES


def is_chatdlm1_checkpoint(model_path: str) -> bool:
    """Return True for a ChatDLM1 incremental (LoRA) checkpoint directory."""
    path = Path(model_path)
    if not path.is_dir():
        return False
    if read_model_type(model_path) != "chatdlm1":
        return False
    return (path / "trainable_model.safetensors").is_file() or (
        path / "training_state.pt"
    ).is_file()
