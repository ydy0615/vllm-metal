# SPDX-License-Identifier: Apache-2.0
"""Fast tests for Modilify protocol helpers (no 26B weights)."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from vllm_metal.modilify.commit_policy import (
    bounded_prefix_failure_commit_lengths,
    fused_commit_failure_rate,
    prefix_failure_commit_lengths,
    select_commit_lengths,
)
from vllm_metal.modilify.config import (
    CHATDLM1_DENOISE_TEMPERATURE,
    CHATDLM1_MODEL_TYPE,
    COMMIT_FAILURE_BUDGET,
    MK1_MODEL_TYPE,
    ModilifyRuntimeConfig,
    require_compatible_chatdlm1_protocol,
)
from vllm_metal.modilify.detection import (
    is_chatdlm1_checkpoint,
    is_modilify_model,
    read_model_type,
)
from vllm_metal.modilify.fused_ops import geglu, gelu_pytorch_tanh, residual_rms_norm
from vllm_metal.modilify.latent_deliberation import LatentDeliberationState
from vllm_metal.modilify.remap import remap_state_dict, should_keep_source_key
from vllm_metal.modilify.vocab_ops import (
    chunked_vocab_statistics,
    oneshot_vocab_statistics,
)


def _to_list(array: mx.array) -> list:
    mx.eval(array)
    return np.array(array).tolist()


class TestDetection:
    def test_mk1_model_type(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "modilify_mk1"}))
        assert read_model_type(str(tmp_path)) == "modilify_mk1"
        assert is_modilify_model(str(tmp_path))
        assert not is_chatdlm1_checkpoint(str(tmp_path))

    def test_chatdlm1_lora_checkpoint(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "chatdlm1"}))
        (tmp_path / "trainable_model.safetensors").write_bytes(b"x")
        assert is_modilify_model(str(tmp_path))
        assert is_chatdlm1_checkpoint(str(tmp_path))

    def test_unknown_type(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "gemma4"}))
        assert not is_modilify_model(str(tmp_path))


class TestConfig:
    def test_chatdlm1_locks_temperature(self) -> None:
        cfg = ModilifyRuntimeConfig.from_dict(
            {
                "model_type": "chatdlm1",
                "text_config": {"vocab_size": 128, "hidden_size": 32},
            }
        )
        assert cfg.model_type == CHATDLM1_MODEL_TYPE
        assert cfg.denoise_temperature == CHATDLM1_DENOISE_TEMPERATURE
        assert cfg.temperature_locked
        assert cfg.commit_failure_budget == COMMIT_FAILURE_BUDGET
        locked = cfg.with_denoise_temperature(0.2)
        assert locked.denoise_temperature == CHATDLM1_DENOISE_TEMPERATURE

    def test_mk1_allows_temperature_override(self) -> None:
        cfg = ModilifyRuntimeConfig.from_dict(
            {
                "model_type": "modilify_mk1",
                "text_config": {"vocab_size": 128},
                "denoise_temperature": 0.8,
            }
        )
        assert cfg.model_type == MK1_MODEL_TYPE
        assert not cfg.temperature_locked
        assert cfg.with_denoise_temperature(0.4).denoise_temperature == pytest.approx(0.4)

    def test_schema18_rejected(self) -> None:
        with pytest.raises(RuntimeError, match="schema-v20"):
            require_compatible_chatdlm1_protocol(
                {
                    "state_schema_version": 18,
                    "training_scheme": (
                        "gold_prefix_random_commit_0_canvas_x0_ce_"
                        "confidence_calibration_fused_throughput_sft"
                    ),
                    "memory_scheme": "latent_transformer",
                }
            )

    def test_schema19_accepted(self) -> None:
        require_compatible_chatdlm1_protocol(
            {
                "state_schema_version": 19,
                "training_scheme": (
                    "gold_prefix_random_commit_0_canvas_x0_ce_"
                    "confidence_calibration_fused_throughput_sft"
                ),
                "memory_scheme": "latent_transformer",
            }
        )


class TestCommitPolicy:
    def test_strict_prefix_failure_sum(self) -> None:
        failure_rate = mx.array(
            [
                [0.05, 0.04, 0.10, 0.01],
                [0.01, 0.01, 0.01, 0.01],
            ],
            dtype=mx.float32,
        )
        lengths = prefix_failure_commit_lengths(failure_rate, failure_budget=0.1)
        equality = prefix_failure_commit_lengths(
            mx.array([[0.25, 0.0, 0.0, 0.0]], dtype=mx.float32),
            failure_budget=0.25,
        )
        assert _to_list(lengths) == [2, 4]
        assert _to_list(equality) == [0]

    def test_invalid_tail_cannot_extend_prefix(self) -> None:
        confidence = mx.full((2, 5), 0.99, dtype=mx.float32)
        entropy = mx.zeros((2, 5), dtype=mx.float32)
        valid = mx.array(
            [
                [True, True, False, True, True],
                [True, True, True, True, False],
            ]
        )
        failure = fused_commit_failure_rate(confidence, entropy, vocab_size=256)
        lengths = prefix_failure_commit_lengths(
            failure, failure_budget=0.1, valid_mask=valid
        )
        assert _to_list(lengths) == [2, 4]

    def test_turn_or_eos_clips_commit(self) -> None:
        proposal = mx.array([[7, 106, 1, 9], [7, 1, 106, 9]], dtype=mx.int32)
        lengths = bounded_prefix_failure_commit_lengths(
            proposal,
            mx.full((2, 4), 0.01, dtype=mx.float32),
            failure_budget=0.2,
            remaining_lengths=mx.array([4, 4], dtype=mx.int32),
            stop_token_id=(106, 1),
        )
        assert _to_list(lengths) == [2, 2]

    def test_jump_only_after_ponder_budget(self) -> None:
        proposal = mx.array([[7, 8, 9, 10]], dtype=mx.int32)
        confidence = mx.full((1, 4), 0.25, dtype=mx.float32)
        entropy = mx.zeros((1, 4), dtype=mx.float32)
        greedy = mx.array([[17, 18, 19, 106]], dtype=mx.int32)
        greedy_confidence = mx.full((1, 4), 0.95, dtype=mx.float32)
        decision = select_commit_lengths(
            proposal,
            fused_commit_failure_rate(confidence, entropy, vocab_size=256),
            fused_commit_failure_rate(
                mx.full((1, 4), 0.5, dtype=mx.float32), entropy, vocab_size=256
            ),
            greedy,
            fused_commit_failure_rate(greedy_confidence, entropy, vocab_size=256),
            ponder_steps=mx.array([64], dtype=mx.int32),
            stagnation_steps=mx.array([0], dtype=mx.int32),
            active_rows=mx.array([True]),
            remaining_lengths=mx.array([4], dtype=mx.int32),
            failure_budget=0.2,
            stop_token_id=106,
            max_ponder_steps=64,
            stagnation_threshold=12,
            min_progress=0.005,
        )
        mx.eval(decision.commit_lengths, decision.jump_rows)
        assert bool(np.array(decision.jump_rows)[0])
        assert int(np.array(decision.commit_lengths)[0]) == 4


class TestGenerateShift:
    def test_prefix_shift_is_row_local(self) -> None:
        from vllm_metal.modilify.generate import _shift_prefix

        tensor = mx.arange(8, dtype=mx.float32).reshape((2, 4))
        shifted = _shift_prefix(tensor, 1, 0)
        mx.eval(shifted)
        np.testing.assert_allclose(
            np.array(shifted),
            [[1, 2, 3, 0], [5, 6, 7, 0]],
        )


class TestLatentShift:
    def test_shift_keeps_memory_slots(self) -> None:
        state = LatentDeliberationState.empty(
            batch_size=1,
            canvas_length=4,
            latent_dim=8,
            memory_slots=3,
            dtype=mx.float32,
        )
        state.token_latents = mx.arange(32, dtype=mx.float32).reshape((1, 4, 8))
        state.memory_slots = mx.ones((1, 3, 8), dtype=mx.float32)
        state.confidence = mx.array([[0.1, 0.2, 0.3, 0.4]], dtype=mx.float32)
        shifted = state.shift(2, entropy_fill_value=9.0)
        mx.eval(shifted.token_latents, shifted.memory_slots, shifted.confidence)
        np.testing.assert_allclose(
            np.array(shifted.token_latents)[0, 0], np.array(state.token_latents)[0, 2]
        )
        np.testing.assert_allclose(np.array(shifted.memory_slots), np.array(state.memory_slots))
        np.testing.assert_allclose(
            np.array(shifted.confidence), [[0.3, 0.4, 0.0, 0.0]], rtol=1e-6
        )
        assert _to_list(shifted.ponder_steps) == [0]


class TestVocabOps:
    def test_chunked_matches_full_softmax(self) -> None:
        mx.random.seed(0)
        rows, dim, vocab = 6, 8, 32
        hidden = mx.random.normal((rows, dim)).astype(mx.float32)
        weight = mx.random.normal((vocab, dim)).astype(mx.float32)
        temperature = 1.0
        softcap = 30.0
        scores = mx.tanh((hidden @ weight.T) / softcap) * softcap
        sample = scores / temperature
        probs = mx.softmax(sample, axis=-1, precise=True)
        entropy = -mx.sum(probs * mx.log(mx.maximum(probs, 1e-30)), axis=-1)
        greedy = mx.argmax(probs, axis=-1)
        greedy_conf = mx.take_along_axis(probs, greedy[:, None], axis=-1).reshape((rows,))
        stats = chunked_vocab_statistics(
            hidden,
            weight,
            temperature=temperature,
            softcap=softcap,
            chunk_size=7,
            gumbel_noise=mx.zeros((rows, vocab), dtype=mx.float32),
        )
        mx.eval(stats.greedy_proposal, stats.token_entropy, stats.greedy_confidence)
        np.testing.assert_array_equal(np.array(stats.greedy_proposal), np.array(greedy))
        np.testing.assert_allclose(
            np.array(stats.token_entropy), np.array(entropy), rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            np.array(stats.greedy_confidence),
            np.array(greedy_conf),
            rtol=1e-4,
            atol=1e-4,
        )
        # Zero Gumbel noise → sample equals greedy.
        np.testing.assert_array_equal(
            np.array(stats.proposal), np.array(stats.greedy_proposal)
        )

    def test_oneshot_matches_chunked_greedy(self) -> None:
        mx.random.seed(1)
        rows, dim, vocab = 6, 8, 32
        hidden = mx.random.normal((rows, dim)).astype(mx.float32)
        weight = mx.random.normal((vocab, dim)).astype(mx.float32)
        chunked = chunked_vocab_statistics(
            hidden,
            weight,
            temperature=0.8,
            softcap=30.0,
            chunk_size=9,
            gumbel_noise=mx.zeros((rows, vocab), dtype=mx.float32),
        )
        mx.random.seed(1)
        oneshot = oneshot_vocab_statistics(
            hidden, weight, temperature=0.8, softcap=30.0
        )
        mx.eval(
            chunked.greedy_proposal,
            chunked.token_entropy,
            chunked.greedy_confidence,
            oneshot.greedy_proposal,
            oneshot.token_entropy,
            oneshot.greedy_confidence,
        )
        np.testing.assert_array_equal(
            np.array(chunked.greedy_proposal), np.array(oneshot.greedy_proposal)
        )
        np.testing.assert_allclose(
            np.array(chunked.token_entropy),
            np.array(oneshot.token_entropy),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            np.array(chunked.greedy_confidence),
            np.array(oneshot.greedy_confidence),
            rtol=1e-4,
            atol=1e-4,
        )


class TestRemap:
    def test_skips_encoder_language_and_vision(self) -> None:
        assert not should_keep_source_key("model.encoder.language_model.layers.0.mlp.weight")
        assert should_keep_source_key("model.encoder.language_model.layers.0.layer_scalar")
        assert not should_keep_source_key(
            "model.encoder.vision_tower.encoder.layers.0.self_attn.q_proj.weight"
        )

    def test_splits_latent_qkv_and_renames_experts(self) -> None:
        dim = 6
        weight = mx.arange(3 * dim * dim, dtype=mx.float32).reshape((3 * dim, dim))
        remapped = remap_state_dict(
            [
                ("latent_deliberation.blocks.0.local_attention.in_proj_weight", weight),
                (
                    "model.decoder.layers.0.experts.gate_up_proj",
                    mx.ones((4, 8, 8), dtype=mx.float32),
                ),
                (
                    "latent_deliberation.blocks.0.token_ff.0.weight",
                    mx.ones((8, 6), dtype=mx.float32),
                ),
            ]
        )
        assert "latent_deliberation.blocks.0.local_attention.query_proj.weight" in remapped
        assert "model.decoder.layers.0.experts.gate_up_proj.weight" in remapped
        assert "latent_deliberation.blocks.0.token_ff.layers.0.weight" in remapped


class TestAdapter:
    def test_yoco_mapping_is_none(self) -> None:
        from vllm_metal.modilify.adapter import ModilifyModelAdapter

        adapter = ModilifyModelAdapter()
        args = {
            "model_type": "modilify_mk1",
            "num_hidden_layers": 30,
            "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
            "num_kv_shared_layers": 4,
        }
        assert adapter.build_yoco_cache_mapping(args) is None
        assert adapter.should_force_text_backbone(None) is True

    def test_per_layer_kv_matches_gemma4_layout(self) -> None:
        from vllm_metal.modilify.adapter import ModilifyModelAdapter

        adapter = ModilifyModelAdapter()
        layer_types = (["sliding_attention"] * 5 + ["full_attention"]) * 5
        args = {
            "layer_types": layer_types,
            "global_head_dim": 512,
            "num_global_key_value_heads": 2,
            "sliding_window": 1024,
        }
        shapes = adapter.build_per_layer_kv_shapes(
            args, num_layers=30, num_kv_heads=8, head_dim=256
        )
        assert shapes is not None
        kv_heads, head_dims = shapes
        assert kv_heads[0] == 8 and head_dims[0] == 256
        assert kv_heads[5] == 2 and head_dims[5] == 512
        windows = adapter.build_sliding_window_per_layer(args, num_layers=30)
        assert windows is not None
        assert windows[0] == 1024
        assert windows[5] == -1


class TestLatentMerge:
    def test_rms_cap_does_not_exceed_half_token_rms(self) -> None:
        from vllm_metal.modilify.language import merge_latent_context

        class _Mapper:
            def pre_norm(self, x):
                return x

            def gate_proj(self, x):
                return x

            def up_proj(self, x):
                return mx.full(x.shape, 10.0, dtype=x.dtype)

            def down_proj(self, x):
                return x

            def post_norm(self, x):
                return x

        tokens = mx.ones((1, 4, 8), dtype=mx.float32)
        context = mx.full((1, 4, 8), 50.0, dtype=mx.float32)
        merged = merge_latent_context(_Mapper(), tokens, context, rms_ratio_cap=0.5)
        residual = merged - tokens
        mapped_rms = mx.sqrt(mx.mean(mx.square(residual), axis=-1))
        token_rms = mx.sqrt(mx.mean(mx.square(tokens), axis=-1))
        mx.eval(mapped_rms, token_rms)
        assert float(mx.max(mapped_rms / token_rms).item()) <= 0.5 + 1e-5


class TestFusedOps:
    def test_geglu_matches_eager(self) -> None:
        gate = mx.array([[0.2, -0.5, 1.0]], dtype=mx.float32)
        up = mx.array([[1.0, 2.0, 0.5]], dtype=mx.float32)
        out = geglu(gate, up)
        expected = gelu_pytorch_tanh(gate) * up
        mx.eval(out, expected)
        np.testing.assert_allclose(np.array(out), np.array(expected), rtol=1e-5)

    def test_residual_rms_norm(self) -> None:
        residual = mx.ones((2, 4), dtype=mx.float32)
        hidden = mx.full((2, 4), 0.5, dtype=mx.float32)
        weight = mx.ones((4,), dtype=mx.float32)
        out = residual_rms_norm(residual, hidden, weight, 1e-6)
        combined = residual + hidden
        mean_sq = mx.mean(mx.square(combined.astype(mx.float32)), axis=-1, keepdims=True)
        expected = combined.astype(mx.float32) * mx.rsqrt(mean_sq + 1e-6) * weight
        mx.eval(out, expected)
        np.testing.assert_allclose(np.array(out), np.array(expected), rtol=1e-5, atol=1e-5)
