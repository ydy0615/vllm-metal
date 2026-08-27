# SPDX-License-Identifier: Apache-2.0
"""Tests for Modilify high-batch packing, prefix cache, and chunked prefill."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from vllm_metal.modilify.attention import _slice_prefix_kv, build_decoder_masks
from vllm_metal.modilify.continuous_batch import (
    left_pad_prefix,
    pack_encoder_caches,
    prefix_valid_mask,
    shift_row_canvas,
)
from vllm_metal.modilify.prefix_cache import PromptPrefixCache, prompt_block_keys


class _Layer:
    def __init__(self, max_size=16, step=256):
        self.max_size = max_size
        self.step = step
        self.keys = None
        self.values = None
        self.offset = 0

    @property
    def decoder_state(self):
        return self.keys, self.values


def _layer_with(length: int, fill: float, heads: int = 2, dim: int = 4) -> _Layer:
    layer = _Layer(max_size=max(length, 8))
    layer.keys = mx.full((1, heads, length, dim), fill, dtype=mx.float32)
    layer.values = mx.full((1, heads, length, dim), fill + 1, dtype=mx.float32)
    layer.offset = length
    return layer


class TestLeftPadAndPack:
    def test_left_pad_puts_real_tokens_on_the_right(self) -> None:
        keys = mx.arange(6, dtype=mx.float32).reshape((1, 1, 6, 1))
        values = keys + 10
        padded_k, padded_v = left_pad_prefix(keys, values, logical_length=3, max_length=6)
        mx.eval(padded_k, padded_v)
        np.testing.assert_allclose(
            np.array(padded_k[0, 0, :, 0]), [0, 0, 0, 0, 1, 2]
        )
        np.testing.assert_allclose(
            np.array(padded_v[0, 0, :, 0]), [0, 0, 0, 10, 11, 12]
        )

    def test_uniform_pack_skips_left_pad(self) -> None:
        left = [_layer_with(4, 1.0)]
        right = [_layer_with(4, 3.0)]
        packed, max_len = pack_encoder_caches([left, right], [4, 4])
        assert max_len == 4
        keys = np.array(packed[0].keys)
        assert keys.shape[0] == 2
        assert keys[0, 0, 0, 0] == 1.0
        assert keys[1, 0, 0, 0] == 3.0

    def test_pack_stacks_ragged_prefixes(self) -> None:
        short = [_layer_with(2, 1.0)]
        long = [_layer_with(4, 3.0)]
        packed, max_len = pack_encoder_caches([short, long], [2, 4])
        assert max_len == 4
        keys = np.array(packed[0].keys)
        assert keys.shape == (2, 2, 4, 4)
        assert keys[0, 0, 0, 0] == 0
        assert keys[0, 0, -1, 0] == 1.0
        assert keys[1, 0, 0, 0] == 3.0

    def test_valid_mask_is_right_aligned(self) -> None:
        mask = prefix_valid_mask(mx.array([2, 4], dtype=mx.int32), 4)
        mx.eval(mask)
        np.testing.assert_array_equal(
            np.array(mask),
            [[False, False, True, True], [True, True, True, True]],
        )


class TestRaggedMasks:
    def test_batched_prefix_mask_hides_left_pad(self) -> None:
        full, slide = build_decoder_masks(
            prefix_len=mx.array([2, 4], dtype=mx.int32),
            canvas_length=2,
            cache_capacity=4,
            sliding_window=1024,
            batch_size=2,
        )
        mx.eval(full, slide)
        # full[..., :4] is prefix; last 2 are canvas (always true)
        prefix = np.array(full[:, 0, 0, :4])
        np.testing.assert_array_equal(
            prefix, [[False, False, True, True], [True, True, True, True]]
        )
        assert bool(np.array(full[:, 0, 0, 4:]).all())

    def test_uniform_prefix_needs_no_mask(self) -> None:
        full, slide = build_decoder_masks(
            prefix_len=16,
            canvas_length=4,
            cache_capacity=272,
            sliding_window=1024,
        )
        assert full is None and slide is None

    def test_slice_drops_unpopulated_and_out_of_window(self) -> None:
        keys = mx.arange(8, dtype=mx.float32).reshape((1, 1, 8, 1))
        values = keys + 10
        sliced_k, sliced_v = _slice_prefix_kv(keys, values, 3, sliding_window=None)
        mx.eval(sliced_k, sliced_v)
        assert int(sliced_k.shape[2]) == 3
        np.testing.assert_allclose(np.array(sliced_k[0, 0, :, 0]), [0, 1, 2])
        windowed_k, _ = _slice_prefix_kv(keys, values, 8, sliding_window=4)
        mx.eval(windowed_k)
        # sliding_window-1 = 3 last populated keys
        np.testing.assert_allclose(np.array(windowed_k[0, 0, :, 0]), [5, 6, 7])


class TestPrefixCache:
    def test_lookup_longest_block_prefix(self) -> None:
        cache = PromptPrefixCache(block_size=2, max_entries=8)
        stored = [_layer_with(4, 5.0)]
        tokens = [1, 2, 3, 4, 5]
        cache.store(tokens[:4], stored)
        hit_n, hit = cache.lookup(tokens)
        assert hit_n == 4
        assert hit is not None
        assert int(hit[0].offset) == 4
        miss_n, miss = cache.lookup([9, 9, 9, 9])
        assert miss_n == 0
        assert miss is None

    def test_block_keys(self) -> None:
        assert prompt_block_keys([1, 2, 3, 4, 5], 2) == [(1, 2), (1, 2, 3, 4)]


class TestShift:
    def test_uniform_shift(self) -> None:
        tensor = mx.arange(8, dtype=mx.float32).reshape((2, 4))
        shifted = shift_row_canvas(tensor, 1, 0)
        mx.eval(shifted)
        np.testing.assert_allclose(
            np.array(shifted), [[1, 2, 3, 0], [5, 6, 7, 0]]
        )
