# SPDX-License-Identifier: Apache-2.0
"""Prompt-prefix KV cache for Modilify encoder states.

Stores unpadded per-request encoder caches. Hits return a copy so a later
commit cannot mutate a shared prefix entry.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import mlx.core as mx


def _copy_array(value: mx.array | None) -> mx.array | None:
    if value is None:
        return None
    return value + 0


def clone_encoder_cache(cache: list[Any]) -> list[Any]:
    """Deep-copy one encoder cache list without sharing buffers."""
    cloned = []
    for layer in cache:
        replica = type(layer)(
            int(layer.max_size), int(getattr(layer, "step", 256))
        )
        if hasattr(replica, "read_only"):
            replica.read_only = False
        replica.keys = _copy_array(getattr(layer, "keys", None))
        replica.values = _copy_array(getattr(layer, "values", None))
        replica.offset = int(getattr(layer, "offset", 0) or 0)
        cloned.append(replica)
    return cloned


def prompt_block_keys(token_ids: list[int], block_size: int) -> list[tuple[int, ...]]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    keys: list[tuple[int, ...]] = []
    for end in range(block_size, len(token_ids) + 1, block_size):
        keys.append(tuple(token_ids[:end]))
    return keys


class PromptPrefixCache:
    """LRU of full-block prompt prefixes → cloned encoder caches."""

    def __init__(self, *, block_size: int = 128, max_entries: int = 64) -> None:
        if block_size <= 0 or max_entries <= 0:
            raise ValueError("block_size and max_entries must be positive")
        self.block_size = int(block_size)
        self.max_entries = int(max_entries)
        self._store: OrderedDict[tuple[int, ...], list[Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def lookup(self, token_ids: list[int]) -> tuple[int, list[Any] | None]:
        """Return ``(hit_tokens, cloned_cache)`` for the longest cached prefix."""
        best_n = 0
        best_cache: list[Any] | None = None
        for key in reversed(prompt_block_keys(token_ids, self.block_size)):
            cache = self._store.get(key)
            if cache is not None:
                self._store.move_to_end(key)
                best_n = len(key)
                best_cache = clone_encoder_cache(cache)
                break
        if best_n:
            self.hits += 1
        else:
            self.misses += 1
        return best_n, best_cache

    def store(self, token_ids: list[int], cache: list[Any]) -> None:
        """Remember the encoder cache for exactly *token_ids*."""
        if not token_ids:
            return
        key = tuple(token_ids)
        self._store[key] = clone_encoder_cache(cache)
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)
