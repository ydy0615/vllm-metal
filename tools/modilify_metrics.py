#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Modilify speed metrics: denoise/s, TTFD, batch-8, prefix cache, chunked prefill, CB.

Primary numbers:
  * denoise/s  — steady heavy-denoise throughput after the first step
  * TTFD       — prefill + first denoise (time to first denoise)

Secondary:
  * prefix-cache TTFD hit vs miss
  * chunked vs one-shot prefill on a long prompt
  * packed ragged-batch denoise vs sequential
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROMPT = "Explain why the sky is blue."


def _prompt_ids(model_path: Path, prompt: str) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], (list, tuple)):
        token_ids = token_ids[0]
    return [int(t) for t in token_ids]


def _eval_cache(cache) -> None:
    items = []
    for block in cache:
        state = getattr(block, "state", None)
        if state is None:
            continue
        items.extend([x for x in state if x is not None])
    if items:
        mx.eval(*items)


def _eval_step(output) -> None:
    mx.eval(
        output.proposal,
        output.heavy_hidden_state,
        output.proposal_confidence,
        output.token_entropy,
    )


def _load_model(model_path: Path):
    from vllm_metal.modilify.config import ModilifyRuntimeConfig
    from vllm_metal.modilify.loader import _load_weight_files
    from vllm_metal.modilify.modeling import ModilifyForBlockDiffusion
    from vllm_metal.modilify.remap import remap_state_dict

    print("[metrics] building graph", flush=True)
    config = ModilifyRuntimeConfig.from_json(model_path / "config.json")
    model = ModilifyForBlockDiffusion(config)
    print("[metrics] reading shards", flush=True)
    weights = _load_weight_files(model_path)
    remapped = remap_state_dict(weights.items(), skip_vision=True)
    del weights
    model.load_weights(list(remapped.items()), strict=False)
    del remapped
    mx.eval(tree_flatten(model.parameters()))
    return model, config


def _empty_state(model, batch: int):
    from vllm_metal.modilify.latent_deliberation import LatentDeliberationState

    config = model.config
    dtype = model.model.decoder.embed_tokens.weight.dtype
    canvas = int(config.canvas_length)
    vocab = int(config.vocab_size)
    unknown = math.log(vocab)
    latent = LatentDeliberationState.empty(
        batch_size=batch,
        canvas_length=canvas,
        latent_dim=config.latent_dim,
        memory_slots=config.latent_memory_slots,
        dtype=dtype,
    )
    return {
        "canvas": mx.random.randint(0, vocab, (batch, canvas)),
        "confidence": mx.zeros((batch, canvas), dtype=mx.float32),
        "entropy": mx.full((batch, canvas), unknown, dtype=mx.float32),
        "age": mx.zeros((batch, canvas), dtype=mx.int32),
        "latent": latent,
        "history": None,
        "unknown": unknown,
        "vocab": vocab,
        "canvas_len": canvas,
        "temp": float(config.denoise_temperature),
    }


def _cache_capacity(cache) -> int:
    from vllm_metal.modilify.attention import cache_meta

    return int(cache_meta(cache)[1])


def _denoise(model, cache, state, prefix_len: int):
    output = model(
        decoder_input_ids=state["canvas"],
        cache=cache,
        previous_confidence=state["confidence"],
        previous_entropy=state["entropy"],
        token_age=state["age"],
        latent_state=state["latent"],
        history_hidden_state=state["history"],
        denoise_temperature=state["temp"],
        prefix_len=prefix_len,
        cache_capacity=_cache_capacity(cache),
    )
    _eval_step(output)
    state["canvas"] = output.proposal
    state["confidence"] = output.proposal_confidence.astype(mx.float32)
    state["entropy"] = output.token_entropy.astype(mx.float32)
    state["age"] = state["age"] + 1
    state["latent"] = output.next_latent_state
    state["history"] = output.heavy_hidden_state
    return output


def _report(name: str, payload: dict) -> None:
    print(f"\n=== {name} ===", flush=True)
    for key, value in payload.items():
        if isinstance(value, float):
            print(f"  {key:24s} {value:.4f}", flush=True)
        else:
            print(f"  {key:24s} {value}", flush=True)


def measure_ttfd_and_decode(
    model,
    input_ids: mx.array,
    *,
    n_steady: int,
    label: str,
) -> dict:
    """Warm kernels, then time TTFD and steady denoise/s on *input_ids*."""
    batch, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
    # Same-shape warmup so TTFD is steady serving, not first-time JIT.
    wcache = model.make_cache(max_size=seq + 256)
    wcache = model.prefill(input_ids, cache=wcache)
    _eval_cache(wcache)
    wstate = _empty_state(model, batch)
    _denoise(model, wcache, wstate, seq)

    cache = model.make_cache(max_size=seq + 256)
    t0 = time.perf_counter()
    cache = model.prefill(input_ids, cache=cache)
    _eval_cache(cache)
    prefill_s = time.perf_counter() - t0

    state = _empty_state(model, batch)
    t1 = time.perf_counter()
    _denoise(model, cache, state, seq)
    first_s = time.perf_counter() - t1
    ttfd = prefill_s + first_s

    t2 = time.perf_counter()
    for _ in range(n_steady):
        _denoise(model, cache, state, seq)
    steady_s = time.perf_counter() - t2
    dps = n_steady / steady_s if steady_s > 0 else 0.0
    payload = {
        "batch": batch,
        "prompt_tokens": seq,
        "prefill_s": prefill_s,
        "first_denoise_s": first_s,
        "ttfd_s": ttfd,
        "n_steady": n_steady,
        "steady_s": steady_s,
        "denoise_per_sec": dps,
        "row_denoise_per_sec": dps * batch,
        "ms_per_denoise": 1000.0 / dps if dps else 0.0,
        "canvas_tokens": batch * int(model.config.canvas_length),
    }
    _report(label, payload)
    return payload


def measure_denoise_breakdown(model, input_ids: mx.array) -> dict:
    """Split one heavy denoise into latent / decoder / vocab wall times."""
    from vllm_metal.modilify.attention import (
        build_decoder_masks,
        cache_meta,
        decoder_hidden_states,
    )
    from vllm_metal.modilify.vocab_ops import canvas_vocab_statistics

    seq = int(input_ids.shape[1])
    cache = model.make_cache(max_size=seq + 8)
    cache = model.prefill(input_ids, cache=cache)
    _eval_cache(cache)
    state = _empty_state(model, int(input_ids.shape[0]))
    # Prime graphs.
    _denoise(model, cache, dict(state), seq)

    dtype = model.model.decoder.embed_tokens.weight.dtype
    t0 = time.perf_counter()
    latent_context, next_state = model._prepare_latent_context(
        state["canvas"],
        history_hidden_state=state["history"],
        confidence=state["confidence"],
        entropy=state["entropy"],
        age=state["age"],
        latent_state=state["latent"],
        dtype=dtype,
    )
    mx.eval(latent_context)
    latent_s = time.perf_counter() - t0

    prefix_len, capacity = cache_meta(cache)
    full_mask, slide_mask = build_decoder_masks(
        prefix_len=int(prefix_len),
        canvas_length=int(state["canvas"].shape[1]),
        cache_capacity=int(capacity),
        sliding_window=int(model.config.text_config.sliding_window),
        batch_size=int(state["canvas"].shape[0]),
    )
    t1 = time.perf_counter()
    hidden = decoder_hidden_states(
        model.model.decoder,
        state["canvas"],
        latent_context,
        cache,
        int(prefix_len),
        full_mask,
        slide_mask,
    )
    mx.eval(hidden)
    decoder_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    stats = canvas_vocab_statistics(
        hidden,
        model.model.decoder.embed_tokens.weight,
        temperature=state["temp"],
        softcap=model.final_logit_softcapping,
        chunk_size=model.config.vocab_chunk_size,
    )
    mx.eval(stats.proposal, stats.proposal_confidence, stats.token_entropy)
    vocab_s = time.perf_counter() - t2
    total = latent_s + decoder_s + vocab_s
    payload = {
        "latent_s": latent_s,
        "decoder_s": decoder_s,
        "vocab_s": vocab_s,
        "total_s": total,
        "latent_frac": latent_s / total if total else 0.0,
        "decoder_frac": decoder_s / total if total else 0.0,
        "vocab_frac": vocab_s / total if total else 0.0,
    }
    _report("denoise breakdown B=1", payload)
    return payload


def measure_prefix_cache(model, prompt_ids: list[int], *, n_steady: int) -> dict:
    from vllm_metal.modilify.continuous_batch import prefill_with_prefix_cache
    from vllm_metal.modilify.prefix_cache import PromptPrefixCache

    ids = mx.array([prompt_ids], dtype=mx.int32)
    # Populate cache (not timed as the hit path).
    store = PromptPrefixCache(block_size=max(len(prompt_ids), 1), max_entries=8)
    cache, _ = prefill_with_prefix_cache(
        model,
        prompt_ids,
        prefix_cache=store,
        chunk_size=None,
        max_size=len(prompt_ids) + 256,
    )
    _eval_cache(cache)

    t0 = time.perf_counter()
    hit_cache, hit_n = prefill_with_prefix_cache(
        model,
        prompt_ids,
        prefix_cache=store,
        chunk_size=None,
        max_size=len(prompt_ids) + 256,
    )
    _eval_cache(hit_cache)
    hit_prefill_s = time.perf_counter() - t0
    assert hit_n == len(prompt_ids)

    empty = PromptPrefixCache(block_size=max(len(prompt_ids), 1), max_entries=8)
    t1 = time.perf_counter()
    miss_cache, _ = prefill_with_prefix_cache(
        model,
        prompt_ids,
        prefix_cache=empty,
        chunk_size=None,
        max_size=len(prompt_ids) + 256,
    )
    _eval_cache(miss_cache)
    miss_prefill_s = time.perf_counter() - t1

    state = _empty_state(model, 1)
    t2 = time.perf_counter()
    _denoise(model, hit_cache, state, len(prompt_ids))
    hit_first = time.perf_counter() - t2

    payload = {
        "prompt_tokens": len(prompt_ids),
        "miss_prefill_s": miss_prefill_s,
        "hit_prefill_s": hit_prefill_s,
        "hit_ttfd_s": hit_prefill_s + hit_first,
        "miss_ttfd_s": miss_prefill_s + hit_first,
        "prefill_speedup": (
            miss_prefill_s / hit_prefill_s if hit_prefill_s > 0 else 0.0
        ),
        "cache_hits": store.hits,
        "n_steady_unused": n_steady,
    }
    _report("prefix cache", payload)
    return payload


def measure_chunked_prefill(model, prompt_ids: list[int], *, chunk_size: int) -> dict:
    from vllm_metal.modilify.continuous_batch import chunked_prefill

    long_ids = prompt_ids * (max(chunk_size * 2 // max(len(prompt_ids), 1), 2))
    long_ids = long_ids[: chunk_size * 2]
    ids = mx.array([long_ids], dtype=mx.int32)

    t0 = time.perf_counter()
    cache_a = model.make_cache(max_size=len(long_ids) + 8)
    cache_a = model.prefill(ids, cache=cache_a)
    _eval_cache(cache_a)
    one_shot = time.perf_counter() - t0

    t1 = time.perf_counter()
    cache_b = model.make_cache(max_size=len(long_ids) + 8)
    cache_b = chunked_prefill(model, ids, chunk_size=chunk_size, cache=cache_b)
    _eval_cache(cache_b)
    chunked = time.perf_counter() - t1

    offset_a = int(getattr(cache_a[0], "offset", 0))
    offset_b = int(getattr(cache_b[0], "offset", 0))
    payload = {
        "prompt_tokens": len(long_ids),
        "chunk_size": chunk_size,
        "oneshot_prefill_s": one_shot,
        "chunked_prefill_s": chunked,
        "offset_oneshot": offset_a,
        "offset_chunked": offset_b,
        "offsets_match": offset_a == offset_b,
    }
    _report("chunked prefill", payload)
    return payload


def measure_continuous_batch(model, prompt_ids: list[int], *, batch: int) -> dict:
    from vllm_metal.modilify.continuous_batch import pack_encoder_caches

    lengths = [max(8, (i + 1) * max(len(prompt_ids) // batch, 2)) for i in range(batch)]
    lengths = [min(length, len(prompt_ids)) for length in lengths]
    if len(set(lengths)) == 1:
        lengths[-1] = max(lengths[0] // 2, 4)

    caches = []
    for length in lengths:
        ids = mx.array([prompt_ids[:length]], dtype=mx.int32)
        cache = model.make_cache(max_size=length + 8)
        cache = model.prefill(ids, cache=cache)
        _eval_cache(cache)
        caches.append(cache)

    packed, max_prefix = pack_encoder_caches(caches, lengths)
    # Packed denoise
    canvases = []
    latents = []
    confs = []
    ents = []
    ages = []
    for _length in lengths:
        st = _empty_state(model, 1)
        canvases.append(st["canvas"])
        latents.append(st["latent"])
        confs.append(st["confidence"])
        ents.append(st["entropy"])
        ages.append(st["age"])
    from vllm_metal.modilify.continuous_batch import pack_latents

    packed_latent = pack_latents(latents)
    canvas = mx.concatenate(canvases, axis=0)
    t0 = time.perf_counter()
    out = model(
        decoder_input_ids=canvas,
        cache=packed,
        previous_confidence=mx.concatenate(confs, axis=0),
        previous_entropy=mx.concatenate(ents, axis=0),
        token_age=mx.concatenate(ages, axis=0),
        latent_state=packed_latent,
        denoise_temperature=float(model.config.denoise_temperature),
        prefix_len=mx.array(lengths, dtype=mx.int32),
        cache_capacity=max_prefix,
    )
    _eval_step(out)
    packed_s = time.perf_counter() - t0

    sequential_s = 0.0
    for cache, length in zip(caches, lengths):
        st = _empty_state(model, 1)
        t1 = time.perf_counter()
        _denoise(model, cache, st, length)
        sequential_s += time.perf_counter() - t1

    payload = {
        "batch": batch,
        "prefix_lengths": lengths,
        "packed_denoise_s": packed_s,
        "sequential_denoise_s": sequential_s,
        "packed_denoise_per_sec": 1.0 / packed_s if packed_s else 0.0,
        "sequential_denoise_per_sec": batch / sequential_s if sequential_s else 0.0,
        "pack_speedup_vs_seq": sequential_s / packed_s if packed_s else 0.0,
    }
    _report("continuous batch (ragged pack)", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path.home() / "Modilify-Mk1")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-steady", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--decode-only",
        action="store_true",
        help="Only TTFD + denoise/s (skip prefix/chunked/CB). Use for huge batches.",
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "tools" / "modilify_metrics_last.json",
    )
    args = parser.parse_args()
    mx.random.seed(int(args.seed))

    model, config = _load_model(args.model)
    model.config = model.config.with_denoise_temperature(args.temperature)
    prompt_ids = _prompt_ids(args.model, PROMPT)
    print(
        f"[metrics] model={args.model} temp={args.temperature} "
        f"prompt_tokens={len(prompt_ids)} canvas={config.canvas_length}",
        flush=True,
    )
    ids_b1 = mx.array([prompt_ids], dtype=mx.int32)
    ids_b = mx.array([prompt_ids] * int(args.batch), dtype=mx.int32)

    results = {
        "temperature": args.temperature,
        "seed": args.seed,
        "prompt": PROMPT,
        "prompt_tokens": len(prompt_ids),
        "decode_b1": measure_ttfd_and_decode(
            model, ids_b1, n_steady=args.n_steady, label="decode B=1"
        ),
        f"decode_b{args.batch}": measure_ttfd_and_decode(
            model, ids_b, n_steady=args.n_steady, label=f"decode B={args.batch}"
        ),
    }
    if not args.decode_only:
        results.update(
            {
                "denoise_breakdown": measure_denoise_breakdown(model, ids_b1),
                "prefix_cache": measure_prefix_cache(
                    model, prompt_ids, n_steady=args.n_steady
                ),
                "chunked_prefill": measure_chunked_prefill(
                    model, prompt_ids, chunk_size=args.chunk_size
                ),
                "continuous_batch": measure_continuous_batch(
                    model, prompt_ids, batch=min(int(args.batch), 8)
                ),
            }
        )
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\n[metrics] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
