#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Modilify-Mk1 benchmark suite with Hybrid KV Cache, Compiled Attention & Uniform Batch Branch."""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vllm_metal.modilify.continuous_batch import pack_encoder_caches, pack_latents
from vllm_metal.modilify.latent_deliberation import LatentDeliberationState
from vllm_metal.modilify.loader import load_modilify, load_tokenizer
from mlx_vlm.models.cache import StaticPrefixKVCache, RotatingKVCache, KVCache, create_causal_mask

RotatingKVCache.decoder_state = property(lambda self: (self.keys, self.values))

CORPUS_PATH = Path.home() / "omlx" / "omlx" / "admin" / "bench_corpora" / "code_python.txt"


def _load_corpus(path: Path) -> str:
    if not path.exists():
        fallback = ROOT / "tools" / "code_python.txt"
        if fallback.exists():
            return fallback.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Benchmark corpus not found at {path} or {fallback}")
    return path.read_text(encoding="utf-8")


def _generate_prompt(
    tokenizer: Any,
    target_tokens: int,
    corpus: str,
    chars_per_token: float = 4.0,
    max_attempts: int = 16,
) -> list[int]:
    """Generate exactly ``target_tokens`` benchmark-corpus token IDs matching omlx logic."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")

    unique_prefix = f"BENCH-{uuid.uuid4().hex} "
    target_chars = max(round(target_tokens * chars_per_token), 1)
    for _ in range(max_attempts):
        repeats = (target_chars + len(corpus) - 1) // len(corpus)
        body = (corpus * repeats)[:target_chars]
        tokens = [int(token) for token in tokenizer.encode(unique_prefix + body)]
        if len(tokens) >= target_tokens:
            return tokens[:target_tokens]
        if not tokens:
            raise RuntimeError("Benchmark corpus tokenized to 0 tokens")
        target_chars = max(
            target_chars + 1,
            (target_chars * target_tokens + len(tokens) - 1) // len(tokens) + 1,
        )
    return tokens[:target_tokens]


def get_peak_memory_bytes() -> int:
    """Get peak memory in bytes from MLX Metal or OS rusage."""
    peak = 0
    if hasattr(mx, "get_peak_memory"):
        try:
            peak = mx.get_peak_memory()
        except Exception:
            pass
    elif hasattr(mx.metal, "get_peak_memory"):
        try:
            peak = mx.metal.get_peak_memory()
        except Exception:
            pass
    if peak <= 0:
        rusage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            peak = rusage
        else:
            peak = rusage * 1024
    return peak


def reset_peak_memory() -> None:
    if hasattr(mx, "reset_peak_memory"):
        try:
            mx.reset_peak_memory()
        except Exception:
            pass
    elif hasattr(mx.metal, "reset_peak_memory"):
        try:
            mx.metal.reset_peak_memory()
        except Exception:
            pass


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


def _empty_state(model, batch: int):
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


def make_hybrid_cache(config: Any, max_size: int | None = None) -> list[Any]:
    """Hybrid KV Cache: RotatingKVCache (1024) for SWA, Static/Full for global attention."""
    caches = []
    window = config.text_config.sliding_window
    for layer_type in config.text_config.layer_types:
        if layer_type == "full_attention":
            if max_size is not None:
                caches.append(StaticPrefixKVCache(max_size=max_size))
            else:
                caches.append(KVCache())
        else:
            caches.append(RotatingKVCache(max_size=window))
    return caches


def _prefill_sequence(model: Any, ids: mx.array, chunk_threshold: int = 8192, chunk_size: int = 4096) -> Any:
    seq_len = int(ids.shape[1])
    cache = make_hybrid_cache(model.config, max_size=seq_len + 256)
    if seq_len <= chunk_threshold:
        cache = model.prefill(ids, cache=cache)
        _eval_cache(cache)
        return cache

    # Chunked prefill with per-chunk eval to prevent intermediate activation buildup
    for start in range(0, seq_len, chunk_size):
        stop = min(start + chunk_size, seq_len)
        chunk = ids[:, start:stop]
        cache = model.prefill(chunk, cache=cache)
        _eval_cache(cache)
    return cache


def run_single_request_bench(
    model: Any,
    tokenizer: Any,
    corpus: str,
    prompt_tokens_len: int,
    denoise_steps: int,
    temperature: float,
) -> dict[str, Any]:
    print(f"\n[单请求] 正在测试 pp{prompt_tokens_len}/tg{denoise_steps} (Prompt={prompt_tokens_len}, Denoise={denoise_steps}) ...", flush=True)
    prompt_ids = _generate_prompt(tokenizer, prompt_tokens_len, corpus)
    ids = mx.array([prompt_ids], dtype=mx.int32)
    seq = len(prompt_ids)

    # 1. Warmup step
    wcache = make_hybrid_cache(model.config, max_size=min(seq, 2048) + 256)
    wids = ids[:, :min(seq, 2048)]
    wcache = model.prefill(wids, cache=wcache)
    _eval_cache(wcache)
    model.compile_attention(wcache)
    wstate = _empty_state(model, 1)
    _denoise(model, wcache, wstate, min(seq, 2048))

    # 2. Timed Prefill
    reset_peak_memory()
    t0 = time.perf_counter()
    cache = _prefill_sequence(model, ids, chunk_threshold=8192, chunk_size=4096)
    prefill_s = time.perf_counter() - t0

    # 3. Compile Attention on populated cache
    model.compile_attention(cache)

    # 4. First Denoise (TTFD / TTFT)
    state = _empty_state(model, 1)
    t1 = time.perf_counter()
    _denoise(model, cache, state, seq)
    first_denoise_s = time.perf_counter() - t1
    ttft_s = prefill_s + first_denoise_s
    ttft_ms = ttft_s * 1000.0

    # 5. Steady Denoise Steps
    remaining_steps = max(denoise_steps - 1, 0)
    t2 = time.perf_counter()
    for _ in range(remaining_steps):
        _denoise(model, cache, state, seq)
    steady_s = time.perf_counter() - t2
    total_gen_s = first_denoise_s + steady_s

    peak_mem_bytes = get_peak_memory_bytes()
    peak_mem_gb = round(peak_mem_bytes / (1024**3), 2)

    gen_dps = denoise_steps / max(total_gen_s, 1e-9)
    tpot_ms = (total_gen_s / denoise_steps) * 1000.0 if denoise_steps > 0 else 0.0
    prefill_tps = prompt_tokens_len / max(prefill_s, 1e-9)
    e2e_s = prefill_s + total_gen_s
    throughput_tps = (prompt_tokens_len + denoise_steps) / max(e2e_s, 1e-9)

    res = {
        "test": f"pp{prompt_tokens_len}/tg{denoise_steps}",
        "prompt_tokens": prompt_tokens_len,
        "denoise_steps": denoise_steps,
        "ttft_ms": round(ttft_ms, 1),
        "tpot_ms": round(tpot_ms, 2),
        "prefill_tps": round(prefill_tps, 1),
        "gen_dps": round(gen_dps, 1),
        "e2e_latency_s": round(e2e_s, 3),
        "throughput_tps": round(throughput_tps, 1),
        "peak_memory_gb": peak_mem_gb,
    }
    print(
        f"  TTFT: {res['ttft_ms']:.1f} ms | TPOT: {res['tpot_ms']:.2f} ms/Denoise | "
        f"预处理 TPS: {res['prefill_tps']:.1f} tok/s | 生成 dPS: {res['gen_dps']:.1f} dPS | "
        f"端到端延迟: {res['e2e_latency_s']:.3f}s | 吞吐量: {res['throughput_tps']:.1f} tok/s | "
        f"峰值内存: {res['peak_memory_gb']:.2f} GB",
        flush=True,
    )
    return res


def run_continuous_batch_bench(
    model: Any,
    tokenizer: Any,
    corpus: str,
    batch_size: int,
    prompt_tokens_len: int,
    denoise_steps: int,
    baseline_dps: float | None = None,
) -> dict[str, Any]:
    print(f"\n[连续批处理] 正在测试批大小={batch_size}x @ pp{prompt_tokens_len}/tg{denoise_steps} ...", flush=True)
    lengths = [prompt_tokens_len] * batch_size
    prompts = [
        _generate_prompt(tokenizer, prompt_tokens_len, corpus)
        for _ in range(batch_size)
    ]

    reset_peak_memory()
    t0 = time.perf_counter()
    caches = []
    for prompt_ids in prompts:
        ids = mx.array([prompt_ids], dtype=mx.int32)
        cache = _prefill_sequence(model, ids, chunk_threshold=8192, chunk_size=4096)
        caches.append(cache)
    prefill_s = time.perf_counter() - t0

    packed_cache, max_prefix = pack_encoder_caches(caches, lengths)

    # Compile attention on packed cache for batch
    model.compile_attention(packed_cache)

    canvases = []
    latents = []
    confs = []
    ents = []
    ages = []
    for _ in range(batch_size):
        st = _empty_state(model, 1)
        canvases.append(st["canvas"])
        latents.append(st["latent"])
        confs.append(st["confidence"])
        ents.append(st["entropy"])
        ages.append(st["age"])

    packed_latent = pack_latents(latents)
    canvas = mx.concatenate(canvases, axis=0)
    confidence = mx.concatenate(confs, axis=0)
    entropy = mx.concatenate(ents, axis=0)
    age = mx.concatenate(ages, axis=0)
    
    # Use uniform Python int prefix_len for compiled batch branch!
    uniform_prefix_len = prompt_tokens_len

    # First packed denoise
    t1 = time.perf_counter()
    out = model(
        decoder_input_ids=canvas,
        cache=packed_cache,
        previous_confidence=confidence,
        previous_entropy=entropy,
        token_age=age,
        latent_state=packed_latent,
        denoise_temperature=float(model.config.denoise_temperature),
        prefix_len=uniform_prefix_len,
        cache_capacity=max_prefix,
    )
    _eval_step(out)
    first_denoise_s = time.perf_counter() - t1

    # Steady packed denoise
    canvas = out.proposal
    confidence = out.proposal_confidence.astype(mx.float32)
    entropy = out.token_entropy.astype(mx.float32)
    age = age + 1
    packed_latent = out.next_latent_state
    history = out.heavy_hidden_state

    remaining_steps = max(denoise_steps - 1, 0)
    t2 = time.perf_counter()
    for _ in range(remaining_steps):
        out = model(
            decoder_input_ids=canvas,
            cache=packed_cache,
            previous_confidence=confidence,
            previous_entropy=entropy,
            token_age=age,
            latent_state=packed_latent,
            history_hidden_state=history,
            denoise_temperature=float(model.config.denoise_temperature),
            prefix_len=uniform_prefix_len,
            cache_capacity=max_prefix,
        )
        _eval_step(out)
        canvas = out.proposal
        confidence = out.proposal_confidence.astype(mx.float32)
        entropy = out.token_entropy.astype(mx.float32)
        age = age + 1
        packed_latent = out.next_latent_state
        history = out.heavy_hidden_state
    steady_s = time.perf_counter() - t2
    total_gen_s = first_denoise_s + steady_s

    peak_mem_bytes = get_peak_memory_bytes()
    peak_mem_gb = round(peak_mem_bytes / (1024**3), 2)

    batch_gen_dps = (batch_size * denoise_steps) / max(total_gen_s, 1e-9)
    speedup = round(batch_gen_dps / baseline_dps, 2) if baseline_dps and baseline_dps > 0 else 1.00

    prefill_tps = (batch_size * prompt_tokens_len) / max(prefill_s, 1e-9)
    pp_tps_per_req = prefill_tps / batch_size
    avg_ttft_ms = (prefill_s + first_denoise_s) * 1000.0
    e2e_s = prefill_s + total_gen_s

    res = {
        "batch_size": batch_size,
        "label": f"{batch_size}x" if batch_size > 1 else "1x (基准线)",
        "gen_dps": round(batch_gen_dps, 1),
        "speedup": speedup,
        "prefill_tps": round(prefill_tps, 1),
        "pp_tps_per_req": round(pp_tps_per_req, 1),
        "avg_ttft_ms": round(avg_ttft_ms, 1),
        "e2e_latency_s": round(e2e_s, 3),
        "peak_memory_gb": peak_mem_gb,
    }
    print(
        f"  生成 dPS: {res['gen_dps']:.1f} dPS | 加速比: {res['speedup']:.2f}x | "
        f"预处理 TPS: {res['prefill_tps']:.1f} tok/s | pp TPS/请求: {res['pp_tps_per_req']:.1f} tok/s | "
        f"平均 TTFT: {res['avg_ttft_ms']:.1f} ms | 端到端延迟: {res['e2e_latency_s']:.3f}s",
        flush=True,
    )
    return res


def format_markdown_tables(single_results: list[dict], batch_results: list[dict]) -> str:
    lines = []
    lines.append("### ⚡ 单请求结果")
    lines.append("")
    lines.append("| 测试 | TTFT (毫秒) | TPOT (毫秒/Denoise) | 预处理 TPS | 生成 dPS | 端到端延迟 | 吞吐量 | 峰值内存 |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in single_results:
        tpot_str = f"{r['tpot_ms']:.2f}" if r.get('tpot_ms') is not None else "N/A"
        gen_dps_str = f"**{r['gen_dps']:.1f} dPS**" if r.get('gen_dps') is not None else "N/A"
        lines.append(
            f"| **{r['test']}** | {r['ttft_ms']:.1f} | {tpot_str} | "
            f"{r['prefill_tps']:.1f} tok/s | {gen_dps_str} | "
            f"{r['e2e_latency_s']:.3f}s | {r['throughput_tps']:.1f} tok/s | {r['peak_memory_gb']:.2f} GB |"
        )
    lines.append("")
    lines.append("### 🥞 连续批处理")
    lines.append("")
    lines.append("*预处理1024 / 生成128*")
    lines.append("")
    lines.append("| 批大小 | 生成 dPS | 加速比 | 预处理 TPS | pp TPS/请求 | 平均 TTFT (ms) | 端到端延迟 |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in batch_results:
        speedup_str = f"**{r['speedup']:.2f}x**" if r['speedup'] > 1.0 else f"{r['speedup']:.2f}x"
        lines.append(
            f"| **{r['label']}** | **{r['gen_dps']:.1f} dPS** | {speedup_str} | "
            f"{r['prefill_tps']:.1f} tok/s | {r['pp_tps_per_req']:.1f} tok/s | "
            f"{r['avg_ttft_ms']:.1f} | {r['e2e_latency_s']:.3f}s |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path.home() / "Modilify-Mk1")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--denoise-steps", type=int, default=128)
    parser.add_argument(
        "--single-prompts",
        type=int,
        nargs="+",
        default=[1024, 4096, 8192, 16384, 32768, 65536, 131072, 200000],
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
    )
    parser.add_argument("--smoke-test", action="store_true", help="Quick verification test")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "tools" / "modilify_benchmark_results.json",
    )
    args = parser.parse_args()

    if args.smoke_test:
        args.single_prompts = [512, 1024]
        args.batch_sizes = [1, 2]
        args.denoise_steps = 16

    print("=================================================================", flush=True)
    print(f"Modilify-Mk1 Benchmark Suite (vLLM-Metal)")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature} | Denoise Steps: {args.denoise_steps}")
    print(f"Single Prompts: {args.single_prompts}")
    print(f"Batch Sizes: {args.batch_sizes}")
    print("=================================================================", flush=True)

    print("\n[Init] 加载模型与分词器 ...", flush=True)
    t0 = time.perf_counter()
    model, config = load_modilify(args.model)
    tokenizer = load_tokenizer(args.model)
    model.config = model.config.with_denoise_temperature(args.temperature)
    corpus = _load_corpus(CORPUS_PATH)

    # Patch encoder masks to support hybrid rotating / static KV caches properly
    def make_chunk_masks(h, cache, attention_mask=None, mm_token_type_ids=None):
        N = h.shape[1]
        masks = []
        for layer, c in zip(model.model.encoder.decoder.layers, cache):
            window = config.text_config.sliding_window if layer.layer_type == "sliding_attention" else None
            if hasattr(c, "make_mask"):
                mask = c.make_mask(N, window_size=window)
            else:
                offset = getattr(c, "offset", 0)
                mask = create_causal_mask(N, offset=offset, window_size=window)
            masks.append(mask)
        return masks

    model.model.encoder._make_encoder_masks = make_chunk_masks
    model.make_cache = lambda max_size=None: make_hybrid_cache(config, max_size)

    print(f"[Init] 模型加载完成 ({time.perf_counter() - t0:.2f}s) | 语料库大小: {len(corpus):,} 字符", flush=True)

    single_results = []
    for pp in args.single_prompts:
        res = run_single_request_bench(
            model,
            tokenizer,
            corpus,
            prompt_tokens_len=pp,
            denoise_steps=args.denoise_steps,
            temperature=args.temperature,
        )
        single_results.append(res)

    pp1024_res = next((r for r in single_results if r["prompt_tokens"] == 1024), None)
    baseline_dps = pp1024_res["gen_dps"] if pp1024_res else None

    batch_results = []
    for b in args.batch_sizes:
        if b == 1 and pp1024_res:
            b_res = {
                "batch_size": 1,
                "label": "1x (基准线)",
                "gen_dps": pp1024_res["gen_dps"],
                "speedup": 1.00,
                "prefill_tps": pp1024_res["prefill_tps"],
                "pp_tps_per_req": pp1024_res["prefill_tps"],
                "avg_ttft_ms": pp1024_res["ttft_ms"],
                "e2e_latency_s": pp1024_res["e2e_latency_s"],
                "peak_memory_gb": pp1024_res["peak_memory_gb"],
            }
        else:
            b_res = run_continuous_batch_bench(
                model,
                tokenizer,
                corpus,
                batch_size=b,
                prompt_tokens_len=1024,
                denoise_steps=args.denoise_steps,
                baseline_dps=baseline_dps,
            )
            if b == 1 and baseline_dps is None:
                baseline_dps = b_res["gen_dps"]
        batch_results.append(b_res)

    markdown_report = format_markdown_tables(single_results, batch_results)
    print("\n" + "=" * 65)
    print("FINAL BENCHMARK REPORT")
    print("=" * 65 + "\n")
    print(markdown_report)

    output_payload = {
        "model": str(args.model),
        "temperature": args.temperature,
        "denoise_steps": args.denoise_steps,
        "single_request_results": single_results,
        "continuous_batching_results": batch_results,
        "markdown_report": markdown_report,
    }
    args.out.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[Done] 评测结果已保存至 {args.out}")


if __name__ == "__main__":
    main()
