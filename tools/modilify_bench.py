#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Speed + accuracy trial for Modilify on real Mk1 weights (temperature 0.8)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROMPT = "Explain why the sky is blue."


def _prompt_ids(model_path: Path, prompt: str, enable_thinking: bool) -> tuple[list[int], object]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
        return_dict=False,
    )
    if hasattr(token_ids, "input_ids"):
        token_ids = token_ids.input_ids
    elif isinstance(token_ids, dict):
        token_ids = token_ids["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], (list, tuple)):
        token_ids = token_ids[0]
    return [int(t) for t in token_ids], tokenizer


def _report_weight_overlap(model, remapped: dict) -> dict[str, int]:
    from mlx.utils import tree_flatten

    param_names = {name for name, _ in tree_flatten(model.parameters())}
    mapped = set(remapped)
    missing = sorted(param_names - mapped)
    unused = sorted(mapped - param_names)
    print(
        f"[bench] params={len(param_names)} remapped={len(mapped)} "
        f"missing={len(missing)} unused={len(unused)}",
        flush=True,
    )
    if missing[:12]:
        print("[bench] missing sample:", missing[:12], flush=True)
    if unused[:12]:
        print("[bench] unused sample:", unused[:12], flush=True)
    return {
        "n_params": len(param_names),
        "n_remapped": len(mapped),
        "n_missing": len(missing),
        "n_unused": len(unused),
        "missing_head": missing[:20],
        "unused_head": unused[:20],
    }


def run_ours(
    model_path: Path,
    *,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    enable_thinking: bool,
) -> dict:
    from mlx.utils import tree_flatten

    from vllm_metal.modilify.config import ModilifyRuntimeConfig
    from vllm_metal.modilify.generate import generate
    from vllm_metal.modilify.loader import _load_weight_files
    from vllm_metal.modilify.modeling import ModilifyForBlockDiffusion
    from vllm_metal.modilify.remap import remap_state_dict

    print("[ours] building graph", flush=True)
    t0 = time.perf_counter()
    config = ModilifyRuntimeConfig.from_json(model_path / "config.json")
    config = config.with_denoise_temperature(temperature)
    model = ModilifyForBlockDiffusion(config)
    print(f"[ours] graph {time.perf_counter() - t0:.1f}s", flush=True)

    print("[ours] reading shards", flush=True)
    t1 = time.perf_counter()
    weights = _load_weight_files(model_path)
    remapped = remap_state_dict(weights.items(), skip_vision=config.skip_vision)
    del weights
    overlap = _report_weight_overlap(model, remapped)
    model.load_weights(list(remapped.items()), strict=False)
    del remapped
    mx.eval(tree_flatten(model.parameters()))
    print(f"[ours] load {time.perf_counter() - t1:.1f}s", flush=True)

    prompt_ids, tokenizer = _prompt_ids(model_path, prompt, enable_thinking)
    print(
        f"[ours] prompt_tokens={len(prompt_ids)} temp={config.denoise_temperature} "
        f"canvas={config.canvas_length} seed={seed}",
        flush=True,
    )
    try:
        from mlx_vlm.generate.common import wired_limit
    except ImportError:
        wired_limit = None
    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
    )
    if wired_limit is not None:
        with wired_limit(model, None):
            output = generate(
                model,
                mx.array([prompt_ids], dtype=mx.int32),
                **generate_kwargs,
            )
    else:
        output = generate(
            model,
            mx.array([prompt_ids], dtype=mx.int32),
            **generate_kwargs,
        )
    text = tokenizer.decode(output.generated_ids[0], skip_special_tokens=False)
    visible = tokenizer.decode(output.generated_ids[0], skip_special_tokens=True)
    n_tokens = len(output.generated_ids[0])
    result = {
        "runtime": "vllm_metal.modilify",
        "temperature": config.denoise_temperature,
        "temperature_locked": config.temperature_locked,
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": n_tokens,
        "denoise_steps": output.denoise_steps,
        "jump_count": output.jump_count,
        "stop": output.stop_reasons[0],
        "prefill_seconds": output.prefill_seconds,
        "first_denoise_seconds": output.first_denoise_seconds,
        "generate_seconds": output.generate_seconds,
        "compile_seconds": output.compile_seconds,
        "tokens_per_second": output.tokens_per_second,
        "heavy_denoise_per_second": output.heavy_denoise_per_second,
        "tokens_per_forward": (
            n_tokens / output.denoise_steps if output.denoise_steps else 0.0
        ),
        "raw": text,
        "visible": visible,
        "token_ids": output.generated_ids[0],
        "weight_overlap": overlap,
    }
    _print_result("ours", result)
    return result


def run_mk1_mlx(
    model_path: Path,
    mlx_root: Path,
    *,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    enable_thinking: bool,
) -> dict:
    sys.path.insert(0, str(mlx_root))
    from modilify_mlx.generate import generate as mlx_generate
    from modilify_mlx.modeling import load

    print("[mlx] loading", flush=True)
    t0 = time.perf_counter()
    model, config = load(model_path, expert_bits=16)
    print(f"[mlx] load {time.perf_counter() - t0:.1f}s", flush=True)
    prompt_ids, tokenizer = _prompt_ids(model_path, prompt, enable_thinking)
    print(
        f"[mlx] prompt_tokens={len(prompt_ids)} temp={temperature} "
        f"canvas={config.canvas_length} seed={seed}",
        flush=True,
    )
    output = mlx_generate(
        model,
        mx.array([prompt_ids], dtype=mx.int32),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
    )
    text = tokenizer.decode(output.generated_ids, skip_special_tokens=False)
    visible = tokenizer.decode(output.generated_ids, skip_special_tokens=True)
    result = {
        "runtime": "modilify_mlx",
        "temperature": temperature,
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": output.generated_length,
        "denoise_steps": output.denoise_steps,
        "jump_count": output.jump_count,
        "stop": output.stop_reason,
        "prefill_seconds": output.prefill_seconds,
        "first_denoise_seconds": output.first_denoise_seconds,
        "generate_seconds": output.generate_seconds,
        "tokens_per_second": output.tokens_per_second,
        "heavy_denoise_per_second": output.heavy_denoise_per_second,
        "tokens_per_forward": output.tokens_per_forward,
        "raw": text,
        "visible": visible,
        "token_ids": list(output.generated_ids),
    }
    _print_result("mlx", result)
    return result


def _print_result(tag: str, result: dict) -> None:
    print(
        f"[{tag}] stop={result['stop']} denoise={result['denoise_steps']} "
        f"committed={result['generated_tokens']} jumps={result['jump_count']} "
        f"tpf={result['tokens_per_forward']:.2f}",
        flush=True,
    )
    print(
        f"[{tag}] prefill={result['prefill_seconds']:.3f}s "
        f"compile={result.get('compile_seconds', 0):.3f}s "
        f"first_denoise={result['first_denoise_seconds']:.3f}s "
        f"generate={result['generate_seconds']:.3f}s "
        f"hd/s={result['heavy_denoise_per_second']:.3f} "
        f"tok/s={result['tokens_per_second']:.2f}",
        flush=True,
    )
    print(f"[{tag}] --- visible ---", flush=True)
    print(result["visible"], flush=True)


def _token_prefix_match(a: list[int], b: list[int]) -> dict[str, float | int]:
    n = min(len(a), len(b))
    matched = 0
    for i in range(n):
        if a[i] != b[i]:
            break
        matched += 1
    return {
        "ours_len": len(a),
        "ref_len": len(b),
        "prefix_match": matched,
        "prefix_frac_of_shorter": (matched / n) if n else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path.home() / "Modilify-Mk1")
    parser.add_argument(
        "--mlx-root", type=Path, default=Path.home() / "Modilify-Mk1-MLX"
    )
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--runtime",
        choices=("ours", "mlx", "both"),
        default="ours",
        help="ours=vllm-metal, mlx=Modilify-Mk1-MLX, both=sequential (not simultaneous).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "tools" / "modilify_bench_last.json",
    )
    args = parser.parse_args()

    payload: dict = {
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "enable_thinking": args.enable_thinking,
        "model": str(args.model),
    }
    if args.runtime in ("ours", "both"):
        payload["ours"] = run_ours(
            args.model,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,
            enable_thinking=args.enable_thinking,
        )
    if args.runtime in ("mlx", "both"):
        # Sequential: free ours first if both were requested by running mlx
        # in a separate process. In-process both will OOM on 26B×2.
        if args.runtime == "both":
            raise SystemExit(
                "Run --runtime ours and --runtime mlx in separate processes; "
                "two 26B copies will not fit."
            )
        payload["mlx"] = run_mk1_mlx(
            args.model,
            args.mlx_root,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,
            enable_thinking=args.enable_thinking,
        )
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[bench] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
