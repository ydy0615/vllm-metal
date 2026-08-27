#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Simple chat UI for Modilify Mk1 / ChatDLM1.

vLLM's OpenAI server has no built-in chat page. This is a thin Gradio
frontend over the same ``generate()`` path used by the Metal runner.

    PYTHONPATH="/path/to/vllm-metal:/path/to/vllm" python tools/modilify_chat.py
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
from pathlib import Path

import gradio as gr
import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_ENGINE: "ModilifyEngine | None" = None


def _as_token_ids(token_ids) -> list[int]:
    if hasattr(token_ids, "input_ids"):
        token_ids = token_ids.input_ids
    elif isinstance(token_ids, dict):
        token_ids = token_ids["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], (list, tuple)):
        token_ids = token_ids[0]
    return [int(t) for t in token_ids]


def _user_text(message) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        return str(message.get("text") or message.get("content") or "")
    return str(message)


def _history_messages(history) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if not history:
        return messages
    for item in history:
        if isinstance(item, dict) and "role" in item:
            content = item.get("content", "")
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            role = str(item.get("role") or "user")
            if role in {"user", "assistant", "system"} and content:
                messages.append({"role": role, "content": str(content)})
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            user, assistant = item[0], item[1]
            if user:
                messages.append({"role": "user", "content": str(user)})
            if assistant:
                messages.append({"role": "assistant", "content": str(assistant)})
    return messages


class ModilifyEngine:
    """Load and generate on one thread so MLX keeps a live GPU stream."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.status = ""
        self.temperature_locked = False
        self.default_temp = 0.8
        self._jobs: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        thread = threading.Thread(target=self._loop, name="modilify-infer", daemon=True)
        thread.start()
        if not self._ready.wait(timeout=600):
            raise RuntimeError("Timed out loading Modilify.")
        if self._error is not None:
            raise self._error

    def _loop(self) -> None:
        from vllm_metal.modilify.generate import generate
        from vllm_metal.modilify.loader import load_modilify, load_tokenizer

        try:
            print(f"[chat] loading {self.model_path}", flush=True)
            model, config = load_modilify(self.model_path)
            tokenizer = load_tokenizer(self.model_path)
            dummy = mx.ones((1, 1))
            mx.eval(dummy)
            self.model = model
            self.config = config
            self.tokenizer = tokenizer
            self.generate = generate
            self.temperature_locked = bool(config.temperature_locked)
            self.default_temp = float(config.denoise_temperature)
            self.status = (
                f"Loaded `{self.model_path}` · {config.model_type} · "
                f"canvas {config.canvas_length} · default temp {config.denoise_temperature}"
            )
            print(
                f"[chat] ready type={config.model_type} canvas={config.canvas_length} "
                f"temp={config.denoise_temperature}",
                flush=True,
            )
            self._ready.set()
        except BaseException as exc:  # noqa: BLE001 — surface load errors to UI
            self._error = exc
            self._ready.set()
            return

        while True:
            job = self._jobs.get()
            if job is None:
                return
            messages, temperature, max_new_tokens, seed, enable_thinking, reply_q = job
            try:
                reply_q.put(("ok", self._generate(
                    messages, temperature, max_new_tokens, seed, enable_thinking
                )))
            except BaseException as exc:  # noqa: BLE001
                reply_q.put(("err", exc))

    def _generate(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_new_tokens: int,
        seed: int,
        enable_thinking: bool,
    ) -> str:
        token_ids = _as_token_ids(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=bool(enable_thinking),
                return_dict=False,
            )
        )
        temp = None if self.temperature_locked else float(temperature)
        max_new = max(8, int(max_new_tokens))
        seed_value = None if int(seed) < 0 else int(seed)
        if seed_value is not None:
            mx.random.seed(seed_value)
        output = self.generate(
            self.model,
            mx.array([token_ids], dtype=mx.int32),
            max_new_tokens=max_new,
            temperature=temp,
            seed=seed_value,
        )
        reply = self.tokenizer.decode(output.generated_ids[0], skip_special_tokens=True)
        n_tokens = len(output.generated_ids[0])
        ttfd = output.prefill_seconds + output.first_denoise_seconds
        print(
            f"[chat] tokens={n_tokens} denoise={output.denoise_steps} "
            f"ttfd={ttfd:.2f}s denoise/s={output.heavy_denoise_per_second:.2f} "
            f"tok/s={output.tokens_per_second:.2f} stop={output.stop_reasons[0]}",
            flush=True,
        )
        return reply.strip() or "(empty reply)"

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_new_tokens: int,
        seed: int,
        enable_thinking: bool,
    ) -> str:
        reply_q: queue.Queue = queue.Queue()
        self._jobs.put(
            (messages, temperature, max_new_tokens, seed, enable_thinking, reply_q)
        )
        kind, payload = reply_q.get()
        if kind == "err":
            raise payload
        return str(payload)


def chat(
    message: str,
    history: list,
    temperature: float,
    max_new_tokens: int,
    seed: int,
    enable_thinking: bool,
) -> str:
    if _ENGINE is None:
        raise gr.Error("Model is not loaded.")
    text = _user_text(message).strip()
    if not text:
        raise gr.Error("Type a message first.")
    messages = _history_messages(history)
    messages.append({"role": "user", "content": text})
    try:
        return _ENGINE.chat(
            messages, temperature, max_new_tokens, seed, enable_thinking
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def build_ui(status: str) -> gr.ChatInterface:
    locked = bool(_ENGINE.temperature_locked) if _ENGINE is not None else False
    default_temp = _ENGINE.default_temp if _ENGINE is not None else 0.8
    return gr.ChatInterface(
        fn=chat,
        title="Modilify",
        description=status,
        additional_inputs=[
            gr.Slider(
                0.05,
                1.5,
                value=default_temp,
                step=0.05,
                label="Temperature",
                interactive=not locked,
                info="Locked at 1.0 for ChatDLM1" if locked else "Mk1 default is 0.8",
            ),
            gr.Slider(32, 1024, value=256, step=32, label="Max new tokens"),
            gr.Number(value=0, precision=0, label="Seed (−1 = random)"),
            gr.Checkbox(value=False, label="Enable thinking"),
        ],
        examples=[
            ["Explain why the sky is blue.", default_temp, 256, 0, False],
            [
                "Write a short poem about rain on a metal roof.",
                default_temp,
                256,
                0,
                False,
            ],
            [
                "Give me a 3-step recipe for scrambled eggs.",
                default_temp,
                256,
                0,
                False,
            ],
        ],
        flagging_mode="never",
        concurrency_limit=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path.home() / "Modilify-Mk1")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model path does not exist: {args.model}")

    global _ENGINE
    _ENGINE = ModilifyEngine(args.model)
    demo = build_ui(_ENGINE.status)
    print(f"[chat] http://{args.host}:{args.port}", flush=True)
    demo.launch(
        server_name=args.host,
        server_port=int(args.port),
        inbrowser=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
