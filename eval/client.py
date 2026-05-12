"""HTTP client for llama-server's OpenAI-compatible /v1/chat/completions endpoint."""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path

import httpx

from common.config import EVAL_MAX_TOKENS, EVAL_SEED, EVAL_TEMPERATURE
from eval.prompts import EVAL_SYSTEM_PROMPT, render_user

log = logging.getLogger(__name__)


def _image_to_data_url(image_path: str | Path) -> str:
    raw = Path(image_path).read_bytes()
    b64 = base64.b64encode(raw).decode()
    return f"data:image/jpeg;base64,{b64}"


def query_chat_completion(
    base_url: str,
    image_path: str | Path,
    question: str,
    *,
    max_tokens: int = EVAL_MAX_TOKENS,
    temperature: float = EVAL_TEMPERATURE,
    seed: int = EVAL_SEED,
    timeout_s: float = 120.0,
) -> tuple[str, float]:
    image_data_url = _image_to_data_url(image_path)
    payload = {
        "model": "smolvlm",
        "messages": [
            {"role": "system", "content": EVAL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": render_user(question)},
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "top_p": 1.0,
    }
    start = time.monotonic()
    response = httpx.post(
        f"{base_url}/v1/chat/completions", json=payload, timeout=timeout_s
    )
    response.raise_for_status()
    elapsed_ms = (time.monotonic() - start) * 1000.0
    data = response.json()
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unexpected llama-server response: {data}") from exc
    return text, elapsed_ms
