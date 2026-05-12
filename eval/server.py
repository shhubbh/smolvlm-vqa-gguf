"""Lifecycle for `llama-server` (C binary from llama.cpp).

Start in a child process, wait for /health to return 200, run, then graceful shutdown
(SIGTERM with 10s wait, fall back to SIGKILL). Frees the port between variants.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import httpx
import psutil

from common.config import EVAL_CHAT_TEMPLATE, NUM_THREADS, PATHS

log = logging.getLogger(__name__)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(base_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{base_url}/health", timeout=2.0)
            if r.status_code == 200:
                return
        except Exception as exc:
            last_err = exc
        time.sleep(0.5)
    raise RuntimeError(f"llama-server did not become healthy within {timeout_s}s: {last_err}")


def _max_rss_sampler(pid: int, stop_event: threading.Event, out: dict) -> None:
    out["peak_rss_mb"] = 0.0
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    while not stop_event.is_set():
        try:
            rss = proc.memory_info().rss / (1024**2)
            for child in proc.children(recursive=True):
                rss += child.memory_info().rss / (1024**2)
            out["peak_rss_mb"] = max(out["peak_rss_mb"], rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return
        time.sleep(1.0)


@contextmanager
def llama_server(
    *,
    text_gguf: str | Path,
    mmproj_gguf: str | Path,
    chat_template: str = EVAL_CHAT_TEMPLATE,
    threads: int = NUM_THREADS,
    n_ctx: int = 4096,
    seed: int = 42,
    health_timeout_s: float = 120.0,
    llama_cpp_dir: str | Path = PATHS["llama_cpp_dir"],
) -> Generator[dict, None, None]:
    bin_path = Path(llama_cpp_dir) / "build" / "bin" / "llama-server"
    if not bin_path.is_file():
        raise RuntimeError(f"llama-server not built at {bin_path}")

    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    cmd = [
        str(bin_path),
        "--model", str(text_gguf),
        "--mmproj", str(mmproj_gguf),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--threads", str(threads),
        "--threads-batch", str(threads),
        "--ctx-size", str(n_ctx),
        "--n-gpu-layers", "0",
        "--chat-template", chat_template,
        "--seed", str(seed),
        "--no-warmup",
    ]
    log.info("starting llama-server: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    rss_holder: dict = {"peak_rss_mb": 0.0}
    stop_event = threading.Event()
    sampler = threading.Thread(
        target=_max_rss_sampler, args=(proc.pid, stop_event, rss_holder), daemon=True
    )
    sampler.start()

    try:
        _wait_for_health(base_url, health_timeout_s)
        yield {"base_url": base_url, "pid": proc.pid, "rss": rss_holder}
    finally:
        stop_event.set()
        if proc.poll() is None:
            try:
                proc.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                log.warning("llama-server did not exit after SIGTERM; sending SIGKILL")
                try:
                    proc.send_signal(signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    log.error("llama-server still alive after SIGKILL")
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind(("127.0.0.1", port))
                    break
                except OSError:
                    time.sleep(0.2)
