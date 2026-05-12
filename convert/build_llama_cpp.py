"""Clone llama.cpp at a pinned tag and build CPU-only.

On the VPS, use `--native` to compile with the actual CPU's feature flags.
On a portable build (e.g. Mac), default flags target AVX2 only.
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

DEVICE = "cpu"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from common.config import LLAMA_CPP_TAG, NUM_THREADS, PATHS
from common.logging_setup import setup_logging

log = logging.getLogger("convert.build")

REPO_URL = "https://github.com/ggml-org/llama.cpp.git"

REQUIRED_BINARIES = ("llama-server", "llama-quantize", "llama-mtmd-cli")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default=LLAMA_CPP_TAG)
    p.add_argument("--dir", default=PATHS["llama_cpp_dir"])
    p.add_argument("--native", action="store_true", help="enable -DGGML_NATIVE=ON")
    p.add_argument("--jobs", type=int, default=NUM_THREADS)
    return p.parse_args()


def run(cmd: list[str], cwd: str | None = None) -> None:
    log.info("$ %s%s", " ".join(cmd), f"  (cwd={cwd})" if cwd else "")
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_clone(repo_dir: Path, tag: str) -> None:
    if repo_dir.exists() and (repo_dir / ".git").exists():
        log.info("llama.cpp already cloned at %s; fetching", repo_dir)
        run(["git", "fetch", "--tags", "--depth=1", "origin", tag], cwd=str(repo_dir))
    else:
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        run(
            ["git", "clone", "--depth=1", "--branch", tag, REPO_URL, str(repo_dir)],
        )
    run(["git", "checkout", "--detach", tag], cwd=str(repo_dir))


def cmake_flags(native: bool) -> list[str]:
    flags = [
        "-DGGML_CUDA=OFF",
        "-DGGML_BLAS=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=ON",
        "-DLLAMA_BUILD_SERVER=ON",
    ]
    if native:
        flags.append("-DGGML_NATIVE=ON")
    else:
        flags.extend(["-DGGML_NATIVE=OFF", "-DGGML_AVX2=ON"])
    return flags


def build(repo_dir: Path, native: bool, jobs: int) -> None:
    build_dir = repo_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    run(["cmake", "..", *cmake_flags(native)], cwd=str(build_dir))
    run(["cmake", "--build", ".", "--config", "Release", "-j", str(jobs)], cwd=str(build_dir))


def verify_binaries(repo_dir: Path) -> None:
    bin_dir = repo_dir / "build" / "bin"
    missing = []
    for name in REQUIRED_BINARIES:
        path = bin_dir / name
        if not path.is_file() or not os.access(path, os.X_OK):
            missing.append(name)
    if missing:
        raise RuntimeError(f"missing binaries after build: {missing}")
    for name in REQUIRED_BINARIES:
        subprocess.run([str(bin_dir / name), "--version"], check=False, timeout=10)


def main() -> int:
    setup_logging()
    args = parse_args()
    repo_dir = Path(args.dir)
    native = args.native or (platform.system() == "Linux" and platform.machine() in ("x86_64", "AMD64"))
    log.info("llama.cpp tag=%s native=%s", args.tag, native)

    if shutil.which("cmake") is None or shutil.which("git") is None:
        log.error("cmake and git are required")
        return 2

    ensure_clone(repo_dir, args.tag)
    build(repo_dir, native, args.jobs)
    verify_binaries(repo_dir)
    log.info("llama.cpp built at %s", repo_dir / "build" / "bin")
    return 0


if __name__ == "__main__":
    sys.exit(main())
