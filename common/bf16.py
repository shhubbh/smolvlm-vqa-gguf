"""BF16 capability detection."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def cpu_supports_bf16() -> bool:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return False
    try:
        text = cpuinfo.read_text()
    except OSError:
        return False
    return "avx512_bf16" in text


def choose_compute_dtype():
    import torch

    if cpu_supports_bf16():
        log.info("BF16 available (avx512_bf16). Using torch.bfloat16 for weights.")
        return torch.bfloat16
    log.warning("BF16 unsupported on this CPU. Falling back to FP32.")
    return torch.float32
