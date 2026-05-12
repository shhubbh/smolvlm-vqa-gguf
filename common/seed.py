"""Seeding helpers for deterministic CPU runs."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from common.config import NUM_THREADS


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(NUM_THREADS)
