"""Randomness controls."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def derived_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable 32-bit seed without Python's randomized hash."""
    value = int(base_seed) & 0xFFFFFFFF
    for part in parts:
        for byte in str(part).encode("utf-8"):
            value = ((value * 16777619) ^ byte) & 0xFFFFFFFF
    return value
