"""Synthetic-prior registry."""

from __future__ import annotations

import numpy as np

from .generic import generate_generic
from .types import SyntheticSeries
from .volatility import generate_volatility


def generate_series(prior: str, length: int, seed: int) -> SyntheticSeries:
    """Generate one deterministic sequence from a named implemented prior."""
    rng = np.random.default_rng(seed)
    if prior == "generic":
        return generate_generic(length, rng)
    if prior == "volatility":
        return generate_volatility(length, rng)
    if prior in {"regime_tail", "mixed"}:
        raise NotImplementedError(
            f"{prior} is intentionally gated until the Generic-vs-Volatility pilot passes"
        )
    raise ValueError(f"Unknown prior: {prior}")
