"""Dependence-aware comparison helpers."""

from __future__ import annotations

import numpy as np


def block_bootstrap_mean_interval(
    values: np.ndarray,
    *,
    block_length: int,
    replications: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Circular block-bootstrap interval for a dependent mean."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or np.any(~np.isfinite(array)):
        raise ValueError("values must be a finite 1D array with at least two elements")
    if not 1 <= block_length <= len(array) or replications <= 0:
        raise ValueError("Invalid block length or replication count")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0, 1)")
    rng = np.random.default_rng(seed)
    block_count = int(np.ceil(len(array) / block_length))
    offsets = np.arange(block_length)
    means = np.empty(replications, dtype=np.float64)
    for replication in range(replications):
        starts = rng.integers(0, len(array), size=block_count)
        indices = ((starts[:, None] + offsets[None, :]) % len(array)).reshape(-1)[: len(array)]
        means[replication] = array[indices].mean()
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))
