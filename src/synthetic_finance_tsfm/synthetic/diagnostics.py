"""Small descriptor set for comparing synthetic prior behavior."""

from __future__ import annotations

import numpy as np


def _autocorrelation(values: np.ndarray, lag: int) -> float:
    if len(values) <= lag or np.std(values[:-lag]) == 0 or np.std(values[lag:]) == 0:
        return float("nan")
    return float(np.corrcoef(values[:-lag], values[lag:])[0, 1])


def describe_variance(values: np.ndarray, epsilon: float = 1e-10) -> dict[str, float]:
    """Return auditable persistence, scale, and tail descriptors."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 8 or np.any(array <= 0):
        raise ValueError("Expected a positive 1D variance sequence with at least eight values")
    logs = np.log(array + epsilon)
    centered = logs - logs.mean()
    scale = float(logs.std())
    kurtosis = float(np.mean(centered**4) / max(np.mean(centered**2) ** 2, 1e-20))
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "log_std": scale,
        "log_acf_1": _autocorrelation(logs, 1),
        "log_acf_5": _autocorrelation(logs, 5),
        "log_kurtosis": kurtosis,
        "max_to_median": float(array.max() / max(np.median(array), epsilon)),
    }
