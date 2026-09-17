"""Positive naive cumulative-variance forecasts."""

from __future__ import annotations

import numpy as np


def _validate_history(history: np.ndarray) -> np.ndarray:
    values = np.asarray(history, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("history must be a non-empty finite positive 1D array")
    return values


def last_value_forecast(history: np.ndarray, horizons: tuple[int, ...]) -> np.ndarray:
    """Assume daily variance remains at its last observed value."""
    values = _validate_history(history)
    return np.asarray(horizons, dtype=np.float64) * values[-1]


def historical_mean_forecast(history: np.ndarray, horizons: tuple[int, ...]) -> np.ndarray:
    """Assume future daily variance equals the historical arithmetic mean."""
    values = _validate_history(history)
    return np.asarray(horizons, dtype=np.float64) * values.mean()
