"""Numerically stable realized-variance forecast metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def qlike(
    actual: np.ndarray, forecast: np.ndarray, *, floor: float = 1e-12
) -> tuple[np.ndarray, int]:
    """Return Patton-style QLIKE and the number of clipped forecasts.

    The convention is `actual / forecast - log(actual / forecast) - 1`, which is non-negative and
    differs from some reported QLIKE conventions only by terms that do not change model rankings.
    """
    y = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(forecast, dtype=np.float64)
    if y.shape != predicted.shape:
        raise ValueError("actual and forecast must have identical shapes")
    if np.any(~np.isfinite(y)) or np.any(y <= 0):
        raise ValueError("actual values must be finite and positive")
    if floor <= 0:
        raise ValueError("floor must be positive")
    clipped_mask = (~np.isfinite(predicted)) | (predicted < floor)
    safe = np.where(clipped_mask, floor, predicted)
    ratio = y / safe
    return ratio - np.log(ratio) - 1.0, int(clipped_mask.sum())


def evaluate_point_forecasts(
    actual: np.ndarray, forecast: np.ndarray, *, floor: float = 1e-12
) -> dict[str, Any]:
    """Aggregate QLIKE and scale-aware secondary metrics."""
    losses, clipped = qlike(actual, forecast, floor=floor)
    y = np.asarray(actual, dtype=np.float64)
    predicted = np.maximum(np.asarray(forecast, dtype=np.float64), floor)
    log_error = np.log(predicted) - np.log(y)
    return {
        "mean_qlike": float(losses.mean()),
        "median_qlike": float(np.median(losses)),
        "log_mse": float(np.mean(log_error**2)),
        "log_mae": float(np.mean(np.abs(log_error))),
        "mae": float(np.mean(np.abs(predicted - y))),
        "clipped_predictions": clipped,
        "count": int(y.size),
    }
