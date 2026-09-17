"""Leakage-safe input windows and cumulative targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForecastWindows:
    contexts: np.ndarray
    normalized_targets: np.ndarray
    cumulative_targets: np.ndarray
    context_locations: np.ndarray
    context_scales: np.ndarray
    origins: np.ndarray
    horizons: tuple[int, ...]


def build_forecast_windows(
    realized_variance: np.ndarray,
    *,
    context_length: int,
    horizons: tuple[int, ...],
    epsilon: float,
    origins: np.ndarray | None = None,
    scale_floor: float = 1e-6,
) -> ForecastWindows:
    """Construct windows using context-only log-space location and scale.

    Origin `t` means the context ends at `RV[t]` and every target begins at `RV[t + 1]`.
    """
    rv = np.asarray(realized_variance, dtype=np.float64)
    if rv.ndim != 1 or np.any(~np.isfinite(rv)) or np.any(rv <= 0):
        raise ValueError("realized_variance must be a finite positive 1D array")
    if context_length <= 1 or epsilon <= 0:
        raise ValueError("context_length must exceed one and epsilon must be positive")
    if not horizons or tuple(sorted(set(horizons))) != horizons or min(horizons) <= 0:
        raise ValueError("horizons must be positive, unique, and sorted")

    first_origin = context_length - 1
    last_origin = len(rv) - max(horizons) - 1
    if last_origin < first_origin:
        raise ValueError("Series is too short for the requested context and horizons")
    selected = (
        np.arange(first_origin, last_origin + 1, dtype=np.int64)
        if origins is None
        else np.asarray(origins, dtype=np.int64)
    )
    if selected.ndim != 1 or np.any(selected < first_origin) or np.any(selected > last_origin):
        raise ValueError("One or more forecast origins are outside the valid range")

    contexts: list[np.ndarray] = []
    normalized_targets: list[list[float]] = []
    cumulative_targets: list[list[float]] = []
    locations: list[float] = []
    scales: list[float] = []
    log_rv = np.log(rv + epsilon)

    for origin in selected:
        context = log_rv[origin - context_length + 1 : origin + 1]
        location = float(context.mean())
        scale = max(float(context.std(ddof=0)), scale_floor)
        contexts.append((context - location) / scale)
        target_row: list[float] = []
        cumulative_row: list[float] = []
        for horizon in horizons:
            cumulative = float(rv[origin + 1 : origin + horizon + 1].sum())
            log_average = float(np.log(cumulative / horizon + epsilon))
            target_row.append((log_average - location) / scale)
            cumulative_row.append(cumulative)
        normalized_targets.append(target_row)
        cumulative_targets.append(cumulative_row)
        locations.append(location)
        scales.append(scale)

    return ForecastWindows(
        contexts=np.asarray(contexts, dtype=np.float32),
        normalized_targets=np.asarray(normalized_targets, dtype=np.float32),
        cumulative_targets=np.asarray(cumulative_targets, dtype=np.float64),
        context_locations=np.asarray(locations, dtype=np.float64),
        context_scales=np.asarray(scales, dtype=np.float64),
        origins=selected,
        horizons=horizons,
    )


def reconstruct_cumulative_forecasts(
    normalized_predictions: np.ndarray,
    *,
    locations: np.ndarray,
    scales: np.ndarray,
    horizons: tuple[int, ...],
) -> np.ndarray:
    """Transform normalized log-average outputs into positive cumulative forecasts."""
    predictions = np.asarray(normalized_predictions, dtype=np.float64)
    if predictions.shape != (len(locations), len(horizons)):
        raise ValueError("Prediction shape does not match locations and horizons")
    log_average = predictions * scales[:, None] + locations[:, None]
    return np.asarray(horizons, dtype=np.float64)[None, :] * np.exp(log_average)
