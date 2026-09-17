"""Generic positive time-series prior without explicit volatility dynamics."""

from __future__ import annotations

import numpy as np

from .types import SyntheticSeries


def generate_generic(length: int, rng: np.random.Generator) -> SyntheticSeries:
    """Generate a generic AR/trend/seasonality/change-point series in log space."""
    if length < 32:
        raise ValueError("Generic sequences require at least 32 observations")

    phi = float(rng.uniform(-0.35, 0.85))
    innovation_scale = float(rng.uniform(0.08, 0.35))
    trend = float(rng.normal(0.0, 0.0015))
    seasonal_period = int(rng.choice([5, 7, 12, 22, 30]))
    seasonal_amplitude = float(rng.uniform(0.0, 0.25))
    use_student = bool(rng.random() < 0.35)

    innovations = (
        rng.standard_t(df=float(rng.uniform(5.0, 15.0)), size=length)
        if use_student
        else rng.normal(size=length)
    )
    innovations *= innovation_scale

    log_values = np.empty(length, dtype=np.float64)
    log_values[0] = float(rng.normal(-7.0, 1.0))
    center = log_values[0]
    time = np.arange(length, dtype=np.float64)
    seasonal = seasonal_amplitude * np.sin(2.0 * np.pi * time / seasonal_period)
    for index in range(1, length):
        log_values[index] = (
            center
            + phi * (log_values[index - 1] - center)
            + trend * index
            + seasonal[index]
            + innovations[index]
        )

    change_points: list[int] = []
    if length >= 64 and rng.random() < 0.65:
        count = int(rng.integers(1, 3))
        possible = np.arange(length // 4, 3 * length // 4)
        points = np.sort(rng.choice(possible, size=count, replace=False))
        for point in points:
            shift = float(rng.normal(0.0, 0.45))
            log_values[point:] += shift
            change_points.append(int(point))

    log_values = np.clip(log_values, -18.0, 4.0)
    values = np.exp(log_values)
    result = SyntheticSeries(
        values=values,
        latent_variance=values.copy(),
        time_index=np.arange(length, dtype=np.int64),
        family="generic",
        parameters={
            "phi": phi,
            "innovation_scale": innovation_scale,
            "trend": trend,
            "seasonal_period": seasonal_period,
            "seasonal_amplitude": seasonal_amplitude,
            "innovation": "student_t" if use_student else "gaussian",
        },
        events={"change_points": np.asarray(change_points, dtype=np.int64)},
    )
    result.validate()
    return result
