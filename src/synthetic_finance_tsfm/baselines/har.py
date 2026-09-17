"""Direct Log-HAR baseline with daily, weekly, and monthly log-RV features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _har_features(log_rv: np.ndarray, origin: int) -> np.ndarray:
    if origin < 21:
        raise ValueError("Log-HAR requires at least 22 historical observations")
    return np.asarray(
        [
            1.0,
            log_rv[origin],
            log_rv[origin - 4 : origin + 1].mean(),
            log_rv[origin - 21 : origin + 1].mean(),
        ],
        dtype=np.float64,
    )


@dataclass
class DirectLogHAR:
    """One direct OLS model per cumulative forecast horizon."""

    horizons: tuple[int, ...] = (1, 5, 22)
    epsilon: float = 1e-10
    coefficients_: np.ndarray | None = None

    def fit(self, realized_variance: np.ndarray, end_origin: int | None = None) -> DirectLogHAR:
        rv = np.asarray(realized_variance, dtype=np.float64)
        if rv.ndim != 1 or np.any(~np.isfinite(rv)) or np.any(rv <= 0):
            raise ValueError("realized_variance must be a finite positive 1D array")
        maximum_origin = len(rv) - max(self.horizons) - 1
        if end_origin is not None:
            maximum_origin = min(maximum_origin, int(end_origin))
        if maximum_origin < 25:
            raise ValueError("Not enough observations to fit DirectLogHAR")
        origins = np.arange(21, maximum_origin + 1)
        log_rv = np.log(rv + self.epsilon)
        design = np.stack([_har_features(log_rv, int(origin)) for origin in origins])
        targets = np.empty((len(origins), len(self.horizons)), dtype=np.float64)
        for row, origin in enumerate(origins):
            for column, horizon in enumerate(self.horizons):
                average = rv[origin + 1 : origin + horizon + 1].mean()
                targets[row, column] = np.log(average + self.epsilon)
        self.coefficients_ = np.linalg.lstsq(design, targets, rcond=None)[0]
        return self

    def predict(self, history: np.ndarray) -> np.ndarray:
        if self.coefficients_ is None:
            raise RuntimeError("DirectLogHAR must be fitted before prediction")
        rv = np.asarray(history, dtype=np.float64)
        if rv.ndim != 1 or len(rv) < 22 or np.any(rv <= 0):
            raise ValueError("history must contain at least 22 positive observations")
        features = _har_features(np.log(rv + self.epsilon), len(rv) - 1)
        log_average = features @ self.coefficients_
        return np.asarray(self.horizons, dtype=np.float64) * np.exp(log_average)
