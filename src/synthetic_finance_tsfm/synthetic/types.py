"""Synthetic generator data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SyntheticSeries:
    """One positive realized-variance-like synthetic sequence."""

    values: np.ndarray
    latent_variance: np.ndarray
    time_index: np.ndarray
    family: str
    parameters: dict[str, Any]
    events: dict[str, np.ndarray]

    def validate(self) -> None:
        n = self.values.shape[0]
        if self.values.ndim != 1 or self.latent_variance.shape != (n,):
            raise ValueError("Synthetic values and latent variance must be aligned 1D arrays")
        if self.time_index.shape != (n,):
            raise ValueError("Synthetic time index must align with values")
        if not np.all(np.isfinite(self.values)) or np.any(self.values <= 0):
            raise ValueError("Synthetic observed variance must be finite and positive")
        if not np.all(np.isfinite(self.latent_variance)) or np.any(self.latent_variance <= 0):
            raise ValueError("Synthetic latent variance must be finite and positive")
