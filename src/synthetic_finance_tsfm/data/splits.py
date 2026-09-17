"""Chronological split construction with target embargoes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChronologicalSplits:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def split_origins_with_embargo(
    origins: np.ndarray,
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    embargo: int = 22,
) -> ChronologicalSplits:
    """Split sorted origins while removing an embargo around both boundaries."""
    values = np.asarray(origins, dtype=np.int64)
    if values.ndim != 1 or len(values) < 10 or np.any(np.diff(values) <= 0):
        raise ValueError("origins must be a strictly increasing 1D array of length at least ten")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must lie in (0, 1)")
    if train_fraction + validation_fraction >= 1 or embargo < 0:
        raise ValueError("Split fractions leave no test set, or embargo is negative")

    train_boundary = int(len(values) * train_fraction)
    validation_boundary = int(len(values) * (train_fraction + validation_fraction))
    train = values[: max(train_boundary - embargo, 0)]
    validation = values[
        min(train_boundary + embargo, len(values)) : max(validation_boundary - embargo, 0)
    ]
    test = values[min(validation_boundary + embargo, len(values)) :]
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("The requested embargo leaves an empty split")
    return ChronologicalSplits(train=train, validation=validation, test=test)
