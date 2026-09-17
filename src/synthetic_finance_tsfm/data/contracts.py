"""Canonical real-data schema validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

CANONICAL_COLUMNS = (
    "asset_id",
    "date",
    "realized_variance",
    "log_realized_variance",
    "observation_count",
    "expected_observation_count",
    "coverage_ratio",
    "data_source",
    "pilot_only",
    "quality_flags",
)


def validate_daily_frame(frame: pd.DataFrame) -> None:
    """Validate a canonical daily realized-variance frame."""
    missing = set(CANONICAL_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing canonical columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Daily realized-variance frame is empty")
    if frame[["asset_id", "date"]].duplicated().any():
        raise ValueError("Duplicate asset/date rows are not allowed")
    rv = frame["realized_variance"].to_numpy(dtype=float)
    if np.any(~np.isfinite(rv)) or np.any(rv <= 0):
        raise ValueError("realized_variance must be finite and positive")
    coverage = frame["coverage_ratio"].to_numpy(dtype=float)
    if np.any(~np.isfinite(coverage)) or np.any((coverage < 0) | (coverage > 1)):
        raise ValueError("coverage_ratio must be finite and within [0, 1]")
