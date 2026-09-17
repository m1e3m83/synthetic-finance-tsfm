"""Intraday-price to daily realized-variance conversion."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import CANONICAL_COLUMNS, validate_daily_frame


def realized_variance_from_prices(
    frame: pd.DataFrame,
    *,
    asset_id: str,
    data_source: str,
    interval_minutes: int,
    epsilon: float = 1e-10,
    minimum_coverage: float = 0.95,
    pilot_only: bool = True,
) -> pd.DataFrame:
    """Aggregate close-to-close squared log returns by UTC ending day.

    The input must contain `timestamp` and `close`. Timestamps may be timezone-aware or parseable by
    pandas. The return between two prices is assigned to the timestamp of the later price.
    """
    if interval_minutes <= 0 or 1440 % interval_minutes != 0:
        raise ValueError("interval_minutes must be a positive divisor of 1440")
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be in (0, 1]")
    required = {"timestamp", "close"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Intraday data must contain {sorted(required)}")

    work = frame.loc[:, ["timestamp", "close"]].copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="raise")
    if work["timestamp"].duplicated().any():
        raise ValueError("Duplicate intraday timestamps are not allowed")
    work = work.sort_values("timestamp", kind="stable").reset_index(drop=True)
    work["close"] = pd.to_numeric(work["close"], errors="coerce")
    if work["close"].isna().any() or (work["close"] <= 0).any():
        raise ValueError("Close prices must be finite and positive")

    expected_delta = pd.Timedelta(minutes=interval_minutes)
    delta = work["timestamp"].diff()
    consecutive = delta.eq(expected_delta)
    log_price = np.log(work["close"].to_numpy(dtype=float))
    returns = np.empty(len(work), dtype=np.float64)
    returns.fill(np.nan)
    returns[1:] = np.diff(log_price)
    returns[~consecutive.to_numpy()] = np.nan
    work["squared_return"] = returns**2
    work["date"] = work["timestamp"].dt.floor("D")

    expected_observations = 1440 // interval_minutes
    grouped = work.groupby("date", sort=True, observed=True)
    daily = grouped.agg(
        realized_variance=("squared_return", "sum"),
        observation_count=("squared_return", "count"),
    ).reset_index()
    daily["expected_observation_count"] = expected_observations
    daily["coverage_ratio"] = daily["observation_count"] / expected_observations
    daily["quality_flags"] = np.where(
        daily["coverage_ratio"] >= minimum_coverage, "", "insufficient_coverage"
    )
    daily = daily.loc[daily["coverage_ratio"] >= minimum_coverage].copy()
    daily = daily.loc[daily["realized_variance"] > 0].copy()
    daily["asset_id"] = asset_id
    daily["log_realized_variance"] = np.log(daily["realized_variance"] + epsilon)
    daily["data_source"] = data_source
    daily["pilot_only"] = bool(pilot_only)
    daily = daily.loc[:, CANONICAL_COLUMNS].reset_index(drop=True)
    validate_daily_frame(daily)
    return daily
