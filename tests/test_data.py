import numpy as np
import pandas as pd
import pytest

from synthetic_finance_tsfm.data.pilot import load_pilot_config
from synthetic_finance_tsfm.data.realized_variance import realized_variance_from_prices
from synthetic_finance_tsfm.data.splits import split_origins_with_embargo
from synthetic_finance_tsfm.data.windows import (
    build_forecast_windows,
    reconstruct_cumulative_forecasts,
)


def test_target_indices_and_reconstruction_are_exact() -> None:
    rv = np.arange(1.0, 11.0)
    windows = build_forecast_windows(
        rv,
        context_length=4,
        horizons=(1, 3),
        epsilon=1e-12,
        origins=np.asarray([3]),
    )
    np.testing.assert_allclose(windows.cumulative_targets[0], [5.0, 5.0 + 6.0 + 7.0])
    reconstructed = reconstruct_cumulative_forecasts(
        windows.normalized_targets,
        locations=windows.context_locations,
        scales=windows.context_scales,
        horizons=(1, 3),
    )
    np.testing.assert_allclose(reconstructed, windows.cumulative_targets, rtol=1e-6, atol=1e-9)


def test_context_scaling_does_not_use_future_values() -> None:
    original = np.arange(1.0, 12.0)
    changed = original.copy()
    changed[4:] *= 1000.0
    first = build_forecast_windows(
        original,
        context_length=4,
        horizons=(1,),
        epsilon=1e-10,
        origins=np.asarray([3]),
    )
    second = build_forecast_windows(
        changed,
        context_length=4,
        horizons=(1,),
        epsilon=1e-10,
        origins=np.asarray([3]),
    )
    np.testing.assert_array_equal(first.contexts, second.contexts)
    np.testing.assert_array_equal(first.context_locations, second.context_locations)
    np.testing.assert_array_equal(first.context_scales, second.context_scales)
    assert first.normalized_targets[0, 0] != second.normalized_targets[0, 0]


def test_split_embargo_separates_boundaries() -> None:
    splits = split_origins_with_embargo(np.arange(100), embargo=5)
    assert splits.validation.min() - splits.train.max() >= 11
    assert splits.test.min() - splits.validation.max() >= 11


def test_realized_variance_uses_ending_timestamp_and_flags_gaps() -> None:
    timestamps = pd.date_range("2024-01-01", periods=4, freq="12h", tz="UTC")
    frame = pd.DataFrame({"timestamp": timestamps, "close": np.exp(np.arange(4.0))})
    daily = realized_variance_from_prices(
        frame,
        asset_id="TEST",
        data_source="fixture",
        interval_minutes=720,
        minimum_coverage=0.5,
    )
    np.testing.assert_allclose(daily["realized_variance"].to_numpy(), [1.0, 2.0])
    np.testing.assert_array_equal(daily["observation_count"].to_numpy(), [1, 2])
    assert daily["pilot_only"].all()


def test_duplicate_intraday_timestamp_is_rejected() -> None:
    frame = pd.DataFrame({"timestamp": ["2024-01-01", "2024-01-01"], "close": [100.0, 101.0]})
    with pytest.raises(ValueError, match="Duplicate"):
        realized_variance_from_prices(
            frame,
            asset_id="TEST",
            data_source="fixture",
            interval_minutes=5,
        )


def test_frozen_binance_manifest_is_pilot_only() -> None:
    config = load_pilot_config("configs/data/binance_pilot.yaml")
    assert config["role"] == "pilot_only"
    assert config["symbols"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert config["start_month"] == "2024-01"
    assert config["end_month"] == "2025-06"
