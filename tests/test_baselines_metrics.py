import numpy as np
import pytest

from synthetic_finance_tsfm.baselines import (
    DirectLogHAR,
    historical_mean_forecast,
    last_value_forecast,
)
from synthetic_finance_tsfm.evaluation.metrics import evaluate_point_forecasts, qlike
from synthetic_finance_tsfm.evaluation.statistics import block_bootstrap_mean_interval


def test_naive_baselines_return_cumulative_positive_forecasts() -> None:
    history = np.asarray([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(last_value_forecast(history, (1, 5)), [3.0, 15.0])
    np.testing.assert_array_equal(historical_mean_forecast(history, (1, 5)), [2.0, 10.0])


def test_qlike_is_zero_at_perfect_forecast() -> None:
    actual = np.asarray([0.5, 1.0, 3.0])
    loss, clipped = qlike(actual, actual)
    np.testing.assert_allclose(loss, 0.0)
    assert clipped == 0


def test_qlike_counts_clipping_and_metrics_are_finite() -> None:
    actual = np.asarray([1.0, 2.0])
    forecast = np.asarray([0.0, 2.5])
    metrics = evaluate_point_forecasts(actual, forecast, floor=1e-6)
    assert metrics["clipped_predictions"] == 1
    assert np.isfinite(metrics["mean_qlike"])


def test_direct_log_har_fits_and_predicts_positive_values() -> None:
    time = np.arange(120)
    rv = np.exp(-7.0 + 0.2 * np.sin(time / 8.0))
    model = DirectLogHAR(horizons=(1, 5, 22)).fit(rv)
    forecast = model.predict(rv)
    assert forecast.shape == (3,)
    assert np.all(np.isfinite(forecast))
    assert np.all(forecast > 0)


def test_block_bootstrap_is_deterministic() -> None:
    values = np.linspace(-1.0, 1.0, 100)
    first = block_bootstrap_mean_interval(values, block_length=5, replications=50, seed=7)
    second = block_bootstrap_mean_interval(values, block_length=5, replications=50, seed=7)
    assert first == second
    assert first[0] <= first[1]


def test_invalid_baseline_history_is_rejected() -> None:
    with pytest.raises(ValueError):
        last_value_forecast(np.asarray([1.0, -1.0]), (1,))
