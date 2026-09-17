import numpy as np
import pytest

from synthetic_finance_tsfm.synthetic import generate_series
from synthetic_finance_tsfm.synthetic.diagnostics import describe_variance


@pytest.mark.parametrize("prior", ["generic", "volatility"])
def test_generators_are_deterministic_positive_and_aligned(prior: str) -> None:
    first = generate_series(prior, 128, 123)
    second = generate_series(prior, 128, 123)
    np.testing.assert_array_equal(first.values, second.values)
    assert first.values.shape == (128,)
    assert first.latent_variance.shape == (128,)
    assert np.all(np.isfinite(first.values))
    assert np.all(first.values > 0)


def test_generators_change_with_seed() -> None:
    first = generate_series("generic", 128, 1)
    second = generate_series("generic", 128, 2)
    assert not np.array_equal(first.values, second.values)


def test_descriptors_are_finite_for_regular_sequence() -> None:
    descriptor = describe_variance(np.exp(np.linspace(-8, -6, 64)))
    assert descriptor["mean"] > 0
    assert np.isfinite(descriptor["log_kurtosis"])


def test_gated_prior_fails_loudly() -> None:
    with pytest.raises(NotImplementedError, match="gated"):
        generate_series("mixed", 128, 5)
