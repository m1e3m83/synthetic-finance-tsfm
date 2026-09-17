"""Finance-aware variance priors with clustering and asymmetric shocks."""

from __future__ import annotations

import numpy as np

from .types import SyntheticSeries


def _gjr_garch(length: int, rng: np.random.Generator) -> SyntheticSeries:
    alpha = float(rng.uniform(0.03, 0.10))
    gamma = float(rng.uniform(0.02, 0.10))
    persistence = float(rng.uniform(0.90, 0.985))
    beta = persistence - alpha - 0.5 * gamma
    if beta < 0.65:
        beta = 0.65
        persistence = alpha + beta + 0.5 * gamma
    unconditional = float(np.exp(rng.uniform(-9.5, -5.5)))
    omega = unconditional * max(1.0 - persistence, 1e-4)
    degrees_freedom = float(rng.uniform(5.0, 12.0))
    intraday_count = int(rng.choice([24, 48, 78]))

    variance = np.empty(length, dtype=np.float64)
    returns = np.zeros(length, dtype=np.float64)
    variance[0] = unconditional
    standardized = rng.standard_t(degrees_freedom, size=length)
    standardized /= np.sqrt(degrees_freedom / (degrees_freedom - 2.0))
    for index in range(1, length):
        shock = returns[index - 1] ** 2
        asymmetric = shock if returns[index - 1] < 0 else 0.0
        variance[index] = omega + alpha * shock + gamma * asymmetric + beta * variance[index - 1]
        variance[index] = float(np.clip(variance[index], 1e-14, 1e2))
        returns[index] = np.sqrt(variance[index]) * standardized[index]

    measurement = rng.chisquare(df=intraday_count, size=length) / intraday_count
    realized = np.maximum(variance * measurement, 1e-14)
    result = SyntheticSeries(
        values=realized,
        latent_variance=variance,
        time_index=np.arange(length, dtype=np.int64),
        family="volatility_gjr_garch",
        parameters={
            "omega": omega,
            "alpha": alpha,
            "gamma": gamma,
            "beta": beta,
            "persistence": persistence,
            "degrees_freedom": degrees_freedom,
            "intraday_count": intraday_count,
        },
        events={"negative_return": (returns < 0).astype(np.int8)},
    )
    result.validate()
    return result


def _stochastic_volatility(length: int, rng: np.random.Generator) -> SyntheticSeries:
    phi = float(rng.uniform(0.92, 0.995))
    sigma = float(rng.uniform(0.08, 0.25))
    mean_log_variance = float(rng.uniform(-9.5, -5.5))
    leverage = float(rng.uniform(-0.75, -0.15))
    intraday_count = int(rng.choice([24, 48, 78]))

    log_variance = np.empty(length, dtype=np.float64)
    log_variance[0] = mean_log_variance
    return_shock = 0.0
    for index in range(1, length):
        innovation = float(rng.normal())
        log_variance[index] = (
            mean_log_variance
            + phi * (log_variance[index - 1] - mean_log_variance)
            + sigma * (innovation + leverage * return_shock)
        )
        return_shock = float(rng.normal())
    log_variance = np.clip(log_variance, -22.0, 4.0)
    variance = np.exp(log_variance)
    measurement = rng.chisquare(df=intraday_count, size=length) / intraday_count
    realized = np.maximum(variance * measurement, 1e-14)
    result = SyntheticSeries(
        values=realized,
        latent_variance=variance,
        time_index=np.arange(length, dtype=np.int64),
        family="volatility_stochastic",
        parameters={
            "phi": phi,
            "sigma": sigma,
            "mean_log_variance": mean_log_variance,
            "leverage": leverage,
            "intraday_count": intraday_count,
        },
        events={},
    )
    result.validate()
    return result


def generate_volatility(length: int, rng: np.random.Generator) -> SyntheticSeries:
    """Draw a volatility-specific sequence from a GJR-GARCH or SV family."""
    if length < 32:
        raise ValueError("Volatility sequences require at least 32 observations")
    if rng.random() < 0.65:
        return _gjr_garch(length, rng)
    return _stochastic_volatility(length, rng)
