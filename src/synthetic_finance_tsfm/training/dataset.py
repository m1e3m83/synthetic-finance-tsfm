"""Deterministic synthetic forecasting-task construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from synthetic_finance_tsfm.data.windows import build_forecast_windows
from synthetic_finance_tsfm.synthetic import generate_series
from synthetic_finance_tsfm.synthetic.diagnostics import describe_variance
from synthetic_finance_tsfm.utils.seed import derived_seed


@dataclass(frozen=True)
class SyntheticTaskMetadata:
    prior: str
    seed: int
    sample_count: int
    descriptor_means: dict[str, float]
    family_counts: dict[str, int]


class SyntheticWindowDataset(Dataset[tuple[torch.Tensor, ...]]):
    """In-memory task set used for deterministic pilot and smoke runs."""

    def __init__(
        self,
        *,
        contexts: np.ndarray,
        normalized_targets: np.ndarray,
        cumulative_targets: np.ndarray,
        locations: np.ndarray,
        scales: np.ndarray,
        metadata: SyntheticTaskMetadata,
    ) -> None:
        self.contexts = torch.as_tensor(contexts, dtype=torch.float32)
        self.normalized_targets = torch.as_tensor(normalized_targets, dtype=torch.float32)
        self.cumulative_targets = torch.as_tensor(cumulative_targets, dtype=torch.float64)
        self.locations = torch.as_tensor(locations, dtype=torch.float64)
        self.scales = torch.as_tensor(scales, dtype=torch.float64)
        self.metadata = metadata

    def __len__(self) -> int:
        return self.contexts.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            self.contexts[index],
            self.normalized_targets[index],
            self.cumulative_targets[index],
            self.locations[index],
            self.scales[index],
        )


def make_synthetic_dataset(
    *,
    prior: str,
    sample_count: int,
    series_length: int,
    context_length: int,
    horizons: tuple[int, ...],
    epsilon: float,
    seed: int,
) -> SyntheticWindowDataset:
    """Generate deterministic independent tasks and retain prior diagnostics."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    contexts: list[np.ndarray] = []
    normalized_targets: list[np.ndarray] = []
    cumulative_targets: list[np.ndarray] = []
    locations: list[float] = []
    scales: list[float] = []
    descriptors: list[dict[str, float]] = []
    family_counts: dict[str, int] = {}
    origin_rng = np.random.default_rng(derived_seed(seed, prior, "origins"))
    first_origin = context_length - 1
    last_origin = series_length - max(horizons) - 1

    for sample_index in range(sample_count):
        sample_seed = derived_seed(seed, prior, sample_index)
        series = generate_series(prior, series_length, sample_seed)
        origin = int(origin_rng.integers(first_origin, last_origin + 1))
        windows = build_forecast_windows(
            series.values,
            context_length=context_length,
            horizons=horizons,
            epsilon=epsilon,
            origins=np.asarray([origin], dtype=np.int64),
        )
        contexts.append(windows.contexts[0])
        normalized_targets.append(windows.normalized_targets[0])
        cumulative_targets.append(windows.cumulative_targets[0])
        locations.append(float(windows.context_locations[0]))
        scales.append(float(windows.context_scales[0]))
        descriptors.append(describe_variance(series.values, epsilon))
        family_counts[series.family] = family_counts.get(series.family, 0) + 1

    descriptor_means = {
        key: float(np.nanmean([descriptor[key] for descriptor in descriptors]))
        for key in descriptors[0]
    }
    metadata = SyntheticTaskMetadata(
        prior=prior,
        seed=seed,
        sample_count=sample_count,
        descriptor_means=descriptor_means,
        family_counts=family_counts,
    )
    return SyntheticWindowDataset(
        contexts=np.stack(contexts),
        normalized_targets=np.stack(normalized_targets),
        cumulative_targets=np.stack(cumulative_targets),
        locations=np.asarray(locations),
        scales=np.asarray(scales),
        metadata=metadata,
    )
