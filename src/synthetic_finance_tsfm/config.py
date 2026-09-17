"""Validated experiment configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    output_dir: str
    seed: int
    priors: tuple[str, ...]
    train_samples: int
    validation_samples: int
    series_length: int


@dataclass(frozen=True)
class DataConfig:
    context_length: int
    horizons: tuple[int, ...]
    epsilon: float


@dataclass(frozen=True)
class ModelConfig:
    patch_length: int
    patch_stride: int
    num_hidden_layers: int
    d_model: int
    num_attention_heads: int
    ffn_dim: int
    dropout: float


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    optimizer_steps: int
    learning_rate: float
    weight_decay: float
    validation_interval: int
    device: str


@dataclass(frozen=True)
class ProjectConfig:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate a YAML experiment configuration."""
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")

    try:
        experiment_raw = dict(raw["experiment"])
        experiment_raw["priors"] = tuple(experiment_raw["priors"])
        data_raw = dict(raw["data"])
        data_raw["horizons"] = tuple(int(h) for h in data_raw["horizons"])
        config = ProjectConfig(
            experiment=ExperimentConfig(**experiment_raw),
            data=DataConfig(**data_raw),
            model=ModelConfig(**raw["model"]),
            training=TrainingConfig(**raw["training"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid configuration in {source}: {exc}") from exc

    validate_config(config)
    return config


def validate_config(config: ProjectConfig) -> None:
    """Fail early on invalid or computationally incoherent settings."""
    supported_priors = {"generic", "volatility", "regime_tail", "mixed"}
    unknown = set(config.experiment.priors) - supported_priors
    if unknown:
        raise ValueError(f"Unsupported priors: {sorted(unknown)}")
    if not config.experiment.priors:
        raise ValueError("At least one prior is required")
    if len(set(config.experiment.priors)) != len(config.experiment.priors):
        raise ValueError("Each prior must appear exactly once")
    if config.data.context_length < config.model.patch_length:
        raise ValueError("context_length must be at least patch_length")
    if config.model.patch_stride <= 0 or config.model.patch_length <= 0:
        raise ValueError("Patch length and stride must be positive")
    if config.model.d_model % config.model.num_attention_heads != 0:
        raise ValueError("d_model must be divisible by num_attention_heads")
    if not config.data.horizons or min(config.data.horizons) <= 0:
        raise ValueError("Horizons must be positive")
    if tuple(sorted(set(config.data.horizons))) != config.data.horizons:
        raise ValueError("Horizons must be unique and sorted")
    minimum_length = config.data.context_length + max(config.data.horizons)
    if config.experiment.series_length < minimum_length:
        raise ValueError(f"series_length must be at least {minimum_length}")
    positive_ints = {
        "train_samples": config.experiment.train_samples,
        "validation_samples": config.experiment.validation_samples,
        "batch_size": config.training.batch_size,
        "optimizer_steps": config.training.optimizer_steps,
        "validation_interval": config.training.validation_interval,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if config.data.epsilon <= 0:
        raise ValueError("epsilon must be positive")
