from pathlib import Path

import pytest

from synthetic_finance_tsfm.config import load_config


def test_smoke_config_is_valid() -> None:
    config = load_config(Path("configs/pilot/smoke.yaml"))
    assert config.experiment.priors == ("generic", "volatility")
    assert config.data.horizons == (1, 5, 22)
    assert config.training.optimizer_steps == 20


def test_full_config_is_gated_but_structurally_valid() -> None:
    config = load_config(Path("configs/full/main.yaml"))
    assert "mixed" in config.experiment.priors


def test_calibration_config_has_medium_equal_budget() -> None:
    config = load_config(Path("configs/pilot/calibration.yaml"))
    assert config.experiment.priors == ("generic", "volatility")
    assert config.training.optimizer_steps == 500
    assert config.experiment.train_samples == 2048


def test_benchmark_uses_the_predeclared_pilot_architecture() -> None:
    benchmark = load_config(Path("configs/pilot/benchmark.yaml"))
    pilot = load_config(Path("configs/pilot/pilot.yaml"))
    assert benchmark.data == pilot.data
    assert benchmark.model == pilot.model
    assert benchmark.training.batch_size == pilot.training.batch_size


def test_duplicate_prior_is_rejected(tmp_path: Path) -> None:
    original = Path("configs/pilot/smoke.yaml").read_text(encoding="utf-8")
    invalid = original.replace("[generic, volatility]", "[generic, generic]")
    path = tmp_path / "invalid.yaml"
    path.write_text(invalid, encoding="utf-8")
    with pytest.raises(ValueError, match="exactly once"):
        load_config(path)
