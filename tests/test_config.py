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


def test_duplicate_prior_is_rejected(tmp_path: Path) -> None:
    original = Path("configs/pilot/smoke.yaml").read_text(encoding="utf-8")
    invalid = original.replace("[generic, volatility]", "[generic, generic]")
    path = tmp_path / "invalid.yaml"
    path.write_text(invalid, encoding="utf-8")
    with pytest.raises(ValueError, match="exactly once"):
        load_config(path)
