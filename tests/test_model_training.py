import json
from pathlib import Path

import numpy as np
import torch

from synthetic_finance_tsfm.config import load_config
from synthetic_finance_tsfm.models import PatchTSTStyleConfig, PatchTSTStyleRegressor
from synthetic_finance_tsfm.training.dataset import make_synthetic_dataset
from synthetic_finance_tsfm.training.runner import build_model, summarize_calibration


def _tiny_model() -> PatchTSTStyleRegressor:
    return PatchTSTStyleRegressor(
        PatchTSTStyleConfig(
            context_length=32,
            patch_length=8,
            patch_stride=4,
            num_hidden_layers=1,
            d_model=16,
            num_attention_heads=4,
            ffn_dim=32,
            dropout=0.0,
            num_targets=3,
        )
    )


def test_model_forward_backward_and_head_shape() -> None:
    model = _tiny_model()
    inputs = torch.randn(4, 32)
    targets = torch.randn(4, 3)
    outputs = model(inputs)
    assert outputs.shape == (4, 3)
    loss = torch.nn.functional.mse_loss(outputs, targets)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(4)
    model = _tiny_model().eval()
    inputs = torch.randn(2, 32)
    expected = model(inputs).detach()
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = _tiny_model().eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(restored(inputs), expected)


def test_synthetic_dataset_is_deterministic() -> None:
    kwargs = {
        "prior": "volatility",
        "sample_count": 8,
        "series_length": 80,
        "context_length": 32,
        "horizons": (1, 5, 22),
        "epsilon": 1e-10,
        "seed": 31,
    }
    first = make_synthetic_dataset(**kwargs)
    second = make_synthetic_dataset(**kwargs)
    torch.testing.assert_close(first.contexts, second.contexts, rtol=0, atol=0)
    torch.testing.assert_close(first.normalized_targets, second.normalized_targets, rtol=0, atol=0)


def test_configured_model_parameterization() -> None:
    config = load_config(Path("configs/pilot/smoke.yaml"))
    model = build_model(config)
    output = model(torch.zeros(2, config.data.context_length))
    assert output.shape == (2, len(config.data.horizons))
    assert model.parameter_count > 0
    assert np.isfinite(output.detach().numpy()).all()


def test_calibration_summary_checks_majority_gates(tmp_path: Path) -> None:
    run_directories = []
    for index, seed in enumerate((1, 2, 3)):
        run_directory = tmp_path / f"run-{seed}"
        run_directory.mkdir()
        generic_own = 0.5 if index < 2 else 1.2
        volatility_own = 0.4 if index < 2 else 1.1
        payload = {
            "equal_budget": {
                "seed": seed,
                "train_samples_per_prior": 10,
                "validation_samples_per_prior": 5,
                "optimizer_steps_per_prior": 2,
                "batch_size": 2,
            },
            "cross_prior": {
                "generic": {
                    "generic": {
                        "mean_qlike": generic_own,
                        "beats_last_value_qlike": index < 2,
                    },
                    "volatility": {
                        "mean_qlike": 0.9,
                        "beats_last_value_qlike": False,
                    },
                },
                "volatility": {
                    "generic": {
                        "mean_qlike": 0.8,
                        "beats_last_value_qlike": False,
                    },
                    "volatility": {
                        "mean_qlike": volatility_own,
                        "beats_last_value_qlike": index < 2,
                    },
                },
            },
        }
        (run_directory / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
        run_directories.append(run_directory)

    output = tmp_path / "summary.json"
    summarize_calibration(run_directories, output)
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["gate_status"]["synthetic_learning"] == "pass"
    assert summary["gate_status"]["prior_distinguishability"] == "pass"
