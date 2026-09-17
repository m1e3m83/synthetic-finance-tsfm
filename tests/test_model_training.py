from pathlib import Path

import numpy as np
import torch

from synthetic_finance_tsfm.config import load_config
from synthetic_finance_tsfm.models import PatchTSTStyleConfig, PatchTSTStyleRegressor
from synthetic_finance_tsfm.training.dataset import make_synthetic_dataset
from synthetic_finance_tsfm.training.runner import build_model


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
