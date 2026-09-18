"""One-way evaluation of frozen synthetic checkpoints on sealed pilot-only data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from synthetic_finance_tsfm.baselines import (
    DirectLogHAR,
    historical_mean_forecast,
    last_value_forecast,
)
from synthetic_finance_tsfm.config import ProjectConfig, load_config
from synthetic_finance_tsfm.data.contracts import validate_daily_frame
from synthetic_finance_tsfm.data.windows import (
    ForecastWindows,
    build_forecast_windows,
    reconstruct_cumulative_forecasts,
)
from synthetic_finance_tsfm.evaluation.metrics import evaluate_point_forecasts, qlike
from synthetic_finance_tsfm.models import PatchTSTStyleConfig, PatchTSTStyleRegressor
from synthetic_finance_tsfm.utils.io import atomic_write_text, environment_metadata, write_json


def _load_panel(manifest_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest["panel"].get("pilot_only") or manifest.get("final_evaluation_eligible"):
        raise ValueError("Real pilot evaluation requires a pilot-only, final-ineligible manifest")
    panel_path = Path(manifest["panel"]["path"])
    panel = pd.read_csv(panel_path, parse_dates=["date"])
    validate_daily_frame(panel)
    if not panel["pilot_only"].all():
        raise ValueError("Every real-pilot row must be marked pilot_only")
    return panel, manifest


def _load_frozen_model(
    checkpoint_path: Path, expected_config: ProjectConfig
) -> tuple[PatchTSTStyleRegressor, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_config = PatchTSTStyleConfig(**payload["model_config"])
    if model_config.context_length != expected_config.data.context_length:
        raise ValueError("Checkpoint context length does not match its run configuration")
    if model_config.num_targets != len(expected_config.data.horizons):
        raise ValueError("Checkpoint target count does not match configured horizons")
    model = PatchTSTStyleRegressor(model_config)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


@torch.no_grad()
def _model_forecasts(
    model: PatchTSTStyleRegressor,
    windows: ForecastWindows,
    horizons: tuple[int, ...],
    batch_size: int = 128,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for start in range(0, len(windows.contexts), batch_size):
        context = torch.as_tensor(windows.contexts[start : start + batch_size], dtype=torch.float32)
        predictions.append(model(context).numpy())
    normalized = np.concatenate(predictions)
    return reconstruct_cumulative_forecasts(
        normalized,
        locations=windows.context_locations,
        scales=windows.context_scales,
        horizons=horizons,
    )


def _baseline_forecasts(
    rv: np.ndarray,
    windows: ForecastWindows,
    *,
    horizons: tuple[int, ...],
    epsilon: float,
) -> dict[str, np.ndarray]:
    last_rows: list[np.ndarray] = []
    mean_rows: list[np.ndarray] = []
    har_rows: list[np.ndarray] = []
    for origin in windows.origins:
        history = rv[: int(origin) + 1]
        last_rows.append(last_value_forecast(history, horizons))
        mean_rows.append(historical_mean_forecast(history, horizons))
        har_rows.append(
            DirectLogHAR(horizons=horizons, epsilon=epsilon).fit(history).predict(history)
        )
    return {
        "last_value": np.stack(last_rows),
        "historical_mean": np.stack(mean_rows),
        "direct_log_har": np.stack(har_rows),
    }


def _metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    keys = ["model", "prior", "seed", "asset_id", "horizon"]
    for group_key, group in predictions.groupby(keys, dropna=False, sort=True):
        model, prior, seed, asset_id, horizon = group_key
        metrics = evaluate_point_forecasts(group["actual"].to_numpy(), group["forecast"].to_numpy())
        records.append(
            {
                "model": model,
                "prior": prior,
                "seed": seed,
                "asset_id": asset_id,
                "horizon": int(horizon),
                **metrics,
            }
        )
    result = pd.DataFrame.from_records(records)
    har = result.loc[result["model"] == "direct_log_har", ["asset_id", "horizon", "mean_qlike"]]
    har = har.rename(columns={"mean_qlike": "har_mean_qlike"})
    result = result.merge(har, on=["asset_id", "horizon"], how="left", validate="many_to_one")
    result["loss_ratio_to_har"] = result["mean_qlike"] / result["har_mean_qlike"]
    return result


def evaluate_real_pilot(
    *,
    manifest_path: str | Path,
    run_directories: list[str | Path],
    output_dir: str | Path,
) -> Path:
    """Evaluate every supplied frozen checkpoint once, without real-data model selection."""
    if len(run_directories) < 2:
        raise ValueError("Real pilot requires multiple frozen seed runs")
    panel, manifest = _load_panel(Path(manifest_path))
    run_paths = [Path(path) for path in run_directories]
    run_configs = [load_config(path / "config.yaml") for path in run_paths]
    reference = run_configs[0]
    for config in run_configs[1:]:
        if config.data != reference.data or config.model != reference.model:
            raise ValueError(
                "All frozen pilot checkpoints must share data and model configurations"
            )
        if config.training != reference.training:
            raise ValueError("All frozen pilot checkpoints must share the same training budget")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prediction_records: list[dict[str, Any]] = []
    asset_windows: dict[str, tuple[np.ndarray, pd.Series, ForecastWindows]] = {}

    for asset_id, asset_frame in panel.groupby("asset_id", sort=True):
        asset_frame = asset_frame.sort_values("date").reset_index(drop=True)
        rv = asset_frame["realized_variance"].to_numpy(dtype=np.float64)
        windows = build_forecast_windows(
            rv,
            context_length=reference.data.context_length,
            horizons=reference.data.horizons,
            epsilon=reference.data.epsilon,
        )
        asset_windows[str(asset_id)] = (rv, asset_frame["date"], windows)
        baselines = _baseline_forecasts(
            rv,
            windows,
            horizons=reference.data.horizons,
            epsilon=reference.data.epsilon,
        )
        for baseline_name, forecast in baselines.items():
            losses, _ = qlike(windows.cumulative_targets, forecast)
            for row, origin in enumerate(windows.origins):
                for column, horizon in enumerate(reference.data.horizons):
                    prediction_records.append(
                        {
                            "model": baseline_name,
                            "prior": "real_history",
                            "seed": -1,
                            "asset_id": str(asset_id),
                            "origin_date": str(asset_frame.loc[int(origin), "date"]),
                            "horizon": horizon,
                            "actual": windows.cumulative_targets[row, column],
                            "forecast": forecast[row, column],
                            "qlike": losses[row, column],
                        }
                    )

    checkpoint_records: list[dict[str, Any]] = []
    for run_path, config in zip(run_paths, run_configs, strict=True):
        for prior in config.experiment.priors:
            checkpoint_path = run_path / "checkpoints" / f"{prior}.pt"
            model, payload = _load_frozen_model(checkpoint_path, config)
            checkpoint_records.append(
                {
                    "run": str(run_path),
                    "checkpoint": str(checkpoint_path),
                    "prior": prior,
                    "seed": config.experiment.seed,
                    "checkpoint_step": payload["step"],
                    "synthetic_validation_log_mse": payload["validation_log_mse"],
                }
            )
            for asset_id, (_, dates, windows) in asset_windows.items():
                forecast = _model_forecasts(model, windows, config.data.horizons)
                losses, _ = qlike(windows.cumulative_targets, forecast)
                for row, origin in enumerate(windows.origins):
                    for column, horizon in enumerate(config.data.horizons):
                        prediction_records.append(
                            {
                                "model": "patchtst_style",
                                "prior": prior,
                                "seed": config.experiment.seed,
                                "asset_id": asset_id,
                                "origin_date": str(dates.iloc[int(origin)]),
                                "horizon": horizon,
                                "actual": windows.cumulative_targets[row, column],
                                "forecast": forecast[row, column],
                                "qlike": losses[row, column],
                            }
                        )

    predictions = pd.DataFrame.from_records(prediction_records)
    metric_table = _metric_rows(predictions)
    model_metrics = metric_table.loc[metric_table["model"] == "patchtst_style"].copy()
    five_day = model_metrics.loc[model_metrics["horizon"] == 5]
    paired = five_day.pivot_table(
        index=["seed", "asset_id"], columns="prior", values="mean_qlike", aggfunc="first"
    ).dropna()
    volatility_wins = int((paired["volatility"] < paired["generic"]).sum())
    comparison_count = int(len(paired))
    median_ratios = (
        model_metrics.groupby(["prior", "horizon"])["loss_ratio_to_har"]
        .median()
        .rename("median_asset_loss_ratio_to_har")
        .reset_index()
    )
    pilot_summary = {
        "result_label": "pilot_only_real_evaluation",
        "manifest": str(manifest_path),
        "panel_sha256": manifest["panel"]["sha256"],
        "checkpoints": checkpoint_records,
        "forecast_origins_per_asset": {
            asset: len(windows.origins) for asset, (_, _, windows) in asset_windows.items()
        },
        "primary_five_day": {
            "volatility_prior_wins": volatility_wins,
            "paired_asset_seed_comparisons": comparison_count,
            "directional_majority": bool(volatility_wins > comparison_count / 2),
        },
        "median_asset_loss_ratios_to_har": median_ratios.to_dict(orient="records"),
        "gate_status": {
            "pipeline": "pass",
            "synthetic_learning": "pass_before_real_evaluation",
            "prior_contrast": "pass_before_real_evaluation",
            "real_pilot_direction": ("pass" if volatility_wins > comparison_count / 2 else "fail"),
            "final_evaluation": "not_run",
        },
        "environment": environment_metadata(Path.cwd()),
    }
    atomic_write_text(output / "predictions.csv", predictions.to_csv(index=False))
    atomic_write_text(output / "asset_metrics.csv", metric_table.to_csv(index=False))
    write_json(output / "metrics.json", pilot_summary)
    return output / "metrics.json"
