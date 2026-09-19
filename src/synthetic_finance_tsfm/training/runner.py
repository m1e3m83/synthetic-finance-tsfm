"""Configuration-driven equal-budget synthetic training and evaluation."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, replace
from itertools import cycle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from synthetic_finance_tsfm.config import ProjectConfig, load_config
from synthetic_finance_tsfm.data.windows import reconstruct_cumulative_forecasts
from synthetic_finance_tsfm.evaluation.metrics import evaluate_point_forecasts, qlike
from synthetic_finance_tsfm.models import PatchTSTStyleConfig, PatchTSTStyleRegressor
from synthetic_finance_tsfm.training.dataset import SyntheticWindowDataset, make_synthetic_dataset
from synthetic_finance_tsfm.utils.io import (
    atomic_write_text,
    environment_metadata,
    write_json,
    write_yaml,
)
from synthetic_finance_tsfm.utils.seed import derived_seed, set_global_seed


def _device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    selected = torch.device(requested)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if selected.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return selected


def build_model(config: ProjectConfig) -> PatchTSTStyleRegressor:
    model_config = PatchTSTStyleConfig(
        context_length=config.data.context_length,
        patch_length=config.model.patch_length,
        patch_stride=config.model.patch_stride,
        num_hidden_layers=config.model.num_hidden_layers,
        d_model=config.model.d_model,
        num_attention_heads=config.model.num_attention_heads,
        ffn_dim=config.model.ffn_dim,
        dropout=config.model.dropout,
        num_targets=len(config.data.horizons),
    )
    return PatchTSTStyleRegressor(model_config)


def _atomic_torch_save(value: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


@torch.no_grad()
def _validation_log_mse(
    model: nn.Module, dataset: SyntheticWindowDataset, device: torch.device, batch_size: int
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for contexts, targets, *_ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        outputs = model(contexts.to(device))
        loss = nn.functional.mse_loss(outputs, targets.to(device), reduction="sum")
        total += float(loss.cpu())
        count += targets.numel()
    return total / count


def _train_one(
    *,
    prior: str,
    config: ProjectConfig,
    train_dataset: SyntheticWindowDataset,
    validation_dataset: SyntheticWindowDataset,
    run_dir: Path,
    device: torch.device,
) -> tuple[PatchTSTStyleRegressor, dict[str, Any]]:
    set_global_seed(config.experiment.seed)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    loader_generator = torch.Generator().manual_seed(
        derived_seed(config.experiment.seed, prior, "loader")
    )
    loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        generator=loader_generator,
        drop_last=False,
    )
    batches = cycle(loader)
    best_validation = float("inf")
    best_step = 0
    checkpoint = run_dir / "checkpoints" / f"{prior}.pt"
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for step in range(1, config.training.optimizer_steps + 1):
        contexts, targets, *_ = next(batches)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        predictions = model(contexts.to(device))
        loss = nn.functional.mse_loss(predictions, targets.to(device))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss for {prior} at step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        should_validate = (
            step % config.training.validation_interval == 0
            or step == config.training.optimizer_steps
        )
        if should_validate:
            validation_loss = _validation_log_mse(
                model, validation_dataset, device, config.training.batch_size
            )
            history.append(
                {
                    "step": step,
                    "train_log_mse": float(loss.detach().cpu()),
                    "validation_log_mse": validation_loss,
                }
            )
            if validation_loss < best_validation:
                best_validation = validation_loss
                best_step = step
                _atomic_torch_save(
                    {
                        "model_state": model.state_dict(),
                        "model_config": asdict(model.config),
                        "prior": prior,
                        "step": step,
                        "validation_log_mse": validation_loss,
                    },
                    checkpoint,
                )

    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(payload["model_state"])
    elapsed = time.perf_counter() - started
    return model, {
        "prior": prior,
        "parameter_count": model.parameter_count,
        "optimizer_steps": config.training.optimizer_steps,
        "best_step": best_step,
        "best_validation_log_mse": best_validation,
        "runtime_seconds": elapsed,
        "history": history,
        "checkpoint": str(checkpoint),
        "train_generator": asdict(train_dataset.metadata),
        "validation_generator": asdict(validation_dataset.metadata),
    }


@torch.no_grad()
def _evaluate_cross_prior(
    *,
    model: PatchTSTStyleRegressor,
    model_prior: str,
    evaluation_prior: str,
    dataset: SyntheticWindowDataset,
    horizons: tuple[int, ...],
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    normalized: list[np.ndarray] = []
    actual: list[np.ndarray] = []
    locations: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    for context, _, cumulative, location, scale in DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    ):
        normalized.append(model(context.to(device)).cpu().numpy())
        actual.append(cumulative.numpy())
        locations.append(location.numpy())
        scales.append(scale.numpy())
        contexts.append(context.numpy())
    normalized_array = np.concatenate(normalized)
    actual_array = np.concatenate(actual)
    location_array = np.concatenate(locations)
    scale_array = np.concatenate(scales)
    context_array = np.concatenate(contexts)
    forecast = reconstruct_cumulative_forecasts(
        normalized_array,
        locations=location_array,
        scales=scale_array,
        horizons=horizons,
    )
    metrics = evaluate_point_forecasts(actual_array, forecast)

    last_log_value = context_array[:, -1] * scale_array + location_array
    last_daily_variance = np.exp(last_log_value)
    last_forecast = last_daily_variance[:, None] * np.asarray(horizons)[None, :]
    metrics["last_value"] = evaluate_point_forecasts(actual_array, last_forecast)
    metrics["beats_last_value_qlike"] = bool(
        metrics["mean_qlike"] < metrics["last_value"]["mean_qlike"]
    )

    losses, _ = qlike(actual_array, forecast)
    records: list[dict[str, Any]] = []
    for sample in range(actual_array.shape[0]):
        for horizon_index, horizon in enumerate(horizons):
            records.append(
                {
                    "model_prior": model_prior,
                    "evaluation_prior": evaluation_prior,
                    "sample": sample,
                    "horizon": horizon,
                    "actual": actual_array[sample, horizon_index],
                    "forecast": forecast[sample, horizon_index],
                    "qlike": losses[sample, horizon_index],
                }
            )
    return metrics, pd.DataFrame.from_records(records)


def run_experiment(config_path: str | Path, *, seed_override: int | None = None) -> Path:
    """Train each implemented prior and write a cross-prior smoke/pilot artifact set."""
    config = load_config(config_path)
    if seed_override is not None:
        if seed_override < 0:
            raise ValueError("seed_override must be non-negative")
        config = replace(
            config,
            experiment=replace(config.experiment, seed=seed_override),
        )
    unimplemented = set(config.experiment.priors) - {"generic", "volatility"}
    if unimplemented:
        raise NotImplementedError(
            "The gated first delivery supports Generic and Volatility only; requested "
            f"{sorted(unimplemented)}"
        )
    run_dir = Path(config.experiment.output_dir) / (
        f"{config.experiment.name}-seed{config.experiment.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(run_dir / "config.yaml", config.to_dict())
    write_json(run_dir / "environment.json", environment_metadata(Path.cwd()))
    device = _device(config.training.device)

    training_sets: dict[str, SyntheticWindowDataset] = {}
    validation_sets: dict[str, SyntheticWindowDataset] = {}
    for prior in config.experiment.priors:
        training_sets[prior] = make_synthetic_dataset(
            prior=prior,
            sample_count=config.experiment.train_samples,
            series_length=config.experiment.series_length,
            context_length=config.data.context_length,
            horizons=config.data.horizons,
            epsilon=config.data.epsilon,
            seed=derived_seed(config.experiment.seed, "train"),
        )
        validation_sets[prior] = make_synthetic_dataset(
            prior=prior,
            sample_count=config.experiment.validation_samples,
            series_length=config.experiment.series_length,
            context_length=config.data.context_length,
            horizons=config.data.horizons,
            epsilon=config.data.epsilon,
            seed=derived_seed(config.experiment.seed, "validation"),
        )

    models: dict[str, PatchTSTStyleRegressor] = {}
    training_metrics: dict[str, Any] = {}
    for prior in config.experiment.priors:
        model, summary = _train_one(
            prior=prior,
            config=config,
            train_dataset=training_sets[prior],
            validation_dataset=validation_sets[prior],
            run_dir=run_dir,
            device=device,
        )
        models[prior] = model
        training_metrics[prior] = summary
        write_json(run_dir / "training" / f"{prior}.json", summary)

    cross_prior: dict[str, Any] = {}
    prediction_frames: list[pd.DataFrame] = []
    for model_prior, model in models.items():
        cross_prior[model_prior] = {}
        for evaluation_prior, dataset in validation_sets.items():
            metrics, predictions = _evaluate_cross_prior(
                model=model,
                model_prior=model_prior,
                evaluation_prior=evaluation_prior,
                dataset=dataset,
                horizons=config.data.horizons,
                batch_size=config.training.batch_size,
                device=device,
            )
            cross_prior[model_prior][evaluation_prior] = metrics
            prediction_frames.append(predictions)

    result = {
        "result_label": "smoke" if "smoke" in config.experiment.name else "synthetic_pilot",
        "device": str(device),
        "equal_budget": {
            "train_samples_per_prior": config.experiment.train_samples,
            "validation_samples_per_prior": config.experiment.validation_samples,
            "optimizer_steps_per_prior": config.training.optimizer_steps,
            "batch_size": config.training.batch_size,
            "seed": config.experiment.seed,
        },
        "training": training_metrics,
        "training_completion_verified": True,
        "cross_prior": cross_prior,
        "gate_status": {
            "pipeline": "partial_smoke_only",
            "synthetic_learning": "inspect_cross_prior_metrics",
            "prior_contrast": "not_evaluable_from_one_smoke_seed",
            "real_pilot": "not_run",
        },
    }
    write_json(run_dir / "metrics.json", result)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    atomic_write_text(run_dir / "predictions.csv", predictions.to_csv(index=False))
    return run_dir


def evaluate_synthetic_checkpoints(
    config_path: str | Path, *, seed_override: int | None = None
) -> Path:
    """Finish synthetic evaluation from existing checkpoints without retraining."""
    config = load_config(config_path)
    if seed_override is not None:
        if seed_override < 0:
            raise ValueError("seed_override must be non-negative")
        config = replace(
            config,
            experiment=replace(config.experiment, seed=seed_override),
        )
    unimplemented = set(config.experiment.priors) - {"generic", "volatility"}
    if unimplemented:
        raise NotImplementedError(
            "Synthetic checkpoint evaluation currently supports Generic and Volatility only; "
            f"requested {sorted(unimplemented)}"
        )

    run_dir = Path(config.experiment.output_dir) / (
        f"{config.experiment.name}-seed{config.experiment.seed}"
    )
    recorded_config_path = run_dir / "config.yaml"
    if not recorded_config_path.exists():
        raise FileNotFoundError(recorded_config_path)
    if load_config(recorded_config_path) != config:
        raise ValueError(
            f"The requested configuration does not match the recorded run in {run_dir}"
        )

    device = _device(config.training.device)
    validation_sets: dict[str, SyntheticWindowDataset] = {}
    for prior in config.experiment.priors:
        validation_sets[prior] = make_synthetic_dataset(
            prior=prior,
            sample_count=config.experiment.validation_samples,
            series_length=config.experiment.series_length,
            context_length=config.data.context_length,
            horizons=config.data.horizons,
            epsilon=config.data.epsilon,
            seed=derived_seed(config.experiment.seed, "validation"),
        )

    models: dict[str, PatchTSTStyleRegressor] = {}
    training_metrics: dict[str, Any] = {}
    completion_verified = True
    for prior in config.experiment.priors:
        checkpoint = run_dir / "checkpoints" / f"{prior}.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        if payload.get("prior") != prior:
            raise ValueError(f"Prior mismatch in {checkpoint}")
        model = build_model(config).to(device)
        if payload.get("model_config") != asdict(model.config):
            raise ValueError(f"Model configuration mismatch in {checkpoint}")
        model.load_state_dict(payload["model_state"])
        models[prior] = model
        completion_record = run_dir / "training" / f"{prior}.json"
        if completion_record.exists():
            summary = json.loads(completion_record.read_text(encoding="utf-8"))
            if summary.get("prior") != prior:
                raise ValueError(f"Prior mismatch in {completion_record}")
            if summary.get("optimizer_steps") != config.training.optimizer_steps:
                raise ValueError(f"Training-budget mismatch in {completion_record}")
            summary["recovered_from_checkpoint"] = True
            training_metrics[prior] = summary
        else:
            completion_verified = False
            training_metrics[prior] = {
                "prior": prior,
                "parameter_count": model.parameter_count,
                "optimizer_steps": config.training.optimizer_steps,
                "best_step": payload["step"],
                "best_validation_log_mse": payload["validation_log_mse"],
                "checkpoint": str(checkpoint),
                "recovered_from_checkpoint": True,
                "completion_record_missing": True,
                "validation_generator": asdict(validation_sets[prior].metadata),
            }

    cross_prior: dict[str, Any] = {}
    prediction_frames: list[pd.DataFrame] = []
    for model_prior, model in models.items():
        cross_prior[model_prior] = {}
        for evaluation_prior, dataset in validation_sets.items():
            metrics, predictions = _evaluate_cross_prior(
                model=model,
                model_prior=model_prior,
                evaluation_prior=evaluation_prior,
                dataset=dataset,
                horizons=config.data.horizons,
                batch_size=config.training.batch_size,
                device=device,
            )
            cross_prior[model_prior][evaluation_prior] = metrics
            prediction_frames.append(predictions)

    result = {
        "result_label": "synthetic_pilot",
        "device": str(device),
        "equal_budget": {
            "train_samples_per_prior": config.experiment.train_samples,
            "validation_samples_per_prior": config.experiment.validation_samples,
            "optimizer_steps_per_prior": config.training.optimizer_steps,
            "batch_size": config.training.batch_size,
            "seed": config.experiment.seed,
        },
        "training": training_metrics,
        "training_completion_verified": completion_verified,
        "cross_prior": cross_prior,
        "gate_status": {
            "pipeline": (
                "checkpoint_evaluation_complete"
                if completion_verified
                else "checkpoint_evaluation_complete_unverified_training"
            ),
            "synthetic_learning": "inspect_cross_prior_metrics",
            "prior_contrast": "aggregate_across_seeds",
            "real_pilot": "not_run",
        },
    }
    write_json(run_dir / "metrics.json", result)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    atomic_write_text(run_dir / "predictions.csv", predictions.to_csv(index=False))
    return run_dir


def summarize_calibration(run_directories: list[str | Path], output_path: str | Path) -> Path:
    """Aggregate equal-budget seed runs and evaluate predeclared synthetic gates."""
    if len(run_directories) < 2:
        raise ValueError("At least two seed runs are required for calibration summary")
    runs: list[dict[str, Any]] = []
    for directory in run_directories:
        metrics_path = Path(directory) / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        run = json.loads(metrics_path.read_text(encoding="utf-8"))
        if run.get("training_completion_verified") is False:
            raise ValueError(f"Training completion is not verified for {directory}")
        runs.append(run)

    reference_budget = {
        key: value for key, value in runs[0]["equal_budget"].items() if key != "seed"
    }
    for run in runs[1:]:
        comparison = {key: value for key, value in run["equal_budget"].items() if key != "seed"}
        if comparison != reference_budget:
            raise ValueError("Calibration runs do not have equal non-seed budgets")

    priors = sorted(runs[0]["cross_prior"])
    matrix: dict[str, dict[str, dict[str, float]]] = {}
    for model_prior in priors:
        matrix[model_prior] = {}
        for evaluation_prior in priors:
            losses = np.asarray(
                [run["cross_prior"][model_prior][evaluation_prior]["mean_qlike"] for run in runs],
                dtype=np.float64,
            )
            matrix[model_prior][evaluation_prior] = {
                "mean_qlike": float(losses.mean()),
                "std_qlike": float(losses.std(ddof=1)),
            }

    own_domain_wins: dict[str, int] = {}
    matched_model_wins: dict[str, int] = {}
    for prior in priors:
        own_domain_wins[prior] = sum(
            bool(run["cross_prior"][prior][prior]["beats_last_value_qlike"]) for run in runs
        )
        alternatives = [candidate for candidate in priors if candidate != prior]
        matched_model_wins[prior] = sum(
            all(
                run["cross_prior"][prior][prior]["mean_qlike"]
                < run["cross_prior"][alternative][prior]["mean_qlike"]
                for alternative in alternatives
            )
            for run in runs
        )

    required_wins = len(runs) // 2 + 1
    synthetic_learning_pass = all(wins >= required_wins for wins in own_domain_wins.values())
    distinguishability_pass = all(wins >= required_wins for wins in matched_model_wins.values())
    summary = {
        "result_label": "synthetic_calibration",
        "run_count": len(runs),
        "seeds": [run["equal_budget"]["seed"] for run in runs],
        "equal_non_seed_budget": reference_budget,
        "cross_prior_mean_matrix": matrix,
        "own_domain_wins_against_last_value": own_domain_wins,
        "matched_model_wins_on_each_prior": matched_model_wins,
        "required_majority_wins": required_wins,
        "gate_status": {
            "synthetic_learning": "pass" if synthetic_learning_pass else "fail",
            "prior_distinguishability": "pass" if distinguishability_pass else "fail",
            "real_pilot": "not_run",
        },
    }
    destination = Path(output_path)
    write_json(destination, summary)
    return destination
