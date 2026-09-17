# Synthetic Pre-training for Financial Time-Series Models

Reproducible research code for testing whether finance-aware synthetic priors improve zero-shot
realized-volatility forecasts relative to generic synthetic time-series priors.

## Status

The repository currently supports the first CPU-safe research delivery:

- deterministic Generic and Volatility synthetic priors;
- leakage-safe cumulative 1-, 5-, and 22-day variance targets;
- context-only log normalization and positive forecast reconstruction;
- Last Value, Historical Mean, and direct Log-HAR baselines;
- QLIKE and secondary point metrics;
- a randomly initialized PatchTST-style encoder with a custom volatility head;
- equal-budget training and a 2-by-2 synthetic cross-prior evaluation;
- configuration, checkpoints, predictions, metrics, and environment metadata;
- unit tests for target indexing, leakage, generators, metrics, baselines, and model behavior.

No real-data result is included yet. Binance remains a pilot-only engineering source, and VOLARE
access and schema have not been verified.

## Research question

Can a model trained entirely on synthetic time series forecast realized volatility on unseen real
financial series, and which components of the synthetic pretraining distribution enable transfer?

The main comparison holds architecture and training budget fixed while changing only the synthetic
prior. The strict zero-shot rules are recorded in
[`docs/research_protocol.md`](docs/research_protocol.md).

## Installation

Python 3.11 is recommended.

```bash
uv sync --extra dev
```

If compatible NumPy, pandas, PyYAML, and PyTorch installations already exist, an editable install can
be made without resolving dependencies:

```bash
python3 -m pip install -e . --no-deps
```

No pretrained model weights are downloaded.

## Validate the configuration

```bash
uv run synthetic-finance-tsfm validate-config --config configs/pilot/smoke.yaml
```

## Inspect the priors

```bash
uv run synthetic-finance-tsfm describe-prior --prior generic --length 512 --seed 42
uv run synthetic-finance-tsfm describe-prior --prior volatility --length 512 --seed 42
```

## Run tests and linting

```bash
uv run pytest
uv run ruff check .
```

## Run the end-to-end smoke experiment

```bash
uv run synthetic-finance-tsfm run --config configs/pilot/smoke.yaml
```

The run writes to:

```text
runs/smoke/smoke_equal_budget-seed17/
    config.yaml
    environment.json
    metrics.json
    predictions.csv
    checkpoints/
```

Smoke results verify plumbing only. A single small run cannot establish a prior advantage or support a
real-data claim.

## Repository structure

```text
configs/                         Smoke, pilot, and gated full configurations
docs/                            Protocol, implementation plan, and data documentation
src/synthetic_finance_tsfm/
    baselines/                   Naive and Log-HAR baselines
    data/                        Schemas, realized variance, windows, splits, Binance adapter
    evaluation/                  Metrics and dependence-aware summaries
    models/                      PatchTST-style encoder and custom head
    synthetic/                   Generic and Volatility prior bank
    training/                    Dataset construction and equal-budget runner
tests/                           Unit and integration-oriented tests
```

## Experimental progression

1. Pass data, target, baseline, and smoke tests.
2. Freeze a pilot-only Binance manifest and verify data quality.
3. Train Generic and Volatility models with equal budgets across three seeds.
4. Evaluate the frozen checkpoints once on the real pilot.
5. Scale to Regime-and-Tail and Mixed priors only if the pilot gates pass.
6. Add ForecastPFN and public foundation models as contextual baselines after the core result is
   stable.

## Important limitations

- The local smoke configuration is intentionally tiny and does not represent final model capacity.
- Regime-and-Tail and Mixed priors are deliberately gated and not yet implemented.
- ForecastPFN, GARCH, quantile heads, Diebold–Mariano tests, and Model Confidence Set analysis are
  later-stage work.
- No final dataset has been accessed, frozen, or evaluated.
- The clean PyTorch implementation follows the PatchTST architectural pattern but does not load or
  copy pretrained PatchTST weights.
