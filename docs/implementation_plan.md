# Implementation plan

## Environment audit

- Repository began with one README and a clean `main` branch tracking `origin/main`.
- Local development machine: Apple M2, 8 CPU cores, 8 GB unified memory, no CUDA GPU.
- Python 3.11 and `uv` are available.
- The smoke configuration is intentionally CPU-safe. Pilot and full configurations must be
  benchmarked on suitable GPU hardware before use.

## Delivery checklist

- [x] Record the zero-shot and equal-budget research contract.
- [x] Add installable package, validated YAML configuration, logging, and deterministic seeds.
- [x] Implement canonical realized-variance data processing and leakage-safe windows.
- [x] Implement deterministic Generic and Volatility priors.
- [x] Implement Last Value, Historical Mean, and direct Log-HAR baselines.
- [x] Implement QLIKE, log errors, and asset-level aggregation primitives.
- [x] Implement a randomly initialized PatchTST-style encoder with a custom multi-horizon head.
- [x] Implement checkpointed equal-budget training and cross-prior evaluation.
- [x] Add unit tests and a CPU smoke configuration.
- [x] Pass a three-seed, 500-step Generic-versus-Volatility synthetic calibration.
- [x] Validate a public Binance pilot dataset and freeze its pilot-only manifest.
- [x] Evaluate the frozen calibration checkpoints once on pilot-only real data.
- [x] Benchmark the predeclared pilot architecture and estimate three-seed compute.
- [ ] Run the predeclared three-seed pilot architecture using synthetic data only.
- [ ] Request and verify VOLARE access and its redistribution terms.
- [ ] Add Regime-and-Tail and Mixed priors only after pilot gates pass.
- [ ] Add ForecastPFN and public-foundation-model baselines after the core comparison is stable.

## Acceptance criteria for the first delivery

The package installs without downloading model weights; unit tests pass; both pilot priors are
reproducible and positive; target indexing and context-only scaling are tested; baselines produce
positive forecasts; a tiny model completes forward/backward and checkpoint round-trip tests; and a
configuration-driven smoke run writes metrics, predictions, checkpoints, environment metadata, and
the resolved configuration.

## Risks and responses

- **Limited local memory:** keep smoke inputs small and benchmark before pilot scaling.
- **Interrupted long runs:** write a completion record after each prior finishes its full optimizer
  budget; never infer completion from a best-checkpoint file alone.
- **Synthetic/real mismatch:** export descriptor summaries and the cross-prior holdout matrix.
- **Loss instability:** train the smoke model in normalized log space and evaluate with positive-floor
  QLIKE on the original scale.
- **Data leakage:** make context location/scale explicit and test split embargoes and target indices.
- **Provider access:** keep the canonical schema provider-independent and never infer unavailable
  VOLARE fields.

## Opened pilot status

The Binance engineering panel has been opened for evaluation. It cannot be used for any further
generator, architecture, hyperparameter, or checkpoint choice. Detailed results and their limitations
are recorded in `docs/pilot_results.md`.

## Pilot architecture benchmark

The predeclared 796,163-parameter architecture required approximately 8.4--8.5 seconds per 100
optimizer steps per prior on the local CPU. A 10,000-step run is therefore estimated at roughly
14--17 minutes per prior before allowing for larger synthetic-set generation and validation. Six runs
(two priors across three seeds) are budgeted at approximately 1.5--2 hours on this machine. This is an
extrapolation, so actual runtime and thermal throttling must be recorded.

## Current full-pilot status

The local three-seed attempt produced complete Generic checkpoints, but its three Volatility jobs
were interrupted without completion records. Their recovered cross-prior metrics are diagnostic only
and are not accepted by the calibration summarizer. The clean rerun is defined in
`configs/pilot/pilot_gpu.yaml` and `docs/gpu_runbook.md`; it uses a new output directory and requires
verified CUDA execution.
