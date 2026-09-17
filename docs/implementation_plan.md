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
- [ ] Validate a public Binance pilot dataset and freeze its pilot-only manifest.
- [ ] Run the three-seed controlled pilot on GPU hardware.
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
- **Synthetic/real mismatch:** export descriptor summaries and the cross-prior holdout matrix.
- **Loss instability:** train the smoke model in normalized log space and evaluate with positive-floor
  QLIKE on the original scale.
- **Data leakage:** make context location/scale explicit and test split embargoes and target indices.
- **Provider access:** keep the canonical schema provider-independent and never infer unavailable
  VOLARE fields.

