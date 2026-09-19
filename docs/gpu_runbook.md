# GPU runbook for the three-seed synthetic pilot

This run is synthetic-only. Do not point it at the opened Binance pilot data, and do not run
`evaluate-real-pilot` after training. The dedicated configuration writes to `runs/pilot_gpu`, so it
cannot overwrite the interrupted local run.

## 1. Open the repository root

```bash
cd /path/to/synthetic-finance-tsfm
```

All remaining commands should be run from this directory.

## 2. Create the environment and confirm CUDA

```bash
uv sync --extra dev
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not detected')"
```

Continue only if the first value is `True`. If it is `False`, save the complete output and fix the
CUDA-enabled PyTorch installation before starting a long run.

## 3. Run a fast preflight

```bash
uv run pytest
uv run synthetic-finance-tsfm validate-config --config configs/pilot/pilot_gpu.yaml
```

## 4. Train the three seeds

Use one process at a time on a single GPU. This avoids memory competition and keeps runtime
comparisons interpretable.

```bash
for seed in 17 29 43; do
  CUDA_VISIBLE_DEVICES=0 uv run synthetic-finance-tsfm run \
    --config configs/pilot/pilot_gpu.yaml \
    --seed "$seed"
done
```

Each prior writes a completion record only after all 10,000 optimizer steps finish. If a process is
interrupted, rerun that seed; do not treat the presence of a checkpoint alone as proof of completion.

## 5. Aggregate and verify the gate

```bash
uv run synthetic-finance-tsfm summarize-calibration \
  --runs \
    runs/pilot_gpu/pilot_generic_vs_volatility_gpu-seed17 \
    runs/pilot_gpu/pilot_generic_vs_volatility_gpu-seed29 \
    runs/pilot_gpu/pilot_generic_vs_volatility_gpu-seed43 \
  --output runs/pilot_gpu/pilot_summary.json
```

The aggregation command refuses any run explicitly marked as having unverified training. The files
to retain are the three complete run directories plus `runs/pilot_gpu/pilot_summary.json`.

## 6. Return the artifacts when training is remote

Copy the entire `runs/pilot_gpu` directory back to the project machine. It contains the resolved
configuration, environment metadata, completion records, best checkpoints, predictions, per-seed
metrics, and the aggregate summary needed for the next decision.
