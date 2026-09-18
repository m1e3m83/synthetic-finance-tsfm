# Data split registry

The Binance pilot-only split is frozen in `configs/data/binance_pilot.yaml`:

- BTCUSDT, ETHUSDT, and SOLUSDT;
- five-minute public spot klines;
- January 2024 through June 2025, inclusive;
- UTC day boundary;
- 95% minimum daily interval coverage;
- 22-day maximum horizon;
- final-evaluation eligibility: false.

The generated manifest and checksums live under `data/processed/binance_pilot/`, which is intentionally
not committed because it describes locally retrieved data. This split has been opened and must not
influence future model or generator choices.

No final real-data split is frozen yet.

Before using real data, add a machine-readable manifest containing:

- provider and dataset version;
- retrieval timestamp and checksums;
- asset identifiers;
- UTC date boundaries;
- sampling interval and day-coverage threshold;
- development, pilot, and final roles;
- exclusions and their reasons;
- embargo length;
- confirmation that pilot-only assets and windows do not appear in final results.

The final split must be frozen before model checkpoints are evaluated on it.
