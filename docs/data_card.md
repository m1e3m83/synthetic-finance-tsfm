# Data card

## Canonical daily schema

Processed real data use these fields: `asset_id`, `date`, `realized_variance`,
`log_realized_variance`, `observation_count`, `expected_observation_count`, `coverage_ratio`,
`data_source`, `pilot_only`, and `quality_flags`.

## Pilot source

The planned engineering pilot uses Binance public spot klines. Timestamps are converted to UTC.
Close-to-close log returns are assigned to their ending timestamps, and squared returns are summed
within UTC calendar days. A day is usable only when its coverage ratio meets the configured threshold.
Duplicate timestamps and non-positive prices are rejected.

No Binance data have been downloaded or validated in the initial smoke delivery. When a download is
performed, the exact symbol, interval, dates, archive URL, retrieval time, checksum, exclusions, and
coverage summary must be added to the split registry. All Binance engineering data remain
`pilot_only` unless a later protocol explicitly changes that role before final evaluation.

## Final source

VOLARE is the preferred final evaluation source. Access, actual file schemas, calendar conventions,
and redistribution rights have not yet been verified. The repository therefore defines a canonical
schema but does not invent a VOLARE adapter.

## Storage and release

Raw and processed datasets are ignored by Git. Reproducibility should use retrieval code, manifests,
checksums, and transformations. Data files are released only when their source license permits it.

