# Pilot-only results

These results are an engineering pilot, not final evidence. They use BTCUSDT, ETHUSDT, and SOLUSDT
from January 2024 through June 2025. All rows are permanently marked `pilot_only` and are ineligible
for final evaluation.

## Synthetic calibration

Three seeds (`17`, `29`, and `43`) used identical budgets: 2,048 training tasks, 512 validation
tasks, batch size 32, and 500 optimizer steps per prior. Both matched models beat Last Value on their
own synthetic holdouts in all three seeds. Each matched model also outperformed the other-prior model
on its own holdout in all three seeds. The synthetic-learning and prior-distinguishability gates passed.

Mean cross-prior QLIKE across seeds:

| Training prior | Generic holdout | Volatility holdout |
|---|---:|---:|
| Generic | 0.0328 | 0.0860 |
| Volatility | 0.0419 | 0.0603 |

## Real pilot

Each asset contains 547 daily observations and 398 valid forecast origins at context length 128.
Daily interval coverage is at least 99.65%, with no duplicate asset/date rows or missing realized
variance values.

At the primary five-day horizon, the Volatility prior had lower QLIKE than the Generic prior in 6 of
9 paired asset/seed comparisons. The direction therefore passed the predeclared majority rule, but it
was not uniform.

Median asset-level QLIKE ratios relative to Direct Log-HAR:

| Prior | 1 day | 5 days | 22 days |
|---|---:|---:|---:|
| Generic | 1.021 | 1.301 | 1.355 |
| Volatility | 1.001 | 1.125 | 1.193 |

A ratio below one would beat Log-HAR. The small Volatility model was approximately tied at one day
but did not beat Log-HAR at five or 22 days. This is a meaningful limitation, not a failed software
run.

## Descriptor diagnostic

The synthetic Volatility holdout had higher persistence than the real pilot at lag five, but much less
dispersion and fewer extreme values. For example, its average max-to-median variance ratio was about
4.7, whereas the three real pilot assets were around 24--27. This diagnostic may explain part of the
gap, but it will not be used to tune the prior because the real pilot has already been opened.

## Protocol consequence

The six evaluated calibration checkpoints remain frozen. No further checkpoint or generator choice
may use this Binance panel. The next stage benchmarks and trains the architecture already declared in
`configs/pilot/pilot.yaml` using synthetic training and validation only. Its eventual reported test must
use a separate, sealed final dataset.

## Next compute stage

The already-declared pilot architecture contains 796,163 parameters. A representative 100-step
benchmark took 8.39 seconds for Generic and 8.52 seconds for Volatility on CPU. The planned six
10,000-step runs are expected to require approximately 1.5--2 hours locally. This training will remain
synthetic-only and will not be re-evaluated on the opened Binance panel.
