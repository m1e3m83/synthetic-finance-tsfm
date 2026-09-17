# Research protocol

## Objective

Determine whether finance-aware synthetic priors improve zero-shot realized-variance forecasts
relative to a generic synthetic prior when architecture, optimization, sample count, and random-seed
policy are held fixed.

## Forecasting task

For daily realized variance `RV[t]`, the target at horizon `h` is the cumulative future variance
`sum(RV[t+1:t+h+1])`, for `h` in `{1, 5, 22}`. Inputs are
`log(RV + epsilon)`. The model predicts the context-standardized log average future daily variance;
forecasts are transformed back to positive cumulative variance for scoring.

## Strict zero-shot rules

1. Neural-model weights are learned only from simulated series.
2. Model selection, early stopping, generator design, and hyperparameter selection use synthetic
   training and validation data only.
3. Final real assets and periods cannot influence the prior, checkpoint, architecture, loss, context
   length, or any other modeling choice.
4. Real observations before a forecast origin may be inference context but cannot produce gradient
   updates.
5. Engineering data must be marked `pilot_only` and excluded from final results.
6. The final evaluation is run only after configurations, splits, and checkpoints are frozen.
7. Chronological partitions use an embargo of at least 22 observations to keep overlapping targets
   from crossing boundaries.
8. Smoke, pilot, development, and final results are labeled separately.

## Equal-budget comparison

Prior arms use the same model configuration, number of examples, optimizer steps, batch size,
optimizer, validation schedule, seed policy, and checkpoint-selection rule. Only the generator family
changes. A run is invalid if these invariants are not recorded or cannot be verified.

## Baseline interpretation

Naive and Log-HAR forecasts may estimate parameters from real observations strictly before each
forecast origin. They are practical reference methods and do not share the synthetic-only training
condition. Public pretrained models are reported separately because their original data exposure may
not be auditable.

## Primary evaluation

QLIKE is primary. Losses are first aggregated within asset and horizon. Cross-asset comparisons use
asset-level loss ratios against Log-HAR. Log-MSE and log-MAE are secondary. Any positive-floor
clipping is counted and reported.

## Decision gates

1. **Pipeline gate:** target alignment, data quality, and baseline outputs pass independent tests.
2. **Synthetic-learning gate:** the neural model learns held-out synthetic tasks better than trivial
   forecasts.
3. **Prior-contrast gate:** the equal-budget contrast is stable across seeds and not caused by a
   numerical failure.
4. **Real-pilot gate:** any directional result is not driven by one asset or one seed.

Failed gates are reported and diagnosed before any increase in model or experiment scale.

