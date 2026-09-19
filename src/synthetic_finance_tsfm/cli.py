"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthetic_finance_tsfm.config import load_config
from synthetic_finance_tsfm.data.pilot import prepare_binance_pilot
from synthetic_finance_tsfm.evaluation.real_pilot import evaluate_real_pilot
from synthetic_finance_tsfm.synthetic import generate_series
from synthetic_finance_tsfm.synthetic.diagnostics import describe_variance
from synthetic_finance_tsfm.training.runner import (
    evaluate_synthetic_checkpoints,
    run_experiment,
    summarize_calibration,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synthetic-finance-tsfm")
    subcommands = parser.add_subparsers(dest="command", required=True)

    validate = subcommands.add_parser("validate-config", help="Validate a YAML configuration")
    validate.add_argument("--config", required=True, type=Path)

    generate = subcommands.add_parser("describe-prior", help="Generate and describe one sequence")
    generate.add_argument("--prior", required=True, choices=("generic", "volatility"))
    generate.add_argument("--length", type=int, default=512)
    generate.add_argument("--seed", type=int, default=42)

    smoke = subcommands.add_parser("run", help="Run a configured synthetic experiment")
    smoke.add_argument("--config", required=True, type=Path)
    smoke.add_argument("--seed", type=int, help="Override the configured seed")

    recover = subcommands.add_parser(
        "evaluate-synthetic", help="Evaluate existing synthetic checkpoints without retraining"
    )
    recover.add_argument("--config", required=True, type=Path)
    recover.add_argument("--seed", type=int, help="Override the configured seed")

    summarize = subcommands.add_parser(
        "summarize-calibration", help="Aggregate completed equal-budget seed runs"
    )
    summarize.add_argument("--runs", nargs="+", required=True, type=Path)
    summarize.add_argument("--output", required=True, type=Path)

    prepare_pilot = subcommands.add_parser(
        "prepare-binance-pilot", help="Download and aggregate the frozen pilot-only panel"
    )
    prepare_pilot.add_argument("--config", required=True, type=Path)

    evaluate_pilot = subcommands.add_parser(
        "evaluate-real-pilot", help="Evaluate frozen synthetic checkpoints on pilot-only data"
    )
    evaluate_pilot.add_argument("--manifest", required=True, type=Path)
    evaluate_pilot.add_argument("--runs", nargs="+", required=True, type=Path)
    evaluate_pilot.add_argument("--output-dir", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "validate-config":
        config = load_config(args.config)
        print(json.dumps(config.to_dict(), indent=2))
        return
    if args.command == "describe-prior":
        series = generate_series(args.prior, args.length, args.seed)
        print(
            json.dumps(
                {
                    "family": series.family,
                    "parameters": series.parameters,
                    "descriptors": describe_variance(series.values),
                },
                indent=2,
            )
        )
        return
    if args.command == "run":
        destination = run_experiment(args.config, seed_override=args.seed)
        print(destination)
        return
    if args.command == "evaluate-synthetic":
        destination = evaluate_synthetic_checkpoints(args.config, seed_override=args.seed)
        print(destination)
        return
    if args.command == "summarize-calibration":
        destination = summarize_calibration(args.runs, args.output)
        print(destination)
        return
    if args.command == "prepare-binance-pilot":
        destination = prepare_binance_pilot(args.config)
        print(destination)
        return
    if args.command == "evaluate-real-pilot":
        destination = evaluate_real_pilot(
            manifest_path=args.manifest,
            run_directories=args.runs,
            output_dir=args.output_dir,
        )
        print(destination)
        return
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
