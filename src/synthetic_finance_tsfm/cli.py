"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthetic_finance_tsfm.config import load_config
from synthetic_finance_tsfm.synthetic import generate_series
from synthetic_finance_tsfm.synthetic.diagnostics import describe_variance
from synthetic_finance_tsfm.training.runner import run_experiment


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
        destination = run_experiment(args.config)
        print(destination)
        return
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
