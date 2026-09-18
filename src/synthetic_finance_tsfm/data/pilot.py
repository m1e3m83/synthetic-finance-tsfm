"""Configuration-driven creation of the sealed pilot-only Binance panel."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from synthetic_finance_tsfm.data.binance import download_monthly_klines
from synthetic_finance_tsfm.data.contracts import validate_daily_frame
from synthetic_finance_tsfm.data.realized_variance import realized_variance_from_prices
from synthetic_finance_tsfm.utils.io import write_json


def _month_range(start: str, end: str) -> list[str]:
    values = pd.period_range(start=start, end=end, freq="M")
    if len(values) == 0:
        raise ValueError("Pilot month range is empty")
    return [str(value) for value in values]


def load_pilot_config(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {
        "source",
        "role",
        "symbols",
        "interval",
        "interval_minutes",
        "start_month",
        "end_month",
        "minimum_coverage",
        "epsilon",
        "cache_dir",
        "output_dir",
    }
    missing = required - set(raw)
    if missing:
        raise ValueError(f"Missing pilot configuration fields: {sorted(missing)}")
    if raw["role"] != "pilot_only":
        raise ValueError("Binance engineering data must remain pilot_only")
    if not raw["symbols"]:
        raise ValueError("At least one symbol is required")
    _month_range(str(raw["start_month"]), str(raw["end_month"]))
    return raw


def prepare_binance_pilot(config_path: str | Path) -> Path:
    """Download, validate, aggregate, and manifest the pilot panel."""
    config = load_pilot_config(config_path)
    months = _month_range(str(config["start_month"]), str(config["end_month"]))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    all_daily: list[pd.DataFrame] = []
    archives: list[dict[str, str]] = []
    asset_summaries: dict[str, Any] = {}

    for symbol in config["symbols"]:
        monthly_frames: list[pd.DataFrame] = []
        for month in months:
            intraday, metadata = download_monthly_klines(
                symbol=str(symbol),
                interval=str(config["interval"]),
                year_month=month,
                cache_dir=config["cache_dir"],
            )
            monthly_frames.append(intraday)
            archives.append({"symbol": str(symbol), "month": month, **metadata})
        combined = pd.concat(monthly_frames, ignore_index=True)
        daily = realized_variance_from_prices(
            combined,
            asset_id=str(symbol),
            data_source=str(config["source"]),
            interval_minutes=int(config["interval_minutes"]),
            epsilon=float(config["epsilon"]),
            minimum_coverage=float(config["minimum_coverage"]),
            pilot_only=True,
        )
        validate_daily_frame(daily)
        all_daily.append(daily)
        asset_summaries[str(symbol)] = {
            "daily_rows": len(daily),
            "first_date": str(daily["date"].min()),
            "last_date": str(daily["date"].max()),
            "minimum_coverage": float(daily["coverage_ratio"].min()),
            "median_coverage": float(daily["coverage_ratio"].median()),
        }

    panel = pd.concat(all_daily, ignore_index=True).sort_values(["asset_id", "date"])
    panel_path = output_dir / "daily_realized_variance.csv"
    panel.to_csv(panel_path, index=False)
    panel_checksum = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    manifest = {
        "configuration": config,
        "months": months,
        "archives": archives,
        "assets": asset_summaries,
        "panel": {
            "path": str(panel_path),
            "sha256": panel_checksum,
            "rows": len(panel),
            "pilot_only": True,
        },
        "exclusions": "Days below configured coverage are excluded by the processing pipeline.",
        "final_evaluation_eligible": False,
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path
