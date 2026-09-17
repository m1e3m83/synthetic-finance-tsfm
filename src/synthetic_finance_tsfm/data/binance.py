"""Small, explicit Binance public archive adapter for pilot-only data."""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
)


def monthly_archive_url(symbol: str, interval: str, year_month: str) -> str:
    normalized_symbol = symbol.upper()
    return (
        "https://data.binance.vision/data/spot/monthly/klines/"
        f"{normalized_symbol}/{interval}/{normalized_symbol}-{interval}-{year_month}.zip"
    )


def download_monthly_klines(
    *, symbol: str, interval: str, year_month: str, cache_dir: str | Path
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Download one public monthly archive and return normalized timestamps and closes."""
    url = monthly_archive_url(symbol, interval, year_month)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    archive_path = cache / Path(url).name
    if not archive_path.exists():
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
            payload = response.read()
        archive_path.write_bytes(payload)
    payload = archive_path.read_bytes()
    checksum = hashlib.sha256(payload).hexdigest()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"Expected exactly one CSV in {archive_path}")
        with archive.open(members[0]) as stream:
            frame = pd.read_csv(stream, header=None, names=KLINE_COLUMNS)
    timestamp_unit = "us" if float(frame["open_time"].iloc[0]) > 1e14 else "ms"
    normalized = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["close_time"], unit=timestamp_unit, utc=True),
            "close": pd.to_numeric(frame["close"], errors="raise"),
        }
    )
    return normalized, {"url": url, "sha256": checksum, "archive": str(archive_path)}
