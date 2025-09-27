#!/usr/bin/env python3
"""Download Bybit OHLCV data for a given symbol/timeframe range.

Usage:
    PYTHONPATH=src python3 scripts/download_bybit_ohlcv.py \
        --symbol BTCUSDT --timeframe 1m \
        --since 2024-01-01T00:00:00Z --until 2025-09-18T23:59:00Z \
        --outfile data/btcusdt_1m_20240101_20250918.csv

Requires ccxt (already listed in requirements).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

import ccxt
import pandas as pd


def _parse_iso8601(value: str) -> int:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _chunks(total: int, size: int) -> Iterable[tuple[int, int]]:
    steps = math.ceil(total / size)
    cursor = 0
    for _ in range(steps):
        yield cursor, min(cursor + size, total)
        cursor += size


def _fetch_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    limit: int,
    pause_seconds: float,
) -> List[List[float]]:
    candles: List[List[float]] = []
    cursor = since_ms
    while cursor <= until_ms:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        except ccxt.BaseError as exc:  # pragma: no cover - network
            print(f"fetch error at {cursor}: {exc}", file=sys.stderr)
            time.sleep(pause_seconds)
            continue
        if not batch:
            break

        candles.extend(batch)
        last_ts = batch[-1][0]
        if last_ts == cursor:
            # no progress; avoid infinite loop
            cursor += limit * exchange.parse_timeframe(timeframe) * 1000
        else:
            cursor = last_ts + exchange.parse_timeframe(timeframe) * 1000

        if cursor > until_ms:
            break

        time.sleep(pause_seconds)

    return [row for row in candles if since_ms <= row[0] <= until_ms]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Bybit OHLCV via ccxt")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1m")
    parser.add_argument("--since", required=True, help="ISO8601 start (UTC)")
    parser.add_argument("--until", required=True, help="ISO8601 end (UTC)")
    parser.add_argument("--outfile", required=True, help="CSV output path")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--pause", type=float, default=0.2)
    args = parser.parse_args()

    since_ms = _parse_iso8601(args.since)
    until_ms = _parse_iso8601(args.until)
    if since_ms >= until_ms:
        raise ValueError("since must be earlier than until")

    exchange = ccxt.bybit({"enableRateLimit": True})
    exchange.load_markets()
    # ensure 시장 정보 확인
    market = exchange.market(args.symbol)
    if not market:
        raise ValueError(f"symbol {args.symbol} not found on Bybit")

    raw = _fetch_ohlcv(
        exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        since_ms=since_ms,
        until_ms=until_ms,
        limit=args.limit,
        pause_seconds=args.pause,
    )

    if not raw:
        print("No candles fetched", file=sys.stderr)
        return

    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()

    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outfile)
    print(f"Saved {len(df)} rows to {outfile}")


if __name__ == "__main__":
    main()
