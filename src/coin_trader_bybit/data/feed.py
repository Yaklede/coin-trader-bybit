from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import pandas as pd

from ..exchange.bybit import BybitClient


@dataclass
class KlineRecord:
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


class DataFeed:
    """Abstract interface for candle feeds."""

    def fetch(self, limit: int) -> pd.DataFrame:  # pragma: no cover - interface
        raise NotImplementedError


class BybitDataFeed(DataFeed):
    """Fetches OHLCV candles from Bybit via the REST API."""

    def __init__(
        self,
        client: BybitClient,
        *,
        symbol: str,
        category: str = "linear",
        interval: str = "1",
    ) -> None:
        self.client = client
        self.symbol = symbol
        self.category = category
        self.interval = interval

    def fetch(self, limit: int) -> pd.DataFrame:
        raw = self.client.get_kline(
            symbol=self.symbol,
            category=self.category,
            interval=self.interval,
            limit=limit,
        )
        records: List[KlineRecord] = []
        for entry in raw:
            if isinstance(entry, dict):
                start_raw = entry.get("startTime") or entry.get("start")
                if start_raw is None:
                    continue
                timestamp = pd.to_datetime(int(start_raw), unit="ms", utc=True)
                records.append(
                    KlineRecord(
                        timestamp=timestamp,
                        open=float(entry["open"]),
                        high=float(entry["high"]),
                        low=float(entry["low"]),
                        close=float(entry["close"]),
                        volume=float(
                            entry.get("volume") or entry.get("turnover") or 0.0
                        ),
                    )
                )
            elif isinstance(entry, (list, tuple)) and len(entry) >= 6:
                timestamp = pd.to_datetime(int(entry[0]), unit="ms", utc=True)
                records.append(
                    KlineRecord(
                        timestamp=timestamp,
                        open=float(entry[1]),
                        high=float(entry[2]),
                        low=float(entry[3]),
                        close=float(entry[4]),
                        volume=float(entry[5]),
                    )
                )
        if not records:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame([r.__dict__ for r in records]).set_index("timestamp")
        return df.sort_index()


class MemoryDataFeed(DataFeed):
    """In-memory feed used for testing."""

    def __init__(self, candles: Sequence[KlineRecord]) -> None:
        self.candles = list(candles)

    def fetch(self, limit: int) -> pd.DataFrame:
        selected = self.candles[-limit:]
        if not selected:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame([r.__dict__ for r in selected]).set_index("timestamp")
        return df.sort_index()
