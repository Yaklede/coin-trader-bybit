from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd

from ..core.config import StrategyConfig

Side = Literal["Buy", "Sell"]


@dataclass
class Signal:
    """Represents a discretionary trade idea."""

    timestamp: pd.Timestamp
    side: Side
    entry_price: float
    stop_price: float
    reason: str


def timeframe_to_pandas_freq(value: str) -> str:
    unit = value.strip().lower()
    if not unit:
        raise ValueError("timeframe cannot be empty")
    suffix = unit[-1]
    amount = unit[:-1] or "1"
    if not amount.isdigit():
        raise ValueError(f"Unsupported timeframe amount: {value}")
    suffix_map = {"s": "S", "m": "min", "h": "H", "d": "D"}
    if suffix not in suffix_map:
        raise ValueError(f"Unsupported timeframe unit: {value}")
    return f"{int(amount)}{suffix_map[suffix]}"


def _compute_atr(data: pd.DataFrame, period: int) -> pd.Series:
    high = data["high"]
    low = data["low"]
    close = data["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


class Scalper:
    """Implements the breakout-with-trend rules used by the bot."""

    def __init__(self, cfg: StrategyConfig) -> None:
        self.cfg = cfg

    def compute_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            raise ValueError("data must not be empty")
        missing = {"open", "high", "low", "close"} - set(data.columns)
        if missing:
            raise ValueError(f"data missing required columns: {missing}")

        df = data.copy()
        anchor_freq = timeframe_to_pandas_freq(self.cfg.timeframe_anchor)
        close_anchor = df["close"].resample(anchor_freq).last()
        ema_fast = close_anchor.ewm(span=self.cfg.ema_fast, adjust=False).mean()
        ema_slow = close_anchor.ewm(span=self.cfg.ema_slow, adjust=False).mean()
        trend_up = (ema_fast > ema_slow).reindex(df.index, method="ffill").fillna(False)

        df["trend_up"] = trend_up
        df["atr"] = _compute_atr(df, self.cfg.atr_period)

        lookback = max(self.cfg.micro_high_lookback, 1)
        rolling_high = df["high"].rolling(window=lookback, min_periods=1).max()
        df["micro_high"] = rolling_high.shift(1)
        df["long_breakout"] = (df["high"] > df["micro_high"]) & df["trend_up"]
        return df

    def maybe_signal(self) -> Optional[Signal]:
        """Placeholder for live wiring (not yet connected to streaming data)."""

        return None


__all__ = ["Scalper", "Signal", "timeframe_to_pandas_freq"]
