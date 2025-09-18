from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd
import numpy as np

from ..core.config import StrategyConfig

Side = Literal["Buy", "Sell"]


@dataclass
class Signal:
    """Represents a trade opportunity with pricing context."""

    timestamp: pd.Timestamp
    side: Side
    entry_price: float
    stop_price: float
    atr: float
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
        missing = {"open", "high", "low", "close", "volume"} - set(data.columns)
        if missing:
            raise ValueError(f"data missing required columns: {missing}")

        df = data.copy()
        if self.cfg.use_trend_filter:
            anchor_freq = timeframe_to_pandas_freq(self.cfg.timeframe_anchor)
            close_anchor = df["close"].resample(anchor_freq).last()
            ema_fast = close_anchor.ewm(span=self.cfg.ema_fast, adjust=False).mean()
            ema_slow = close_anchor.ewm(span=self.cfg.ema_slow, adjust=False).mean()
            trend_up = (
                (ema_fast > ema_slow).reindex(df.index, method="ffill").fillna(False)
            )
        else:
            trend_up = pd.Series(True, index=df.index)

        df["trend_up"] = trend_up
        df["atr"] = _compute_atr(df, self.cfg.atr_period)

        lookback = max(self.cfg.micro_high_lookback, 1)
        rolling_high = df["high"].rolling(window=lookback, min_periods=1).max()
        df["micro_high"] = rolling_high.shift(1)
        df["long_breakout"] = (df["high"] > df["micro_high"]) & df["trend_up"]

        if self.cfg.use_volume_filter:
            volume_ma = (
                df["volume"]
                    .rolling(
                        window=self.cfg.volume_ma_period,
                        min_periods=self.cfg.volume_ma_period,
                    )
                    .mean()
            )
            df["volume_ma"] = volume_ma
            df["volume_ok"] = (
                df["volume"] >= df["volume_ma"] * self.cfg.volume_threshold_ratio
            )
        else:
            df["volume_ma"] = pd.Series(np.nan, index=df.index)
            df["volume_ok"] = True
        return df

    def generate_signal(self, features: pd.DataFrame) -> Optional[Signal]:
        if features.empty:
            return None
        latest = features.iloc[-1]
        if not bool(latest.get("long_breakout", False)):
            return None
        if self.cfg.use_volume_filter and not bool(latest.get("volume_ok", False)):
            return None
        atr_val = float(latest.get("atr", 0.0) or 0.0)
        if self.cfg.stop_loss_pct is None and atr_val <= 0:
            return None
        entry_price = float(latest["close"])
        if self.cfg.entry_buffer_pct > 0:
            entry_price = entry_price * (1 - self.cfg.entry_buffer_pct)
        stop_price: float
        if self.cfg.stop_loss_pct is not None and self.cfg.stop_loss_pct > 0:
            stop_price = entry_price * (1 - self.cfg.stop_loss_pct)
        else:
            stop_price = entry_price - atr_val * self.cfg.atr_mult_stop
        if stop_price <= 0:
            return None
        timestamp = features.index[-1]
        reason = "long breakout with trend and volume confirmation"
        return Signal(
            timestamp=timestamp,
            side="Buy",
            entry_price=entry_price,
            stop_price=stop_price,
            atr=atr_val,
            reason=reason,
        )


__all__ = ["Scalper", "Signal", "timeframe_to_pandas_freq"]
