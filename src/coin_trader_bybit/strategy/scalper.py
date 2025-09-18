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
    suffix_map = {"s": "s", "m": "min", "h": "h", "d": "d"}
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
        rolling_low = df["low"].rolling(window=lookback, min_periods=1).min()
        df["micro_high"] = rolling_high.shift(1)
        df["micro_low"] = rolling_low.shift(1)
        df["long_breakout"] = (df["high"] > df["micro_high"]) & df["trend_up"]
        df["short_breakout"] = (df["low"] < df["micro_low"]) & (~df["trend_up"])
        df["weak_short_breakout"] = df["low"] < df["micro_low"]

        if self.cfg.use_volume_filter:
            volume_checks = []
            for timeframe in self.cfg.volume_timeframes or [self.cfg.timeframe_entry]:
                freq = timeframe_to_pandas_freq(timeframe)
                vol_series = (
                    df["volume"].resample(freq).sum()
                    if timeframe != self.cfg.timeframe_entry
                    else df["volume"]
                )
                ma = vol_series.rolling(
                    window=self.cfg.volume_ma_period,
                    min_periods=self.cfg.volume_ma_period,
                ).mean()
                ratio_ok = vol_series >= ma * self.cfg.volume_threshold_ratio
                ratio_ok = ratio_ok.reindex(df.index, method="ffill").fillna(False)
                volume_checks.append(ratio_ok.rename(f"volume_ok_{timeframe}"))

            if volume_checks:
                volume_matrix = pd.concat(volume_checks, axis=1)
                if self.cfg.volume_tf_mode.lower() == "all":
                    volume_ok_combined = volume_matrix.all(axis=1)
                else:
                    volume_ok_combined = volume_matrix.any(axis=1)
                df = df.join(volume_matrix, how="left")
                df["volume_ma"] = (
                    df["volume"].rolling(
                        window=self.cfg.volume_ma_period,
                        min_periods=self.cfg.volume_ma_period,
                    ).mean()
                )
                df["volume_ok"] = volume_ok_combined
            else:
                df["volume_ma"] = pd.Series(np.nan, index=df.index)
                df["volume_ok"] = True
        else:
            df["volume_ma"] = pd.Series(np.nan, index=df.index)
            df["volume_ok"] = True
        return df

    def generate_signal(self, features: pd.DataFrame) -> Optional[Signal]:
        if features.empty:
            return None
        latest = features.iloc[-1]
        if self.cfg.use_volume_filter and not bool(latest.get("volume_ok", False)):
            return None

        long_candidate = bool(latest.get("long_breakout", False))
        short_candidate = bool(latest.get("short_breakout", False))
        if not short_candidate and self.cfg.allow_counter_trend_shorts:
            short_candidate = bool(latest.get("weak_short_breakout", False))
        if not long_candidate and not short_candidate:
            return None

        atr_val = float(latest.get("atr", 0.0) or 0.0)
        if self.cfg.stop_loss_pct is None and atr_val <= 0:
            return None

        close_price = float(latest["close"])
        side, entry_price, stop_price = self._build_signal_prices(
            close_price=close_price,
            atr_val=atr_val,
            long_candidate=long_candidate,
            short_candidate=short_candidate,
        )
        if side is None or entry_price is None or stop_price is None:
            return None

        timestamp = features.index[-1]
        reason = "long breakout" if side == "Buy" else "short breakdown"
        return Signal(
            timestamp=timestamp,
            side=side,
            entry_price=entry_price,
            stop_price=stop_price,
            atr=atr_val,
            reason=reason,
        )

    def _build_signal_prices(
        self,
        *,
        close_price: float,
        atr_val: float,
        long_candidate: bool,
        short_candidate: bool,
    ) -> tuple[Optional[Side], Optional[float], Optional[float]]:
        entry_price = close_price
        buffer_pct = max(self.cfg.entry_buffer_pct, 0.0)
        stop_pct = self.cfg.stop_loss_pct

        if long_candidate and short_candidate:
            if self.cfg.use_trend_filter:
                short_candidate = False
            else:
                # prefer stronger breakout distance
                long_move = close_price - (close_price * (1 - buffer_pct))
                short_move = (close_price * (1 + buffer_pct)) - close_price
                if long_move >= short_move:
                    short_candidate = False
                else:
                    long_candidate = False

        if long_candidate:
            if buffer_pct > 0:
                entry_price = close_price * (1 - buffer_pct)
            stop_price = self._compute_stop_price(
                entry_price=entry_price,
                atr_val=atr_val,
                pct=stop_pct,
                direction="long",
            )
            return "Buy", entry_price, stop_price

        if short_candidate:
            if buffer_pct > 0:
                entry_price = close_price * (1 + buffer_pct)
            stop_price = self._compute_stop_price(
                entry_price=entry_price,
                atr_val=atr_val,
                pct=stop_pct,
                direction="short",
            )
            return "Sell", entry_price, stop_price

        return None, None, None

    def _compute_stop_price(
        self,
        *,
        entry_price: float,
        atr_val: float,
        pct: Optional[float],
        direction: str,
    ) -> Optional[float]:
        if pct is not None and pct > 0:
            if direction == "long":
                return entry_price * (1 - pct)
            return entry_price * (1 + pct)
        stop_offset = atr_val * self.cfg.atr_mult_stop
        if stop_offset <= 0:
            return None
        if direction == "long":
            return entry_price - stop_offset
        return entry_price + stop_offset


__all__ = ["Scalper", "Signal", "timeframe_to_pandas_freq"]
