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
    fast_market: bool = False


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


def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


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
        df = df.sort_index()

        df["ema_fast"] = df["close"].ewm(span=self.cfg.ema_fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.cfg.ema_slow, adjust=False).mean()
        trend_strength = (df["ema_fast"] - df["ema_slow"]) / df["ema_slow"].replace(0.0, np.nan)
        df["trend_strength"] = trend_strength.fillna(0.0)
        slope_lookback = max(self.cfg.trend_confirm_lookback, 1)
        ema_slow_prev = df["ema_slow"].shift(slope_lookback)
        trend_slope_pct = (df["ema_slow"] - ema_slow_prev) / ema_slow_prev.replace(0.0, np.nan)
        df["trend_slope_pct"] = trend_slope_pct.fillna(0.0)
        df["ema_slow_prev"] = ema_slow_prev

        if self.cfg.use_trend_filter:
            anchor_freq = timeframe_to_pandas_freq(self.cfg.timeframe_anchor)
            anchor_close = df["close"].resample(anchor_freq).last()
            anchor_fast = anchor_close.ewm(span=self.cfg.ema_fast, adjust=False).mean()
            anchor_slow = anchor_close.ewm(span=self.cfg.ema_slow, adjust=False).mean()
            anchor_fast_i = anchor_fast.reindex(df.index, method="ffill").fillna(df["close"])
            anchor_slow_i = anchor_slow.reindex(df.index, method="ffill").fillna(df["close"])
            df["anchor_ema_fast"] = anchor_fast_i
            df["anchor_ema_slow"] = anchor_slow_i
            # Anchor trend strength and slope (measured on resampled series but aligned to entry index)
            anchor_strength = (anchor_fast_i - anchor_slow_i) / anchor_slow_i.replace(0.0, np.nan)
            df["anchor_trend_strength"] = anchor_strength.fillna(0.0)
            slope_lookback = max(self.cfg.trend_confirm_lookback, 1)
            anchor_slow_prev = anchor_slow_i.shift(slope_lookback)
            anchor_trend_slope_pct = (anchor_slow_i - anchor_slow_prev) / anchor_slow_prev.replace(0.0, np.nan)
            df["anchor_trend_slope_pct"] = anchor_trend_slope_pct.fillna(0.0)
        else:
            df["anchor_ema_fast"] = df["ema_fast"]
            df["anchor_ema_slow"] = df["ema_slow"]
            df["anchor_trend_strength"] = df["trend_strength"]
            df["anchor_trend_slope_pct"] = df["trend_slope_pct"]

        df["atr"] = _compute_atr(df, self.cfg.atr_period)
        df["atr_pct"] = (df["atr"] / df["close"].replace(0.0, np.nan)).fillna(0.0)
        df["rsi"] = _compute_rsi(df["close"], self.cfg.rsi_period)

        volume_ma = df["volume"].rolling(window=self.cfg.volume_ma_period, min_periods=1).mean()
        df["volume_ma"] = volume_ma
        df["volume_ratio"] = (df["volume"] / volume_ma.replace(0.0, np.nan)).fillna(0.0)

        fast_market = (
            (df["volume_ratio"] >= self.cfg.fast_market_volume_ratio)
            & (df["atr_pct"] >= self.cfg.fast_market_atr_pct)
        )
        df["fast_market"] = fast_market.fillna(False)

        trend_up = df["trend_strength"] >= self.cfg.regime_trend_min
        trend_down = df["trend_strength"] <= -self.cfg.regime_trend_min
        atr_ok = (df["atr_pct"] >= self.cfg.regime_atr_min_pct) & (df["atr_pct"] <= self.cfg.regime_atr_max_pct)
        impulse_ok = df["volume_ratio"] >= self.cfg.impulse_volume_ratio
        slope_ok_up = df["trend_slope_pct"] >= self.cfg.trend_slope_min_pct
        slope_ok_down = df["trend_slope_pct"] <= -self.cfg.trend_slope_min_pct

        # Optional higher timeframe slope gate
        anchor_slope_min = float(self.cfg.anchor_trend_slope_min_pct)
        anchor_slope_ok_up = df["anchor_trend_slope_pct"] >= anchor_slope_min
        anchor_slope_ok_down = df["anchor_trend_slope_pct"] <= -anchor_slope_min if anchor_slope_min > 0 else True

        gate_up = trend_up & atr_ok & impulse_ok & slope_ok_up
        gate_down = trend_down & atr_ok & impulse_ok & slope_ok_down
        if self.cfg.use_trend_filter:
            gate_up = gate_up & anchor_slope_ok_up
            gate_down = gate_down & anchor_slope_ok_down

        df["regime_trend_up"] = gate_up.fillna(False)
        df["regime_trend_down"] = gate_down.fillna(False)
        df["regime_range"] = (~df["regime_trend_up"] & ~df["regime_trend_down"]).fillna(False)

        range_lookback = max(self.cfg.range_lookback, 20)
        range_high = df["close"].rolling(window=range_lookback, min_periods=range_lookback).max().shift(1)
        range_low = df["close"].rolling(window=range_lookback, min_periods=range_lookback).min().shift(1)
        df["range_high"] = range_high
        df["range_low"] = range_low

        # Micro structure highs/lows for breakout logic
        mh = max(self.cfg.micro_high_lookback, 3)
        ml = max(self.cfg.micro_low_lookback, 3)
        df["micro_high"] = df["high"].rolling(window=mh, min_periods=mh).max().shift(1)
        df["micro_low"] = df["low"].rolling(window=ml, min_periods=ml).min().shift(1)

        return df

    def generate_signal(self, features: pd.DataFrame) -> Optional[Signal]:
        if features.empty:
            return None
        latest = features.iloc[-1]
        ts = features.index[-1]

        # Time-of-day gate (UTC). If allowed_hours is provided, require ts.hour to be inside any window.
        if self.cfg.allowed_hours:
            hour = int(pd.Timestamp(ts).tz_convert("UTC").hour if getattr(ts, "tzinfo", None) else pd.Timestamp(ts, tz="UTC").hour)
            in_any = False
            for win in self.cfg.allowed_hours:
                if len(win) != 2:
                    continue
                start, end = int(win[0]) % 24, int(win[1]) % 24
                if start <= end:
                    if start <= hour <= end:
                        in_any = True
                        break
                else:
                    if hour >= start or hour <= end:
                        in_any = True
                        break
            if not in_any:
                return None

        volume_ratio = float(latest.get("volume_ratio", 0.0))
        if self.cfg.use_volume_filter and volume_ratio < self.cfg.volume_threshold_ratio:
            return None

        atr_val = float(latest.get("atr", 0.0) or 0.0)
        if atr_val <= 0:
            return None

        close_price = float(latest["close"])
        if close_price <= 0:
            return None

        atr_pct = atr_val / close_price
        if self.cfg.min_atr_pct and atr_pct < self.cfg.min_atr_pct:
            return None
        if atr_pct < self.cfg.regime_atr_min_pct:
            return None
        if self.cfg.regime_atr_max_pct and atr_pct > self.cfg.regime_atr_max_pct:
            return None

        ema_fast = float(latest.get("ema_fast", close_price))
        ema_slow = float(latest.get("ema_slow", close_price))
        trend_strength = float(latest.get("trend_strength", 0.0) or 0.0)

        max_strength = self.cfg.max_trend_strength
        if max_strength is not None and abs(trend_strength) > max_strength:
            return None

        trend_up_flag = bool(latest.get("regime_trend_up", False))
        trend_down_flag = bool(latest.get("regime_trend_down", False))
        range_flag = bool(latest.get("regime_range", False))

        rsi_val = float(latest.get("rsi", 50.0))
        pullback_buffer = max(self.cfg.ema_pullback_pct, 0.0)
        close_to_ema_fast = abs(close_price - ema_fast) / close_price <= pullback_buffer

        # ATR-normalized pullback distance gate
        if self.cfg.ema_pullback_atr_mult and atr_val > 0:
            close_to_ema_fast = close_to_ema_fast and (
                abs(close_price - ema_fast) <= self.cfg.ema_pullback_atr_mult * atr_val
            )

        # Recent impulse gate: allow impulse on any of the last W bars, optionally requiring at least N occurrences
        win = max(int(self.cfg.impulse_window_bars), 1)
        recent = features.iloc[-win:]
        recent_impulses = (recent["volume_ratio"] >= self.cfg.impulse_volume_ratio).sum()
        impulse_ok = recent_impulses >= max(int(self.cfg.impulse_min_persist), 1)

        fast_market = bool(latest.get("fast_market", False))
        if self.cfg.avoid_fast_market_entries and fast_market:
            return None

        side: Optional[Side] = None
        reason = ""

        if (
            trend_up_flag
            and impulse_ok
            and close_price <= ema_fast
            and close_to_ema_fast
            and rsi_val <= self.cfg.rsi_buy_threshold
        ):
            side = "Buy"
            reason = "trend pullback long"
        elif (
            self.cfg.allow_counter_trend_shorts
            and trend_down_flag
            and impulse_ok
            and close_price >= ema_fast
            and close_to_ema_fast
            and rsi_val >= self.cfg.rsi_sell_threshold
        ):
            side = "Sell"
            reason = "trend pullback short"
        elif self.cfg.enable_range_trades and range_flag and volume_ratio <= self.cfg.volume_threshold_ratio:
            range_high = float(latest.get("range_high", float("nan")))
            range_low = float(latest.get("range_low", float("nan")))
            band_pct = max(self.cfg.range_band_pct, 0.0)
            if not np.isnan(range_low) and close_price <= range_low * (1 + band_pct) and rsi_val <= self.cfg.rsi_buy_threshold:
                side = "Buy"
                reason = "range long"
            elif self.cfg.allow_counter_trend_shorts and not np.isnan(range_high) and close_price >= range_high * (1 - band_pct) and rsi_val >= self.cfg.rsi_sell_threshold:
                side = "Sell"
                reason = "range short"
        elif self.cfg.enable_breakout_entries:
            band = max(self.cfg.breakout_band_pct, 0.0)
            micro_high = float(latest.get("micro_high", float("nan")))
            micro_low = float(latest.get("micro_low", float("nan")))
            if trend_up_flag and impulse_ok and not np.isnan(micro_high):
                if close_price >= micro_high * (1 + band) and close_price >= ema_fast:
                    side = "Buy"
                    reason = "breakout long"
            elif self.cfg.allow_counter_trend_shorts and trend_down_flag and impulse_ok and not np.isnan(micro_low):
                if close_price <= micro_low * (1 - band) and close_price <= ema_fast:
                    side = "Sell"
                    reason = "breakout short"

        if side is None:
            return None

        entry_price = close_price
        min_stop_pct = max(self.cfg.min_stop_distance_pct or 0.0, 0.0)
        min_stop_offset = entry_price * min_stop_pct
        atr_offset = atr_val * self.cfg.atr_mult_stop
        ema_offset = abs(entry_price - ema_fast) * max(self.cfg.support_stop_weight, 0.0)
        range_high = float(latest.get("range_high", float("nan")))
        range_low = float(latest.get("range_low", float("nan")))
        range_offset = 0.0
        if side == "Buy" and not np.isnan(range_low):
            range_offset = max(range_offset, abs(entry_price - range_low))
        if side == "Sell" and not np.isnan(range_high):
            range_offset = max(range_offset, abs(range_high - entry_price))
        stop_offset = max(min_stop_offset, atr_offset, ema_offset, range_offset)
        stop_price = entry_price - stop_offset if side == "Buy" else entry_price + stop_offset
        if stop_price <= 0:
            return None

        timestamp = features.index[-1]
        return Signal(
            timestamp=timestamp,
            side=side,
            entry_price=entry_price,
            stop_price=stop_price,
            atr=atr_val,
            reason=reason,
            fast_market=fast_market,
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
