from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..core.config import StrategyConfig
from .scalper import Signal, timeframe_to_pandas_freq


@dataclass
class _SwingParams:
    # Defaults tuned conservatively; override via params_swing.yaml
    rsi_buy_threshold: float = 45.0
    atr_mult_stop: float = 2.0
    min_stop_distance_pct: float = 0.005
    pullback_pct: float = 0.0025
    atr_min_pct: float = 0.0008
    atr_max_pct: float = 0.0030
    anchor_slope_min_pct: float = 0.0005


def _resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    return df.resample(freq).agg(agg).dropna(how="any")


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    # Wilder smoothing via EMA(alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)).fillna(0.0)
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx.fillna(0.0)


class Swing:
    """Lower-frequency trend-pullback strategy using higher-timeframe regime.

    - Entry timeframe: cfg.timeframe_entry (e.g., 15m)
    - Anchor timeframe: cfg.timeframe_anchor (e.g., 1h)
    - Long-only by default; shorts optional via cfg.allow_counter_trend_shorts
    """

    def __init__(self, cfg: StrategyConfig) -> None:
        self.cfg = cfg
        self.p = _SwingParams(
            rsi_buy_threshold=getattr(cfg, "rsi_buy_threshold", 45.0),
            atr_mult_stop=getattr(cfg, "atr_mult_stop", 2.0),
            min_stop_distance_pct=getattr(cfg, "min_stop_distance_pct", 0.005),
            pullback_pct=getattr(cfg, "ema_pullback_pct", 0.0025),
            atr_min_pct=getattr(cfg, "min_atr_pct", 0.0008),
            atr_max_pct=getattr(cfg, "regime_atr_max_pct", 0.0030),
            anchor_slope_min_pct=getattr(cfg, "anchor_trend_slope_min_pct", 0.0005),
        )

    def compute_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data.index, pd.DatetimeIndex) or data.empty:
            raise ValueError("data must be a non-empty DatetimeIndex DataFrame")

        entry_freq = timeframe_to_pandas_freq(self.cfg.timeframe_entry)
        df = _resample_ohlcv(data, entry_freq)
        if df.empty:
            raise ValueError("resampling produced empty dataset; check timeframe")

        df["ema_fast"] = _ema(df["close"], max(self.cfg.ema_fast, 2))
        df["ema_slow"] = _ema(df["close"], max(self.cfg.ema_slow, 3))
        df["atr"] = _atr(df, max(self.cfg.atr_period, 5))
        df["atr_pct"] = (df["atr"] / df["close"].replace(0.0, np.nan)).fillna(0.0)
        # keep a simple volume MA to satisfy downstream filters
        vol_ma = df["volume"].rolling(window=max(self.cfg.volume_ma_period, 5), min_periods=1).mean()
        df["volume_ma"] = vol_ma
        df["volume_ratio"] = (df["volume"] / vol_ma.replace(0.0, np.nan)).fillna(0.0)
        # Rolling ATR% percentile for contraction detection
        pct_win = 24  # 24 entry bars ~ 1 day on 1h
        rolling = df["atr_pct"].rolling(window=pct_win, min_periods=pct_win)
        rank = rolling.apply(lambda x: (x.rank(pct=True).iloc[-1] if len(x) == pct_win else np.nan))
        df["atr_pct_pctile"] = rank

        # Simple RSI(14) on entry close
        delta = df["close"].diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        df["rsi"] = (100 - (100 / (1 + rs))).fillna(50.0)

        # Anchor regime on higher timeframe
        if self.cfg.use_trend_filter:
            anchor_freq = timeframe_to_pandas_freq(self.cfg.timeframe_anchor)
            a = _resample_ohlcv(data, anchor_freq)
            a_fast = _ema(a["close"], max(self.cfg.ema_fast, 2))
            a_slow = _ema(a["close"], max(self.cfg.ema_slow, 3))
            # Compute anchor slope over confirm window and project back to entry index
            look = max(self.cfg.trend_confirm_lookback, 10)
            a_slow_prev = a_slow.shift(look)
            a_slope = ((a_slow - a_slow_prev) / a_slow_prev.replace(0.0, np.nan)).fillna(0.0)
            # Anchor ADX
            adx_period = max(int(getattr(self.cfg, "adx_period", 14)), 5)
            a_adx = _adx(a, adx_period)
            df["anchor_fast"] = a_fast.reindex(df.index, method="ffill").fillna(df["close"])  # type: ignore[assignment]
            df["anchor_slow"] = a_slow.reindex(df.index, method="ffill").fillna(df["close"])  # type: ignore[assignment]
            df["anchor_slope_pct"] = a_slope.reindex(df.index, method="ffill").fillna(0.0)  # type: ignore[assignment]
            df["anchor_adx"] = a_adx.reindex(df.index, method="ffill").fillna(0.0)  # type: ignore[assignment]
        else:
            df["anchor_fast"] = df["ema_fast"]
            df["anchor_slow"] = df["ema_slow"]
            df["anchor_slope_pct"] = (df["ema_slow"] - df["ema_slow"].shift(max(self.cfg.trend_confirm_lookback, 10))) / df[
                "ema_slow"
            ].shift(max(self.cfg.trend_confirm_lookback, 10)).replace(0.0, np.nan)
            df["anchor_slope_pct"] = df["anchor_slope_pct"].fillna(0.0)
            df["anchor_adx"] = 25.0

        # Regime definitions
        uptrend = (df["anchor_fast"] > df["anchor_slow"]) & (df["anchor_slope_pct"] >= self.p.anchor_slope_min_pct)
        adx_ok = True
        if getattr(self.cfg, "anchor_adx_min", None) is not None:
            adx_ok = df["anchor_adx"] >= float(self.cfg.anchor_adx_min)
        atr_ok = (df["atr_pct"] >= self.p.atr_min_pct) & (df["atr_pct"] <= self.p.atr_max_pct)
        df["regime_up"] = (uptrend & adx_ok & atr_ok).fillna(False)
        downtrend = (df["anchor_fast"] < df["anchor_slow"]) & (df["anchor_slope_pct"] <= -self.p.anchor_slope_min_pct)
        df["regime_down"] = (downtrend & adx_ok & atr_ok).fillna(False)

        # Previous day high/low for daily breakout entry
        if getattr(self.cfg, "enable_daily_breakout", False):
            daily = _resample_ohlcv(data, "1D")
            prev_high = daily["high"].shift(1)
            prev_low = daily["low"].shift(1)
            df["prev_day_high"] = prev_high.reindex(df.index, method="ffill")
            df["prev_day_low"] = prev_low.reindex(df.index, method="ffill")
            # NR7: previous day's range is the lowest of last 7 days
            prev_range = (daily["high"].shift(1) - daily["low"].shift(1))
            rolling_min = (daily["high"] - daily["low"]).rolling(7, min_periods=7).min().shift(1)
            nr7 = (prev_range <= rolling_min)
            df["prev_day_nr7"] = nr7.reindex(df.index, method="ffill").fillna(False)

        # Rolling micro highs/lows on entry TF for multi-day breakout
        mh = max(int(getattr(self.cfg, "micro_high_lookback", 60)), 3)
        ml = max(int(getattr(self.cfg, "micro_low_lookback", 60)), 3)
        df["micro_high"] = df["high"].rolling(window=mh, min_periods=mh).max().shift(1)
        df["micro_low"] = df["low"].rolling(window=ml, min_periods=ml).min().shift(1)

        return df

    def generate_signal(self, features: pd.DataFrame) -> Optional[Signal]:
        if features.empty:
            return None
        latest = features.iloc[-1]
        ts = features.index[-1]

        # Time-of-day gate (UTC)
        if getattr(self.cfg, "allowed_hours", None):
            hour = int(pd.Timestamp(ts).tz_convert("UTC").hour if getattr(ts, "tzinfo", None) else pd.Timestamp(ts, tz="UTC").hour)
            ok = False
            for win in self.cfg.allowed_hours:
                if len(win) != 2:
                    continue
                s, e = int(win[0]) % 24, int(win[1]) % 24
                if s <= e and s <= hour <= e:
                    ok = True
                    break
                if s > e and (hour >= s or hour <= e):
                    ok = True
                    break
            if not ok:
                return None

        close = float(latest["close"])  # entry timeframe close
        atr = float(latest.get("atr", 0.0) or 0.0)
        if close <= 0 or atr <= 0:
            return None

        # Daily breakout first (at most once per day)
        if getattr(self.cfg, "enable_daily_breakout", False) and bool(latest.get("regime_up", False)):
            # Pre-breakout contraction gate (optional)
            max_pctile = getattr(self.cfg, "prebreakout_atr_pct_max_pctile", None)
            if max_pctile is not None:
                pctile = float(latest.get("atr_pct_pctile", np.nan))
                if not np.isnan(pctile) and pctile > max_pctile:
                    return None
            if getattr(self.cfg, "require_nr7", False):
                if not bool(latest.get("prev_day_nr7", False)):
                    return None
            pdh = float(latest.get("prev_day_high", np.nan))
            band = max(getattr(self.cfg, "daily_breakout_band_pct", 0.0), 0.0)
            breakout_level = pdh * (1 + band) if not np.isnan(pdh) else float("nan")
            high = float(latest.get("high", close))
            if not np.isnan(breakout_level) and high >= breakout_level:
                entry = breakout_level
                stop = max(entry - max(entry * self.p.min_stop_distance_pct, atr * self.p.atr_mult_stop), 0.0)
                if stop > 0:
                    return Signal(
                        timestamp=features.index[-1],
                        side="Buy",
                        entry_price=entry,
                        stop_price=stop,
                        atr=atr,
                        reason="swing daily breakout long",
                        fast_market=False,
                    )

        # Daily breakdown short (if allowed and downtrend regime)
        if getattr(self.cfg, "enable_daily_breakout", False) and bool(latest.get("regime_down", False)) and getattr(self.cfg, "allow_counter_trend_shorts", False):
            max_pctile = getattr(self.cfg, "prebreakout_atr_pct_max_pctile", None)
            if max_pctile is not None:
                pctile = float(latest.get("atr_pct_pctile", np.nan))
                if not np.isnan(pctile) and pctile > max_pctile:
                    return None
            if getattr(self.cfg, "require_nr7", False):
                if not bool(latest.get("prev_day_nr7", False)):
                    return None
            pdl = float(latest.get("prev_day_low", np.nan))
            band = max(getattr(self.cfg, "daily_breakout_band_pct", 0.0), 0.0)
            breakdown_level = pdl * (1 - band) if not np.isnan(pdl) else float("nan")
            low = float(latest.get("low", close))
            if not np.isnan(breakdown_level) and low <= breakdown_level:
                entry = breakdown_level
                stop = entry + max(entry * self.p.min_stop_distance_pct, atr * self.p.atr_mult_stop)
                if stop > 0:
                    return Signal(
                        timestamp=features.index[-1],
                        side="Sell",
                        entry_price=entry,
                        stop_price=stop,
                        atr=atr,
                        reason="swing daily breakout short",
                        fast_market=False,
                    )

        # Rolling breakout on entry timeframe (more 기회)
        if getattr(self.cfg, "enable_breakout_entries", False):
            band = max(getattr(self.cfg, "breakout_band_pct", 0.0), 0.0)
            micro_high = float(latest.get("micro_high", np.nan))
            micro_low = float(latest.get("micro_low", np.nan))
            high = float(latest.get("high", close))
            low = float(latest.get("low", close))
            # Apply same contraction gate if configured
            max_pctile = getattr(self.cfg, "prebreakout_atr_pct_max_pctile", None)
            if max_pctile is not None:
                pctile = float(latest.get("atr_pct_pctile", np.nan))
                if not np.isnan(pctile) and pctile > max_pctile:
                    return None
            # Long in uptrend regime
            if bool(latest.get("regime_up", False)) and not np.isnan(micro_high) and high >= micro_high * (1 + band):
                entry = micro_high * (1 + band)
                stop = max(entry - max(entry * self.p.min_stop_distance_pct, atr * self.p.atr_mult_stop), 0.0)
                if stop > 0:
                    return Signal(
                        timestamp=features.index[-1],
                        side="Buy",
                        entry_price=entry,
                        stop_price=stop,
                        atr=atr,
                        reason="swing breakout long",
                        fast_market=False,
                    )
            # Short in downtrend regime
            if getattr(self.cfg, "allow_counter_trend_shorts", False) and bool(latest.get("regime_down", False)) and not np.isnan(micro_low) and low <= micro_low * (1 - band):
                entry = micro_low * (1 - band)
                stop = entry + max(entry * self.p.min_stop_distance_pct, atr * self.p.atr_mult_stop)
                if stop > 0:
                    return Signal(
                        timestamp=features.index[-1],
                        side="Sell",
                        entry_price=entry,
                        stop_price=stop,
                        atr=atr,
                        reason="swing breakout short",
                        fast_market=False,
                    )

        # Otherwise: pullback in uptrend regime (optional)
        if not getattr(self.cfg, "enable_pullback_entries", True):
            return None
        if not bool(latest.get("regime_up", False)):
            return None
        ema_fast = float(latest.get("ema_fast", close))
        rsi = float(latest.get("rsi", 50.0))
        pullback = (close <= ema_fast) and (abs(close - ema_fast) / close <= max(self.p.pullback_pct, 0.0)) and (
            rsi <= self.p.rsi_buy_threshold
        )
        if not pullback:
            return None

        entry = close
        stop = max(entry - max(entry * self.p.min_stop_distance_pct, atr * self.p.atr_mult_stop), 0.0)
        if stop <= 0:
            return None
        return Signal(
            timestamp=features.index[-1],
            side="Buy",
            entry_price=entry,
            stop_price=stop,
            atr=atr,
            reason="swing pullback long",
            fast_market=False,
        )


__all__ = ["Swing"]
