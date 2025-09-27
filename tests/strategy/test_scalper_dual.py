import pandas as pd

from coin_trader_bybit.core.config import StrategyConfig
from coin_trader_bybit.strategy.scalper import Scalper


def _cfg(**updates) -> StrategyConfig:
    base = StrategyConfig(
        timeframe_entry="1m",
        timeframe_anchor="1m",
        ema_fast=2,
        ema_slow=4,
        atr_period=3,
        atr_mult_stop=1.5,
        partial_tp_r=1.0,
        trail_atr_mult=1.5,
        micro_high_lookback=2,
        time_stop_minutes=30,
        volume_ma_period=3,
        volume_threshold_ratio=1.4,
        use_trend_filter=False,
        use_volume_filter=True,
        entry_buffer_pct=0.0,
        stop_loss_pct=None,
        allow_counter_trend_shorts=True,
        volume_timeframes=["1m"],
        volume_tf_mode="any",
        trend_slope_threshold=0.0,
        min_atr_pct=0.0,
        support_resistance_lookback=3,
        pattern_ticks=3,
        bounce_tolerance_pct=0.001,
        min_stop_distance_pct=0.005,
        max_pre_bearish_for_bounce=1,
        min_pre_bullish_for_reject=3,
    )
    return base.model_copy(update=updates)


def _df(rows):
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="1min")
    return pd.DataFrame(rows, index=idx)


def test_shorts_blocked_when_flag_disabled():
    df = _df(
        [
            {"open": 100.2, "high": 100.9, "low": 100.0, "close": 100.7, "volume": 900},
            {"open": 100.7, "high": 101.5, "low": 100.3, "close": 101.3, "volume": 950},
            {"open": 101.3, "high": 102.3, "low": 101.0, "close": 101.9, "volume": 980},
            {"open": 102.0, "high": 102.9, "low": 101.4, "close": 101.4, "volume": 1200},
            {"open": 101.4, "high": 102.8, "low": 101.1, "close": 100.9, "volume": 2200},
            {"open": 100.9, "high": 103.0, "low": 100.6, "close": 100.2, "volume": 3600},
        ]
    )
    scalper = Scalper(_cfg(allow_counter_trend_shorts=False, min_pre_bullish_for_reject=1))
    features = scalper.compute_features(df)
    assert scalper.generate_signal(features) is None


def test_short_reject_signal_emitted_when_allowed():
    df = _df(
        [
            {"open": 100.2, "high": 100.9, "low": 100.0, "close": 100.7, "volume": 900},
            {"open": 100.7, "high": 101.5, "low": 100.3, "close": 101.3, "volume": 950},
            {"open": 101.3, "high": 102.3, "low": 101.0, "close": 101.9, "volume": 980},
            {"open": 102.0, "high": 102.9, "low": 101.4, "close": 101.4, "volume": 1200},
            {"open": 101.4, "high": 102.8, "low": 101.1, "close": 100.9, "volume": 2200},
            {"open": 100.9, "high": 103.0, "low": 100.6, "close": 100.2, "volume": 3800},
        ]
    )
    scalper = Scalper(_cfg(min_pre_bullish_for_reject=1))
    features = scalper.compute_features(df)
    signal = scalper.generate_signal(features)
    assert signal is not None and signal.side == "Sell"
