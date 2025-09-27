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
        time_stop_minutes=30,
        long_partial_tp_r=1.0,
        long_partial_fraction=0.5,
        long_trail_atr_mult=1.5,
        long_time_stop_minutes=60,
        short_partial_tp_r=1.0,
        short_partial_fraction=0.5,
        short_trail_atr_mult=1.5,
        short_time_stop_minutes=60,
        range_lookback=60,
        range_band_pct=0.01,
        range_partial_tp_r=1.0,
        range_partial_fraction=0.5,
        range_trail_atr_mult=1.5,
        range_time_stop_minutes=60,
        volume_ma_period=3,
        volume_threshold_ratio=0.5,
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
        max_trend_strength=None,
        regime_trend_min=0.0,
        regime_atr_min_pct=0.0,
        regime_atr_max_pct=1.0,
        impulse_volume_ratio=0.1,
        fast_market_volume_ratio=10.0,
        fast_market_atr_pct=10.0,
        fast_market_tp_r=1.0,
        fast_market_partial_fraction=1.0,
        fast_market_time_stop_minutes=30,
        fast_market_trail_atr_mult=1.0,
        rsi_period=3,
        rsi_buy_threshold=80.0,
        rsi_sell_threshold=20.0,
        ema_pullback_pct=0.05,
        trend_confirm_lookback=2,
        trend_slope_min_pct=0.0,
    )
    return base.model_copy(update=updates)


def _make_df(rows: list[dict]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="1min")
    return pd.DataFrame(rows, index=idx)


def _assert_stop(signal, entry: float, atr: float, cfg: StrategyConfig, *, is_long: bool) -> None:
    min_offset = max(
        entry * cfg.min_stop_distance_pct,
        atr * cfg.atr_mult_stop,
    )
    actual = (entry - signal.stop_price) if is_long else (signal.stop_price - entry)
    assert actual >= min_offset - 1e-6


def test_long_bounce_signal_sets_stop_with_buffer() -> None:
    df = _make_df(
        [
            {"open": 100.5, "high": 101.0, "low": 99.8, "close": 100.2, "volume": 900},
            {"open": 100.1, "high": 100.6, "low": 99.7, "close": 100.4, "volume": 950},
            {"open": 100.0, "high": 100.5, "low": 99.6, "close": 100.5, "volume": 1000},
            {"open": 99.8, "high": 100.3, "low": 99.55, "close": 100.6, "volume": 1100},
            {"open": 100.2, "high": 101.2, "low": 99.52, "close": 101.1, "volume": 1800},
            {"open": 100.8, "high": 101.2, "low": 99.8, "close": 100.0, "volume": 2600},
        ]
    )
    scalper = Scalper(_cfg(rsi_buy_threshold=100.0, impulse_volume_ratio=0.0, volume_threshold_ratio=0.0, regime_trend_min=-1.0, trend_slope_min_pct=-1.0))
    features = scalper.compute_features(df)
    signal = scalper.generate_signal(features)
    assert signal is not None and signal.side == "Buy"
    entry = features["close"].iloc[-1]
    atr = features["atr"].iloc[-1]
    _assert_stop(signal, entry, atr, scalper.cfg, is_long=True)


def test_short_reject_signal_sets_stop_with_buffer() -> None:
    df = _make_df(
        [
            {"open": 100.2, "high": 100.9, "low": 100.0, "close": 100.7, "volume": 900},
            {"open": 100.7, "high": 101.5, "low": 100.3, "close": 101.3, "volume": 950},
            {"open": 101.3, "high": 102.3, "low": 101.0, "close": 101.9, "volume": 980},
            {"open": 102.0, "high": 102.9, "low": 101.4, "close": 101.4, "volume": 1200},
            {"open": 101.4, "high": 102.8, "low": 101.1, "close": 100.9, "volume": 2200},
            {"open": 101.8, "high": 103.5, "low": 101.0, "close": 103.0, "volume": 3600},
        ]
    )
    scalper = Scalper(_cfg(rsi_sell_threshold=0.0, impulse_volume_ratio=0.0, volume_threshold_ratio=0.0, regime_trend_min=-1.0, trend_slope_min_pct=-1.0))
    features = scalper.compute_features(df)
    signal = scalper.generate_signal(features)
    assert signal is not None and signal.side == "Sell"
    entry = features["close"].iloc[-1]
    atr = features["atr"].iloc[-1]
    _assert_stop(signal, entry, atr, scalper.cfg, is_long=False)


def test_volume_spike_required_for_signal() -> None:
    df = _make_df(
        [
            {"open": 100.0, "high": 100.4, "low": 99.8, "close": 100.2, "volume": 900},
            {"open": 100.2, "high": 100.6, "low": 100.0, "close": 100.4, "volume": 900},
            {"open": 100.4, "high": 100.8, "low": 100.2, "close": 100.6, "volume": 900},
        ]
    )
    scalper = Scalper(_cfg())
    features = scalper.compute_features(df)
    assert scalper.generate_signal(features) is None
