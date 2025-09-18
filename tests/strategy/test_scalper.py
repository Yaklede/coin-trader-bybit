import pandas as pd
import pytest

from coin_trader_bybit.core.config import StrategyConfig
from coin_trader_bybit.strategy.scalper import Scalper


def _sample_candles(volume_last: float = 1500.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=5, freq="1min")
    data = {
        "open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "high": [101.0, 102.0, 103.0, 104.0, 110.0],
        "low": [99.0, 100.0, 101.0, 102.0, 103.0],
        "close": [100.0, 101.0, 102.0, 103.0, 109.0],
        "volume": [1000.0, 900.0, 1100.0, 1000.0, volume_last],
    }
    return pd.DataFrame(data, index=idx)


def _base_cfg(**overrides) -> StrategyConfig:
    cfg = StrategyConfig(
        timeframe_entry="1m",
        timeframe_anchor="1m",
        ema_fast=2,
        ema_slow=3,
        atr_period=3,
        atr_mult_stop=1.0,
        partial_tp_r=1.0,
        trail_atr_mult=1.5,
        micro_high_lookback=2,
        time_stop_minutes=15,
        volume_ma_period=2,
        volume_threshold_ratio=1.1,
        use_trend_filter=False,
        use_volume_filter=True,
        entry_buffer_pct=0.01,
        stop_loss_pct=0.02,
        volume_timeframes=["1m"],
        volume_tf_mode="any",
        allow_counter_trend_shorts=True,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_scalper_applies_entry_buffer_and_percent_stop():
    cfg = _base_cfg()
    scalper = Scalper(cfg)
    features = scalper.compute_features(_sample_candles())
    signal = scalper.generate_signal(features)
    assert signal is not None
    expected_entry = features["close"].iloc[-1] * (1 - cfg.entry_buffer_pct)
    assert signal.entry_price == pytest.approx(expected_entry)
    expected_stop = expected_entry * (1 - cfg.stop_loss_pct)
    assert signal.stop_price == pytest.approx(expected_stop)


def test_scalper_ignores_volume_filter_when_disabled():
    cfg = _base_cfg(use_volume_filter=False)
    scalper = Scalper(cfg)
    features = scalper.compute_features(_sample_candles(volume_last=100.0))
    signal = scalper.generate_signal(features)
    assert signal is not None


def test_scalper_generates_signal_without_trend_filter():
    cfg = _base_cfg(use_trend_filter=False)
    scalper = Scalper(cfg)
    features = scalper.compute_features(_sample_candles())
    signal = scalper.generate_signal(features)
    assert signal is not None
