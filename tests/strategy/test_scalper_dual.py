import pandas as pd
import pytest

from coin_trader_bybit.core.config import StrategyConfig
from coin_trader_bybit.strategy.scalper import Scalper


@pytest.fixture
def base_cfg() -> StrategyConfig:
    return StrategyConfig(
        timeframe_entry="1m",
        timeframe_anchor="1m",
        ema_fast=2,
        ema_slow=3,
        atr_period=3,
        atr_mult_stop=1.5,
        partial_tp_r=3.0,
        trail_atr_mult=2.0,
        micro_high_lookback=2,
        time_stop_minutes=30,
        volume_ma_period=2,
        volume_threshold_ratio=1.0,
        use_trend_filter=False,
        use_volume_filter=True,
        entry_buffer_pct=0.01,
        stop_loss_pct=0.03,
        allow_counter_trend_shorts=True,
        volume_timeframes=["1m"],
        volume_tf_mode="any",
    )


def _candles_for_long(volume_last: float = 1500.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=5, freq="1min")
    data = {
        "open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "high": [101.0, 102.0, 103.0, 104.0, 110.0],
        "low": [99.0, 100.0, 101.0, 102.0, 103.0],
        "close": [100.0, 101.0, 102.0, 103.0, 109.0],
        "volume": [1000.0, 900.0, 1100.0, 1000.0, volume_last],
    }
    return pd.DataFrame(data, index=idx)


def _candles_for_short(volume_last: float = 1500.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=5, freq="1min")
    data = {
        "open": [110.0, 109.0, 108.0, 107.0, 106.0],
        "high": [111.0, 110.0, 109.0, 108.0, 107.0],
        "low": [109.0, 108.0, 107.0, 98.0, 95.0],
        "close": [110.0, 109.0, 108.0, 101.0, 96.0],
        "volume": [1000.0, 900.0, 1100.0, 1200.0, volume_last],
    }
    return pd.DataFrame(data, index=idx)


def test_generate_long_signal(base_cfg: StrategyConfig):
    scalper = Scalper(base_cfg)
    features = scalper.compute_features(_candles_for_long())
    signal = scalper.generate_signal(features)
    assert signal is not None
    assert signal.side == "Buy"
    # buffer lowers entry
    expected_entry = features["close"].iloc[-1] * (1 - base_cfg.entry_buffer_pct)
    assert signal.entry_price == pytest.approx(expected_entry)
    expected_stop = expected_entry * (1 - base_cfg.stop_loss_pct)
    assert signal.stop_price == pytest.approx(expected_stop)


def test_generate_short_signal(base_cfg: StrategyConfig):
    scalper = Scalper(base_cfg)
    features = scalper.compute_features(_candles_for_short())
    signal = scalper.generate_signal(features)
    assert signal is not None
    assert signal.side == "Sell"
    expected_entry = features["close"].iloc[-1] * (1 + base_cfg.entry_buffer_pct)
    assert signal.entry_price == pytest.approx(expected_entry)
    expected_stop = expected_entry * (1 + base_cfg.stop_loss_pct)
    assert signal.stop_price == pytest.approx(expected_stop)


def test_volume_filter_blocks_signal(base_cfg: StrategyConfig):
    cfg = base_cfg
    scalper = Scalper(cfg)
    features = scalper.compute_features(_candles_for_long(volume_last=10.0))
    signal = scalper.generate_signal(features)
    assert signal is None


def test_multi_timeframe_volume_any_passes(base_cfg: StrategyConfig):
    cfg = base_cfg.model_copy(
        update={
            "volume_timeframes": ["1m", "5m"],
            "volume_tf_mode": "any",
            "volume_ma_period": 1,
        }
    )
    scalper = Scalper(cfg)
    features = scalper.compute_features(_candles_for_long())
    signal = scalper.generate_signal(features)
    assert signal is not None


def test_multi_timeframe_volume_all_requires_all(base_cfg: StrategyConfig):
    cfg = base_cfg.model_copy(
        update={
            "volume_timeframes": ["1m", "5m"],
            "volume_tf_mode": "all",
            "volume_ma_period": 2,
        }
    )
    scalper = Scalper(cfg)
    features = scalper.compute_features(_candles_for_long())
    signal = scalper.generate_signal(features)
    assert signal is None
