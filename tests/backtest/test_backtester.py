import pandas as pd
import pytest

from coin_trader_bybit.backtest import Backtester
from coin_trader_bybit.core.config import AppConfig


def _base_config() -> AppConfig:
    cfg = AppConfig()
    cfg.strategy.ema_fast = 3
    cfg.strategy.ema_slow = 8
    cfg.strategy.atr_period = 3
    cfg.strategy.partial_tp_r = 1.0
    cfg.strategy.trail_atr_mult = 1.5
    cfg.strategy.time_stop_minutes = 5
    cfg.strategy.micro_high_lookback = 4
    cfg.strategy.volume_ma_period = 3
    cfg.strategy.volume_threshold_ratio = 1.0
    cfg.risk.max_risk_per_trade_pct = 1.0
    cfg.execution.taker_fee_bps = 6.0
    cfg.execution.slippage_bps = 5.0
    return cfg


def _make_partial_dataset() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01 00:00", periods=20, freq="1min")
    close = [100.0] * 20
    open_ = [99.8] * 20
    high = [100.5] * 20
    low = [99.5] * 20
    volume = [100.0] * 20

    entry_idx = 12
    close[entry_idx] = 103.0
    high[entry_idx] = 104.0
    low[entry_idx] = 102.0
    volume[entry_idx] = 500.0

    close[entry_idx + 1] = 105.0
    high[entry_idx + 1] = 107.0
    low[entry_idx + 1] = 104.0
    volume[entry_idx + 1] = 500.0

    close[entry_idx + 2] = 104.0
    high[entry_idx + 2] = 105.0
    low[entry_idx + 2] = 103.0
    volume[entry_idx + 2] = 400.0

    for j in range(entry_idx + 3, 20):
        close[j] = 102.5 - 0.3 * (j - entry_idx - 3)
        high[j] = close[j] + 0.4
        low[j] = close[j] - 0.6
        volume[j] = 150.0

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


def _make_time_stop_dataset() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01 00:00", periods=25, freq="1min")
    close = [100.0] * 25
    open_ = [99.8] * 25
    high = [100.5] * 25
    low = [99.5] * 25
    volume = [120.0] * 25

    entry_idx = 12
    close[entry_idx] = 103.0
    high[entry_idx] = 104.0
    low[entry_idx] = 102.0
    volume[entry_idx] = 600.0

    for j in range(entry_idx + 1, 25):
        close[j] = 103.2 - 0.05 * (j - entry_idx)
        high[j] = close[j] + 0.3
        low[j] = close[j] - 0.3
        volume[j] = 150.0

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


def test_backtest_partial_then_stop_exit() -> None:
    df = _make_partial_dataset()

    cfg = _base_config()
    backtester = Backtester(cfg)
    report = backtester.run(df)

    assert len(report.trades) == 1
    trade = report.trades[0]
    assert trade.partial_taken is True
    assert trade.exit_reason in {"trail", "stop"}
    assert trade.pnl > 0
    assert report.metrics.win_rate == pytest.approx(1.0)
    assert report.metrics.total_pnl == pytest.approx(trade.pnl)


def test_backtest_time_stop_triggers_exit() -> None:
    df = _make_time_stop_dataset()

    cfg = _base_config()
    backtester = Backtester(cfg)
    report = backtester.run(df)

    assert len(report.trades) == 1
    trade = report.trades[0]
    assert trade.exit_reason == "time"
    assert trade.partial_taken is False
    assert trade.pnl < 0
    assert report.metrics.total_pnl == pytest.approx(trade.pnl)
    assert report.metrics.win_rate == pytest.approx(0.0)


def test_backtest_fee_impact() -> None:
    df = _make_partial_dataset()

    cfg_no_fee = _base_config()
    cfg_no_fee.execution.taker_fee_bps = 0.0
    cfg_no_fee.execution.slippage_bps = 0.0
    pnl_no_fee = Backtester(cfg_no_fee).run(df).metrics.total_pnl

    cfg_fee = _base_config()
    cfg_fee.execution.taker_fee_bps = 10.0
    cfg_fee.execution.slippage_bps = 0.0
    pnl_fee = Backtester(cfg_fee).run(df).metrics.total_pnl

    assert pnl_fee < pnl_no_fee
