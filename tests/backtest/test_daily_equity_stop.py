import pandas as pd

from coin_trader_bybit.backtest import Backtester
from coin_trader_bybit.core.config import AppConfig


def _lossy_config() -> AppConfig:
    cfg = AppConfig()
    # Loosen filters to force frequent entries
    cfg.strategy.timeframe_entry = "1m"
    cfg.strategy.timeframe_anchor = "1m"
    cfg.strategy.use_trend_filter = False
    cfg.strategy.ema_fast = 2
    cfg.strategy.ema_slow = 4
    cfg.strategy.atr_period = 3
    cfg.strategy.atr_mult_stop = 1.5
    cfg.strategy.partial_tp_r = 10.0  # ensure partial tp won't trigger
    cfg.strategy.trail_atr_mult = 0.0
    cfg.strategy.time_stop_minutes = 120
    cfg.strategy.volume_ma_period = 3
    cfg.strategy.volume_threshold_ratio = 0.0
    cfg.strategy.impulse_volume_ratio = 0.0
    cfg.strategy.min_atr_pct = 0.0
    cfg.strategy.regime_atr_min_pct = 0.0
    cfg.strategy.regime_atr_max_pct = 1.0
    cfg.strategy.regime_trend_min = -1.0
    cfg.strategy.trend_slope_min_pct = -1.0
    cfg.strategy.rsi_period = 3
    cfg.strategy.rsi_buy_threshold = 100.0  # always pass
    cfg.strategy.ema_pullback_pct = 1.0
    cfg.strategy.allow_counter_trend_shorts = False

    # Risk: 1% per trade, 2% daily equity stop
    cfg.risk.starting_equity = 10_000.0
    cfg.risk.max_risk_per_trade_pct = 1.0
    cfg.risk.daily_max_trades = 100
    cfg.risk.daily_loss_limit_pct = 0.02

    # Remove frictions to make outcome deterministic
    cfg.execution.taker_fee_bps = 0.0
    cfg.execution.slippage_bps = 0.0
    cfg.execution.post_only = False
    return cfg


def _make_three_loss_dataset() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01 00:00", periods=60, freq="1min")
    close = [100.0] * 60
    open_ = [100.0] * 60
    high = [100.5] * 60
    low = [99.5] * 60
    volume = [100.0] * 60

    # Craft three long entries that each stop out on the following bar
    for n, base in enumerate([10, 25, 40]):
        # pre-entry bullish context so EMA is above price at entry bar
        close[base - 3] = 101.0
        close[base - 2] = 101.5
        close[base - 1] = 102.0
        high[base - 3] = 101.4
        high[base - 2] = 101.9
        high[base - 1] = 102.4
        low[base - 3] = 100.6
        low[base - 2] = 101.1
        low[base - 1] = 101.6
        volume[base - 1] = 500.0

        # entry bar: pullback into EMA, volume spike
        close[base] = 100.2
        open_[base] = 100.6
        high[base] = 100.8
        low[base] = 99.9
        volume[base] = 1000.0

        # next bar: plunge below any reasonable stop distance to realize full -1R
        close[base + 1] = 98.0
        open_[base + 1] = 100.0
        high[base + 1] = 100.1
        low[base + 1] = 95.0
        volume[base + 1] = 1200.0

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


def test_daily_equity_stop_blocks_third_entry_same_day() -> None:
    cfg = _lossy_config()
    df = _make_three_loss_dataset()
    report = Backtester(cfg).run(df)

    # Two back-to-back -1R losses should hit the -2% equity stop,
    # preventing the third entry on the same UTC day.
    assert report.metrics.num_trades == 2

