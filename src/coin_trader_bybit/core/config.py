from typing import List

from pydantic import BaseModel, Field


class RiskConfig(BaseModel):
    max_risk_per_trade_pct: float = 0.5
    daily_stop_r_multiple: float = -2.0
    cooldown_minutes: int = 3
    max_live_order_notional_krw: float = 50_000.0
    usdt_krw_rate: float = 1_350.0
    starting_equity: float = 10_000.0


class StrategyConfig(BaseModel):
    name: str = "scalper_v1"
    timeframe_entry: str = "1m"
    timeframe_anchor: str = "5m"
    ema_fast: int = 50
    ema_slow: int = 200
    atr_period: int = 14
    atr_mult_stop: float = 1.0
    partial_tp_r: float = 1.0
    trail_atr_mult: float = 1.5
    micro_high_lookback: int = 3
    time_stop_minutes: int = 15
    volume_ma_period: int = 20
    volume_threshold_ratio: float = 1.2
    use_trend_filter: bool = True
    use_volume_filter: bool = True
    entry_buffer_pct: float = 0.0
    stop_loss_pct: float | None = None
    allow_counter_trend_shorts: bool = False
    volume_timeframes: List[str] = Field(
        default_factory=lambda: ["1m", "5m", "15m", "1h", "4h"]
    )
    volume_tf_mode: str = "any"


class ExecutionConfig(BaseModel):
    post_only: bool = False
    reduce_only_exits: bool = True
    max_position: int = 1
    min_qty: float = 0.001
    qty_step: float = 0.001
    margin_mode: str = "ISOLATED_MARGIN"
    leverage: float = 3.0
    position_mode: str = "ONE_WAY"
    testnet: bool = True
    poll_interval_seconds: int = 30
    lookback_candles: int = 200
    taker_fee_bps: float = 6.0
    slippage_bps: float = 5.0


class LoggingConfig(BaseModel):
    level: str = "INFO"
    dir: str = "logs"
    json_output: bool = True


class MonitoringConfig(BaseModel):
    enable_metrics: bool = True
    host: str = "0.0.0.0"
    port: int = 9000
    recent_trades: int = 5


class AppConfig(BaseModel):
    symbol: str = "BTCUSDT"
    category: str = "linear"
    mode: str = "paper"  # paper | live
    risk: RiskConfig = RiskConfig()
    strategy: StrategyConfig = StrategyConfig()
    execution: ExecutionConfig = ExecutionConfig()
    logging: LoggingConfig = LoggingConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
