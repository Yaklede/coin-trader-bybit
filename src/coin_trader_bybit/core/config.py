from pydantic import BaseModel


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


class ExecutionConfig(BaseModel):
    post_only: bool = False
    reduce_only_exits: bool = True
    max_position: int = 1
    min_qty: float = 0.001
    testnet: bool = True
    poll_interval_seconds: int = 30
    lookback_candles: int = 200


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
