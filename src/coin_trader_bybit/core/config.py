from typing import List

from pydantic import BaseModel, Field


class RiskConfig(BaseModel):
    max_risk_per_trade_pct: float = 0.2
    daily_stop_r_multiple: float = -1.5
    cooldown_minutes: int = 3
    max_live_order_notional_krw: float = 50_000.0
    usdt_krw_rate: float = 1_350.0
    starting_equity: float = 10_000.0
    daily_max_trades: int | None = 100
    daily_loss_limit_pct: float | None = None
    # Optional: stop opening new trades for the rest of the UTC day
    # once the equity gain from the day's start reaches this percentage.
    daily_gain_limit_pct: float | None = None


class StrategyConfig(BaseModel):
    name: str = "scalper_v1"
    timeframe_entry: str = "1m"
    timeframe_anchor: str = "5m"
    ema_fast: int = 50
    ema_slow: int = 200
    atr_period: int = 14
    atr_mult_stop: float = 1.5
    partial_tp_r: float = 1.0
    partial_take_fraction: float = 0.5
    trail_atr_mult: float = 1.5
    soft_stop_r: float | None = None
    breakeven_buffer_r: float = 0.0
    time_stop_minutes: int = 15
    volume_ma_period: int = 20
    volume_threshold_ratio: float = 1.2
    use_trend_filter: bool = True
    use_volume_filter: bool = True
    entry_buffer_pct: float = 0.0
    stop_loss_pct: float | None = None
    allow_counter_trend_shorts: bool = True
    volume_timeframes: List[str] = Field(
        default_factory=lambda: ["1m", "5m", "15m", "1h", "4h"]
    )
    volume_tf_mode: str = "any"
    trend_slope_threshold: float = 0.0015
    min_atr_pct: float = 0.0008
    support_resistance_lookback: int = 60
    pattern_ticks: int = 3
    bounce_tolerance_pct: float = 0.001
    min_stop_distance_pct: float = 0.005
    support_stop_weight: float = 1.0
    max_trend_strength: float | None = 0.04
    rsi_period: int = 14
    rsi_buy_threshold: float = 35.0
    rsi_sell_threshold: float = 65.0
    ema_pullback_pct: float = 0.002
    regime_trend_min: float = 0.002
    regime_atr_min_pct: float = 0.0006
    regime_atr_max_pct: float = 0.0018
    impulse_volume_ratio: float = 1.8
    fast_market_volume_ratio: float = 2.5
    fast_market_atr_pct: float = 0.0018
    fast_market_tp_r: float = 0.3
    fast_market_partial_fraction: float = 1.0
    fast_market_time_stop_minutes: int = 45
    fast_market_trail_atr_mult: float = 1.0
    long_partial_tp_r: float = 0.8
    long_partial_fraction: float = 0.4
    long_trail_atr_mult: float = 2.5
    long_time_stop_minutes: int = 180
    short_partial_tp_r: float = 0.6
    short_partial_fraction: float = 0.4
    short_trail_atr_mult: float = 2.2
    short_time_stop_minutes: int = 150
    range_lookback: int = 120
    range_band_pct: float = 0.0025
    range_partial_tp_r: float = 0.4
    range_partial_fraction: float = 0.6
    range_trail_atr_mult: float = 1.5
    range_time_stop_minutes: int = 60
    trend_confirm_lookback: int = 30
    trend_slope_min_pct: float = 0.0005
    # Minimum anchor (higher timeframe) EMA slow slope required to allow trend entries.
    # Kept permissive by default to avoid breaking existing behavior; tune in params.yaml.
    anchor_trend_slope_min_pct: float = -1.0
    # Additional gating and microstructure controls
    avoid_fast_market_entries: bool = False
    enable_range_trades: bool = True
    impulse_min_persist: int = 1  # Require N recent bars meeting impulse volume
    impulse_window_bars: int = 5  # Lookback window to detect any recent impulse
    ema_pullback_atr_mult: float = 0.0  # ATR-normalized distance to fast EMA
    signal_cooldown_minutes: int = 0  # Minimum minutes between new entries
    allowed_hours: List[List[int]] = Field(default_factory=list)  # [[start,end], ...] in UTC
    # Breakout entries
    enable_breakout_entries: bool = False
    micro_high_lookback: int = 60
    micro_low_lookback: int = 60
    breakout_band_pct: float = 0.0005
    # Daily breakout (prev-day H/L) option for swing
    enable_daily_breakout: bool = False
    daily_breakout_band_pct: float = 0.0010
    # Pullback entry toggle
    enable_pullback_entries: bool = True
    # ADX filter
    adx_period: int = 14
    anchor_adx_min: float | None = None
    # Pre-breakout contraction filter: require ATR% percentile below this threshold (0-1)
    prebreakout_atr_pct_max_pctile: float | None = None
    # Daily NR7 filter
    require_nr7: bool = False
    # Swing multi-target settings (opt-in)
    swing_partial_tp_r1: float = 0.8
    swing_partial_fraction1: float = 0.33
    swing_partial_tp_r2: float = 1.5
    swing_partial_fraction2: float = 0.33
    swing_trail_atr_mult: float = 2.5
    # Pyramiding
    enable_pyramiding: bool = False
    pyramid_r_levels: list[float] = Field(default_factory=lambda: [0.8, 1.6])
    pyramid_add_fractions: list[float] = Field(default_factory=lambda: [0.5, 0.5])
    pyramid_move_stop_to_breakeven: bool = True
    # Dynamic risk multiplier (affects backtester sizing and live)
    risk_mult_base: float = 1.0
    risk_mult_breakout: float = 1.2
    risk_mult_pullback: float = 1.0
    risk_mult_long: float = 1.0
    risk_mult_short: float = 0.9
    risk_mult_fast_market: float = 0.8
    risk_atr_low: float = 0.0006
    risk_atr_high: float = 0.0020
    risk_mult_atr_low: float = 1.1
    risk_mult_atr_high: float = 0.85
    risk_mult_cap: float = 2.0
    # Performance-adaptive risk
    risk_mult_win_gamma: float = 0.0
    risk_mult_loss_gamma: float = 0.0
    risk_mult_floor: float = 0.5
    # Dynamic leverage schedule (live only; backtest PnL unaffected)
    allow_dynamic_leverage: bool = True
    leverage_base: float = 2.0
    leverage_uptrend: float = 3.0
    leverage_downtrend: float = 2.0
    leverage_fast_market: float = 1.0
    leverage_cap: float = 5.0


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
    maker_fee_bps: float = 0.0


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
