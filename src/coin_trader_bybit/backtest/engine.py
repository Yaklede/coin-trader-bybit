from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from ..core.config import AppConfig, RiskConfig
from ..risk.position import size_by_risk
from ..strategy import create_strategy, Signal, timeframe_to_pandas_freq


@dataclass
class BacktestTrade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    entry_reason: str
    entry_price: float
    exit_price: float
    qty: float
    realized_r: float
    pnl: float
    hold_bars: int
    exit_reason: str
    partial_taken: bool


@dataclass
class BacktestMetrics:
    total_pnl: float
    ending_equity: float
    profit_factor: float | None
    win_rate: float | None
    avg_r: float | None
    max_drawdown: float
    cagr: float | None
    num_trades: int


@dataclass
class BacktestReport:
    trades: List[BacktestTrade]
    metrics: BacktestMetrics
    equity_curve: pd.Series


@dataclass
class PositionState:
    entry_time: pd.Timestamp
    side: str
    reason: str
    entry_price: float
    entry_fill_price: float
    stop_price: float
    qty_total: float
    qty_open: float
    risk_per_unit: float
    partial_taken: bool
    realized_pnl: float
    bars_held: int
    fees: float
    partial_tp_r: float
    partial_fraction: float
    partials_taken: int
    partial_tp_r2: float
    partial_fraction2: float
    trail_mult: float
    fast_market: bool
    time_stop_bars: int
    # Pyramiding
    pyramids_taken: int
    pyramid_levels: list[float]
    pyramid_fractions: list[float]


@dataclass
class ExitEvent:
    exit_time: pd.Timestamp
    exit_price: float
    qty_to_close: float
    exit_reason: str
    partial_taken: bool
    is_maker: bool = False


class Backtester:
    """Runs a sequential simulation of the scalper strategy on historical data."""

    def __init__(
        self,
        cfg: AppConfig,
        *,
        initial_equity: float = 10_000.0,
        contract_value: float = 1.0,
        live_like: bool = False,
    ) -> None:
        self.cfg = cfg
        self.initial_equity = initial_equity
        self.contract_value = contract_value
        self.strategy = create_strategy(cfg.strategy)
        self.taker_fee_rate = cfg.execution.taker_fee_bps / 10_000.0
        self.slippage_rate = cfg.execution.slippage_bps / 10_000.0
        self.maker_fee_rate = cfg.execution.maker_fee_bps / 10_000.0
        self.post_only = cfg.execution.post_only
        # When True, recompute features on a sliding window of the last N raw candles,
        # where N = cfg.execution.lookback_candles, to emulate live runtime.
        self.live_like = live_like

    def run(self, data: pd.DataFrame) -> BacktestReport:
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("data index must be a DatetimeIndex")
        if data.empty:
            raise ValueError("data cannot be empty")

        if not self.live_like:
            features = self.strategy.compute_features(data)
            features = features.dropna(subset=["atr", "volume_ma"]).copy()
            if features.empty:
                raise ValueError("insufficient data after applying indicators")
            feature_index = list(features.index)
        else:
            # Build a timeline on the entry timeframe by computing features once
            # (to get resampled timestamps), then drive the loop with those timestamps.
            base_features = self.strategy.compute_features(data)
            base_features = base_features.dropna(subset=["atr", "volume_ma"]).copy()
            if base_features.empty:
                raise ValueError("insufficient data after applying indicators")
            feature_index = list(base_features.index)

        risk_cfg = self.cfg.risk
        daily_limit = risk_cfg.daily_max_trades or 0
        daily_counts: dict[pd.Timestamp, int] = {}
        # Track equity at the start of each UTC day and whether the daily equity stop is hit
        daily_start_equity: dict[pd.Timestamp, float] = {}
        daily_stop_hit: dict[pd.Timestamp, bool] = {}
        daily_gain_hit: dict[pd.Timestamp, bool] = {}

        trades: List[BacktestTrade] = []
        equity = self.initial_equity
        first_ts = feature_index[0]
        equity_points: list[tuple[pd.Timestamp, float]] = [(first_ts, equity)]
        position: Optional[PositionState] = None
        last_open_ts: Optional[pd.Timestamp] = None
        win_streak = 0
        loss_streak = 0

        feature_items: Iterable[tuple[int, pd.Timestamp]] = list(enumerate(feature_index))
        def _risk_multiplier(row: pd.Series, reason: str, side: str) -> float:
            cfgs = self.cfg.strategy
            m = max(cfgs.risk_mult_base, 0.0)
            reason_l = (reason or "").lower()
            if "breakout" in reason_l:
                m *= max(cfgs.risk_mult_breakout, 0.0)
            if "pullback" in reason_l:
                m *= max(cfgs.risk_mult_pullback, 0.0)
            if side == "Buy":
                m *= max(cfgs.risk_mult_long, 0.0)
            else:
                m *= max(cfgs.risk_mult_short, 0.0)
            if bool(row.get("fast_market", False)):
                m *= max(cfgs.risk_mult_fast_market, 0.0)
            atr_pct = float(row.get("atr_pct", 0.0) or 0.0)
            if atr_pct > 0:
                if atr_pct <= cfgs.risk_atr_low:
                    m *= max(cfgs.risk_mult_atr_low, 0.0)
                elif atr_pct >= cfgs.risk_atr_high:
                    m *= max(cfgs.risk_mult_atr_high, 0.0)
            cap = max(cfgs.risk_mult_cap, 0.1)
            return min(max(m, 0.1), cap)

        last_equity_ts: Optional[pd.Timestamp] = equity_points[-1][0] if equity_points else None
        for idx, timestamp in feature_items:
            # Compute the features for this step
            if not self.live_like:
                row = features.iloc[idx]
            else:
                lookback = max(int(self.cfg.execution.lookback_candles), 100)
                # Select last N raw candles up to current timestamp
                window = data.loc[: timestamp].tail(lookback)
                if window.empty:
                    continue
                step_features = self.strategy.compute_features(window)
                step_features = step_features.dropna(subset=["atr", "volume_ma"]).copy()
                if step_features.empty or timestamp not in step_features.index:
                    continue
                row = step_features.loc[timestamp]

            atr = float(row["atr"])
            if np.isnan(atr) or atr <= 0:
                continue
            # Initialise daily trackers on day change
            trade_day = timestamp.normalize()
            if trade_day not in daily_start_equity:
                daily_start_equity[trade_day] = equity
                daily_stop_hit[trade_day] = False
                daily_gain_hit[trade_day] = False

            if position is None:
                if not self.live_like:
                    window = features.iloc[: idx + 1]
                else:
                    window = step_features
                signal = self.strategy.generate_signal(window)
                if signal is not None and signal.timestamp == timestamp:
                    # Cooldown between entries based on strategy.signal_cooldown_minutes
                    cooldown_min = max(self.cfg.strategy.signal_cooldown_minutes, 0)
                    if cooldown_min > 0 and last_open_ts is not None:
                        min_delta = pd.Timedelta(minutes=cooldown_min)
                        if timestamp - last_open_ts < min_delta:
                            continue
                    # Block new entries if daily max trades reached or daily equity stop is hit
                    if daily_limit and daily_counts.get(trade_day, 0) >= daily_limit:
                        pass
                    elif daily_stop_hit.get(trade_day, False) or daily_gain_hit.get(trade_day, False):
                        pass
                    else:
                        mult = _risk_multiplier(row, signal.reason, signal.side)
                        # performance-adaptive risk
                        if win_streak > 0:
                            mult *= 1 + self.cfg.strategy.risk_mult_win_gamma * win_streak
                        if loss_streak > 0:
                            mult *= max(1 - self.cfg.strategy.risk_mult_loss_gamma * loss_streak, self.cfg.strategy.risk_mult_floor)
                        new_position = self._try_open_position(
                            timestamp=timestamp,
                            signal=signal,
                            equity=equity,
                            risk_cfg=risk_cfg,
                            risk_mult=mult,
                        )
                        if new_position is not None:
                            position = new_position
                            daily_counts[trade_day] = daily_counts.get(trade_day, 0) + 1
                            last_open_ts = timestamp
                            continue

            if position is None:
                # track equity at bar close for accurate daily curve
                if last_equity_ts is None or timestamp > last_equity_ts:
                    equity_points.append((timestamp, equity))
                    last_equity_ts = timestamp
                continue

            exit_event = self._manage_position(position, timestamp, row, atr)
            if exit_event is None:
                continue

            trade_result, equity = self._finalise_trade(position, exit_event, equity)
            trades.append(trade_result)
            equity_points.append((trade_result.exit_time, equity))
            last_equity_ts = trade_result.exit_time
            # Update daily equity stop after each trade closes
            day = trade_result.exit_time.normalize()
            start_eq = daily_start_equity.get(day, None)
            if start_eq is None:
                daily_start_equity[day] = equity
            else:
                if start_eq > 0:
                    # Loss limit
                    loss_limit = risk_cfg.daily_loss_limit_pct or 0.0
                    if loss_limit > 0:
                        drawdown = (equity - start_eq) / start_eq
                        if drawdown <= -abs(loss_limit):
                            daily_stop_hit[day] = True
                    # Gain limit
                    gain_limit = risk_cfg.daily_gain_limit_pct or 0.0
                    if gain_limit > 0:
                        gain = (equity - start_eq) / start_eq
                        if gain >= abs(gain_limit):
                            daily_gain_hit[day] = True
            position = None
            # Update streaks
            if trade_result.pnl > 0:
                win_streak += 1
                loss_streak = 0
            elif trade_result.pnl < 0:
                loss_streak += 1
                win_streak = 0

        equity_curve = pd.Series(
            data=[val for _, val in equity_points],
            index=[ts for ts, _ in equity_points],
        )
        metrics = self._compute_metrics(trades, equity_curve)
        return BacktestReport(trades=trades, metrics=metrics, equity_curve=equity_curve)

    def _try_open_position(
        self,
        timestamp: pd.Timestamp,
        signal: Signal,
        equity: float,
        risk_cfg: RiskConfig,
        risk_mult: float = 1.0,
    ) -> Optional[PositionState]:
        entry_price = float(signal.entry_price)
        stop_price = float(signal.stop_price)
        atr = float(signal.atr)
        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0:
            return None
        side = signal.side
        if side == "Buy" and stop_price >= entry_price:
            stop_price = entry_price - stop_distance
        elif side == "Sell" and stop_price <= entry_price:
            stop_price = entry_price + stop_distance

        risk_pct = max(risk_cfg.max_risk_per_trade_pct * max(risk_mult, 0.1), 0.01)
        qty = size_by_risk(
            equity=equity,
            risk_pct=risk_pct,
            stop_distance=stop_distance,
            contract_value=self.contract_value,
        )
        if qty <= 0:
            return None

        reason = signal.reason.lower()
        partial_tp_r = max(self.cfg.strategy.partial_tp_r, 0.0)
        partial_fraction = max(min(self.cfg.strategy.partial_take_fraction, 1.0), 0.0)
        partial_tp_r2 = 0.0
        partial_fraction2 = 0.0
        trail_mult = max(self.cfg.strategy.trail_atr_mult, 0.0)
        time_stop_minutes = max(self.cfg.strategy.time_stop_minutes, 1)

        if "range" in reason:
            partial_tp_r = max(self.cfg.strategy.range_partial_tp_r, 0.0)
            partial_fraction = max(min(self.cfg.strategy.range_partial_fraction, 1.0), 0.0)
            trail_mult = max(self.cfg.strategy.range_trail_atr_mult, 0.0)
            time_stop_minutes = max(self.cfg.strategy.range_time_stop_minutes, 1)
        elif "long" in reason and "swing" not in reason:
            partial_tp_r = max(self.cfg.strategy.long_partial_tp_r, 0.0)
            partial_fraction = max(min(self.cfg.strategy.long_partial_fraction, 1.0), 0.0)
            trail_mult = max(self.cfg.strategy.long_trail_atr_mult, 0.0)
            time_stop_minutes = max(self.cfg.strategy.long_time_stop_minutes, 1)
        elif "short" in reason:
            partial_tp_r = max(self.cfg.strategy.short_partial_tp_r, 0.0)
            partial_fraction = max(min(self.cfg.strategy.short_partial_fraction, 1.0), 0.0)
            trail_mult = max(self.cfg.strategy.short_trail_atr_mult, 0.0)
            time_stop_minutes = max(self.cfg.strategy.short_time_stop_minutes, 1)
        if "swing" in reason:
            partial_tp_r = max(self.cfg.strategy.swing_partial_tp_r1, 0.0)
            partial_fraction = max(min(self.cfg.strategy.swing_partial_fraction1, 1.0), 0.0)
            partial_tp_r2 = max(self.cfg.strategy.swing_partial_tp_r2, 0.0)
            partial_fraction2 = max(min(self.cfg.strategy.swing_partial_fraction2, 1.0), 0.0)
            trail_mult = max(self.cfg.strategy.swing_trail_atr_mult, trail_mult)

        fast_market = bool(getattr(signal, "fast_market", False))
        if fast_market:
            partial_tp_r = max(self.cfg.strategy.fast_market_tp_r, 0.0)
            partial_fraction = max(min(self.cfg.strategy.fast_market_partial_fraction, 1.0), 0.0)
            trail_mult = max(self.cfg.strategy.fast_market_trail_atr_mult, 0.0)
            time_stop_minutes = max(self.cfg.strategy.fast_market_time_stop_minutes, 1)
        time_stop_bars = self._time_stop_bars(time_stop_minutes)

        if side == "Buy":
            entry_fill_price = entry_price * (1 + self.slippage_rate)
        else:
            entry_fill_price = entry_price * (1 - self.slippage_rate)
        entry_fee = entry_fill_price * qty * self.taker_fee_rate

        pyramids_taken = 0
        pyr_levels = []
        pyr_fracs = []
        if getattr(self.cfg.strategy, "enable_pyramiding", False):
            levels = list(getattr(self.cfg.strategy, "pyramid_r_levels", []) or [])
            fracs = list(getattr(self.cfg.strategy, "pyramid_add_fractions", []) or [])
            if len(fracs) < len(levels):
                fracs = fracs + [0.0] * (len(levels) - len(fracs))
            pyr_levels = [max(float(x), 0.0) for x in levels]
            pyr_fracs = [max(min(float(x), 1.0), 0.0) for x in fracs]

        return PositionState(
            entry_time=timestamp,
            side=side,
            reason=signal.reason,
            entry_price=entry_price,
            entry_fill_price=entry_fill_price,
            stop_price=stop_price,
            qty_total=qty,
            qty_open=qty,
            risk_per_unit=stop_distance,
            partial_taken=False,
            realized_pnl=0.0,
            bars_held=0,
            fees=entry_fee,
            partial_tp_r=partial_tp_r,
            partial_fraction=partial_fraction,
            partials_taken=0,
            partial_tp_r2=partial_tp_r2,
            partial_fraction2=partial_fraction2,
            trail_mult=trail_mult,
            fast_market=fast_market,
            time_stop_bars=time_stop_bars,
            pyramids_taken=pyramids_taken,
            pyramid_levels=pyr_levels,
            pyramid_fractions=pyr_fracs,
        )

    def _manage_position(
        self,
        position: PositionState,
        timestamp: pd.Timestamp,
        row: pd.Series,
        atr: float,
    ) -> Optional[ExitEvent]:
        position.bars_held += 1
        stop_price = position.stop_price
        entry_price = position.entry_price
        qty_total = position.qty_total
        qty_open = position.qty_open
        partial_taken = position.partial_taken
        risk_per_unit = position.risk_per_unit
        partial_fraction = max(min(position.partial_fraction, 1.0), 0.0)
        partial_tp_r = position.partial_tp_r
        partials_taken = position.partials_taken
        partial_tp_r2 = position.partial_tp_r2
        partial_fraction2 = max(min(position.partial_fraction2, 1.0), 0.0)

        low = float(row["low"])
        high = float(row["high"])
        close = float(row["close"])

        if position.side == "Buy":
            stop_hit = low <= stop_price
        else:
            stop_hit = high >= stop_price

        if stop_hit:
            exit_price = max(stop_price, 0.0)
            return ExitEvent(
                exit_time=timestamp,
                exit_price=exit_price,
                qty_to_close=qty_open,
                exit_reason="stop",
                partial_taken=partial_taken,
            )

        # Pyramiding: add as price advances by configured R levels
        if position.pyramid_levels and qty_open > 0:
            next_idx = position.pyramids_taken
            if next_idx < len(position.pyramid_levels):
                level_r = position.pyramid_levels[next_idx]
                add_frac = max(min(position.pyramid_fractions[next_idx], 1.0), 0.0)
                if add_frac > 0:
                    if position.side == "Buy":
                        trigger = entry_price + risk_per_unit * level_r
                        hit = high >= trigger
                        add_fill = trigger if self.post_only else trigger * (1 - self.slippage_rate)
                    else:
                        trigger = entry_price - risk_per_unit * level_r
                        hit = low <= trigger
                        add_fill = trigger if self.post_only else trigger * (1 + self.slippage_rate)
                    if hit:
                        add_qty = min(qty_total * add_frac, max(qty_total * 2, qty_open * 5))
                        if add_qty > 0:
                            fee_rate = self.maker_fee_rate if self.post_only else self.taker_fee_rate
                            position.fees += add_fill * add_qty * fee_rate
                            new_total = qty_total + add_qty
                            if new_total > 0:
                                position.entry_fill_price = (
                                    position.entry_fill_price * qty_total + add_fill * add_qty
                                ) / new_total
                            position.qty_total = new_total
                            position.qty_open = qty_open + add_qty
                            position.pyramids_taken = next_idx + 1
                            qty_total = position.qty_total
                            qty_open = position.qty_open
                            # Optional: move stop to breakeven at first add
                            if next_idx == 0 and getattr(self.cfg.strategy, "pyramid_move_stop_to_breakeven", True):
                                buffer = max(self.cfg.strategy.breakeven_buffer_r, 0.0)
                                if position.side == "Buy":
                                    breakeven_price = entry_price - buffer * risk_per_unit
                                    stop_price = max(stop_price, breakeven_price)
                                else:
                                    breakeven_price = entry_price + buffer * risk_per_unit
                                    stop_price = min(stop_price, breakeven_price)
                                position.stop_price = stop_price

        # Partial 1
        if partials_taken == 0 and partial_tp_r > 0:
            if position.side == "Buy":
                partial_target = entry_price + risk_per_unit * partial_tp_r
                hit_partial = high >= partial_target
                if self.post_only:
                    partial_fill_price = partial_target
                else:
                    partial_fill_price = partial_target * (1 - self.slippage_rate)
            else:
                partial_target = entry_price - risk_per_unit * partial_tp_r
                hit_partial = low <= partial_target
                if self.post_only:
                    partial_fill_price = partial_target
                else:
                    partial_fill_price = partial_target * (1 + self.slippage_rate)

            if hit_partial:
                if partial_fraction > 0.0:
                    qty_to_close = min(qty_open, qty_total * partial_fraction)
                else:
                    qty_to_close = 0.0

                if qty_to_close > 0:
                    direction = 1.0 if position.side == "Buy" else -1.0
                    pnl_partial = (
                        (partial_fill_price - position.entry_fill_price) * direction
                    ) * qty_to_close
                    position.realized_pnl += pnl_partial
                    position.qty_open = max(qty_open - qty_to_close, 0.0)
                    position.partial_taken = True
                    position.partials_taken = 1
                    qty_open = position.qty_open
                    buffer = max(self.cfg.strategy.breakeven_buffer_r, 0.0)
                    if position.side == "Buy":
                        breakeven_price = entry_price - buffer * risk_per_unit
                        stop_price = max(stop_price, breakeven_price)
                    else:
                        breakeven_price = entry_price + buffer * risk_per_unit
                        stop_price = min(stop_price, breakeven_price)
                    position.stop_price = stop_price
                    partial_taken = True
                    fee_rate = self.maker_fee_rate if self.post_only else self.taker_fee_rate
                    position.fees += partial_fill_price * qty_to_close * fee_rate
                    if qty_open <= 1e-9:
                        return ExitEvent(
                            exit_time=timestamp,
                            exit_price=partial_target,
                            qty_to_close=0.0,
                            exit_reason="partial",
                            partial_taken=True,
                            is_maker=self.post_only,
                        )

        # Partial 2
        if position.partials_taken == 1 and partial_tp_r2 > 0 and qty_open > 0:
            if position.side == "Buy":
                target2 = entry_price + risk_per_unit * partial_tp_r2
                hit2 = high >= target2
                fill2 = target2 if self.post_only else target2 * (1 - self.slippage_rate)
            else:
                target2 = entry_price - risk_per_unit * partial_tp_r2
                hit2 = low <= target2
                fill2 = target2 if self.post_only else target2 * (1 + self.slippage_rate)
            if hit2:
                qty_to_close = min(qty_open, qty_total * partial_fraction2) if partial_fraction2 > 0 else 0.0
                if qty_to_close > 0:
                    direction = 1.0 if position.side == "Buy" else -1.0
                    pnl_partial = ((fill2 - position.entry_fill_price) * direction) * qty_to_close
                    position.realized_pnl += pnl_partial
                    position.qty_open = max(qty_open - qty_to_close, 0.0)
                    position.partials_taken = 2
                    qty_open = position.qty_open
                    fee_rate = self.maker_fee_rate if self.post_only else self.taker_fee_rate
                    position.fees += fill2 * qty_to_close * fee_rate
                    if qty_open <= 1e-9:
                        return ExitEvent(
                            exit_time=timestamp,
                            exit_price=target2,
                            qty_to_close=0.0,
                            exit_reason="partial2",
                            partial_taken=True,
                            is_maker=self.post_only,
                        )

        trail_mult = max(position.trail_mult, 0.0)
        if partial_taken and qty_open > 0 and trail_mult > 0:
            if position.side == "Buy":
                trail = high - atr * trail_mult
                if trail > stop_price:
                    stop_price = trail
                    position.stop_price = stop_price
                trail_hit = low <= stop_price
            else:
                trail = low + atr * trail_mult
                if trail < stop_price:
                    stop_price = trail
                    position.stop_price = stop_price
                trail_hit = high >= stop_price

            if trail_hit:
                return ExitEvent(
                    exit_time=timestamp,
                    exit_price=stop_price,
                    qty_to_close=qty_open,
                    exit_reason="trail",
                    partial_taken=True,
                )

        soft_stop_r = self.cfg.strategy.soft_stop_r or 0.0
        if soft_stop_r > 0 and qty_open > 0:
            if position.side == "Buy":
                soft_stop_price = entry_price - risk_per_unit * soft_stop_r
                soft_stop_price = max(soft_stop_price, 0.0)
                if low <= soft_stop_price:
                    return ExitEvent(
                        exit_time=timestamp,
                        exit_price=soft_stop_price,
                        qty_to_close=qty_open,
                        exit_reason="soft_stop",
                        partial_taken=partial_taken,
                    )
            else:
                soft_stop_price = entry_price + risk_per_unit * soft_stop_r
                soft_stop_price = max(soft_stop_price, 0.0)
                if high >= soft_stop_price:
                    return ExitEvent(
                        exit_time=timestamp,
                        exit_price=soft_stop_price,
                        qty_to_close=qty_open,
                        exit_reason="soft_stop",
                        partial_taken=partial_taken,
                    )

        if position.bars_held >= position.time_stop_bars:
            exit_price = entry_price if self.post_only else close
            return ExitEvent(
                exit_time=timestamp,
                exit_price=exit_price,
                qty_to_close=qty_open,
                exit_reason="time",
                partial_taken=partial_taken,
                is_maker=self.post_only,
            )

        return None

    def _finalise_trade(
        self,
        position: PositionState,
        exit_event: ExitEvent,
        equity: float,
    ) -> tuple[BacktestTrade, float]:
        entry_fill_price = position.entry_fill_price
        qty_total = position.qty_total
        qty_open = position.qty_open
        realized_pnl = position.realized_pnl
        risk_per_unit = position.risk_per_unit

        exit_price = exit_event.exit_price
        qty_to_close = exit_event.qty_to_close
        partial_taken = exit_event.partial_taken
        exit_reason = exit_event.exit_reason
        exit_time = exit_event.exit_time

        if exit_event.is_maker:
            exit_fill_price = exit_price
            fee_rate = self.maker_fee_rate
        else:
            if position.side == "Buy":
                exit_fill_price = exit_price * (1 - self.slippage_rate)
            else:
                exit_fill_price = exit_price * (1 + self.slippage_rate)
            fee_rate = self.taker_fee_rate
        direction = 1.0 if position.side == "Buy" else -1.0
        pnl_remaining = (exit_fill_price - entry_fill_price) * qty_to_close * direction
        position.fees += exit_fill_price * qty_to_close * fee_rate
        total_pnl = realized_pnl + pnl_remaining - position.fees
        qty_closed = qty_total - qty_open + qty_to_close
        pnl_r = 0.0
        if risk_per_unit > 0 and qty_total > 0:
            pnl_r = total_pnl / (risk_per_unit * qty_total)

        trade = BacktestTrade(
            entry_time=position.entry_time,
            exit_time=exit_time,
            side=position.side,
            entry_reason=position.reason,
            entry_price=entry_fill_price,
            exit_price=exit_fill_price,
            qty=qty_closed,
            realized_r=pnl_r,
            pnl=total_pnl,
            hold_bars=position.bars_held,
            exit_reason=exit_reason,
            partial_taken=partial_taken,
        )

        new_equity = equity + total_pnl
        return trade, new_equity

    def _compute_metrics(
        self, trades: Iterable[BacktestTrade], equity_curve: pd.Series
    ) -> BacktestMetrics:
        trades = list(trades)
        total_pnl = sum(trade.pnl for trade in trades)
        ending_equity = (
            float(equity_curve.iloc[-1])
            if not equity_curve.empty
            else self.initial_equity
        )

        wins = [trade.pnl for trade in trades if trade.pnl > 0]
        losses = [trade.pnl for trade in trades if trade.pnl < 0]
        profit_factor = None
        if losses:
            denom = sum(losses)
            if denom != 0:
                profit_factor = abs(sum(wins) / denom)
        elif wins:
            profit_factor = float("inf")

        win_rate = len(wins) / len(trades) if trades else None
        avg_r = (
            sum(trade.realized_r for trade in trades) / len(trades) if trades else None
        )

        max_drawdown = (
            self._max_drawdown(equity_curve) if not equity_curve.empty else 0.0
        )
        cagr = (
            self._cagr(equity_curve.index, ending_equity)
            if not equity_curve.empty
            else None
        )

        return BacktestMetrics(
            total_pnl=total_pnl,
            ending_equity=ending_equity,
            profit_factor=profit_factor,
            win_rate=win_rate,
            avg_r=avg_r,
            max_drawdown=max_drawdown,
            cagr=cagr,
            num_trades=len(trades),
        )

    def _time_stop_bars(self, time_stop_minutes: int) -> int:
        entry_freq = timeframe_to_pandas_freq(self.cfg.strategy.timeframe_entry)
        delta = pd.Timedelta(entry_freq)
        if delta.total_seconds() <= 0:
            raise ValueError("entry timeframe must map to a positive duration")
        bars = max(int(np.ceil(time_stop_minutes * 60 / delta.total_seconds())), 1)
        return bars

    def _max_drawdown(self, equity_curve: pd.Series) -> float:
        peaks = equity_curve.cummax()
        drawdowns = (equity_curve - peaks) / peaks
        return float(drawdowns.min()) if not drawdowns.empty else 0.0

    def _cagr(
        self, index: Iterable[pd.Timestamp], ending_equity: float
    ) -> float | None:
        dates = list(index)
        if not dates:
            return None
        start = dates[0]
        end = dates[-1]
        days = max((end - start).days, 1)
        years = days / 365.25
        if years <= 0:
            return None
        return (ending_equity / self.initial_equity) ** (1 / years) - 1


__all__ = ["Backtester", "BacktestReport", "BacktestMetrics", "BacktestTrade"]
