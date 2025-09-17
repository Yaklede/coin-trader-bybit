from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from ..core.config import AppConfig, RiskConfig
from ..risk.position import size_by_risk
from ..strategy.scalper import Scalper, timeframe_to_pandas_freq


@dataclass
class BacktestTrade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
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


@dataclass
class ExitEvent:
    exit_time: pd.Timestamp
    exit_price: float
    qty_to_close: float
    exit_reason: str
    partial_taken: bool


class Backtester:
    """Runs a sequential simulation of the scalper strategy on historical data."""

    def __init__(
        self,
        cfg: AppConfig,
        *,
        initial_equity: float = 10_000.0,
        contract_value: float = 1.0,
    ) -> None:
        self.cfg = cfg
        self.initial_equity = initial_equity
        self.contract_value = contract_value
        self.strategy = Scalper(cfg.strategy)
        self.taker_fee_rate = cfg.execution.taker_fee_bps / 10_000.0
        self.slippage_rate = cfg.execution.slippage_bps / 10_000.0

    def run(self, data: pd.DataFrame) -> BacktestReport:
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("data index must be a DatetimeIndex")
        if data.empty:
            raise ValueError("data cannot be empty")

        features = self.strategy.compute_features(data)
        features = features.dropna(subset=["atr", "micro_high", "volume_ma"]).copy()
        if features.empty:
            raise ValueError("insufficient data after applying indicators")

        risk_cfg = self.cfg.risk
        time_stop_bars = self._time_stop_bars(self.cfg.strategy.time_stop_minutes)

        trades: List[BacktestTrade] = []
        equity = self.initial_equity
        equity_points: list[tuple[pd.Timestamp, float]] = [(features.index[0], equity)]
        position: Optional[PositionState] = None

        for timestamp, row in features.iterrows():
            atr = float(row["atr"])
            if np.isnan(atr) or atr <= 0:
                continue

            if (
                position is None
                and bool(row["long_breakout"])
                and bool(row.get("volume_ok", False))
            ):
                position = self._try_open_long(timestamp, row, atr, equity, risk_cfg)
                continue

            if position is None:
                continue

            exit_event = self._manage_position(
                position, timestamp, row, atr, time_stop_bars
            )
            if exit_event is None:
                continue

            trade_result, equity = self._finalise_trade(position, exit_event, equity)
            trades.append(trade_result)
            equity_points.append((trade_result.exit_time, equity))
            position = None

        equity_curve = pd.Series(
            data=[val for _, val in equity_points],
            index=[ts for ts, _ in equity_points],
        )
        metrics = self._compute_metrics(trades, equity_curve)
        return BacktestReport(trades=trades, metrics=metrics, equity_curve=equity_curve)

    def _try_open_long(
        self,
        timestamp: pd.Timestamp,
        row: pd.Series,
        atr: float,
        equity: float,
        risk_cfg: RiskConfig,
    ) -> Optional[PositionState]:
        entry_price = float(row["close"])
        stop_distance = atr * self.cfg.strategy.atr_mult_stop
        if stop_distance <= 0:
            return None
        stop_price = entry_price - stop_distance
        if stop_price <= 0:
            return None

        qty = size_by_risk(
            equity=equity,
            risk_pct=risk_cfg.max_risk_per_trade_pct,
            stop_distance=stop_distance,
            contract_value=self.contract_value,
        )
        if qty <= 0:
            return None

        entry_fill_price = entry_price * (1 + self.slippage_rate)
        entry_fee = entry_fill_price * qty * self.taker_fee_rate

        return PositionState(
            entry_time=timestamp,
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
        )

    def _manage_position(
        self,
        position: PositionState,
        timestamp: pd.Timestamp,
        row: pd.Series,
        atr: float,
        time_stop_bars: int,
    ) -> Optional[ExitEvent]:
        position.bars_held += 1
        stop_price = position.stop_price
        entry_price = position.entry_price
        qty_total = position.qty_total
        qty_open = position.qty_open
        partial_taken = position.partial_taken
        risk_per_unit = position.risk_per_unit

        low = float(row["low"])
        high = float(row["high"])
        close = float(row["close"])

        if low <= stop_price:
            exit_price = max(stop_price, 0.0)
            return ExitEvent(
                exit_time=timestamp,
                exit_price=exit_price,
                qty_to_close=qty_open,
                exit_reason="stop",
                partial_taken=partial_taken,
            )

        if not partial_taken:
            partial_target = (
                entry_price + risk_per_unit * self.cfg.strategy.partial_tp_r
            )
            if high >= partial_target:
                qty_half = qty_total * 0.5
                partial_fill_price = partial_target * (1 - self.slippage_rate)
                pnl_partial = (
                    partial_fill_price - position.entry_fill_price
                ) * qty_half
                position.realized_pnl += pnl_partial
                position.qty_open = qty_open - qty_half
                position.partial_taken = True
                qty_open = position.qty_open
                stop_price = max(stop_price, entry_price)
                position.stop_price = stop_price
                partial_taken = True
                position.fees += partial_fill_price * qty_half * self.taker_fee_rate
                if qty_open <= 0:
                    return ExitEvent(
                        exit_time=timestamp,
                        exit_price=partial_target,
                        qty_to_close=0.0,
                        exit_reason="partial",
                        partial_taken=True,
                    )

        if partial_taken and qty_open > 0:
            trail = high - atr * self.cfg.strategy.trail_atr_mult
            if trail > stop_price:
                stop_price = trail
                position.stop_price = stop_price

        if partial_taken and qty_open > 0 and low <= stop_price:
            return ExitEvent(
                exit_time=timestamp,
                exit_price=stop_price,
                qty_to_close=qty_open,
                exit_reason="trail",
                partial_taken=True,
            )

        if position.bars_held >= time_stop_bars:
            return ExitEvent(
                exit_time=timestamp,
                exit_price=close,
                qty_to_close=qty_open,
                exit_reason="time",
                partial_taken=partial_taken,
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

        exit_fill_price = exit_price * (1 - self.slippage_rate)
        pnl_remaining = (exit_fill_price - entry_fill_price) * qty_to_close
        position.fees += exit_fill_price * qty_to_close * self.taker_fee_rate
        total_pnl = realized_pnl + pnl_remaining - position.fees
        qty_closed = qty_total - qty_open + qty_to_close
        pnl_r = 0.0
        if risk_per_unit > 0 and qty_total > 0:
            pnl_r = total_pnl / (risk_per_unit * qty_total)

        trade = BacktestTrade(
            entry_time=position.entry_time,
            exit_time=exit_time,
            side="Buy",
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
