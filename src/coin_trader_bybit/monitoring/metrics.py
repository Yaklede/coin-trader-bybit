from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)


@dataclass
class TradeRecord:
    """Snapshot of a completed trade used for metrics exposure."""

    timestamp: datetime
    side: str
    qty: float
    pnl: float
    entry_price: float
    exit_price: float
    r_multiple: float


class MetricsCollector:
    """Aggregation layer exposing trading stats via Prometheus metrics."""

    def __init__(
        self,
        *,
        registry: Optional[CollectorRegistry] = None,
        initial_equity: float = 10_000.0,
        recent_limit: int = 5,
    ) -> None:
        self.registry = registry or CollectorRegistry()
        self.initial_equity = initial_equity
        self.recent_limit = max(recent_limit, 1)

        self.position_qty = Gauge(
            "coin_trader_position_qty",
            "Current position size (positive=long, negative=short)",
            registry=self.registry,
        )
        self.position_entry_price = Gauge(
            "coin_trader_position_entry_price",
            "Average entry price of the current position",
            registry=self.registry,
        )
        self.position_mark_price = Gauge(
            "coin_trader_position_mark_price",
            "Latest mark/last price used for PnL",
            registry=self.registry,
        )
        self.position_notional = Gauge(
            "coin_trader_position_notional",
            "Notional value of the current position (quote currency)",
            registry=self.registry,
        )
        self.open_pnl = Gauge(
            "coin_trader_open_pnl",
            "Unrealised PnL for the current open position",
            registry=self.registry,
        )

        self.account_equity = Gauge(
            "coin_trader_account_equity",
            "Latest reported account equity",
            registry=self.registry,
        )
        self.realized_pnl_total = Gauge(
            "coin_trader_realized_pnl_total",
            "Cumulative realised PnL",
            registry=self.registry,
        )
        self.total_return_pct = Gauge(
            "coin_trader_total_return_pct",
            "Total return since start (%)",
            registry=self.registry,
        )
        self.current_return_pct = Gauge(
            "coin_trader_current_return_pct",
            "Current return including unrealised PnL (%)",
            registry=self.registry,
        )
        self.win_rate = Gauge(
            "coin_trader_win_rate",
            "Win rate across closed trades (0-1)",
            registry=self.registry,
        )

        self.trade_counter = Counter(
            "coin_trader_trades_total",
            "Number of closed trades",
            registry=self.registry,
        )
        self.win_counter = Counter(
            "coin_trader_wins_total",
            "Number of winning trades",
            registry=self.registry,
        )
        self.loss_counter = Counter(
            "coin_trader_losses_total",
            "Number of losing trades",
            registry=self.registry,
        )
        self.trade_pnl_hist = Histogram(
            "coin_trader_trade_pnl",
            "Distribution of trade PnL (quote currency)",
            buckets=(
                -500.0,
                -250.0,
                -100.0,
                -50.0,
                -25.0,
                -10.0,
                0.0,
                10.0,
                25.0,
                50.0,
                100.0,
                250.0,
                500.0,
                1_000.0,
            ),
            registry=self.registry,
        )
        self.trade_r_hist = Histogram(
            "coin_trader_trade_r_multiple",
            "Distribution of trade R multiples",
            buckets=(
                -3.0,
                -2.0,
                -1.5,
                -1.0,
                -0.5,
                -0.25,
                0.0,
                0.25,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                5.0,
                10.0,
            ),
            registry=self.registry,
        )

        self.recent_trade_pnl = Gauge(
            "coin_trader_recent_trade_pnl",
            "Most recent trade PnL (slot 0 = latest)",
            ["slot"],
            registry=self.registry,
        )
        self.recent_trade_side = Gauge(
            "coin_trader_recent_trade_side",
            "Most recent trade side (1=buy, -1=sell)",
            ["slot"],
            registry=self.registry,
        )
        self.recent_trade_qty = Gauge(
            "coin_trader_recent_trade_qty",
            "Most recent trade quantity",
            ["slot"],
            registry=self.registry,
        )
        self.recent_trade_r = Gauge(
            "coin_trader_recent_trade_r",
            "Most recent trade R multiple",
            ["slot"],
            registry=self.registry,
        )
        self.recent_trade_timestamp = Gauge(
            "coin_trader_recent_trade_timestamp",
            "Most recent trade timestamp (unix seconds)",
            ["slot"],
            registry=self.registry,
        )

        self._recent_trades: Deque[TradeRecord] = deque(maxlen=self.recent_limit)
        self._realized_pnl: float = 0.0
        self._open_pnl: float = 0.0
        self._total_trades: int = 0
        self._winning_trades: int = 0
        self._equity: float = initial_equity

        self._prime_recent_trade_slots()

    def update_position(
        self, *, qty: float, entry_price: float, mark_price: float
    ) -> None:
        self.position_qty.set(qty)
        self.position_entry_price.set(entry_price)
        self.position_mark_price.set(mark_price)
        notional = mark_price * qty
        self.position_notional.set(notional)
        self._open_pnl = (mark_price - entry_price) * qty
        self.open_pnl.set(self._open_pnl)
        self._refresh_current_return()

    def set_equity(self, equity: float) -> None:
        self.account_equity.set(equity)
        self._equity = equity
        if self.initial_equity > 0:
            total_return = (
                (equity - self.initial_equity) / self.initial_equity
            ) * 100.0
            self.total_return_pct.set(total_return)
        self._refresh_current_return()

    def set_balance(self, balance: float) -> None:
        self.set_equity(balance)

    def record_trade(self, trade: TradeRecord) -> None:
        self._recent_trades.append(trade)
        self._realized_pnl += trade.pnl
        self.realized_pnl_total.set(self._realized_pnl)

        self.trade_counter.inc()
        self._total_trades += 1

        if trade.pnl > 0:
            self.win_counter.inc()
            self._winning_trades += 1
        elif trade.pnl < 0:
            self.loss_counter.inc()

        self.trade_pnl_hist.observe(trade.pnl)
        self.trade_r_hist.observe(trade.r_multiple)

        win_rate = (
            (self._winning_trades / self._total_trades) if self._total_trades else 0.0
        )
        self.win_rate.set(win_rate)

        self._update_recent_trade_metrics()
        self._refresh_current_return()

    def reset(self) -> None:
        self.position_qty.set(0.0)
        self.position_entry_price.set(0.0)
        self.position_mark_price.set(0.0)
        self.position_notional.set(0.0)
        self.open_pnl.set(0.0)
        self.account_equity.set(self.initial_equity)
        self.realized_pnl_total.set(0.0)
        self.total_return_pct.set(0.0)
        self.current_return_pct.set(0.0)
        self.win_rate.set(0.0)
        self._recent_trades.clear()
        self._realized_pnl = 0.0
        self._open_pnl = 0.0
        self._total_trades = 0
        self._winning_trades = 0
        self._equity = self.initial_equity
        self._prime_recent_trade_slots()

    def _prime_recent_trade_slots(self) -> None:
        for idx in range(self.recent_limit):
            slot = str(idx)
            self.recent_trade_pnl.labels(slot=slot).set(0.0)
            self.recent_trade_side.labels(slot=slot).set(0.0)
            self.recent_trade_qty.labels(slot=slot).set(0.0)
            self.recent_trade_r.labels(slot=slot).set(0.0)
            self.recent_trade_timestamp.labels(slot=slot).set(0.0)

    def _update_recent_trade_metrics(self) -> None:
        recent = list(self._recent_trades)[::-1]
        for idx in range(self.recent_limit):
            slot = str(idx)
            if idx < len(recent):
                trade = recent[idx]
                self.recent_trade_pnl.labels(slot=slot).set(trade.pnl)
                side_value = 1.0 if trade.side.lower() == "buy" else -1.0
                self.recent_trade_side.labels(slot=slot).set(side_value)
                self.recent_trade_qty.labels(slot=slot).set(trade.qty)
                self.recent_trade_r.labels(slot=slot).set(trade.r_multiple)
                self.recent_trade_timestamp.labels(slot=slot).set(
                    trade.timestamp.timestamp()
                )
            else:
                self.recent_trade_pnl.labels(slot=slot).set(0.0)
                self.recent_trade_side.labels(slot=slot).set(0.0)
                self.recent_trade_qty.labels(slot=slot).set(0.0)
                self.recent_trade_r.labels(slot=slot).set(0.0)
                self.recent_trade_timestamp.labels(slot=slot).set(0.0)

    def _refresh_current_return(self) -> None:
        if self.initial_equity <= 0:
            self.current_return_pct.set(0.0)
            return
        total = self._realized_pnl + self._open_pnl
        current_return = (total / self.initial_equity) * 100.0
        self.current_return_pct.set(current_return)


def start_metrics_server(host: str, port: int, registry: CollectorRegistry) -> None:
    """Launch an HTTP server exposing Prometheus metrics."""

    start_http_server(port, addr=host, registry=registry)
