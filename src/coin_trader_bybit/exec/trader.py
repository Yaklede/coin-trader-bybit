from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from ..core.config import AppConfig
from ..data.feed import BybitDataFeed, DataFeed
from ..exchange.bybit import BybitClient
from ..monitoring import MetricsCollector
from ..risk.manager import PositionSnapshot, RiskManager
from ..strategy.scalper import Scalper


def _bybit_interval_from_timeframe(timeframe: str) -> str:
    tf = timeframe.strip().lower()
    mapping = {
        "1m": "1",
        "3m": "3",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "4h": "240",
        "1d": "D",
    }
    if tf not in mapping:
        raise ValueError(f"Unsupported timeframe for Bybit feed: {timeframe}")
    return mapping[tf]


@dataclass
class Trader:
    cfg: AppConfig
    metrics: Optional[MetricsCollector] = None
    feed: Optional[DataFeed] = None
    risk_manager: Optional[RiskManager] = None
    client: Optional[BybitClient] = None

    strategy: Scalper = field(init=False)
    poll_interval: int = field(init=False)
    position: PositionSnapshot = field(init=False)
    last_signal_ts: Optional[pd.Timestamp] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.log = logging.getLogger("trader")
        self.client = self.client or BybitClient(
            testnet=self.cfg.execution.testnet,
            max_live_order_notional_krw=self.cfg.risk.max_live_order_notional_krw,
            usdt_krw_rate=self.cfg.risk.usdt_krw_rate,
        )
        interval = _bybit_interval_from_timeframe(self.cfg.strategy.timeframe_entry)
        self.feed = self.feed or BybitDataFeed(
            self.client,
            symbol=self.cfg.symbol,
            category=self.cfg.category,
            interval=interval,
        )
        self.strategy = Scalper(self.cfg.strategy)
        self.risk_manager = self.risk_manager or RiskManager(self.cfg)
        self.poll_interval = max(int(self.cfg.execution.poll_interval_seconds), 1)
        self.position = PositionSnapshot(qty=0.0, entry_price=0.0, mark_price=0.0)

        if self.metrics is not None:
            self.metrics.set_equity(self.cfg.risk.starting_equity)
            self.metrics.update_position(
                qty=self.position.qty,
                entry_price=self.position.entry_price,
                mark_price=self.position.mark_price,
            )

    def run(self) -> None:
        self.log.info(
            "starting trader loop mode=%s symbol=%s testnet=%s interval=%ss",
            self.cfg.mode,
            self.cfg.symbol,
            self.cfg.execution.testnet,
            self.poll_interval,
        )
        try:
            while True:
                self.step()
                self.risk_manager.tick_cooldown()
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            self.log.info("Received interrupt, shutting down")

    def step(self) -> None:
        self._sync_account_state()
        if self.metrics is not None:
            self.metrics.record_loop(time.time())
        try:
            candles = self.feed.fetch(self.cfg.execution.lookback_candles)
        except Exception as exc:  # pragma: no cover - network failure
            self.log.exception("Failed to fetch candles: %s", exc)
            if self.metrics is not None:
                self.metrics.record_error("fetch")
            return

        if candles.empty:
            self.log.debug("No candle data returned")
            return
        if len(candles) < self.cfg.strategy.ema_slow:
            self.log.debug(
                "Insufficient candles (%s) for EMA slow=%s",
                len(candles),
                self.cfg.strategy.ema_slow,
            )
            return

        try:
            features = self.strategy.compute_features(candles)
        except Exception as exc:
            self.log.exception("Failed to compute features: %s", exc)
            return

        mark_price = float(candles["close"].iloc[-1])
        last_ts = candles.index[-1]
        if self.metrics is not None:
            self.metrics.record_candle(last_ts.timestamp())
        self._update_metrics(mark_price)

        signal = self.strategy.generate_signal(features)
        if signal is None:
            return

        if self.metrics is not None:
            self.metrics.record_signal(
                timestamp=signal.timestamp.timestamp(), side=signal.side
            )
        if self.last_signal_ts is not None and signal.timestamp <= self.last_signal_ts:
            return

        cooldown_active = self.risk_manager.is_cooldown_active()
        if not self.risk_manager.can_open_position(
            existing_position=self.position, cooldown_active=cooldown_active
        ):
            self.log.debug("Risk manager blocked new position")
            return

        try:
            qty = self.risk_manager.position_size(
                entry_price=signal.entry_price, stop_price=signal.stop_price
            )
        except ValueError as exc:
            self.log.warning("Invalid signal sizing: %s", exc)
            return

        limit_qty = self.risk_manager.max_order_qty(signal.entry_price)
        if limit_qty is not None and limit_qty > 0 and qty > limit_qty:
            self.log.debug(
                "Capping qty from %.6f to %.6f due to notional limit",
                qty,
                limit_qty,
            )
            qty = limit_qty

        if qty < self.cfg.execution.min_qty:
            self.log.debug(
                "Calculated qty %.6f below minimum %.6f",
                qty,
                self.cfg.execution.min_qty,
            )
            return

        try:
            order = self.client.place_market_order(
                symbol=self.cfg.symbol,
                side=signal.side,
                qty=qty,
                category=self.cfg.category,
            )
            signed_qty = qty if signal.side == "Buy" else -qty
            self.position = PositionSnapshot(
                qty=signed_qty,
                entry_price=signal.entry_price,
                mark_price=mark_price,
            )
            self.last_signal_ts = signal.timestamp
            self.risk_manager.trigger_cooldown()
            self._update_metrics(mark_price)
            self.log.info(
                "Placed %s order qty=%.6f entry=%.2f stop=%.2f order_id=%s",
                signal.side,
                qty,
                signal.entry_price,
                signal.stop_price,
                order.order_id,
            )
        except Exception as exc:  # pragma: no cover - network failure
            self.log.exception("Order placement failed: %s", exc)
            if self.metrics is not None:
                self.metrics.record_error("order")
            self.risk_manager.trigger_cooldown()

    def _sync_account_state(self) -> None:
        try:
            equity = self.client.get_wallet_equity()
            if equity is not None:
                self.risk_manager.update_equity(equity)
                if self.metrics is not None:
                    self.metrics.set_equity(equity)
        except Exception as exc:  # pragma: no cover - network failure
            self.log.debug("Failed to pull wallet balance: %s", exc)

        try:
            snapshot = self.client.get_linear_position_snapshot(
                symbol=self.cfg.symbol, category=self.cfg.category
            )
            self.position = PositionSnapshot(
                qty=float(snapshot.get("qty", 0.0)),
                entry_price=float(snapshot.get("entry_price", 0.0)),
                mark_price=float(snapshot.get("mark_price", 0.0)),
            )
            self._update_metrics(self.position.mark_price)
        except Exception as exc:  # pragma: no cover - network failure
            self.log.debug("Failed to fetch position snapshot: %s", exc)

    def _update_metrics(self, mark_price: float) -> None:
        if self.metrics is None:
            return
        self.metrics.update_position(
            qty=self.position.qty,
            entry_price=self.position.entry_price,
            mark_price=mark_price,
        )
