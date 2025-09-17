from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from ..core.config import AppConfig
from ..exchange.bybit import BybitClient
from ..strategy.scalper import Scalper
from ..monitoring import MetricsCollector


@dataclass
class Trader:
    cfg: AppConfig
    metrics: Optional[MetricsCollector] = None

    def __post_init__(self) -> None:
        self.log = logging.getLogger("trader")
        self.client = BybitClient(
            testnet=self.cfg.execution.testnet,
            max_live_order_notional_krw=self.cfg.risk.max_live_order_notional_krw,
            usdt_krw_rate=self.cfg.risk.usdt_krw_rate,
        )
        self.strategy = Scalper(self.cfg.strategy)
        if self.metrics is not None:
            self.metrics.update_position(qty=0.0, entry_price=0.0, mark_price=0.0)

    def run(self) -> None:
        self.log.info(
            "mode=%s symbol=%s testnet=%s",
            self.cfg.mode,
            self.cfg.symbol,
            self.cfg.execution.testnet,
        )
        self.log.info("This is a scaffold run; wire data & execution next.")
        # Placeholder: no live loop to avoid accidental orders.
        signal = self.strategy.maybe_signal()
        if signal:
            self.log.info("Signal: %s (%s)", signal.side, signal.reason)
        else:
            self.log.info("No signal generated in scaffold.")
