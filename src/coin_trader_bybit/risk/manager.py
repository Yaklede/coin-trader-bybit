from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from ..core.config import AppConfig
from .position import size_by_risk


@dataclass
class PositionSnapshot:
    qty: float
    entry_price: float
    mark_price: float


class RiskManager:
    """Handles sizing and basic protective rules for live trading."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._equity = cfg.risk.starting_equity
        self._cooldown_counter = 0
        self._daily_r_total = 0.0
        self._daily_snapshot_date = self._current_date()

    @property
    def equity(self) -> float:
        return self._equity

    def update_equity(self, equity: float) -> None:
        self._maybe_reset_daily()
        if equity > 0:
            self._equity = equity

    def max_order_qty(self, entry_price: float) -> float | None:
        limit_krw = self.cfg.risk.max_live_order_notional_krw
        if limit_krw is None or limit_krw <= 0:
            return None
        if entry_price <= 0:
            return None
        limit_usdt = limit_krw / self.cfg.risk.usdt_krw_rate
        return limit_usdt / entry_price

    def position_size(self, entry_price: float, stop_price: float) -> float:
        stop_distance = entry_price - stop_price
        if stop_distance <= 0:
            raise ValueError("stop_price must be below entry for long trades")
        qty = size_by_risk(
            equity=self._equity,
            risk_pct=self.cfg.risk.max_risk_per_trade_pct,
            stop_distance=stop_distance,
            contract_value=1.0,
        )
        return qty

    def can_open_position(
        self,
        *,
        existing_position: PositionSnapshot,
        cooldown_active: bool,
    ) -> bool:
        self._maybe_reset_daily()
        if self._is_daily_stop_hit():
            return False
        if cooldown_active:
            return False
        if abs(existing_position.qty) >= self.cfg.execution.max_position:
            return False
        return True

    def is_cooldown_active(self) -> bool:
        return self._cooldown_counter > 0

    def tick_cooldown(self) -> None:
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1

    def trigger_cooldown(self) -> None:
        self._cooldown_counter = max(
            self._cooldown_counter, self.cfg.risk.cooldown_minutes
        )

    def record_realized_r(self, realized_r: float) -> None:
        self._maybe_reset_daily()
        self._daily_r_total += realized_r

    def daily_r_total(self) -> float:
        self._maybe_reset_daily()
        return self._daily_r_total

    def daily_stop_active(self) -> bool:
        self._maybe_reset_daily()
        return self._is_daily_stop_hit()

    def _maybe_reset_daily(self) -> None:
        today = self._current_date()
        if self._daily_snapshot_date != today:
            self._daily_snapshot_date = today
            self._daily_r_total = 0.0

    @staticmethod
    def _current_date():
        return datetime.now(timezone.utc).date()

    def _is_daily_stop_hit(self) -> bool:
        daily_stop = self.cfg.risk.daily_stop_r_multiple
        if daily_stop is None:
            return False
        if daily_stop < 0:
            return self._daily_r_total <= daily_stop
        if daily_stop > 0:
            return self._daily_r_total >= daily_stop
        return False
