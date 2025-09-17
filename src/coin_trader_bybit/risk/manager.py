from __future__ import annotations

from dataclasses import dataclass
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

    @property
    def equity(self) -> float:
        return self._equity

    def update_equity(self, equity: float) -> None:
        if equity > 0:
            self._equity = equity

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
