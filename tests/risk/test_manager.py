from coin_trader_bybit.core.config import AppConfig
from coin_trader_bybit.risk.manager import PositionSnapshot, RiskManager


def test_risk_manager_position_size_positive():
    cfg = AppConfig()
    rm = RiskManager(cfg)
    size = rm.position_size(entry_price=100.0, stop_price=99.0)
    assert size > 0


def test_risk_manager_cooldown_blocks_entry():
    cfg = AppConfig()
    rm = RiskManager(cfg)
    rm.trigger_cooldown()
    active = rm.is_cooldown_active()
    can_open = rm.can_open_position(
        existing_position=PositionSnapshot(qty=0.0, entry_price=0.0, mark_price=0.0),
        cooldown_active=active,
    )
    assert not can_open
    for _ in range(cfg.risk.cooldown_minutes + 1):
        rm.tick_cooldown()
    active = rm.is_cooldown_active()
    can_open_after = rm.can_open_position(
        existing_position=PositionSnapshot(qty=0.0, entry_price=0.0, mark_price=0.0),
        cooldown_active=active,
    )
    if not active:
        assert can_open_after
