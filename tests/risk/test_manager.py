from datetime import timedelta

from coin_trader_bybit.core.config import AppConfig
from coin_trader_bybit.risk.manager import PositionSnapshot, RiskManager


def test_risk_manager_position_size_positive():
    cfg = AppConfig()
    rm = RiskManager(cfg)
    size = rm.position_size(entry_price=100.0, stop_price=99.0, side="Buy")
    assert size > 0


def test_risk_manager_position_size_supports_shorts():
    cfg = AppConfig()
    rm = RiskManager(cfg)
    size = rm.position_size(entry_price=100.0, stop_price=103.0, side="Sell")
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


def test_daily_stop_blocks_entries() -> None:
    cfg = AppConfig()
    cfg.risk.daily_stop_r_multiple = -2.0
    rm = RiskManager(cfg)
    snapshot = PositionSnapshot(qty=0.0, entry_price=0.0, mark_price=0.0)

    assert rm.can_open_position(existing_position=snapshot, cooldown_active=False)

    rm.record_realized_r(-2.5)

    assert rm.daily_stop_active() is True
    assert not rm.can_open_position(existing_position=snapshot, cooldown_active=False)


def test_daily_stop_resets_each_day() -> None:
    cfg = AppConfig()
    cfg.risk.daily_stop_r_multiple = -1.0
    rm = RiskManager(cfg)

    rm.record_realized_r(-1.5)
    assert rm.daily_stop_active() is True

    rm._daily_snapshot_date = rm._daily_snapshot_date - timedelta(days=1)

    assert rm.daily_stop_active() is False
    assert rm.daily_r_total() == 0.0


def test_daily_trade_limit_blocks_entries() -> None:
    cfg = AppConfig()
    cfg.risk.daily_max_trades = 2
    rm = RiskManager(cfg)
    snapshot = PositionSnapshot(qty=0.0, entry_price=0.0, mark_price=0.0)

    assert rm.can_open_position(existing_position=snapshot, cooldown_active=False)
    rm.record_trade_open()
    assert rm.can_open_position(existing_position=snapshot, cooldown_active=False)
    rm.record_trade_open()
    assert not rm.can_open_position(existing_position=snapshot, cooldown_active=False)

    rm._daily_snapshot_date = rm._daily_snapshot_date - timedelta(days=1)
    assert rm.can_open_position(existing_position=snapshot, cooldown_active=False)
