from datetime import datetime, timezone

from prometheus_client import CollectorRegistry

from coin_trader_bybit.monitoring.metrics import MetricsCollector, TradeRecord


def _registry_value(
    registry: CollectorRegistry, name: str, labels: dict[str, str] | None = None
) -> float:
    labels = labels or {}
    value = registry.get_sample_value(name, labels)
    assert value is not None, f"Metric {name} with labels {labels} not found"
    return value


def test_update_position_updates_open_pnl() -> None:
    registry = CollectorRegistry()
    metrics = MetricsCollector(registry=registry, initial_equity=10_000.0)

    metrics.update_position(qty=2.0, entry_price=100.0, mark_price=105.0)

    assert _registry_value(registry, "coin_trader_open_pnl") == 10.0
    assert _registry_value(registry, "coin_trader_position_qty") == 2.0
    assert _registry_value(registry, "coin_trader_position_notional") == 210.0


def test_record_trade_updates_recent_slots_and_win_rate() -> None:
    registry = CollectorRegistry()
    metrics = MetricsCollector(
        registry=registry, initial_equity=10_000.0, recent_limit=2
    )

    trade = TradeRecord(
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        side="Buy",
        qty=1.5,
        pnl=50.0,
        entry_price=100.0,
        exit_price=105.0,
        r_multiple=1.25,
    )
    metrics.record_trade(trade)

    assert _registry_value(registry, "coin_trader_realized_pnl_total") == 50.0
    assert _registry_value(registry, "coin_trader_win_rate") == 1.0
    assert (
        _registry_value(registry, "coin_trader_recent_trade_pnl", {"slot": "0"}) == 50.0
    )
    assert (
        _registry_value(registry, "coin_trader_recent_trade_side", {"slot": "0"}) == 1.0
    )
    assert (
        _registry_value(registry, "coin_trader_recent_trade_qty", {"slot": "0"}) == 1.5
    )
    assert (
        _registry_value(registry, "coin_trader_recent_trade_r", {"slot": "0"}) == 1.25
    )
    assert (
        _registry_value(registry, "coin_trader_recent_trade_timestamp", {"slot": "0"})
        == trade.timestamp.timestamp()
    )

    second_trade = TradeRecord(
        timestamp=datetime(2024, 1, 2, tzinfo=timezone.utc),
        side="Sell",
        qty=1.0,
        pnl=-25.0,
        entry_price=110.0,
        exit_price=107.5,
        r_multiple=-0.5,
    )
    metrics.record_trade(second_trade)

    # latest slot should now reflect the second trade (slot 0) and prior trade shifted (slot 1)
    assert (
        _registry_value(registry, "coin_trader_recent_trade_pnl", {"slot": "0"})
        == -25.0
    )
    assert (
        _registry_value(registry, "coin_trader_recent_trade_side", {"slot": "0"})
        == -1.0
    )
    assert (
        _registry_value(registry, "coin_trader_recent_trade_pnl", {"slot": "1"}) == 50.0
    )
    assert _registry_value(registry, "coin_trader_win_rate") == 0.5
