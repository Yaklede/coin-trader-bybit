from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from prometheus_client import CollectorRegistry

from coin_trader_bybit.core.config import AppConfig
from coin_trader_bybit.data.feed import KlineRecord, MemoryDataFeed
from coin_trader_bybit.exec.trader import Trader
from coin_trader_bybit.exchange.bybit import OrderResult
from coin_trader_bybit.risk.manager import RiskManager
from coin_trader_bybit.monitoring import MetricsCollector


class DummyClient:
    def __init__(self) -> None:
        self.orders = []
        self._mark_price = 105.0
        self._position_qty = 0.0
        self.margin_calls: list[dict[str, object]] = []

    def place_market_order(self, **kwargs):
        if "position_idx" in kwargs:
            kwargs.setdefault("positionIdx", kwargs.pop("position_idx"))
        if "reduce_only" in kwargs:
            kwargs.setdefault("reduceOnly", kwargs.pop("reduce_only"))
        self.orders.append(kwargs)
        side = kwargs.get("side")
        qty = float(kwargs.get("qty", 0.0))
        reduce_only = bool(kwargs.get("reduceOnly", kwargs.get("reduce_only", False)))
        if qty > 0:
            if reduce_only:
                if side == "Sell":
                    self._position_qty = max(self._position_qty - qty, 0.0)
                elif side == "Buy":
                    self._position_qty = min(self._position_qty + qty, 0.0)
            else:
                if side == "Buy":
                    self._position_qty += qty
                elif side == "Sell":
                    self._position_qty -= qty
        return OrderResult(order_id="test-order", raw={})

    def close_position_market(
        self,
        *,
        symbol: str,
        qty: float,
        category: str = "linear",
        reduce_only: bool = True,
        position_idx: int | None = None,
    ):
        side = "Sell" if qty > 0 else "Buy"
        payload: dict[str, object] = {
            "symbol": symbol,
            "side": side,
            "qty": abs(qty),
            "category": category,
            "reduceOnly": reduce_only,
        }
        if position_idx is not None:
            payload["position_idx"] = position_idx
        return self.place_market_order(**payload)

    def get_wallet_equity(self, *, coin: str = "USDT"):
        return 10_000.0

    def get_linear_position_snapshot(self, *, symbol: str, category: str = "linear"):
        return {
            "qty": self._position_qty,
            "entry_price": 0.0,
            "mark_price": self._mark_price,
        }

    def configure_margin_and_leverage(
        self,
        *,
        category: str,
        symbol: str,
        margin_mode: str,
        leverage: float,
        position_mode: str,
    ) -> None:
        self.margin_calls.append(
            {
                "category": category,
                "symbol": symbol,
                "margin_mode": margin_mode,
                "leverage": leverage,
                "position_mode": position_mode,
            }
        )


def _build_trend_records() -> list[KlineRecord]:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles: list[KlineRecord] = []
    price = 100.0
    for i in range(180):
        ts = base + timedelta(minutes=i)
        high = price + 1.0
        low = price - 1.0
        close = price + 0.8
        volume = 100.0 if i < 170 else 2000.0 + (i - 170) * 100.0
        candles.append(
            KlineRecord(
                timestamp=pd.Timestamp(ts),
                open=price,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )
        )
        price = close
    return candles


def _build_downtrend_records() -> list[KlineRecord]:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles: list[KlineRecord] = []
    price = 100.0
    for i in range(180):
        ts = base + timedelta(minutes=i)
        high = price + 0.8
        close = price - 0.8
        low = close - 1.0
        volume = 100.0 if i < 170 else 2000.0 + (i - 170) * 100.0
        candles.append(
            KlineRecord(
                timestamp=pd.Timestamp(ts),
                open=price,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )
        )
        price = close
    return candles


def _metric_value(
    registry: CollectorRegistry, name: str, labels: dict[str, str] | None = None
) -> float:
    labels = labels or {}
    value = registry.get_sample_value(name, labels)
    assert value is not None, f"Metric {name} with labels {labels} not found"
    return value


def test_trader_places_order_on_breakout():
    cfg = AppConfig()
    cfg.execution.lookback_candles = 120
    cfg.execution.min_qty = 0.0001
    cfg.execution.qty_step = 0.0001
    cfg.strategy.ema_fast = 5
    cfg.strategy.ema_slow = 10
    cfg.strategy.micro_high_lookback = 3
    cfg.strategy.atr_period = 5
    cfg.strategy.timeframe_entry = "1m"
    cfg.strategy.volume_ma_period = 5
    cfg.strategy.volume_threshold_ratio = 1.05

    feed = MemoryDataFeed(_build_trend_records())
    client = DummyClient()
    risk_manager = RiskManager(cfg)

    trader = Trader(
        cfg, metrics=None, feed=feed, risk_manager=risk_manager, client=client
    )
    trader.step()

    assert client.orders, "Expected an order to be placed when breakout occurs"
    order = client.orders[0]
    assert order["symbol"] == cfg.symbol
    assert order["side"] == "Buy"
    assert order.get("positionIdx", order.get("position_idx")) == 0


def test_trader_configures_margin_settings():
    cfg = AppConfig()
    cfg.execution.margin_mode = "ISOLATED_MARGIN"
    cfg.execution.leverage = 2.5
    feed = MemoryDataFeed(_build_trend_records())
    client = DummyClient()
    risk_manager = RiskManager(cfg)

    Trader(cfg, metrics=None, feed=feed, risk_manager=risk_manager, client=client)

    assert client.margin_calls
    call = client.margin_calls[0]
    assert call["margin_mode"] == "ISOLATED_MARGIN"
    assert call["leverage"] == pytest.approx(2.5)
    assert call["position_mode"] == "ONE_WAY"


def test_trader_sets_position_idx_in_hedge_mode():
    cfg = AppConfig()
    cfg.execution.position_mode = "HEDGE"
    cfg.execution.margin_mode = "ISOLATED_MARGIN"
    cfg.execution.min_qty = 0.001
    cfg.execution.qty_step = 0.001
    cfg.execution.lookback_candles = 120
    cfg.strategy.ema_fast = 5
    cfg.strategy.ema_slow = 10
    cfg.strategy.micro_high_lookback = 3
    cfg.strategy.atr_period = 5
    cfg.strategy.timeframe_entry = "1m"
    cfg.strategy.volume_ma_period = 5
    cfg.strategy.volume_threshold_ratio = 1.05
    feed = MemoryDataFeed(_build_trend_records())
    client = DummyClient()
    risk_manager = RiskManager(cfg)

    trader = Trader(
        cfg, metrics=None, feed=feed, risk_manager=risk_manager, client=client
    )
    trader.step()

    assert client.orders
    entry = client.orders[0]
    assert entry.get("positionIdx", entry.get("position_idx")) == 1

    trade = trader.active_trade
    assert trade is not None

    stop_price = trade.stop_price
    next_ts = feed.candles[-1].timestamp + timedelta(minutes=1)
    feed.candles.append(
        KlineRecord(
            timestamp=pd.Timestamp(next_ts),
            open=stop_price,
            high=stop_price + 0.5,
            low=stop_price - 5.0,
            close=stop_price - 2.0,
            volume=500.0,
        )
    )

    trader.step()
    assert len(client.orders) == 2
    exit_order = client.orders[1]
    assert exit_order.get("positionIdx", exit_order.get("position_idx")) == 1
    assert bool(exit_order.get("reduceOnly", exit_order.get("reduce_only"))) is True


def test_trader_quantizes_order_qty():
    class FixedRiskManager(RiskManager):
        def position_size(self, entry_price: float, stop_price: float, *, side: str) -> float:
            return 0.0012685871807996793

    cfg = AppConfig()
    cfg.execution.lookback_candles = 120
    cfg.execution.min_qty = 0.001
    cfg.execution.qty_step = 0.001
    cfg.strategy.ema_fast = 5
    cfg.strategy.ema_slow = 10
    cfg.strategy.micro_high_lookback = 3
    cfg.strategy.atr_period = 5
    cfg.strategy.timeframe_entry = "1m"
    cfg.strategy.volume_ma_period = 5
    cfg.strategy.volume_threshold_ratio = 1.05

    feed = MemoryDataFeed(_build_trend_records())
    client = DummyClient()
    risk_manager = FixedRiskManager(cfg)

    trader = Trader(
        cfg, metrics=None, feed=feed, risk_manager=risk_manager, client=client
    )
    trader.step()

    assert client.orders
    order = client.orders[0]
    assert order["qty"] == pytest.approx(0.001)
    assert order.get("positionIdx", order.get("position_idx")) == 0


def test_trader_caps_qty_by_notional_limit():
    cfg = AppConfig()
    cfg.execution.lookback_candles = 120
    cfg.execution.min_qty = 0.0001
    cfg.execution.qty_step = 0.0001
    cfg.strategy.ema_fast = 5
    cfg.strategy.ema_slow = 10
    cfg.strategy.micro_high_lookback = 3
    cfg.strategy.atr_period = 5
    cfg.strategy.timeframe_entry = "1m"
    cfg.strategy.volume_ma_period = 5
    cfg.strategy.volume_threshold_ratio = 1.05
    cfg.risk.max_live_order_notional_krw = 50_000.0
    cfg.risk.usdt_krw_rate = 1_000.0

    feed = MemoryDataFeed(_build_trend_records())
    client = DummyClient()
    risk_manager = RiskManager(cfg)

    trader = Trader(
        cfg, metrics=None, feed=feed, risk_manager=risk_manager, client=client
    )
    trader.step()

    assert client.orders
    order = client.orders[0]
    # limit_usdt = 50, price close ~ >100 => qty should be < 1
    assert float(order["qty"]) < 1


def test_trader_respects_daily_stop():
    cfg = AppConfig()
    cfg.execution.lookback_candles = 120
    cfg.execution.min_qty = 0.0001
    cfg.execution.qty_step = 0.0001
    cfg.strategy.ema_fast = 5
    cfg.strategy.ema_slow = 10
    cfg.strategy.micro_high_lookback = 3
    cfg.strategy.atr_period = 5
    cfg.strategy.timeframe_entry = "1m"
    cfg.strategy.volume_ma_period = 5
    cfg.strategy.volume_threshold_ratio = 1.05

    feed = MemoryDataFeed(_build_trend_records())
    client = DummyClient()
    risk_manager = RiskManager(cfg)

    trader = Trader(
        cfg, metrics=None, feed=feed, risk_manager=risk_manager, client=client
    )

    risk_manager.record_realized_r(-3.0)

    trader.step()

    assert not client.orders


def test_trader_closes_position_on_stop():
    cfg = AppConfig()
    cfg.execution.lookback_candles = 120
    cfg.execution.min_qty = 0.0001
    cfg.execution.qty_step = 0.0001
    cfg.strategy.ema_fast = 5
    cfg.strategy.ema_slow = 10
    cfg.strategy.micro_high_lookback = 3
    cfg.strategy.atr_period = 5
    cfg.strategy.timeframe_entry = "1m"
    cfg.strategy.volume_ma_period = 5
    cfg.strategy.volume_threshold_ratio = 1.05

    feed = MemoryDataFeed(_build_trend_records())
    client = DummyClient()
    risk_manager = RiskManager(cfg)

    trader = Trader(
        cfg, metrics=None, feed=feed, risk_manager=risk_manager, client=client
    )

    trader.step()

    assert len(client.orders) == 1
    trade = trader.active_trade
    assert trade is not None

    stop_price = trade.stop_price
    next_ts = feed.candles[-1].timestamp + timedelta(minutes=1)
    feed.candles.append(
        KlineRecord(
            timestamp=pd.Timestamp(next_ts),
            open=stop_price,
            high=stop_price + 0.5,
            low=stop_price - 5.0,
            close=stop_price - 2.0,
            volume=500.0,
        )
    )

    trader.step()

    assert len(client.orders) == 2
    exit_order = client.orders[1]
    assert bool(exit_order.get("reduceOnly", exit_order.get("reduce_only"))) is True
    assert exit_order.get("side") == "Sell"
    assert pytest.approx(risk_manager.daily_r_total(), rel=1e-2) == -1.0


def test_trader_places_short_trade_and_closes_on_stop():
    cfg = AppConfig()
    cfg.execution.lookback_candles = 120
    cfg.execution.min_qty = 0.0001
    cfg.execution.qty_step = 0.0001
    cfg.strategy.allow_counter_trend_shorts = True
    cfg.strategy.ema_fast = 5
    cfg.strategy.ema_slow = 10
    cfg.strategy.micro_high_lookback = 3
    cfg.strategy.atr_period = 5
    cfg.strategy.timeframe_entry = "1m"
    cfg.strategy.volume_ma_period = 5
    cfg.strategy.volume_threshold_ratio = 1.05

    feed = MemoryDataFeed(_build_downtrend_records())
    client = DummyClient()
    risk_manager = RiskManager(cfg)

    trader = Trader(
        cfg, metrics=None, feed=feed, risk_manager=risk_manager, client=client
    )
    trader.step()

    assert client.orders
    entry_order = client.orders[0]
    assert entry_order["side"] == "Sell"

    trade = trader.active_trade
    assert trade is not None

    stop_price = trade.stop_price
    next_ts = feed.candles[-1].timestamp + timedelta(minutes=1)
    feed.candles.append(
        KlineRecord(
            timestamp=pd.Timestamp(next_ts),
            open=stop_price,
            high=stop_price + 5.0,
            low=stop_price - 0.5,
            close=stop_price + 1.0,
            volume=500.0,
        )
    )

    trader.step()

    assert len(client.orders) == 2
    exit_order = client.orders[1]
    assert exit_order.get("side") == "Buy"


def test_trader_records_trades_in_metrics():
    registry = CollectorRegistry()
    cfg = AppConfig()
    cfg.execution.lookback_candles = 120
    cfg.execution.min_qty = 0.0001
    cfg.execution.qty_step = 0.0001
    cfg.strategy.ema_fast = 5
    cfg.strategy.ema_slow = 10
    cfg.strategy.micro_high_lookback = 3
    cfg.strategy.atr_period = 5
    cfg.strategy.timeframe_entry = "1m"
    cfg.strategy.volume_ma_period = 5
    cfg.strategy.volume_threshold_ratio = 1.05

    feed = MemoryDataFeed(_build_trend_records())
    client = DummyClient()
    risk_manager = RiskManager(cfg)

    metrics = MetricsCollector(
        registry=registry, initial_equity=cfg.risk.starting_equity, recent_limit=1
    )

    trader = Trader(
        cfg, metrics=metrics, feed=feed, risk_manager=risk_manager, client=client
    )

    trader.step()

    trade = trader.active_trade
    assert trade is not None
    stop_price = trade.stop_price

    next_ts = feed.candles[-1].timestamp + timedelta(minutes=1)
    feed.candles.append(
        KlineRecord(
            timestamp=pd.Timestamp(next_ts),
            open=stop_price,
            high=stop_price + 0.5,
            low=stop_price - 5.0,
            close=stop_price - 2.0,
            volume=500.0,
        )
    )

    trader.step()

    assert _metric_value(registry, "coin_trader_trades_total") == 1.0
    assert _metric_value(registry, "coin_trader_recent_trade_pnl", {"slot": "0"}) <= 0.0
