from datetime import datetime, timedelta, timezone

import pandas as pd

from coin_trader_bybit.core.config import AppConfig
from coin_trader_bybit.data.feed import KlineRecord, MemoryDataFeed
from coin_trader_bybit.exec.trader import Trader
from coin_trader_bybit.exchange.bybit import OrderResult
from coin_trader_bybit.risk.manager import RiskManager


class DummyClient:
    def __init__(self) -> None:
        self.orders = []
        self._mark_price = 105.0

    def place_market_order(self, **kwargs):
        self.orders.append(kwargs)
        return OrderResult(order_id="test-order", raw={})

    def get_wallet_equity(self, *, coin: str = "USDT"):
        return 10_000.0

    def get_linear_position_snapshot(self, *, symbol: str, category: str = "linear"):
        return {"qty": 0.0, "entry_price": 0.0, "mark_price": self._mark_price}


def _build_trend_records() -> list[KlineRecord]:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    candles: list[KlineRecord] = []
    price = 100.0
    for i in range(180):
        ts = base + timedelta(minutes=i)
        high = price + 1.0
        low = price - 1.0
        close = price + 0.8
        candles.append(
            KlineRecord(
                timestamp=pd.Timestamp(ts),
                open=price,
                high=high,
                low=low,
                close=close,
            )
        )
        price = close
    return candles


def test_trader_places_order_on_breakout():
    cfg = AppConfig()
    cfg.execution.lookback_candles = 120
    cfg.execution.min_qty = 0.0001
    cfg.strategy.ema_fast = 5
    cfg.strategy.ema_slow = 10
    cfg.strategy.micro_high_lookback = 3
    cfg.strategy.atr_period = 5
    cfg.strategy.timeframe_entry = "1m"

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
