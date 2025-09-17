import pytest

from coin_trader_bybit.exchange.bybit import BybitClient


class FakeHTTP:
    def __init__(self, *, price: float) -> None:
        self.price = price
        self.place_order_called = False
        self.kwargs: dict[str, str] | None = None

    def get_tickers(
        self, *, category: str, symbol: str
    ) -> dict[str, object]:  # noqa: D401 - simple stub
        return {"result": {"list": [{"lastPrice": str(self.price)}]}}

    def place_order(self, **kwargs: str) -> dict[str, object]:
        self.place_order_called = True
        self.kwargs = kwargs
        return {"result": {"orderId": "stub-123"}}


def make_client(
    *, price: float, max_limit: float, usdt_krw_rate: float, testnet: bool
) -> tuple[BybitClient, FakeHTTP]:
    client = BybitClient(
        api_key="",
        api_secret="",
        testnet=testnet,
        max_live_order_notional_krw=max_limit,
        usdt_krw_rate=usdt_krw_rate,
    )
    fake_http = FakeHTTP(price=price)
    client.http = fake_http  # type: ignore[assignment]
    return client, fake_http


def test_live_order_rejected_when_over_krw_limit() -> None:
    client, fake_http = make_client(
        price=1_000.0, max_limit=50_000.0, usdt_krw_rate=1_000.0, testnet=False
    )

    with pytest.raises(ValueError) as exc:
        client.place_market_order(symbol="BTCUSDT", side="Buy", qty=60.0)

    assert "exceeds configured limit" in str(exc.value)
    assert fake_http.place_order_called is False


def test_live_reduce_only_bypasses_limit() -> None:
    client, fake_http = make_client(
        price=1_000.0, max_limit=50_000.0, usdt_krw_rate=1_000.0, testnet=False
    )

    res = client.place_market_order(
        symbol="BTCUSDT", side="Sell", qty=60.0, reduce_only=True
    )

    assert res.order_id == "stub-123"
    assert fake_http.place_order_called is True


def test_testnet_order_ignores_limit() -> None:
    client, fake_http = make_client(
        price=1_000.0, max_limit=50_000.0, usdt_krw_rate=1_000.0, testnet=True
    )

    res = client.place_market_order(symbol="BTCUSDT", side="Buy", qty=60.0)

    assert res.order_id == "stub-123"
    assert fake_http.place_order_called is True


def test_close_position_market_wraps_reduce_only() -> None:
    client, fake_http = make_client(
        price=1_000.0, max_limit=50_000.0, usdt_krw_rate=1_000.0, testnet=False
    )

    client.close_position_market(symbol="BTCUSDT", qty=3.0, reduce_only=True)

    assert fake_http.place_order_called is True
    assert fake_http.kwargs is not None
    assert fake_http.kwargs.get("reduceOnly") is True
    assert fake_http.kwargs.get("side") == "Sell"

    client.close_position_market(symbol="BTCUSDT", qty=-2.0, reduce_only=False)

    assert fake_http.kwargs is not None
    assert fake_http.kwargs.get("reduceOnly") is False
    assert fake_http.kwargs.get("side") == "Buy"
