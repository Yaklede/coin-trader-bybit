from coin_trader_bybit.data.feed import BybitDataFeed


class StubClient:
    def __init__(self, payload):
        self.payload = payload

    def get_kline(self, **kwargs):
        return self.payload


def test_bybit_feed_handles_list_payload():
    payload = [["1700000000000", "100", "101", "99", "100.5", "123"]]
    feed = BybitDataFeed(StubClient(payload), symbol="BTCUSDT", interval="1")
    df = feed.fetch(limit=1)
    assert not df.empty
    assert df.iloc[0]["close"] == 100.5
    assert df.iloc[0]["volume"] == 123.0


def test_bybit_feed_handles_dict_payload():
    payload = [
        {
            "startTime": "1700000000000",
            "open": "100",
            "high": "101",
            "low": "99",
            "close": "100.5",
            "volume": "456",
        }
    ]
    feed = BybitDataFeed(StubClient(payload), symbol="BTCUSDT", interval="1")
    df = feed.fetch(limit=1)
    assert not df.empty
    assert df.iloc[0]["open"] == 100.0
    assert df.iloc[0]["volume"] == 456.0
