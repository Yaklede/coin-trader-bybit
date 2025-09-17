from coin_trader_bybit.risk.position import size_by_risk


def test_size_by_risk_basic():
    qty = size_by_risk(equity=10_000, risk_pct=0.5, stop_distance=50, contract_value=1)
    # Risk = $50; stop_distance $50 → qty ~ 1
    assert 0.9 < qty < 1.1
