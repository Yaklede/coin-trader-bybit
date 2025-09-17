def size_by_risk(
    equity: float, risk_pct: float, stop_distance: float, contract_value: float = 1.0
) -> float:
    """Risk-based position sizing.

    qty = (equity * risk_pct/100) / (stop_distance * contract_value)
    """
    if stop_distance <= 0:
        raise ValueError("stop_distance must be > 0")
    if equity <= 0:
        raise ValueError("equity must be > 0")
    risk_amount = equity * (risk_pct / 100.0)
    return max(risk_amount / (stop_distance * contract_value), 0.0)
