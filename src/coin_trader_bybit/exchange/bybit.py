from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import os

try:
    # Prefer official/unified pybit if available
    from pybit.unified_trading import HTTP  # type: ignore

    HAVE_PYBIT = True
except Exception:  # pragma: no cover - fallback when pybit missing
    HTTP = None  # type: ignore
    HAVE_PYBIT = False


@dataclass
class OrderResult:
    order_id: str | None
    raw: Dict[str, Any]


class BybitClient:
    """Minimal Bybit adapter.

    Uses `pybit.unified_trading.HTTP` when present. Falls back to a stub that raises
    at call-time if the library is missing. Keep only thin wrappers here; implement
    risk/strategy elsewhere.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        *,
        testnet: bool = True,
        max_live_order_notional_krw: Optional[float] = None,
        usdt_krw_rate: float = 1_350.0,
    ):
        api_key = api_key or os.getenv("BYBIT_API_KEY")
        api_secret = api_secret or os.getenv("BYBIT_API_SECRET")
        self.testnet = testnet
        self.max_live_order_notional_krw = max_live_order_notional_krw
        self.usdt_krw_rate = usdt_krw_rate

        if self.usdt_krw_rate <= 0:
            raise ValueError("usdt_krw_rate must be positive")

        if HAVE_PYBIT:
            self.http = HTTP(testnet=testnet, api_key=api_key, api_secret=api_secret)
        else:  # pragma: no cover
            self.http = None

    def place_market_order(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        category: str = "linear",
        reduce_only: bool = False,
        position_idx: int = 0,
    ) -> OrderResult:
        if not HAVE_PYBIT:  # pragma: no cover
            raise RuntimeError(
                "pybit not installed; install requirements or use ccxt fallback"
            )

        if not self.testnet and not reduce_only:
            self._enforce_live_notional_cap(symbol=symbol, category=category, qty=qty)

        resp = self.http.place_order(
            category=category,
            symbol=symbol,
            side=side,
            orderType="Market",
            qty=str(qty),
            reduceOnly=reduce_only,
            positionIdx=position_idx,
            timeInForce="GoodTillCancel",
        )
        order_id = (
            resp.get("result", {}).get("orderId") if isinstance(resp, dict) else None
        )
        return OrderResult(
            order_id=order_id, raw=resp if isinstance(resp, dict) else {}
        )

    def get_kline(
        self,
        *,
        symbol: str,
        category: str = "linear",
        interval: str = "1",
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        if not HAVE_PYBIT:  # pragma: no cover
            raise RuntimeError("pybit not installed")
        resp = self.http.get_kline(
            category=category,
            symbol=symbol,
            interval=interval,
            limit=limit,
        )
        result = resp.get("result") if isinstance(resp, dict) else None
        if not isinstance(result, dict):
            return []
        data = result.get("list")
        return data if isinstance(data, list) else []

    def _enforce_live_notional_cap(
        self, *, symbol: str, category: str, qty: float
    ) -> None:
        limit = self.max_live_order_notional_krw
        if limit is None or limit <= 0:
            return
        if qty <= 0:
            raise ValueError("qty must be positive for live orders")

        price = self._fetch_last_price(symbol=symbol, category=category)
        notional_usdt = price * qty
        notional_krw = notional_usdt * self.usdt_krw_rate
        if notional_krw > limit:
            raise ValueError(
                f"Live order notional {notional_krw:.2f} KRW exceeds configured limit of {limit:.2f} KRW"
            )

    def _fetch_last_price(self, *, symbol: str, category: str) -> float:
        if not HAVE_PYBIT or self.http is None:  # pragma: no cover
            raise RuntimeError("pybit not installed")

        resp = self.http.get_tickers(category=category, symbol=symbol)
        result = resp.get("result") if isinstance(resp, dict) else None
        if not isinstance(result, dict):
            raise RuntimeError("Unexpected response from get_tickers: missing result")
        items = result.get("list")
        if not items:
            raise RuntimeError("Unexpected response from get_tickers: empty list")
        data = items[0]
        price_str = data.get("lastPrice") or data.get("markPrice")
        if price_str is None:
            raise RuntimeError("Ticker response missing price fields")
        try:
            return float(price_str)
        except (TypeError, ValueError) as exc:  # pragma: no cover
            raise RuntimeError("Ticker price is not numeric") from exc

    def cancel_all(self, *, symbol: str, category: str = "linear") -> Dict[str, Any]:
        if not HAVE_PYBIT:  # pragma: no cover
            raise RuntimeError("pybit not installed")
        return self.http.cancel_all_orders(category=category, symbol=symbol)

    def get_position(self, *, symbol: str, category: str = "linear") -> Dict[str, Any]:
        if not HAVE_PYBIT:  # pragma: no cover
            raise RuntimeError("pybit not installed")
        return self.http.get_positions(category=category, symbol=symbol)

    def get_linear_position_snapshot(
        self, *, symbol: str, category: str = "linear"
    ) -> Dict[str, float]:
        data = self.get_position(symbol=symbol, category=category)
        result = data.get("result") if isinstance(data, dict) else {}
        items = result.get("list") if isinstance(result, dict) else []
        if not isinstance(items, list):
            return {"qty": 0.0, "entry_price": 0.0, "mark_price": 0.0}
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("symbol") != symbol:
                continue
            qty = float(item.get("size", 0.0))
            entry_price = float(item.get("avgPrice", 0.0) or 0.0)
            mark_price = float(item.get("markPrice", entry_price) or entry_price)
            return {"qty": qty, "entry_price": entry_price, "mark_price": mark_price}
        return {"qty": 0.0, "entry_price": 0.0, "mark_price": 0.0}

    def get_wallet_equity(self, *, coin: str = "USDT") -> Optional[float]:
        if not HAVE_PYBIT:  # pragma: no cover
            raise RuntimeError("pybit not installed")
        resp = self.http.get_wallet_balance(accountType="UNIFIED", coin=coin)
        result = resp.get("result") if isinstance(resp, dict) else None
        if not isinstance(result, dict):
            return None
        balances = result.get("list")
        if not isinstance(balances, list):
            return None
        for item in balances:
            if not isinstance(item, dict):
                continue
            if item.get("coin") != coin:
                continue
            equity = item.get("equity") or item.get("walletBalance")
            if equity is None:
                continue
            try:
                return float(equity)
            except (TypeError, ValueError):  # pragma: no cover
                continue
        return None

    def get_last_price(
        self, *, symbol: str, category: str = "linear"
    ) -> Optional[float]:
        if not HAVE_PYBIT:  # pragma: no cover
            raise RuntimeError("pybit not installed")
        resp = self.http.get_tickers(category=category, symbol=symbol)
        result = resp.get("result") if isinstance(resp, dict) else None
        if not isinstance(result, dict):
            return None
        items = result.get("list")
        if not isinstance(items, list) or not items:
            return None
        data = items[0]
        if not isinstance(data, dict):
            return None
        price_str = data.get("lastPrice") or data.get("markPrice")
        if price_str is None:
            return None
        try:
            return float(price_str)
        except (TypeError, ValueError):  # pragma: no cover
            return None
