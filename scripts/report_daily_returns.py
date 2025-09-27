#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from coin_trader_bybit.backtest import Backtester
from coin_trader_bybit.core.config import AppConfig


def _load_cfg(path: Path) -> AppConfig:
    raw: Dict[str, Any] = yaml.safe_load(path.read_text())
    return AppConfig.model_validate(raw)


def _load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Run backtest and summarize daily returns")
    p.add_argument("--config", default="configs/params.yaml")
    p.add_argument("--data", required=True)
    p.add_argument("--since", default=None)
    p.add_argument("--until", default=None)
    args = p.parse_args()

    cfg = _load_cfg(Path(args.config))
    data = _load_data(Path(args.data))
    if args.since:
        data = data[data.index >= pd.to_datetime(args.since, utc=True)]
    if args.until:
        data = data[data.index <= pd.to_datetime(args.until, utc=True)]
    rep = Backtester(cfg).run(data)

    # Build a continuous daily series with forward-filled equity
    eq_raw = rep.equity_curve.copy()
    if not eq_raw.empty:
        daily_index = pd.date_range(eq_raw.index[0].normalize(), eq_raw.index[-1].normalize(), freq="1D", tz=eq_raw.index.tz)
        eq = eq_raw.reindex(daily_index, method="ffill")
    else:
        eq = eq_raw
    daily_ret = eq.pct_change().dropna()
    if daily_ret.empty:
        print("No daily returns to summarize")
        return

    print("Daily returns summary")
    print(f"count={len(daily_ret)} mean={daily_ret.mean()*100:.3f}% median={daily_ret.median()*100:.3f}% std={daily_ret.std()*100:.3f}%")
    q = daily_ret.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    for k, v in q.items():
        print(f"p{k:.0%}: {v*100:.3f}%")
    print(f"min={daily_ret.min()*100:.3f}% max={daily_ret.max()*100:.3f}%")
    print("Backtest metrics:")
    m = rep.metrics
    pf = "∞" if (m.profit_factor == float('inf')) else (f"{m.profit_factor:.4f}" if m.profit_factor is not None else "n/a")
    print(f"Trades={m.num_trades} PnL={m.total_pnl:.2f} EndEq={m.ending_equity:.2f} PF={pf} WinRate={(m.win_rate or 0)*100:.2f}% AvgR={(m.avg_r or 0):.3f} MDD={m.max_drawdown*100:.2f}% CAGR={(m.cagr or 0)*100:.2f}%")


if __name__ == "__main__":
    main()
