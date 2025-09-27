#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import random

import pandas as pd
import yaml

from coin_trader_bybit.backtest import Backtester
from coin_trader_bybit.core.config import AppConfig


@dataclass
class Trial:
    params: Dict[str, Any]
    pf: float | None
    avg_r: float | None
    pnl: float
    num_trades: int
    mdd: float


def _load_cfg(path: Path) -> AppConfig:
    raw: Dict[str, Any] = yaml.safe_load(path.read_text())
    return AppConfig.model_validate(raw)


def _load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    return df


def _apply_updates(cfg: AppConfig, updates: Dict[str, Any]) -> AppConfig:
    cfg = cfg.model_copy(deep=True)
    for k, v in updates.items():
        # only support strategy.* and risk.* simple keys here
        if k.startswith("strategy."):
            setattr(cfg.strategy, k.split(".", 1)[1], v)
        elif k.startswith("risk."):
            setattr(cfg.risk, k.split(".", 1)[1], v)
    return cfg


def _grid(param_grid: Dict[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(param_grid.keys())
    vals = [list(v) for v in param_grid.values()]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))


def main() -> None:
    p = argparse.ArgumentParser(description="Grid-search a small set of strategy params")
    p.add_argument("--config", default="configs/params.yaml")
    p.add_argument("--data", required=True)
    p.add_argument("--since", default=None, help="ISO8601 start (UTC)")
    p.add_argument("--until", default=None, help="ISO8601 end (UTC)")
    p.add_argument("--top", type=int, default=8)
    p.add_argument("--objective", choices=["pf", "growth"], default="growth")
    p.add_argument("--samples", type=int, default=0, help="Random sample size; 0 = exhaustive")
    args = p.parse_args()

    base = _load_cfg(Path(args.config))
    data = _load_data(Path(args.data))
    if args.since:
        data = data[data.index >= pd.to_datetime(args.since, utc=True)]
    if args.until:
        data = data[data.index <= pd.to_datetime(args.until, utc=True)]

    # Focused grid for swing breakout
    if args.objective == "growth":
        param_grid = {
            "risk.max_risk_per_trade_pct": [1.0, 1.2],
            "risk.daily_max_trades": [2, 3],
            "strategy.signal_cooldown_minutes": [60, 120],
            "strategy.breakout_band_pct": [0.0001, 0.0002],
            "strategy.daily_breakout_band_pct": [0.0001, 0.0002],
            "strategy.micro_high_lookback": [64, 96],
            "strategy.anchor_adx_min": [10, 12],
            "strategy.anchor_trend_slope_min_pct": [0.0002, 0.0004],
            "strategy.atr_mult_stop": [1.6, 1.8],
            "strategy.swing_trail_atr_mult": [1.8, 2.2],
            "strategy.risk_mult_breakout": [1.3, 1.6],
        }
    else:
        param_grid = {
            "strategy.anchor_adx_min": [10, 14],
            "strategy.anchor_trend_slope_min_pct": [0.0004, 0.0006],
            "strategy.daily_breakout_band_pct": [0.0, 0.0002],
            "strategy.prebreakout_atr_pct_max_pctile": [0.5, 0.7],
            "strategy.atr_mult_stop": [1.8, 2.0],
            "strategy.min_stop_distance_pct": [0.006],
            "strategy.swing_trail_atr_mult": [2.0, 2.4],
        }

    trials: List[Trial] = []
    combos: Iterable[Dict[str, Any]]
    all_combos = list(_grid(param_grid))
    if args.samples and args.samples > 0 and args.samples < len(all_combos):
        random.seed(42)
        combos = random.sample(all_combos, args.samples)
    else:
        combos = all_combos
    for updates in combos:
        cfg = _apply_updates(base, updates)
        rep = Backtester(cfg).run(data)
        m = rep.metrics
        trials.append(
            Trial(params=updates, pf=m.profit_factor, avg_r=m.avg_r, pnl=m.total_pnl, num_trades=m.num_trades, mdd=m.max_drawdown)
        )

    def _score(t: Trial) -> Tuple[float, float, float]:
        if args.objective == "growth":
            # prioritize CAGR, then PF, then low MDD
            # we proxy CAGR by pnl here (short windows), else compute from report
            # For simplicity, treat pnl as growth proxy and PF as tie-breaker
            pf = (t.pf or 0.0)
            return (t.pnl, pf, -t.mdd)
        pf = t.pf or 0.0
        avg_r = t.avg_r or 0.0
        return (pf, avg_r, -t.mdd)

    trials.sort(key=_score, reverse=True)
    print("Top trials (by PF, AvgR, -MDD):")
    for i, t in enumerate(trials[: args.top], 1):
        print(f"#{i} PF={(t.pf or 0):.3f} AvgR={(t.avg_r or 0):.3f} MDD={t.mdd*100:.2f}% PnL={t.pnl:.2f} Trades={t.num_trades} :: {t.params}")


if __name__ == "__main__":
    main()
