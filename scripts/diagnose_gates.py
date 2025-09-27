#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from coin_trader_bybit.core.config import AppConfig
from coin_trader_bybit.strategy.scalper import Scalper


def _load_cfg(path: Path) -> AppConfig:
    raw: Dict[str, Any] = yaml.safe_load(path.read_text())
    return AppConfig.model_validate(raw)


def _load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    return df


def pct(x: float, denom: int) -> str:
    if denom <= 0:
        return "0.00%"
    return f"{(x/denom)*100:.2f}%"


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnose gate pass rates for current config")
    p.add_argument("--config", default="configs/params.yaml")
    p.add_argument("--data", required=True)
    args = p.parse_args()

    cfg = _load_cfg(Path(args.config))
    df = _load_data(Path(args.data))
    sc = Scalper(cfg.strategy)
    f = sc.compute_features(df).dropna(subset=["atr", "volume_ma"]) 
    n = len(f)
    if n == 0:
        print("No features computed")
        return

    conds = {}
    conds["atr_window"] = (f["atr_pct"] >= cfg.strategy.regime_atr_min_pct) & (f["atr_pct"] <= cfg.strategy.regime_atr_max_pct)
    conds["volume_threshold"] = (f["volume_ratio"] >= cfg.strategy.volume_threshold_ratio)
    conds["impulse"] = (f["volume_ratio"] >= cfg.strategy.impulse_volume_ratio)
    conds["trend_up_gate"] = f["regime_trend_up"]
    conds["trend_down_gate"] = f["regime_trend_down"]

    for name, c in conds.items():
        cnt = int(c.sum())
        print(f"{name:16s} {cnt:8d} / {n:8d} :: {pct(cnt, n)}")

    print("\nPullback proximity (|close-ema_fast| / price <= ema_pullback_pct)")
    prox = (f["close"] - f["ema_fast"]).abs() / f["close"] <= max(cfg.strategy.ema_pullback_pct, 0.0)
    print(f"pullback_pct       {int(prox.sum()):8d} / {n:8d} :: {pct(int(prox.sum()), n)}")
    if cfg.strategy.ema_pullback_atr_mult and (f["atr"] > 0).any():
        prox_atr = (f["close"] - f["ema_fast"]).abs() <= cfg.strategy.ema_pullback_atr_mult * f["atr"]
        print(f"pullback_atr       {int(prox_atr.sum()):8d} / {n:8d} :: {pct(int(prox_atr.sum()), n)}")


if __name__ == "__main__":
    main()

