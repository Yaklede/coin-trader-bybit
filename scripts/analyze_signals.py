#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
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


def pct(x: int, n: int) -> str:
    return f"{(x/n)*100:.2f}%" if n else "0.00%"


def main() -> None:
    p = argparse.ArgumentParser(description="Vectorized analysis of signal gates")
    p.add_argument("--config", default="configs/params.yaml")
    p.add_argument("--data", required=True)
    args = p.parse_args()

    cfg = _load_cfg(Path(args.config))
    df = _load_data(Path(args.data))
    sc = Scalper(cfg.strategy)
    f = sc.compute_features(df).dropna(subset=["atr", "volume_ma"]).copy()
    n = len(f)
    if n == 0:
        print("No features")
        return

    # Base ATR% filter (all three applied in scalper.generate_signal)
    atr_pct = f["atr"] / f["close"].replace(0.0, np.nan)
    c_min_atr = atr_pct >= (cfg.strategy.min_atr_pct or 0.0)
    c_regime_atr = (atr_pct >= cfg.strategy.regime_atr_min_pct) & (atr_pct <= cfg.strategy.regime_atr_max_pct)

    vol_ratio = f["volume_ratio"]
    c_vol_threshold = ~cfg.strategy.use_volume_filter | (vol_ratio >= cfg.strategy.volume_threshold_ratio)
    c_impulse = vol_ratio >= cfg.strategy.impulse_volume_ratio

    # Trend gates
    c_trend_up = f["trend_strength"] >= cfg.strategy.regime_trend_min
    c_slope_up = f["trend_slope_pct"] >= cfg.strategy.trend_slope_min_pct
    if cfg.strategy.use_trend_filter:
        c_anchor = f["anchor_trend_slope_pct"] >= cfg.strategy.anchor_trend_slope_min_pct
    else:
        c_anchor = pd.Series(True, index=f.index)

    # Proximity gates
    c_price_below_fast = f["close"] <= f["ema_fast"]
    c_pullback_pct = (f["close"] - f["ema_fast"]).abs() / f["close"] <= max(cfg.strategy.ema_pullback_pct, 0.0)
    if cfg.strategy.ema_pullback_atr_mult and (f["atr"] > 0).any():
        c_pullback_atr = (f["close"] - f["ema_fast"]).abs() <= cfg.strategy.ema_pullback_atr_mult * f["atr"]
    else:
        c_pullback_atr = pd.Series(True, index=f.index)

    # RSI gate
    c_rsi = f["rsi"] <= cfg.strategy.rsi_buy_threshold

    # Fast market gate
    if cfg.strategy.avoid_fast_market_entries:
        c_fast = ~f["fast_market"]
    else:
        c_fast = pd.Series(True, index=f.index)

    stages = [
        ("atr_min", c_min_atr),
        ("atr_window", c_regime_atr),
        ("vol_threshold", c_vol_threshold),
        ("impulse", c_impulse),
        ("trend_up", c_trend_up),
        ("slope_up", c_slope_up),
        ("anchor", c_anchor),
        ("below_fast", c_price_below_fast),
        ("pullback_pct", c_pullback_pct),
        ("pullback_atr", c_pullback_atr),
        ("rsi", c_rsi),
        ("no_fast", c_fast),
    ]

    mask = pd.Series(True, index=f.index)
    for name, cond in stages:
        mask &= cond.fillna(False)
        cnt = int(mask.sum())
        print(f"{name:14s} {cnt:8d} / {n:8d}  :: {pct(cnt, n)}")

    print("\nEstimated long signal count:", int(mask.sum()))


if __name__ == "__main__":
    main()

