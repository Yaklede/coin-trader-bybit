#!/usr/bin/env python3

"""Utility to run a long-horizon backtest using the current scalper config."""

from __future__ import annotations

import argparse
from numbers import Real
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml

import coin_trader_bybit.backtest.engine as engine
from coin_trader_bybit.core.config import AppConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Bybit scalper backtest")
    parser.add_argument(
        "--config",
        default="configs/params.yaml",
        help="YAML config file to load (default: configs/params.yaml)",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="CSV containing timestamp, open, high, low, close, volume columns",
    )
    return parser.parse_args()


def _load_config(path: Path) -> AppConfig:
    raw = yaml.safe_load(path.read_text())
    return AppConfig.model_validate(raw)


def _load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if "timestamp" not in df.columns:
        raise ValueError("CSV must include a timestamp column")
    return df.set_index("timestamp")


def _safe_size_patch() -> Callable[[], None]:
    original = engine.size_by_risk

    def safe_size_by_risk(
        equity: float,
        risk_pct: float,
        stop_distance: float,
        contract_value: float = 1.0,
    ) -> float:
        if equity <= 0:
            return 0.0
        return original(equity, risk_pct, stop_distance, contract_value)

    engine.size_by_risk = safe_size_by_risk  # type: ignore[assignment]

    def restore() -> None:
        engine.size_by_risk = original  # type: ignore[assignment]

    return restore


def _format_percent(value: Real | complex | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, complex):
        if abs(value.imag) < 1e-9:
            value = value.real
        else:
            return "n/a"
    return f"{value * 100:.2f}%"


def main() -> None:
    args = _parse_args()

    config_path = Path(args.config)
    data_path = Path(args.data)

    cfg = _load_config(config_path)
    candles = _load_data(data_path)

    backtester = engine.Backtester(cfg)
    restore_size = _safe_size_patch()
    try:
        report = backtester.run(candles)
    finally:
        restore_size()

    metrics = report.metrics
    ending_equity = metrics.ending_equity
    if abs(ending_equity) < 1e-6:
        ending_equity = 0.0

    print("Backtest complete")
    print(f"Trades executed: {metrics.num_trades}")
    print(f"Total PnL: {metrics.total_pnl:.2f} USDT")
    print(f"Ending equity: {ending_equity:.2f} USDT")
    if metrics.profit_factor is None:
        pf = "n/a"
    elif metrics.profit_factor == float("inf"):
        pf = "∞"
    else:
        pf = f"{metrics.profit_factor:.4f}"
    print(f"Profit factor: {pf}")
    print(f"Win rate: {_format_percent(metrics.win_rate)}")
    avg_r = f"{metrics.avg_r:.2f}" if metrics.avg_r is not None else "n/a"
    print(f"Average R: {avg_r}")
    print(f"Max drawdown: {_format_percent(metrics.max_drawdown)}")
    print(f"CAGR: {_format_percent(metrics.cagr)}")


if __name__ == "__main__":
    main()
