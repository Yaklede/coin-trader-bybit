import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv
from prometheus_client import CollectorRegistry

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - pandas is mandatory via requirements
    raise RuntimeError("pandas is required to run the application") from exc

from .core.config import AppConfig
from .exec.trader import Trader
from .backtest import Backtester
from .monitoring import MetricsCollector, configure_logging, start_metrics_server


def load_config(path: str | Path) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)
    return AppConfig.model_validate(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC Bybit scalping bot")
    parser.add_argument(
        "--mode",
        default="paper",
        choices=["paper", "live", "backtest"],
    )  # noqa: S603
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--config", default="configs/params.yaml")
    parser.add_argument("--testnet", default=None)
    parser.add_argument(
        "--data", default=None, help="CSV with timestamp, open, high, low, close"
    )
    args = parser.parse_args()

    load_dotenv()

    cfg = load_config(args.config)
    cfg.mode = args.mode  # override
    cfg.symbol = args.symbol  # override

    configure_logging(cfg.logging.level, json_output=cfg.logging.json_output)

    if args.testnet is not None:
        # env or flag can force testnet
        cfg.execution.testnet = str(args.testnet).lower() in {"1", "true", "yes"}
    elif os.getenv("USE_TESTNET") is not None:
        cfg.execution.testnet = os.getenv("USE_TESTNET", "true").lower() in {
            "1",
            "true",
            "yes",
        }
    else:
        cfg.execution.testnet = cfg.mode != "live"

    if cfg.mode == "backtest":
        run_backtest(cfg, data_path=args.data)
        return

    metrics = _setup_metrics(cfg) if cfg.monitoring.enable_metrics else None

    trader = Trader(cfg, metrics=metrics)
    trader.run()


def _setup_metrics(cfg: AppConfig) -> Optional[MetricsCollector]:
    registry = CollectorRegistry()
    start_metrics_server(cfg.monitoring.host, cfg.monitoring.port, registry)
    metrics = MetricsCollector(
        registry=registry,
        initial_equity=cfg.risk.starting_equity,
        recent_limit=cfg.monitoring.recent_trades,
    )
    metrics.set_equity(cfg.risk.starting_equity)
    return metrics


def run_backtest(cfg: AppConfig, *, data_path: str | None) -> None:
    if not data_path:
        raise ValueError("--data path is required in backtest mode")

    df = load_ohlcv(Path(data_path))
    backtester = Backtester(cfg)
    report = backtester.run(df)

    print("Backtest complete")
    print(f"Trades executed: {report.metrics.num_trades}")
    print(f"Total PnL: {report.metrics.total_pnl:.2f} USDT")
    print(f"Ending equity: {report.metrics.ending_equity:.2f} USDT")
    if report.metrics.profit_factor is not None:
        pf = (
            "∞"
            if report.metrics.profit_factor == float("inf")
            else f"{report.metrics.profit_factor:.2f}"
        )
        print(f"Profit factor: {pf}")
    if report.metrics.win_rate is not None:
        print(f"Win rate: {report.metrics.win_rate * 100:.1f}%")
    if report.metrics.avg_r is not None:
        print(f"Average R: {report.metrics.avg_r:.2f}")
    print(f"Max drawdown: {report.metrics.max_drawdown * 100:.2f}%")
    if report.metrics.cagr is not None:
        print(f"CAGR: {report.metrics.cagr * 100:.2f}%")


def load_ohlcv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    timestamp_col = None
    for candidate in ("timestamp", "time", "datetime"):
        if candidate in df.columns:
            timestamp_col = candidate
            break
    if timestamp_col is None:
        raise ValueError("CSV must include a timestamp column")
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    df = df.set_index(timestamp_col)
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required OHLC columns: {missing}")
    return df.sort_index()


if __name__ == "__main__":
    main()
