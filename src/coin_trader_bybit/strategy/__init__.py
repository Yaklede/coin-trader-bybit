from __future__ import annotations

from typing import Protocol

from ..core.config import StrategyConfig
from .scalper import Scalper, Signal, timeframe_to_pandas_freq


class StrategyLike(Protocol):  # pragma: no cover - structural typing helper
    cfg: StrategyConfig
    def compute_features(self, data): ...
    def generate_signal(self, features): ...


def create_strategy(cfg: StrategyConfig) -> StrategyLike:
    name = (cfg.name or "scalper_v1").lower()
    if name.startswith("swing"):
        try:
            from .swing import Swing

            return Swing(cfg)
        except Exception:  # pragma: no cover - fallback
            return Scalper(cfg)
    return Scalper(cfg)


__all__ = [
    "create_strategy",
    "Scalper",
    "Signal",
    "timeframe_to_pandas_freq",
]

