from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from ..core.config import AppConfig
from ..data.feed import BybitDataFeed, DataFeed
from ..exchange.bybit import BybitClient
from ..monitoring import MetricsCollector, TradeRecord
from ..risk.manager import PositionSnapshot, RiskManager
from ..strategy.scalper import Scalper, timeframe_to_pandas_freq, Side


def _bybit_interval_from_timeframe(timeframe: str) -> str:
    tf = timeframe.strip().lower()
    mapping = {
        "1m": "1",
        "3m": "3",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "4h": "240",
        "1d": "D",
    }
    if tf not in mapping:
        raise ValueError(f"Unsupported timeframe for Bybit feed: {timeframe}")
    return mapping[tf]


@dataclass
class ActiveTrade:
    side: Side
    entry_time: pd.Timestamp
    entry_price: float
    qty_total: float
    qty_open: float
    stop_price: float
    risk_per_unit: float
    partial_taken: bool
    last_bar_ts: pd.Timestamp
    bars_held: int = 0
    realized_pnl: float = 0.0
    realized_qty: float = 0.0
    avg_exit_price: float = 0.0


@dataclass
class Trader:
    cfg: AppConfig
    metrics: Optional[MetricsCollector] = None
    feed: Optional[DataFeed] = None
    risk_manager: Optional[RiskManager] = None
    client: Optional[BybitClient] = None

    strategy: Scalper = field(init=False)
    poll_interval: int = field(init=False)
    position: PositionSnapshot = field(init=False)
    last_signal_ts: Optional[pd.Timestamp] = field(default=None, init=False)
    active_trade: Optional[ActiveTrade] = field(default=None, init=False)
    time_stop_bars: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.log = logging.getLogger("trader")
        self.client = self.client or BybitClient(
            testnet=self.cfg.execution.testnet,
            max_live_order_notional_krw=self.cfg.risk.max_live_order_notional_krw,
            usdt_krw_rate=self.cfg.risk.usdt_krw_rate,
        )
        self._apply_margin_settings()
        interval = _bybit_interval_from_timeframe(self.cfg.strategy.timeframe_entry)
        self.feed = self.feed or BybitDataFeed(
            self.client,
            symbol=self.cfg.symbol,
            category=self.cfg.category,
            interval=interval,
        )
        self.strategy = Scalper(self.cfg.strategy)
        self.risk_manager = self.risk_manager or RiskManager(self.cfg)
        self.poll_interval = max(int(self.cfg.execution.poll_interval_seconds), 1)
        self.position = PositionSnapshot(qty=0.0, entry_price=0.0, mark_price=0.0)
        self.time_stop_bars = self._time_stop_bars(self.cfg.strategy.time_stop_minutes)

        if self.metrics is not None:
            self.metrics.set_equity(self.cfg.risk.starting_equity)
            self.metrics.update_position(
                qty=self.position.qty,
                entry_price=self.position.entry_price,
                mark_price=self.position.mark_price,
            )

    def run(self) -> None:
        self.log.info(
            "starting trader loop mode=%s symbol=%s testnet=%s interval=%ss",
            self.cfg.mode,
            self.cfg.symbol,
            self.cfg.execution.testnet,
            self.poll_interval,
        )
        try:
            while True:
                self.step()
                self.risk_manager.tick_cooldown()
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            self.log.info("Received interrupt, shutting down")

    def step(self) -> None:
        self._sync_account_state()
        self._reconcile_active_trade()
        if self.metrics is not None:
            self.metrics.record_loop(time.time())
        try:
            candles = self.feed.fetch(self.cfg.execution.lookback_candles)
        except Exception as exc:  # pragma: no cover - network failure
            self.log.exception("Failed to fetch candles: %s", exc)
            if self.metrics is not None:
                self.metrics.record_error("fetch")
            return

        if candles.empty:
            self.log.debug("No candle data returned")
            return
        if len(candles) < self.cfg.strategy.ema_slow:
            self.log.debug(
                "Insufficient candles (%s) for EMA slow=%s",
                len(candles),
                self.cfg.strategy.ema_slow,
            )
            return

        try:
            features = self.strategy.compute_features(candles)
        except Exception as exc:
            self.log.exception("Failed to compute features: %s", exc)
            return

        mark_price = float(candles["close"].iloc[-1])
        last_ts = candles.index[-1]
        if self.metrics is not None:
            self.metrics.record_candle(last_ts.timestamp())
        self._update_metrics(mark_price)

        self.time_stop_bars = self._time_stop_bars(self.cfg.strategy.time_stop_minutes)

        if self._handle_active_position(features, mark_price):
            return

        if self.position.qty > 0 or (
            self.active_trade is not None and self.active_trade.qty_open > 0
        ):
            return

        signal = self.strategy.generate_signal(features)
        if signal is None:
            return

        if self.metrics is not None:
            self.metrics.record_signal(
                timestamp=signal.timestamp.timestamp(), side=signal.side
            )
        if self.last_signal_ts is not None and signal.timestamp <= self.last_signal_ts:
            return

        cooldown_active = self.risk_manager.is_cooldown_active()
        if self.risk_manager.daily_stop_active():
            self.log.info(
                "Daily stop hit (R=%.2f); blocking new entries",
                self.risk_manager.daily_r_total(),
            )
            return
        if not self.risk_manager.can_open_position(
            existing_position=self.position, cooldown_active=cooldown_active
        ):
            self.log.debug("Risk manager blocked new position")
            return

        try:
            qty = self.risk_manager.position_size(
                entry_price=signal.entry_price,
                stop_price=signal.stop_price,
                side=signal.side,
            )
        except ValueError as exc:
            self.log.warning("Invalid signal sizing: %s", exc)
            return

        limit_qty = self.risk_manager.max_order_qty(signal.entry_price)
        if limit_qty is not None and limit_qty > 0 and qty > limit_qty:
            self.log.debug(
                "Capping qty from %.6f to %.6f due to notional limit",
                qty,
                limit_qty,
            )
            qty = limit_qty

        quantized_qty = self._quantize_qty(qty)
        if quantized_qty <= 0:
            self.log.debug(
                "Calculated qty %.6f below minimum %.6f or step %.6f",
                qty,
                self.cfg.execution.min_qty,
                self.cfg.execution.qty_step,
            )
            return
        qty = quantized_qty

        stop_distance = abs(signal.entry_price - signal.stop_price)

        try:
            order = self.client.place_market_order(
                symbol=self.cfg.symbol,
                side=signal.side,
                qty=qty,
                category=self.cfg.category,
                position_idx=self._position_idx(signal.side, reduce_only=False),
            )
            signed_qty = qty if signal.side == "Buy" else -qty
            self.position = PositionSnapshot(
                qty=signed_qty,
                entry_price=signal.entry_price,
                mark_price=mark_price,
            )
            self.last_signal_ts = signal.timestamp
            self.risk_manager.trigger_cooldown()
            trade = ActiveTrade(
                side=signal.side,
                entry_time=signal.timestamp,
                entry_price=signal.entry_price,
                qty_total=qty,
                qty_open=qty,
                stop_price=signal.stop_price,
                risk_per_unit=stop_distance,
                partial_taken=False,
                last_bar_ts=last_ts,
            )
            self.active_trade = trade
            self._hydrate_active_trade()
            self._update_metrics(mark_price)
            self.log.info(
                "Placed %s order qty=%.6f entry=%.2f stop=%.2f order_id=%s",
                signal.side,
                qty,
                signal.entry_price,
                signal.stop_price,
                order.order_id,
            )
        except Exception as exc:  # pragma: no cover - network failure
            self.log.exception("Order placement failed: %s", exc)
            if self.metrics is not None:
                self.metrics.record_error("order")
            self.risk_manager.trigger_cooldown()

    def _sync_account_state(self) -> None:
        try:
            equity = self.client.get_wallet_equity()
            if equity is not None:
                self.risk_manager.update_equity(equity)
                if self.metrics is not None:
                    self.metrics.set_equity(equity)
        except Exception as exc:  # pragma: no cover - network failure
            self.log.debug("Failed to pull wallet balance: %s", exc)

        try:
            snapshot = self.client.get_linear_position_snapshot(
                symbol=self.cfg.symbol, category=self.cfg.category
            )
            qty_raw = float(snapshot.get("qty", 0.0))
            if self.active_trade is not None and qty_raw != 0.0:
                direction = 1.0 if self.active_trade.side == "Buy" else -1.0
                qty = abs(qty_raw) * direction
            else:
                qty = qty_raw
            self.position = PositionSnapshot(
                qty=qty,
                entry_price=float(snapshot.get("entry_price", 0.0)),
                mark_price=float(snapshot.get("mark_price", 0.0)),
            )
            self._update_metrics(self.position.mark_price)
        except Exception as exc:  # pragma: no cover - network failure
            self.log.debug("Failed to fetch position snapshot: %s", exc)

    def _reconcile_active_trade(self) -> None:
        if self.active_trade is None:
            return
        if self.position.qty == 0:
            self.active_trade = None
            return
        self.active_trade.qty_open = abs(self.position.qty)

    def _handle_active_position(
        self, features: pd.DataFrame, mark_price: float
    ) -> bool:
        if self.active_trade is None:
            return False
        if self.position.qty == 0 or self.active_trade.qty_open <= 0:
            self.active_trade = None
            return False

        trade = self.active_trade
        trade.qty_open = abs(self.position.qty)
        latest = features.iloc[-1]
        timestamp = features.index[-1]
        atr = float(latest.get("atr", 0.0) or 0.0)
        high = float(latest["high"])
        low = float(latest["low"])

        if timestamp > trade.last_bar_ts:
            trade.last_bar_ts = timestamp
            trade.bars_held += 1

        side = trade.side
        stop_hit = (
            low <= trade.stop_price if side == "Buy" else high >= trade.stop_price
        )

        if stop_hit and trade.qty_open > 0:
            closed_qty = trade.qty_open
            stop_level = trade.stop_price
            if self._submit_exit_order(
                closed_qty, "stop_loss", mark_price, stop_level
            ):
                self._finalize_trade_close(mark_price, timestamp=timestamp)
                self.log.info(
                    "Closed %s via stop qty=%.6f stop=%.2f",
                    side.lower(),
                    closed_qty,
                    stop_level,
                )
                return True
            return False

        if not trade.partial_taken and trade.qty_open > 0:
            partial_mul = self.cfg.strategy.partial_tp_r
            if partial_mul > 0:
                if side == "Buy":
                    partial_target = trade.entry_price + (
                        trade.risk_per_unit * partial_mul
                    )
                    hit_target = high >= partial_target
                else:
                    partial_target = trade.entry_price - (
                        trade.risk_per_unit * partial_mul
                    )
                    hit_target = low <= partial_target

                if hit_target:
                    qty_partial = min(trade.qty_total * 0.5, trade.qty_open)
                    qty_partial = self._quantize_qty(qty_partial)
                    if qty_partial <= 0:
                        self.log.debug(
                            "Skipping partial: qty below minimum or step (min=%.6f step=%.6f)",
                            self.cfg.execution.min_qty,
                            self.cfg.execution.qty_step,
                        )
                    elif self._submit_exit_order(
                        qty_partial,
                        "partial_take_profit",
                        mark_price,
                        partial_target,
                    ):
                        trade.partial_taken = True
                        trade.qty_open = abs(self.position.qty)
                        if trade.qty_open <= 0:
                            self._finalize_trade_close(mark_price, timestamp=timestamp)
                            self.log.info(
                                "Closed %s via partial TP qty=%.6f target=%.2f",
                                side.lower(),
                                qty_partial,
                                partial_target,
                            )
                            return True
                        if side == "Buy":
                            trade.stop_price = max(
                                trade.stop_price, trade.entry_price
                            )
                        else:
                            trade.stop_price = min(
                                trade.stop_price, trade.entry_price
                            )
                        self._update_metrics(mark_price)
                        self.log.info(
                            "Partial TP filled qty=%.6f target=%.2f remaining=%.6f",
                            qty_partial,
                            partial_target,
                            trade.qty_open,
                        )
                    else:
                        return False

        if trade.partial_taken and trade.qty_open > 0 and atr > 0:
            if side == "Buy":
                trail_stop = high - atr * self.cfg.strategy.trail_atr_mult
                if trail_stop > trade.stop_price:
                    trade.stop_price = trail_stop
                stop_hit = low <= trade.stop_price
            else:
                trail_stop = low + atr * self.cfg.strategy.trail_atr_mult
                if trail_stop < trade.stop_price:
                    trade.stop_price = trail_stop
                stop_hit = high >= trade.stop_price

            if stop_hit:
                closed_qty = trade.qty_open
                trail_level = trade.stop_price
                if self._submit_exit_order(
                    closed_qty, "trail_stop", mark_price, trail_level
                ):
                    self._finalize_trade_close(mark_price, timestamp=timestamp)
                    self.log.info(
                        "Closed %s via trail stop qty=%.6f stop=%.2f",
                        side.lower(),
                        closed_qty,
                        trail_level,
                    )
                    return True
                return False

        if trade.qty_open > 0 and trade.bars_held >= self.time_stop_bars:
            closed_qty = trade.qty_open
            bars = trade.bars_held
            if self._submit_exit_order(
                closed_qty, "time_stop", mark_price, mark_price
            ):
                self._finalize_trade_close(mark_price, timestamp=timestamp)
                self.log.info(
                    "Closed %s via time stop qty=%.6f bars=%s",
                    side.lower(),
                    closed_qty,
                    bars,
                )
                return True
            return False

        return False

    def _hydrate_active_trade(self) -> None:
        trade = self.active_trade
        if trade is None:
            return
        try:
            self._sync_account_state()
        except Exception:
            return

        qty_abs = abs(self.position.qty)
        if qty_abs <= 0:
            return

        qty_signed = qty_abs if trade.side == "Buy" else -qty_abs

        entry_price = self.position.entry_price or trade.entry_price
        mark_price = self.position.mark_price or entry_price
        risk_per_unit = trade.risk_per_unit or abs(trade.entry_price - trade.stop_price)
        if risk_per_unit <= 0:
            risk_per_unit = abs(entry_price - trade.stop_price)
        if risk_per_unit <= 0:
            risk_per_unit = 1e-8

        trade.entry_price = entry_price
        trade.qty_total = qty_abs
        trade.qty_open = qty_abs
        trade.stop_price = self._stop_from_entry(trade.side, entry_price, risk_per_unit)
        trade.risk_per_unit = abs(entry_price - trade.stop_price)

        self.position = PositionSnapshot(
            qty=qty_signed,
            entry_price=entry_price,
            mark_price=mark_price,
        )

    def _submit_exit_order(
        self, qty: float, reason: str, mark_price: float, exit_price: float
    ) -> bool:
        if qty <= 0:
            return True
        current_qty = self.position.qty
        abs_qty = abs(qty)
        quantized_qty = self._quantize_qty(abs_qty)
        if quantized_qty <= 0:
            self.log.debug(
                "Exit %s skipped: qty %.6f below minimum %.6f or step %.6f",
                reason,
                abs_qty,
                self.cfg.execution.min_qty,
                self.cfg.execution.qty_step,
            )
            return True
        signed_qty = quantized_qty if current_qty >= 0 else -quantized_qty
        exit_side = "Sell" if signed_qty > 0 else "Buy"
        try:
            order = self.client.close_position_market(
                symbol=self.cfg.symbol,
                category=self.cfg.category,
                qty=signed_qty,
                reduce_only=self.cfg.execution.reduce_only_exits,
                position_idx=self._position_idx(exit_side, reduce_only=True),
            )
            if current_qty >= 0:
                remaining_qty = max(current_qty - quantized_qty, 0.0)
            else:
                remaining_qty = min(current_qty + quantized_qty, 0.0)
            remaining_qty = round(remaining_qty, 12)
            self.position = PositionSnapshot(
                qty=remaining_qty,
                entry_price=self.position.entry_price,
                mark_price=mark_price,
            )
            self._update_metrics(mark_price)
            if self.active_trade is not None:
                self._register_realized_pnl(
                    self.active_trade, quantized_qty, exit_price
                )
            self.log.info(
                "Exit order placed reason=%s qty=%.6f order_id=%s",
                reason,
                quantized_qty,
                order.order_id,
            )
            return True
        except Exception as exc:  # pragma: no cover - network failure
            self.log.exception("Failed to close position (%s): %s", reason, exc)
            if self.metrics is not None:
                self.metrics.record_error("exit")
            return False

    def _finalize_trade_close(
        self, mark_price: float, *, timestamp: Optional[pd.Timestamp] = None
    ) -> None:
        trade = self.active_trade
        realized_r = None
        exit_price = mark_price
        qty_closed = 0.0
        side: Optional[Side] = None

        if trade is not None:
            risk_amount = trade.risk_per_unit * trade.qty_total
            if risk_amount > 0:
                realized_r = trade.realized_pnl / risk_amount
            exit_price = trade.avg_exit_price or mark_price
            qty_closed = trade.realized_qty or trade.qty_total
            side = trade.side

        self.active_trade = None
        self.position = PositionSnapshot(qty=0.0, entry_price=0.0, mark_price=mark_price)
        self._update_metrics(mark_price)
        self.risk_manager.trigger_cooldown()

        if realized_r is not None:
            self.risk_manager.record_realized_r(realized_r)

        if (
            trade is not None
            and self.metrics is not None
            and side is not None
            and qty_closed > 0
        ):
            trade_ts = (
                timestamp.to_pydatetime()
                if timestamp is not None
                else datetime.now(timezone.utc)
            )
            record = TradeRecord(
                timestamp=trade_ts,
                side=side,
                qty=qty_closed,
                pnl=trade.realized_pnl,
                entry_price=trade.entry_price,
                exit_price=exit_price,
                r_multiple=realized_r or 0.0,
            )
            self.metrics.record_trade(record)

    def _register_realized_pnl(
        self, trade: ActiveTrade, qty_closed: float, exit_price: float
    ) -> None:
        if qty_closed <= 0:
            return
        direction = 1.0 if trade.side == "Buy" else -1.0
        pnl_segment = (exit_price - trade.entry_price) * qty_closed * direction
        trade.realized_pnl += pnl_segment
        prev_qty = trade.realized_qty
        total_qty = prev_qty + qty_closed
        if total_qty > 0:
            trade.avg_exit_price = (
                (trade.avg_exit_price * prev_qty) + (exit_price * qty_closed)
            ) / total_qty
            trade.realized_qty = total_qty

    def _time_stop_bars(self, time_stop_minutes: int) -> int:
        entry_freq = self.cfg.strategy.timeframe_entry
        try:
            delta = pd.Timedelta(timeframe_to_pandas_freq(entry_freq))
        except ValueError:
            self.log.error(
                "Invalid timeframe_entry=%s; defaulting time stop bars to 1",
                entry_freq,
            )
            return 1
        if delta.total_seconds() <= 0:
            raise ValueError("entry timeframe must map to a positive duration")
        bars = max(int((time_stop_minutes * 60) / delta.total_seconds()), 1)
        return bars

    def _update_metrics(self, mark_price: float) -> None:
        if self.metrics is None:
            return
        self.metrics.update_position(
            qty=self.position.qty,
            entry_price=self.position.entry_price,
            mark_price=mark_price,
        )

    def _quantize_qty(self, qty: float) -> float:
        """Snap raw quantity to the configured lot size and minimum."""
        if qty <= 0:
            return 0.0
        step = self.cfg.execution.qty_step
        min_qty = self.cfg.execution.min_qty
        if step is not None and step > 0:
            scaled = math.floor((qty + 1e-12) / step)
            qty = scaled * step
        qty = round(qty, 12)
        if qty < min_qty:
            return 0.0
        return qty

    @staticmethod
    def _stop_from_entry(side: Side, entry_price: float, risk_per_unit: float) -> float:
        if risk_per_unit <= 0:
            return entry_price
        if side == "Buy":
            return max(entry_price - risk_per_unit, 0.0)
        return entry_price + risk_per_unit

    def _apply_margin_settings(self) -> None:
        if self.client is None:
            return
        configure = getattr(self.client, "configure_margin_and_leverage", None)
        if configure is None:
            return
        try:
            configure(
                category=self.cfg.category,
                symbol=self.cfg.symbol,
                margin_mode=self.cfg.execution.margin_mode,
                leverage=self.cfg.execution.leverage,
                position_mode=self.cfg.execution.position_mode,
            )
        except Exception as exc:  # pragma: no cover - network failure
            self.log.warning("Failed to configure margin/leverage: %s", exc)

    def _position_idx(self, side: str, *, reduce_only: bool) -> Optional[int]:
        mode = (self.cfg.execution.position_mode or "").upper()
        if mode in {"ONE_WAY", "UNIFIED", "MERGED", ""}:
            return 0
        if mode in {"HEDGE", "HEDGE_MODE", "TWO_WAY"}:
            if reduce_only:
                current_qty = self.position.qty
                if current_qty < 0:
                    return 2
                return 1
            return 1 if side.upper() == "BUY" else 2
        return None
