"""
Strategy Layer — entry/exit rules, position sizing, stop-loss management.
"""
from __future__ import annotations
import uuid
import time
from typing import Dict, List, Optional
from utils import config
from utils.logger import get_logger

log = get_logger(__name__)


class Position:
    def __init__(
        self,
        symbol:       str,
        side:         str,    # LONG | SHORT
        entry_price:  float,
        units:        float,
        stop_price:   float,
        target_price: float,
        entry_bar:    int,
        signal:       Dict,
    ) -> None:
        self.id           = str(uuid.uuid4())[:8]
        self.symbol       = symbol
        self.side         = side
        self.entry_price  = entry_price
        self.units        = units
        self.stop_price   = stop_price
        self.target_price = target_price
        self.entry_bar    = entry_bar
        self.signal       = signal
        self.entry_ts     = int(time.time() * 1000)

    def unrealised_pnl(self, price: float) -> float:
        if self.side == "LONG":
            return (price - self.entry_price) * self.units
        return (self.entry_price - price) * self.units

    def as_dict(self) -> Dict:
        return {
            "id": self.id, "symbol": self.symbol, "side": self.side,
            "entry_price": self.entry_price, "units": self.units,
            "stop_price":  self.stop_price,  "target_price": self.target_price,
        }


class TradingStrategy:

    def __init__(self, initial_capital: float = config.INITIAL_CAPITAL) -> None:
        self.portfolio_value  = initial_capital
        self.open_positions:  List[Position] = []
        self.trade_history:   List[Dict]     = []
        self.cooldown_counter = 0
        self.bar_index        = 0

    # ── Entry ─────────────────────────────────────────────────────────────

    def should_enter(self, signal: Dict, feat: Dict) -> bool:
        if signal["action"] == "NEUTRAL":
            return False
        if signal["confidence"] < config.MIN_CONFIDENCE:
            return False
        if len(self.open_positions) >= config.MAX_POSITIONS:
            return False
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False
        if any(p.symbol == signal["symbol"] for p in self.open_positions):
            return False
        # Trend alignment with EMA-50
        ema50 = feat.get("ema_50", signal["price"])
        price = signal["price"]
        if signal["action"] == "BUY"  and price < ema50:
            return False
        if signal["action"] == "SELL" and price > ema50:
            return False
        return True

    def compute_position_size(self, signal: Dict, feat: Dict) -> float:
        atr        = feat.get("atr_14", signal["price"] * 0.01)
        stop_dist  = atr * config.STOP_ATR_MULT
        if stop_dist <= 0:
            return 0.0
        risk_amount = self.portfolio_value * config.MAX_RISK_PER_TRADE
        units       = risk_amount / stop_dist
        max_units   = (self.portfolio_value * 0.25) / (signal["price"] + 1e-8)
        return min(units, max_units)

    def enter(self, signal: Dict, feat: Dict) -> Optional[Position]:
        atr   = feat.get("atr_14", signal["price"] * 0.01)
        units = self.compute_position_size(signal, feat)
        if units <= 0:
            return None

        if signal["action"] == "BUY":
            stop   = signal["price"] - atr * config.STOP_ATR_MULT
            target = signal["price"] + atr * config.TP_ATR_MULT
            side   = "LONG"
        else:
            stop   = signal["price"] + atr * config.STOP_ATR_MULT
            target = signal["price"] - atr * config.TP_ATR_MULT
            side   = "SHORT"

        pos = Position(signal["symbol"], side, signal["price"],
                       units, stop, target, self.bar_index, signal)
        self.open_positions.append(pos)
        log.info("ENTER %s %s @ %.4f | stop=%.4f | target=%.4f | units=%.4f",
                 side, signal["symbol"], signal["price"], stop, target, units)
        return pos

    # ── Exit ──────────────────────────────────────────────────────────────

    def check_exits(self, price: float, current_signal: Optional[Dict] = None) -> List[Dict]:
        closed = []
        for pos in list(self.open_positions):
            reason = None
            if pos.side == "LONG":
                if price <= pos.stop_price:   reason = "STOP_LOSS"
                elif price >= pos.target_price: reason = "TAKE_PROFIT"
            else:
                if price >= pos.stop_price:   reason = "STOP_LOSS"
                elif price <= pos.target_price: reason = "TAKE_PROFIT"

            bars_held = self.bar_index - pos.entry_bar
            if bars_held >= config.MAX_BARS_HELD:
                reason = "TIMEOUT"

            if current_signal and current_signal["confidence"] > 0.70:
                if pos.side == "LONG"  and current_signal["action"] == "SELL": reason = "SIGNAL_REVERSAL"
                if pos.side == "SHORT" and current_signal["action"] == "BUY":  reason = "SIGNAL_REVERSAL"

            if reason:
                trade = self._close(pos, price, reason)
                closed.append(trade)
        return closed

    def _close(self, pos: Position, exit_price: float, reason: str) -> Dict:
        exit_p = exit_price * (1 + config.SLIPPAGE_PCT * (1 if pos.side == "SHORT" else -1))
        if pos.side == "LONG":
            pnl = (exit_p - pos.entry_price) * pos.units
        else:
            pnl = (pos.entry_price - exit_p) * pos.units

        commission = abs(exit_p * pos.units * config.COMMISSION_PCT)
        net_pnl    = pnl - commission
        self.portfolio_value += net_pnl

        if net_pnl < 0:
            self.cooldown_counter = config.COOLDOWN_BARS

        trade = {
            **pos.as_dict(),
            "exit_price":  round(exit_p, 6),
            "exit_reason": reason,
            "pnl":         round(net_pnl, 4),
            "pnl_pct":     round(net_pnl / (pos.entry_price * pos.units + 1e-8) * 100, 4),
            "bars_held":   self.bar_index - pos.entry_bar,
        }
        self.open_positions.remove(pos)
        self.trade_history.append(trade)
        log.info("EXIT %s %s @ %.4f (%s) PnL=%.4f", pos.side, pos.symbol, exit_p, reason, net_pnl)
        return trade

    def update_trailing_stop(self, pos: Position, price: float, atr: float) -> None:
        if pos.side == "LONG":
            candidate = price - atr * config.STOP_ATR_MULT
            if candidate > pos.stop_price:
                pos.stop_price = candidate
        else:
            candidate = price + atr * config.STOP_ATR_MULT
            if candidate < pos.stop_price:
                pos.stop_price = candidate

    def mark_to_market(self, price: float) -> float:
        return self.portfolio_value + sum(p.unrealised_pnl(price) for p in self.open_positions)
