"""
FRIDAY — performance_tracker.py
=================================
Maintains a rolling JSON trade log and calculates per-pattern Win/Loss ratios.
Uses TinyDB for lightweight, file-backed persistence.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Optional

from config import TRADE_LOG_PATH, DATA_DIR, MIN_WIN_RATE

logger = logging.getLogger("FRIDAY.PerformanceTracker")


class PerformanceTracker:
    """
    Persists suggested trades to a local JSON log.
    Tracks outcomes and calculates per-pattern and global win/loss ratios.
    """

    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self._log_path = TRADE_LOG_PATH
        self._trades: list = self._load()
        logger.info(f"PerformanceTracker: {len(self._trades)} historical trades loaded.")

    # ── I/O ──────────────────────────────────────────────────

    def _load(self) -> list:
        try:
            with open(self._log_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save(self):
        with open(self._log_path, "w") as f:
            json.dump(self._trades, f, indent=2)

    # ── Trade Logging ────────────────────────────────────────

    def log_signal(
        self,
        asset_key:       str,
        signal:          str,   # "LONG" | "SHORT" | "HOLD"
        composite_score: float,
        entry_price:     float,
        divergence_type: str,   # from 15m divergence
        wyckoff_phase:   str,
        sentiment_tone:  str,
    ) -> str:
        """Log a new trade suggestion. Returns trade_id."""
        trade_id = f"{asset_key}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        trade = {
            "trade_id":       trade_id,
            "asset_key":      asset_key,
            "signal":         signal,
            "composite_score":composite_score,
            "entry_price":    entry_price,
            "divergence_type":divergence_type,
            "wyckoff_phase":  wyckoff_phase,
            "sentiment_tone": sentiment_tone,
            "entry_time":     datetime.now().isoformat(),
            "exit_price":     None,
            "exit_time":      None,
            "outcome":        "OPEN",   # "WIN" | "LOSS" | "OPEN"
            "pnl_pct":        None,
        }
        self._trades.append(trade)
        self._save()
        logger.info(f"Logged trade {trade_id}: {signal} @ {entry_price}")
        return trade_id

    def close_trade(
        self,
        trade_id:   str,
        exit_price: float,
        outcome:    str,        # "WIN" | "LOSS"
    ) -> bool:
        """Update an open trade with its outcome."""
        for t in self._trades:
            if t["trade_id"] == trade_id and t["outcome"] == "OPEN":
                t["exit_price"] = exit_price
                t["exit_time"]  = datetime.now().isoformat()
                t["outcome"]    = outcome
                if t["entry_price"] and exit_price:
                    direction = 1 if t["signal"] == "LONG" else -1
                    t["pnl_pct"] = direction * (
                        (exit_price - t["entry_price"]) / t["entry_price"]
                    ) * 100
                self._save()
                return True
        return False

    # ── Win/Loss Statistics ──────────────────────────────────

    def global_win_rate(self) -> float:
        """Overall win rate across all closed trades."""
        closed = [t for t in self._trades if t["outcome"] in ("WIN", "LOSS")]
        if not closed:
            return 0.50   # default neutral
        wins = sum(1 for t in closed if t["outcome"] == "WIN")
        return wins / len(closed)

    def pattern_win_rate(self, divergence_type: str) -> float:
        """Win rate for a specific divergence pattern."""
        pattern_trades = [
            t for t in self._trades
            if t.get("divergence_type") == divergence_type
            and t["outcome"] in ("WIN", "LOSS")
        ]
        if len(pattern_trades) < 3:
            return 0.50   # not enough data
        wins = sum(1 for t in pattern_trades if t["outcome"] == "WIN")
        return wins / len(pattern_trades)

    def asset_win_rate(self, asset_key: str) -> float:
        """Win rate for a specific asset."""
        asset_trades = [
            t for t in self._trades
            if t["asset_key"] == asset_key
            and t["outcome"] in ("WIN", "LOSS")
        ]
        if not asset_trades:
            return 0.50
        wins = sum(1 for t in asset_trades if t["outcome"] == "WIN")
        return wins / len(asset_trades)

    def get_confidence_multiplier(self, divergence_type: str) -> float:
        """
        Returns a confidence multiplier [0.5, 1.0] based on pattern win rate.
        If a pattern's win rate drops below MIN_WIN_RATE (45%), confidence is penalised.
        """
        rate = self.pattern_win_rate(divergence_type)
        if rate < MIN_WIN_RATE:
            # Linear scaling: 45% win rate → 0.5x, 0% win rate → 0.5x
            return max(0.5, rate / MIN_WIN_RATE)
        return 1.0

    def summary(self) -> Dict:
        """Return summary statistics dict."""
        all_closed = [t for t in self._trades if t["outcome"] in ("WIN", "LOSS")]
        open_trades = [t for t in self._trades if t["outcome"] == "OPEN"]
        by_asset = {}
        for key in ("CRUDE_OIL", "GOLD", "SILVER"):
            by_asset[key] = {
                "win_rate": round(self.asset_win_rate(key), 3),
                "trades":   sum(1 for t in all_closed if t["asset_key"] == key),
            }
        return {
            "total_trades":  len(self._trades),
            "closed_trades": len(all_closed),
            "open_trades":   len(open_trades),
            "global_win_rate": round(self.global_win_rate(), 3),
            "by_asset": by_asset,
        }
