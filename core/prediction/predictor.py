"""
Prediction Engine — combines Baseline + LSTM via weighted ensemble,
adds reversal zone detection and confidence gating.
"""
from __future__ import annotations
import json
import os
import numpy as np
import time
from typing import Dict, List, Optional
from utils import config
from utils.logger import get_logger
from utils.math_helpers import sigmoid
from core.models.baseline import LogisticRegression
from core.models.lstm import LSTM

log = get_logger(__name__)


class PredictionEngine:

    def __init__(
        self,
        baseline: LogisticRegression,
        lstm:     Optional[LSTM],
        mu:       np.ndarray,
        sigma:    np.ndarray,
        feature_cols: List[str],
    ) -> None:
        self.baseline     = baseline
        self.lstm         = lstm
        self.mu           = mu
        self.sigma        = sigma
        self.feature_cols = feature_cols

    # ── Normalise a single feature row ────────────────────────────────────

    def _norm(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mu) / (self.sigma + 1e-8)

    # ── Reversal zone detection ───────────────────────────────────────────

    def _detect_reversal(self, feat: Dict) -> Dict:
        signals = []
        rsi    = feat.get("rsi_14",   50.0)
        pct_b  = feat.get("bb_pct_b", 0.5)
        macd_h = feat.get("macd_hist", 0.0)

        if rsi >= config.REVERSAL_RSI_HIGH:
            signals.append({"type": "OVERBOUGHT_RSI",
                             "strength": round((rsi - config.REVERSAL_RSI_HIGH) / (100 - config.REVERSAL_RSI_HIGH), 3)})
        if rsi <= config.REVERSAL_RSI_LOW:
            signals.append({"type": "OVERSOLD_RSI",
                             "strength": round((config.REVERSAL_RSI_LOW - rsi) / config.REVERSAL_RSI_LOW, 3)})
        if pct_b >= config.REVERSAL_BB_HIGH:
            signals.append({"type": "BB_OVERBOUGHT",
                             "strength": round((pct_b - config.REVERSAL_BB_HIGH) / (1 - config.REVERSAL_BB_HIGH + 1e-8), 3)})
        if pct_b <= config.REVERSAL_BB_LOW:
            signals.append({"type": "BB_OVERSOLD",
                             "strength": round((config.REVERSAL_BB_LOW - pct_b) / (config.REVERSAL_BB_LOW + 1e-8), 3)})

        is_zone = len(signals) >= 2
        overbought_types = {"OVERBOUGHT_RSI", "BB_OVERBOUGHT"}
        oversold_types   = {"OVERSOLD_RSI",   "BB_OVERSOLD"}
        types = {s["type"] for s in signals}
        if types & overbought_types:
            rev_dir = "DOWN"
        elif types & oversold_types:
            rev_dir = "UP"
        else:
            rev_dir = "UNKNOWN"

        return {"is_reversal_zone": is_zone, "reversal_dir": rev_dir, "signals": signals}

    # ── Ensemble predict ──────────────────────────────────────────────────

    def generate_signal(
        self,
        feat_row:  Dict,
        sequence:  Optional[np.ndarray] = None,   # (seq_len, n_features) normalised
    ) -> Dict:
        # Build feature vector
        x = np.array([feat_row.get(c, 0.0) for c in self.feature_cols], dtype=float)
        x_norm = self._norm(x)

        # Baseline probability
        p_base = float(self.baseline.predict_proba(x_norm.reshape(1, -1))[0])

        # LSTM probability
        if self.lstm is not None and sequence is not None:
            p_lstm = float(self.lstm._forward_sequence(sequence, training=False))
        else:
            p_lstm = p_base   # fall back to baseline if no LSTM

        # Weighted ensemble
        p_ens  = config.WEIGHT_BASELINE * p_base + config.WEIGHT_LSTM * p_lstm
        conf   = abs(p_ens - 0.5) * 2

        # Direction
        direction = "UP" if p_ens > 0.5 else "DOWN"

        # Reversal zone
        reversal = self._detect_reversal(feat_row)

        # Dampen confidence if model and reversal conflict
        if reversal["is_reversal_zone"] and reversal["reversal_dir"] not in ("UNKNOWN", direction):
            conf *= 0.7

        # Gate on minimum confidence
        if conf < config.MIN_CONFIDENCE:
            action = "NEUTRAL"
        elif direction == "UP":
            action = "BUY"
        else:
            action = "SELL"

        return {
            "timestamp":    int(time.time() * 1000),
            "symbol":       feat_row.get("symbol", ""),
            "interval":     feat_row.get("interval", ""),
            "price":        round(feat_row.get("close", 0.0), 4),
            "action":       action,
            "direction":    direction,
            "probability":  round(float(p_ens), 4),
            "confidence":   round(float(conf), 4),
            "reversal_zone": reversal["is_reversal_zone"],
            "reversal_dir":  reversal["reversal_dir"],
            "reversal_signals": reversal["signals"],
            "model_outputs": {
                "baseline": round(p_base, 4),
                "lstm":     round(p_lstm, 4),
            },
            "indicators": {
                "rsi":    round(feat_row.get("rsi_14",    0.0), 2),
                "macd":   round(feat_row.get("macd",      0.0), 4),
                "macd_hist": round(feat_row.get("macd_hist", 0.0), 4),
                "atr":    round(feat_row.get("atr_14",    0.0), 4),
                "bb_pct": round(feat_row.get("bb_pct_b",  0.5), 4),
                "vol_delta": round(feat_row.get("vol_delta", 1.0), 4),
                "roc":    round(feat_row.get("roc_10",    0.0), 4),
                "ema_50": round(feat_row.get("ema_50",    0.0), 4),
            },
        }

    # ── Factory: load from saved models ──────────────────────────────────

    @classmethod
    def from_saved(cls, model_dir: str = config.MODEL_DIR) -> "PredictionEngine":
        import json
        baseline_path = os.path.join(model_dir, "baseline.pkl")
        lstm_path     = os.path.join(model_dir, "lstm.npz")
        norm_path     = os.path.join(model_dir, "norm_stats.npz")
        cols_path     = os.path.join(model_dir, "feature_cols.json")

        baseline = LogisticRegression.load(baseline_path)
        lstm     = LSTM.load(lstm_path) if os.path.exists(lstm_path) else None
        norm     = np.load(norm_path)
        with open(cols_path) as f:
            feature_cols = json.load(f)

        log.info("PredictionEngine loaded from %s", model_dir)
        return cls(baseline, lstm, norm["mu"], norm["sigma"], feature_cols)
