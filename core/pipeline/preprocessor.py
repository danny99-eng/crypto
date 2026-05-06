"""
Data Pipeline — validates, gap-fills, normalises, and splits OHLCV candles.
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple
from utils import config
from utils.logger import get_logger
from utils.math_helpers import rolling_zscore

log = get_logger(__name__)
Candle = Dict[str, float]


# ── Step 1: Validate ──────────────────────────────────────────────────────

def validate(candles: List[Candle]) -> List[Candle]:
    clean = []
    for c in candles:
        issues = []
        for field in ("open", "high", "low", "close", "volume"):
            if c.get(field) is None or (isinstance(c[field], float) and np.isnan(c[field])):
                issues.append(f"null_{field}")
        if not issues:
            if c["high"] < c["low"]:
                issues.append("high_lt_low")
            if c["close"] <= 0 or c["open"] <= 0:
                issues.append("non_positive_price")
            if c["volume"] < 0:
                issues.append("negative_volume")
            if c["open"] > 0:
                pct = abs(c["close"] - c["open"]) / c["open"]
                if pct > 0.40:
                    issues.append("spike")
        if issues:
            log.debug("Discarding candle ts=%s: %s", c.get("timestamp"), issues)
        else:
            clean.append(c)
    log.info("Validation: %d → %d candles", len(candles), len(clean))
    return clean


# ── Step 2: Gap-fill ──────────────────────────────────────────────────────

def fill_gaps(candles: List[Candle], interval: str) -> List[Candle]:
    if not candles:
        return candles
    interval_ms = config.TIMEFRAMES.get(interval, 3_600_000)
    sorted_c    = sorted(candles, key=lambda c: c["timestamp"])
    filled      = [sorted_c[0]]
    expected    = sorted_c[0]["timestamp"] + interval_ms

    for c in sorted_c[1:]:
        while expected < c["timestamp"]:
            prev = filled[-1]
            filled.append({
                **prev,
                "timestamp": expected,
                "open":   prev["close"],
                "high":   prev["close"],
                "low":    prev["close"],
                "volume": 0.0,
                "synthetic": True,
            })
            expected += interval_ms
        filled.append(c)
        expected = c["timestamp"] + interval_ms

    n_synthetic = sum(1 for c in filled if c.get("synthetic"))
    log.info("Gap-fill: %d → %d candles (%d synthetic)", len(candles), len(filled), n_synthetic)
    return filled


# ── Step 3: Normalise (rolling z-score, no look-ahead) ───────────────────

def normalise(candles: List[Candle], window: int = 200) -> List[Candle]:
    closes  = np.array([c["close"]  for c in candles], dtype=float)
    volumes = np.array([c["volume"] for c in candles], dtype=float)

    z_close  = rolling_zscore(closes,  window)
    z_volume = rolling_zscore(volumes, window)

    for i, c in enumerate(candles):
        c["close_z"]  = float(z_close[i])
        c["volume_z"] = float(z_volume[i])
    return candles


# ── Step 4: Temporal train/val/test split ─────────────────────────────────

def temporal_split(
    candles: List[Candle],
    train: float = 0.70,
    val:   float = 0.15,
) -> Dict[str, List[Candle]]:
    n       = len(candles)
    i_train = int(n * train)
    i_val   = int(n * (train + val))
    split = {
        "train": candles[:i_train],
        "val":   candles[i_train:i_val],
        "test":  candles[i_val:],
    }
    for k, v in split.items():
        log.info("Split %s: %d candles", k, len(v))
    return split


# ── Master runner ─────────────────────────────────────────────────────────

def run(
    raw_candles: List[Candle],
    interval: str = "1h",
) -> Dict[str, List[Candle]]:
    s1 = validate(raw_candles)
    s2 = fill_gaps(s1, interval)
    s3 = normalise(s2)
    return temporal_split(s3)
