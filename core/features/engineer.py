"""
Feature Engineering — computes all technical indicators and creates labels.
Pure NumPy. Zero look-ahead bias.
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional
from utils import config
from utils.logger import get_logger

log = get_logger(__name__)
Candle  = Dict[str, float]
FeatRow = Dict[str, float]


# ═══════════════════════════════════════════════════════════════════════════
# Indicator functions
# ═══════════════════════════════════════════════════════════════════════════

def ema(series: np.ndarray, period: int) -> np.ndarray:
    k   = 2.0 / (period + 1)
    out = np.zeros_like(series)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = series[i] * k + out[i - 1] * (1 - k)
    return out


def sma(series: np.ndarray, period: int) -> np.ndarray:
    out = np.full_like(series, np.nan, dtype=float)
    for i in range(len(series)):
        sl     = series[max(0, i - period + 1): i + 1]
        out[i] = np.mean(sl)
    return out


def macd(
    closes: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fast_ema  = ema(closes, fast)
    slow_ema  = ema(closes, slow)
    macd_line = fast_ema - slow_ema
    sig_line  = ema(macd_line, signal)
    hist      = macd_line - sig_line
    return macd_line, sig_line, hist


def rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    delta  = np.diff(closes, prepend=closes[0])
    gains  = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_g  = sma(gains,  period)
    avg_l  = sma(losses, period)
    rs     = np.where(avg_l == 0, 1e8, avg_g / (avg_l + 1e-12))
    return 100.0 - (100.0 / (1.0 + rs))


def bollinger_bands(
    closes: np.ndarray,
    period: int = 20,
    n_std:  float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mid   = sma(closes, period)
    std   = np.array([
        np.std(closes[max(0, i - period + 1): i + 1], ddof=min(1, max(0, i)))
        for i in range(len(closes))
    ], dtype=float)
    std   = np.nan_to_num(std, nan=0.0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    band_w = upper - lower
    pct_b  = np.where(band_w == 0, 0.5, (closes - lower) / (band_w + 1e-12))
    return upper, mid, lower, pct_b


def atr(
    highs: np.ndarray,
    lows:  np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    tr = np.maximum(
        highs - lows,
        np.maximum(
            np.abs(highs  - np.roll(closes, 1)),
            np.abs(lows   - np.roll(closes, 1)),
        ),
    )
    tr[0] = highs[0] - lows[0]
    return sma(tr, period)


def on_balance_volume(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    direction = np.sign(np.diff(closes, prepend=closes[0]))
    return np.cumsum(direction * volumes)


def rate_of_change(series: np.ndarray, period: int = 10) -> np.ndarray:
    out = np.full_like(series, np.nan, dtype=float)
    for i in range(period, len(series)):
        if series[i - period] != 0:
            out[i] = (series[i] - series[i - period]) / series[i - period] * 100
    return out


def volume_delta(volumes: np.ndarray, period: int = 20) -> np.ndarray:
    avg = sma(volumes, period)
    return np.where(avg == 0, 1.0, volumes / (avg + 1e-12))


# ═══════════════════════════════════════════════════════════════════════════
# Labels
# ═══════════════════════════════════════════════════════════════════════════

def create_labels(
    closes: np.ndarray,
    horizon:   int   = 5,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Returns: 1 (UP), 0 (DOWN), or np.nan (neutral / unknown).
    """
    labels = np.full(len(closes), np.nan)
    for i in range(len(closes) - horizon):
        fut_ret = (closes[i + horizon] - closes[i]) / closes[i] * 100
        if fut_ret > threshold:
            labels[i] = 1.0
        elif fut_ret < -threshold:
            labels[i] = 0.0
        # else stays NaN → will be filtered out
    return labels


# ═══════════════════════════════════════════════════════════════════════════
# Master builder
# ═══════════════════════════════════════════════════════════════════════════

def build(candles: List[Candle]) -> List[FeatRow]:
    if len(candles) < config.FEATURE_WINDOW:
        log.warning("Only %d candles — need at least %d", len(candles), config.FEATURE_WINDOW)

    closes  = np.array([c["close"]  for c in candles], dtype=float)
    highs   = np.array([c["high"]   for c in candles], dtype=float)
    lows    = np.array([c["low"]    for c in candles], dtype=float)
    volumes = np.array([c["volume"] for c in candles], dtype=float)
    opens   = np.array([c["open"]   for c in candles], dtype=float)

    ema9        = ema(closes,  9)
    ema21       = ema(closes, 21)
    ema50       = ema(closes, 50)
    macd_l, macd_s, macd_h = macd(closes)
    rsi14       = rsi(closes, 14)
    roc10       = rate_of_change(closes, 10)
    atr14       = atr(highs, lows, closes, 14)
    bb_u, bb_m, bb_l, bb_pct = bollinger_bands(closes, 20)
    obv         = on_balance_volume(closes, volumes)
    vol_d       = volume_delta(volumes, 20)
    labels      = create_labels(closes, config.LABEL_HORIZON, config.LABEL_THRESHOLD_PCT)

    # Lag features (t-1 … t-5)
    lag_closes = {f"lag_{k}": np.roll(closes, k) for k in [1, 2, 3, 5, 10]}

    rows: List[FeatRow] = []
    for i, c in enumerate(candles):
        label = labels[i]
        # Skip rows without a valid label or without enough warmup
        if np.isnan(label) or i < config.FEATURE_WINDOW:
            continue

        row: FeatRow = {
            # Raw price
            "close":       closes[i],
            "open":        opens[i],
            "high":        highs[i],
            "low":         lows[i],
            # Trend
            "ema_9":       ema9[i],
            "ema_21":      ema21[i],
            "ema_50":      ema50[i],
            "price_vs_ema9":  closes[i] / (ema9[i]  + 1e-12) - 1,
            "price_vs_ema50": closes[i] / (ema50[i] + 1e-12) - 1,
            # Momentum
            "macd":        macd_l[i],
            "macd_signal": macd_s[i],
            "macd_hist":   macd_h[i],
            "rsi_14":      rsi14[i],
            "roc_10":      roc10[i] if not np.isnan(roc10[i]) else 0.0,
            # Volatility
            "atr_14":      atr14[i],
            "bb_upper":    bb_u[i],
            "bb_lower":    bb_l[i],
            "bb_pct_b":    float(np.clip(bb_pct[i], -1, 2)),
            "bb_width":    (bb_u[i] - bb_l[i]) / (bb_m[i] + 1e-12),
            # Volume
            "volume":      volumes[i],
            "vol_delta":   vol_d[i],
            "obv":         obv[i],
            # Lags
            **{k: float(v[i]) for k, v in lag_closes.items()},
            # Meta
            "timestamp":   c["timestamp"],
            "symbol":      c.get("symbol", ""),
            "interval":    c.get("interval", ""),
            # Label
            "label":       label,
        }
        rows.append(row)

    log.info("Feature matrix: %d rows, %d features each", len(rows), len(rows[0]) - 4 if rows else 0)
    return rows


def to_arrays(
    rows: List[FeatRow],
    feature_cols: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, y) as numpy arrays, dropping meta columns."""
    META = {"timestamp", "symbol", "interval", "label"}
    if feature_cols is None:
        feature_cols = [k for k in rows[0].keys() if k not in META]

    X = np.array([[r[c] for c in feature_cols] for r in rows], dtype=float)
    y = np.array([r["label"] for r in rows], dtype=float)
    return X, y, feature_cols
