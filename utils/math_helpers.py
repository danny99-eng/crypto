"""Pure-NumPy math helpers used across all modules."""
from __future__ import annotations
import numpy as np
from typing import Sequence


def safe_mean(arr: np.ndarray) -> float:
    return float(np.mean(arr)) if len(arr) > 0 else 0.0


def safe_std(arr: np.ndarray, ddof: int = 1) -> float:
    return float(np.std(arr, ddof=ddof)) if len(arr) > 1 else 1e-8


def rolling_mean(series: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(series, np.nan, dtype=float)
    for i in range(len(series)):
        sl = series[max(0, i - window + 1): i + 1]
        out[i] = np.mean(sl)
    return out


def rolling_std(series: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(series, np.nan, dtype=float)
    for i in range(len(series)):
        sl = series[max(0, i - window + 1): i + 1]
        out[i] = float(np.std(sl, ddof=1)) if len(sl) > 1 else 0.0
    return out


def rolling_zscore(series: np.ndarray, window: int = 200) -> np.ndarray:
    mu  = rolling_mean(series, window)
    sig = rolling_std(series, window)
    sig = np.where(sig == 0, 1e-8, sig)
    return (series - mu) / sig


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def compute_auc_roc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Trapezoidal AUC-ROC."""
    thresholds = np.sort(np.unique(y_prob))[::-1]
    tprs, fprs = [0.0], [0.0]
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return 0.5
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        tprs.append(tp / pos)
        fprs.append(fp / neg)
    tprs.append(1.0); fprs.append(1.0)
    tprs_arr = np.array(tprs); fprs_arr = np.array(fprs)
    return float(np.trapezoid(tprs_arr, fprs_arr))
