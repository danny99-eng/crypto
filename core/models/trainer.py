"""
Trainer — orchestrates the full training pipeline for both models.
"""
from __future__ import annotations
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from utils import config
from utils.logger import get_logger
from core.features.engineer import build, to_arrays
from core.models.baseline import LogisticRegression
from core.models.lstm import LSTM, build_sequences

log = get_logger(__name__)


def normalise_features(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    X_test:  np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalise using train statistics only."""
    mu    = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-8
    return (
        (X_train - mu) / sigma,
        (X_val   - mu) / sigma,
        (X_test  - mu) / sigma,
        mu, sigma,
    )


def train_all(
    split: Dict,
    save_dir: str = config.MODEL_DIR,
) -> Dict:
    """
    Full training pipeline.
    split = {"train": [...candles], "val": [...], "test": [...]}
    Returns evaluation metrics for both models.
    """
    os.makedirs(save_dir, exist_ok=True)

    log.info("=== Building feature matrices ===")
    train_rows = build(split["train"])
    val_rows   = build(split["val"])
    test_rows  = build(split["test"])

    if not train_rows:
        raise ValueError("No training rows — check candle count and config.FEATURE_WINDOW")

    X_tr, y_tr, feat_cols = to_arrays(train_rows)
    X_va, y_va, _         = to_arrays(val_rows,  feat_cols)
    X_te, y_te, _         = to_arrays(test_rows, feat_cols)

    log.info("Shapes — train: %s  val: %s  test: %s", X_tr.shape, X_va.shape, X_te.shape)

    # Normalise
    X_tr_n, X_va_n, X_te_n, mu, sigma = normalise_features(X_tr, X_va, X_te)

    # Save normalisation stats
    np.savez(os.path.join(save_dir, "norm_stats.npz"), mu=mu, sigma=sigma)

    # ── Baseline ──────────────────────────────────────────────────────────
    log.info("=== Training Logistic Regression ===")
    baseline = LogisticRegression(learning_rate=0.01, epochs=200, batch_size=64)
    baseline.feature_cols_ = feat_cols
    baseline.fit(X_tr_n, y_tr, X_va_n, y_va)
    base_metrics = baseline.evaluate(X_te_n, y_te)
    baseline.save(os.path.join(save_dir, "baseline.pkl"))

    # ── LSTM ──────────────────────────────────────────────────────────────
    log.info("=== Training LSTM ===")
    # Combine train+val for sequence building (label already locked in)
    X_full  = np.vstack([X_tr_n, X_va_n])
    y_full  = np.concatenate([y_tr, y_va])
    seq_len = config.LSTM_SEQ_LEN

    Xs_tr, ys_tr = build_sequences(X_tr_n, y_tr, seq_len)
    Xs_va, ys_va = build_sequences(X_va_n, y_va, seq_len)
    Xs_te, ys_te = build_sequences(X_te_n, y_te, seq_len)

    if len(Xs_tr) == 0:
        log.warning("Not enough sequences for LSTM — skipping LSTM training")
        lstm_metrics: Dict = {}
    else:
        lstm = LSTM(
            input_dim=X_tr_n.shape[1],
            hidden1=128, hidden2=64, dense_dim=32,
            seq_len=seq_len,
            learning_rate=5e-4, epochs=50, batch_size=32, patience=8,
        )
        lstm.feature_cols_ = feat_cols
        lstm.fit(Xs_tr, ys_tr, Xs_va, ys_va)
        lstm_metrics = lstm.evaluate(Xs_te, ys_te)
        lstm.save(os.path.join(save_dir, "lstm.npz"))

    # Save feature column list for inference
    import json
    with open(os.path.join(save_dir, "feature_cols.json"), "w") as f:
        json.dump(feat_cols, f)

    log.info("=== Training complete ===")
    return {
        "baseline": base_metrics,
        "lstm":     lstm_metrics,
        "feature_cols": feat_cols,
        "n_features":   len(feat_cols),
        "n_train":      len(X_tr),
        "n_val":        len(X_va),
        "n_test":       len(X_te),
    }
