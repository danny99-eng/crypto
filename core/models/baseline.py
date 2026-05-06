"""
Baseline Model — Logistic Regression trained with mini-batch gradient descent.
Pure NumPy. No scikit-learn dependency.
"""
from __future__ import annotations
import os
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple
from utils.logger import get_logger
from utils.math_helpers import sigmoid, compute_auc_roc

log = get_logger(__name__)


class LogisticRegression:
    """Binary logistic regression with L2 regularisation."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs:        int   = 200,
        batch_size:    int   = 64,
        l2_lambda:     float = 1e-4,
        threshold:     float = 0.5,
        patience:      int   = 15,
    ) -> None:
        self.lr         = learning_rate
        self.epochs     = epochs
        self.batch_size = batch_size
        self.l2         = l2_lambda
        self.threshold  = threshold
        self.patience   = patience

        self.weights_: Optional[np.ndarray] = None
        self.bias_:    float = 0.0
        self.feature_cols_: List[str] = []
        self.eval_metrics_: Dict = {}
        self.train_loss_: List[float] = []

    # ── Training ──────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   Optional[np.ndarray] = None,
        y_val:   Optional[np.ndarray] = None,
    ) -> "LogisticRegression":
        n, d = X_train.shape
        self.weights_ = np.zeros(d)
        self.bias_    = 0.0

        best_val_loss = float("inf")
        best_w = self.weights_.copy()
        best_b = self.bias_
        wait   = 0

        for epoch in range(1, self.epochs + 1):
            idx = np.random.permutation(n)
            epoch_loss = 0.0

            for start in range(0, n, self.batch_size):
                batch = idx[start: start + self.batch_size]
                xb    = X_train[batch]
                yb    = y_train[batch]

                p     = sigmoid(xb @ self.weights_ + self.bias_)
                p     = np.clip(p, 1e-9, 1 - 1e-9)
                loss  = -np.mean(yb * np.log(p) + (1 - yb) * np.log(1 - p))
                loss += 0.5 * self.l2 * np.dot(self.weights_, self.weights_)

                err   = p - yb
                dw    = xb.T @ err / len(batch) + self.l2 * self.weights_
                db    = np.mean(err)

                self.weights_ -= self.lr * dw
                self.bias_    -= self.lr * db
                epoch_loss    += loss

            avg_loss = epoch_loss / max(1, n // self.batch_size)
            self.train_loss_.append(avg_loss)

            # Early stopping on validation loss
            if X_val is not None and y_val is not None:
                val_p    = np.clip(sigmoid(X_val @ self.weights_ + self.bias_), 1e-9, 1-1e-9)
                val_loss = -np.mean(y_val * np.log(val_p) + (1 - y_val) * np.log(1 - val_p))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_w = self.weights_.copy()
                    best_b = self.bias_
                    wait   = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        log.info("Early stop at epoch %d (val_loss=%.4f)", epoch, best_val_loss)
                        break
            if epoch % 20 == 0:
                log.info("Epoch %d — loss: %.4f", epoch, avg_loss)

        if X_val is not None:
            self.weights_ = best_w
            self.bias_    = best_b

        log.info("Baseline model trained (%d features)", d)
        return self

    # ── Inference ─────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.weights_ is not None, "Model not trained"
        return sigmoid(X @ self.weights_ + self.bias_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def predict_single(self, x: np.ndarray) -> Dict:
        p = float(self.predict_proba(x.reshape(1, -1))[0])
        return {
            "probability": round(p, 4),
            "label":       int(p >= self.threshold),
            "confidence":  round(abs(p - 0.5) * 2, 4),
        }

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        probs = self.predict_proba(X)
        preds = (probs >= self.threshold).astype(int)
        tp = int(np.sum((preds == 1) & (y == 1)))
        tn = int(np.sum((preds == 0) & (y == 0)))
        fp = int(np.sum((preds == 1) & (y == 0)))
        fn = int(np.sum((preds == 0) & (y == 1)))
        precision = tp / (tp + fp + 1e-12)
        recall    = tp / (tp + fn + 1e-12)
        f1        = 2 * precision * recall / (precision + recall + 1e-12)
        accuracy  = (tp + tn) / len(y)
        auc       = compute_auc_roc(y, probs)
        metrics = dict(accuracy=round(accuracy,4), precision=round(precision,4),
                       recall=round(recall,4), f1=round(f1,4), auc_roc=round(auc,4),
                       tp=tp, tn=tn, fp=fp, fn=fn)
        self.eval_metrics_ = metrics
        log.info("Baseline eval: %s", metrics)
        return metrics

    # ── Feature importance ────────────────────────────────────────────────

    def feature_importance(self) -> List[Tuple[str, float]]:
        if self.weights_ is None:
            return []
        pairs = sorted(
            zip(self.feature_cols_ or [f"f{i}" for i in range(len(self.weights_))],
                np.abs(self.weights_).tolist()),
            key=lambda x: x[1], reverse=True,
        )
        return pairs

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "weights": self.weights_,
                "bias":    self.bias_,
                "config":  {k: getattr(self, k) for k in
                            ("lr","epochs","batch_size","l2","threshold","patience")},
                "feature_cols": self.feature_cols_,
                "eval_metrics": self.eval_metrics_,
            }, f)
        log.info("Baseline model saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "LogisticRegression":
        with open(path, "rb") as f:
            data = pickle.load(f)
        m = cls(**data["config"])
        m.weights_      = data["weights"]
        m.bias_         = data["bias"]
        m.feature_cols_ = data.get("feature_cols", [])
        m.eval_metrics_ = data.get("eval_metrics", {})
        log.info("Baseline model loaded ← %s", path)
        return m
