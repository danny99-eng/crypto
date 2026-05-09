"""
REST API — Flask server exposing predictions, backtests, and model metrics.
"""
from __future__ import annotations
import sys
import json
import os
import time
import threading
from functools import wraps
from collections import defaultdict
from typing import Dict, Optional

# Ensure project root is on path regardless of working directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, jsonify, request, Response
from flask_cors import CORS

from utils import config
from utils.logger import get_logger
from core.ingestion.fetcher import get_candles
from core.pipeline.preprocessor import run as preprocess
from core.features.engineer import build as build_features
from core.prediction.predictor import PredictionEngine
from core.backtest.engine import run_backtest
from core.models.trainer import train_all

log = get_logger(__name__)

app = Flask(__name__)
CORS(app)   # allow all origins — tighten in production

# ── Global state ──────────────────────────────────────────────────────────
_engine: Optional[PredictionEngine] = None
_rate_store: Dict[str, list] = defaultdict(list)
_signal_history: Dict[str, list] = defaultdict(list)
_retrain_lock = threading.Lock()


def get_engine() -> Optional[PredictionEngine]:
    """Load models — auto-trains with synthetic data if models folder missing."""
    global _engine
    if _engine is None:
        try:
            _engine = PredictionEngine.from_saved(config.MODEL_DIR)
            log.info("Models loaded successfully from %s", config.MODEL_DIR)
        except FileNotFoundError:
            log.warning("No trained models found at %s", config.MODEL_DIR)
            log.warning("Auto-training with synthetic data — this takes ~2 minutes...")
            _auto_train()
        except Exception as exc:
            log.warning("Could not load models: %s", exc)
    return _engine


def _auto_train():
    """Train models automatically with synthetic data on first run."""
    global _engine
    if _retrain_lock.acquire(blocking=False):
        try:
            from core.ingestion.fetcher import get_candles
            from core.pipeline.preprocessor import run as preprocess
            from core.models.trainer import train_all
            log.info("=== Auto-training started (synthetic data) ===")
            candles = get_candles(config.DEFAULT_ASSET, config.DEFAULT_TF,
                                  days=365, use_synthetic=True)
            split   = preprocess(candles, config.DEFAULT_TF)
            train_all(split, save_dir=config.MODEL_DIR)
            _engine = PredictionEngine.from_saved(config.MODEL_DIR)
            log.info("=== Auto-training complete — models ready ===")
        except Exception as exc:
            log.error("Auto-training failed: %s", exc)
            log.error("Run manually: python run.py train --synthetic")
        finally:
            _retrain_lock.release()


# ── Middleware ────────────────────────────────────────────────────────────

def rate_limit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key   = request.remote_addr + request.path
        now   = time.time()
        hits  = _rate_store[key]
        _rate_store[key] = [h for h in hits if now - h < 60]
        if len(_rate_store[key]) >= config.RATE_LIMIT_RPM:
            return jsonify({"error": "rate limit exceeded"}), 429
        _rate_store[key].append(now)
        return f(*args, **kwargs)
    return decorated


# ── Health ────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    engine = get_engine()
    model_dir = config.MODEL_DIR
    baseline_exists = os.path.exists(os.path.join(model_dir, "baseline.pkl"))
    lstm_exists     = os.path.exists(os.path.join(model_dir, "lstm.npz"))
    return jsonify({
        "status":           "ok" if engine is not None else "no_models",
        "model_loaded":     engine is not None,
        "baseline_trained": baseline_exists,
        "lstm_trained":     lstm_exists,
        "model_dir":        model_dir,
        "message":          "Ready" if engine else "Models not trained yet. Auto-training in background or run: python run.py train --synthetic",
        "timestamp":        int(time.time() * 1000),
    })


# ── Predict ───────────────────────────────────────────────────────────────

@app.route("/api/predict")
@rate_limit
def predict():
    symbol   = request.args.get("symbol",   config.DEFAULT_ASSET)
    interval = request.args.get("interval", config.DEFAULT_TF)
    use_syn  = request.args.get("synthetic", "false").lower() == "true"

    engine = get_engine()
    if engine is None:
        return jsonify({"error": "Models not loaded. POST /api/retrain first."}), 503

    try:
        candles  = get_candles(symbol, interval, days=30, use_synthetic=use_syn)
        rows     = build_features(candles)
        if not rows:
            return jsonify({"error": "Not enough data to generate features"}), 422

        import numpy as np
        feat_cols = engine.feature_cols
        X_all = [(r.get(c, 0.0) for c in feat_cols) for r in rows]
        X_norm = (
            (lambda arr: (arr - engine.mu) / (engine.sigma + 1e-8))
            (np.array(list(X_all), dtype=float))
        )

        seq_len  = config.LSTM_SEQ_LEN
        sequence = X_norm[-seq_len:] if len(X_norm) >= seq_len else None
        signal   = engine.generate_signal(rows[-1], sequence)

        # Store in history
        _signal_history[symbol].insert(0, signal)
        _signal_history[symbol] = _signal_history[symbol][:100]

        return jsonify(signal)

    except Exception as exc:
        log.exception("predict error")
        return jsonify({"error": str(exc)}), 500


# ── Batch predict ─────────────────────────────────────────────────────────

@app.route("/api/predict/batch")
@rate_limit
def predict_batch():
    symbols  = request.args.getlist("symbols") or config.ASSETS[:3]
    interval = request.args.get("interval", config.DEFAULT_TF)
    use_syn  = request.args.get("synthetic", "false").lower() == "true"

    results = []
    for sym in symbols:
        r = app.test_client().get(f"/api/predict?symbol={sym}&interval={interval}&synthetic={str(use_syn).lower()}")
        results.append(json.loads(r.data))

    return jsonify({"predictions": results, "count": len(results)})


# ── Backtest ──────────────────────────────────────────────────────────────

@app.route("/api/backtest")
@rate_limit
def backtest():
    symbol   = request.args.get("symbol",   config.DEFAULT_ASSET)
    interval = request.args.get("interval", config.DEFAULT_TF)
    days     = int(request.args.get("days", 90))
    use_syn  = request.args.get("synthetic", "false").lower() == "true"

    engine = get_engine()
    if engine is None:
        return jsonify({"error": "Models not loaded."}), 503

    try:
        candles = get_candles(symbol, interval, days=days, use_synthetic=use_syn)
        rows    = build_features(candles)
        if not rows:
            return jsonify({"error": "Not enough data"}), 422

        result = run_backtest(rows, engine)
        return jsonify(result)

    except Exception as exc:
        log.exception("backtest error")
        return jsonify({"error": str(exc)}), 500


# ── Features (inspection / debug) ────────────────────────────────────────

@app.route("/api/features")
@rate_limit
def features():
    symbol   = request.args.get("symbol",   config.DEFAULT_ASSET)
    interval = request.args.get("interval", config.DEFAULT_TF)
    use_syn  = request.args.get("synthetic", "false").lower() == "true"

    candles = get_candles(symbol, interval, days=7, use_synthetic=use_syn)
    rows    = build_features(candles)
    if not rows:
        return jsonify({"error": "No feature rows"}), 422

    row = {k: v for k, v in rows[-1].items() if k not in ("symbol", "interval")}
    return jsonify({"symbol": symbol, "interval": interval, "features": row,
                    "n_features": len(rows[0]) - 4})


# ── Model metrics ─────────────────────────────────────────────────────────

@app.route("/api/model/metrics")
def model_metrics():
    engine = get_engine()
    if engine is None:
        return jsonify({"error": "Models not loaded."}), 503
    return jsonify({
        "baseline": engine.baseline.eval_metrics_,
        "lstm":     engine.lstm.eval_metrics_ if engine.lstm else {},
        "feature_importance": engine.baseline.feature_importance()[:10],
    })


# ── Signal history ────────────────────────────────────────────────────────

@app.route("/api/signals/history")
def signals_history():
    symbol = request.args.get("symbol", config.DEFAULT_ASSET)
    limit  = int(request.args.get("limit", 50))
    return jsonify({"signals": _signal_history[symbol][:limit], "symbol": symbol})


# ── Retrain ───────────────────────────────────────────────────────────────

@app.route("/api/retrain", methods=["POST"])
def retrain():
    global _engine
    body     = request.get_json(silent=True) or {}
    symbol   = body.get("symbol",   config.DEFAULT_ASSET)
    interval = body.get("interval", config.DEFAULT_TF)
    days     = int(body.get("days", 365))
    use_syn  = body.get("synthetic", True)  # default to synthetic for quick start

    if not _retrain_lock.acquire(blocking=False):
        return jsonify({"error": "Retraining already in progress"}), 409

    def _train_bg():
        global _engine
        try:
            log.info("Retraining started: %s %s %d days", symbol, interval, days)
            candles = get_candles(symbol, interval, days=days, use_synthetic=use_syn)
            split   = preprocess(candles, interval)
            metrics = train_all(split, save_dir=config.MODEL_DIR)
            _engine = PredictionEngine.from_saved(config.MODEL_DIR)
            log.info("Retraining complete: %s", metrics)
        except Exception as exc:
            log.exception("Retraining failed: %s", exc)
        finally:
            _retrain_lock.release()

    t = threading.Thread(target=_train_bg, daemon=True)
    t.start()
    return jsonify({"status": "queued", "symbol": symbol, "interval": interval, "days": days}), 202


# ── Entry point ───────────────────────────────────────────────────────────────────────────

def create_app() -> Flask:
    return app


def _serve():
    """Start production WSGI server (Waitress). Falls back to Flask dev server if not installed."""
    try:
        from waitress import serve
        log.info("=================================================")
        log.info("  CryptoOracle API  —  Production server (Waitress)")
        log.info("  http://localhost:%d", config.API_PORT)
        log.info("  Open webapp/index.html and connect to the URL above")
        log.info("=================================================")
        serve(app, host=config.API_HOST, port=config.API_PORT, threads=8)
    except ImportError:
        log.warning("waitress not found — install it: pip install waitress")
        log.warning("Falling back to Flask development server")
        app.run(host=config.API_HOST, port=config.API_PORT, debug=False)


if __name__ == "__main__":
    _serve()
