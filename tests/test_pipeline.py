"""Unit tests for the CryptoOracle pipeline."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from core.ingestion.fetcher import generate_synthetic
from core.pipeline.preprocessor import validate, fill_gaps, normalise, temporal_split
from core.features.engineer import build, to_arrays, ema, rsi, macd, bollinger_bands, atr
from core.models.baseline import LogisticRegression
from core.models.lstm import LSTM, build_sequences
from utils import config


# ── Fetcher ───────────────────────────────────────────────────────────────

def test_synthetic_generation():
    candles = generate_synthetic("BTCUSDT", "1h", n=300)
    assert len(candles) == 300
    for c in candles:
        assert c["high"] >= c["low"]
        assert c["close"] > 0
        assert c["volume"] >= 0
    print("✓ synthetic generation")


# ── Pipeline ──────────────────────────────────────────────────────────────

def test_validate_removes_bad():
    candles = generate_synthetic("BTCUSDT", "1h", n=50)
    bad = {**candles[0], "high": 0.0, "low": 999999.0}  # high < low
    candles.append(bad)
    clean = validate(candles)
    assert len(clean) == 50
    print("✓ validate removes bad candles")


def test_fill_gaps():
    candles = generate_synthetic("BTCUSDT", "1h", n=100)
    # Remove 5 candles to create gaps
    gapped = [c for i, c in enumerate(candles) if i not in (10, 20, 30, 40, 50)]
    filled = fill_gaps(gapped, "1h")
    assert len(filled) >= len(gapped)
    print("✓ gap fill")


def test_temporal_split():
    candles = generate_synthetic("BTCUSDT", "1h", n=500)
    split   = temporal_split(candles)
    total   = len(split["train"]) + len(split["val"]) + len(split["test"])
    assert total == 500
    assert len(split["train"]) > len(split["test"])
    print("✓ temporal split")


# ── Indicators ────────────────────────────────────────────────────────────

def test_ema():
    data = np.array([float(i) for i in range(1, 21)])
    out  = ema(data, 5)
    assert len(out) == 20
    assert out[-1] > out[0]
    print("✓ EMA")


def test_rsi_bounds():
    data = np.random.randn(200).cumsum() + 100
    out  = rsi(data, 14)
    assert np.all((out >= 0) & (out <= 100))
    print("✓ RSI bounds [0, 100]")


def test_bollinger():
    data = np.random.randn(200).cumsum() + 100
    u, m, l, pct = bollinger_bands(data, 20)
    assert np.all(u >= m)
    assert np.all(m >= l)
    print("✓ Bollinger bands ordering")


# ── Feature engineering ───────────────────────────────────────────────────

def test_feature_build():
    candles = generate_synthetic("BTCUSDT", "1h", n=500)
    rows    = build(candles)
    assert len(rows) > 0
    assert "rsi_14"    in rows[0]
    assert "macd"      in rows[0]
    assert "atr_14"    in rows[0]
    assert "bb_pct_b"  in rows[0]
    assert "label"     in rows[0]
    assert rows[0]["label"] in (0.0, 1.0)
    print(f"✓ feature build ({len(rows)} rows, {len(rows[0])-4} features)")


# ── Baseline model ────────────────────────────────────────────────────────

def test_baseline_train_predict():
    np.random.seed(42)
    X = np.random.randn(300, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    model = LogisticRegression(epochs=50, batch_size=32)
    model.fit(X[:200], y[:200], X[200:250], y[200:250])
    metrics = model.evaluate(X[250:], y[250:])
    assert metrics["accuracy"] > 0.5, f"Accuracy too low: {metrics['accuracy']}"
    pred = model.predict_single(X[0])
    assert 0.0 <= pred["probability"] <= 1.0
    print(f"✓ baseline train/predict (acc={metrics['accuracy']:.3f})")


# ── LSTM sequences ────────────────────────────────────────────────────────

def test_build_sequences():
    X = np.random.randn(200, 10)
    y = np.random.randint(0, 2, 200).astype(float)
    Xs, ys = build_sequences(X, y, seq_len=30)
    assert Xs.shape == (170, 30, 10)
    assert ys.shape == (170,)
    print("✓ LSTM sequence builder")


def test_lstm_forward():
    lstm = LSTM(input_dim=5, hidden1=16, hidden2=8, dense_dim=8, seq_len=10)
    seq  = np.random.randn(10, 5)
    out  = lstm.predict_single(seq)
    assert 0.0 <= out["probability"] <= 1.0
    assert 0.0 <= out["confidence"]  <= 1.0
    print("✓ LSTM forward pass")


# ── Run all ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== CryptoOracle Unit Tests ===\n")
    test_synthetic_generation()
    test_validate_removes_bad()
    test_fill_gaps()
    test_temporal_split()
    test_ema()
    test_rsi_bounds()
    test_bollinger()
    test_feature_build()
    test_baseline_train_predict()
    test_build_sequences()
    test_lstm_forward()
    print("\n✅  All tests passed!\n")
