# CryptoOracle 🔮

A production-grade crypto price prediction system built entirely in **pure Python + NumPy**.
No ML framework dependencies. Runs offline with synthetic data or live via Binance API.

---

## Architecture

```
crypto/
├── core/
│   ├── ingestion/    fetcher.py         — Binance REST + synthetic fallback
│   ├── pipeline/     preprocessor.py    — validate, gap-fill, normalise, split
│   ├── features/     engineer.py        — RSI, MACD, BB, ATR, OBV, lags, labels
│   ├── models/       baseline.py        — Logistic Regression (pure NumPy)
│   │                 lstm.py            — LSTM (pure NumPy, from scratch)
│   │                 trainer.py         — orchestrates full training pipeline
│   ├── prediction/   predictor.py       — weighted ensemble + reversal detection
│   ├── strategy/     strategy.py        — entry/exit rules, position sizing, stops
│   └── backtest/     engine.py          — simulation, Sharpe, Sortino, drawdown
├── api/              server.py          — Flask REST API (CORS enabled)
├── webapp/           index.html         — live dashboard (connect to API or demo)
├── scripts/          train.py           — CLI trainer
│                     backtest.py        — CLI backtest runner
├── tests/            test_pipeline.py   — unit tests
├── utils/            config.py          — all configuration
│                     logger.py          — structured logging
│                     math_helpers.py    — sigmoid, AUC-ROC, rolling stats
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run unit tests
```bash
python tests/test_pipeline.py
```

### 3. Train models (synthetic data — no API key needed)
```bash
python scripts/train.py --synthetic --days 365
```

### 4. Run backtest
```bash
python scripts/backtest.py --synthetic --days 90
```

### 5. Start the API
```bash
python api/server.py
```
API runs at `http://localhost:8080`

### 6. Open the dashboard
Open `webapp/index.html` in your browser.
Paste `http://localhost:8080` in the API connection box → Connect.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | System health + model status |
| GET | `/api/predict?symbol=BTCUSDT&interval=1h` | Latest signal |
| GET | `/api/predict/batch?symbols=BTCUSDT&symbols=ETHUSDT` | Multi-asset signals |
| GET | `/api/backtest?symbol=BTCUSDT&days=90` | Backtest results |
| GET | `/api/features?symbol=BTCUSDT` | Raw feature vector |
| GET | `/api/model/metrics` | Model accuracy, AUC, feature importance |
| GET | `/api/signals/history?symbol=BTCUSDT` | Recent signal log |
| POST | `/api/retrain` | Trigger async model retraining |

### Example signal response
```json
{
  "symbol":        "BTCUSDT",
  "interval":      "1h",
  "price":         67432.10,
  "action":        "BUY",
  "direction":     "UP",
  "probability":   0.7312,
  "confidence":    0.4624,
  "reversal_zone": false,
  "model_outputs": { "baseline": 0.698, "lstm": 0.749 },
  "indicators": {
    "rsi":    48.2,
    "macd":   124.5,
    "atr":    812.3,
    "bb_pct": 0.61,
    "vol_delta": 1.42,
    "roc":    2.1
  }
}
```

---

## Training with Live Data

Remove `--synthetic` to use real Binance data:
```bash
python scripts/train.py --symbol BTCUSDT --interval 1h --days 365
```
No API key required for Binance public endpoints.

---

## Configuration

All settings live in `utils/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `LSTM_SEQ_LEN` | 60 | Bars of history per LSTM input |
| `LABEL_HORIZON` | 5 | Bars ahead to define the label |
| `MIN_CONFIDENCE` | 0.60 | Minimum confidence to emit a signal |
| `MAX_RISK_PER_TRADE` | 0.02 | 2% portfolio risk per trade |
| `STOP_ATR_MULT` | 2.0 | Stop-loss = entry ± 2×ATR |
| `TP_ATR_MULT` | 3.0 | Take-profit = entry ± 3×ATR |
| `WEIGHT_BASELINE` | 0.35 | Ensemble weight for LR model |
| `WEIGHT_LSTM` | 0.65 | Ensemble weight for LSTM |

---

## Features Computed

| Category | Features |
|----------|----------|
| Trend | EMA-9, EMA-21, EMA-50, price vs EMA ratios |
| Momentum | MACD line, signal, histogram, RSI-14, ROC-10 |
| Volatility | ATR-14, Bollinger upper/lower/width/`%B` |
| Volume | Raw volume, volume delta (relative), OBV |
| Lags | Close price at t-1, t-2, t-3, t-5, t-10 |

**Labels**: Binary — 1 (UP) if close rises >0.5% in next 5 bars, 0 (DOWN) if falls >0.5%.

---

## Models

### Logistic Regression (Baseline)
- Mini-batch gradient descent with L2 regularisation
- Early stopping on validation loss
- Interpretable via signed weight feature importance

### LSTM (Advanced)
- 2-layer LSTM (128 → 64 hidden units)
- Dense head: 32 → 1 (sigmoid)
- Adagrad optimiser, dropout, early stopping
- Pure NumPy — no PyTorch/TensorFlow required

### Ensemble
- Weighted average: `0.35 × LR + 0.65 × LSTM`
- Confidence gated: signals below 60% confidence suppressed
- Reversal zone dampening: conflicting technical confluence reduces confidence

---

## ⚠️ Disclaimer

This system is for **educational and research purposes only**.
It is **not financial advice**. Crypto markets are highly volatile.
Never trade with money you cannot afford to lose.
