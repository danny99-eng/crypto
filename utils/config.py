"""Central configuration for CryptoOracle."""
import os

# ── Exchange ──────────────────────────────────────────────────────────────
EXCHANGE_BASE_URL = os.getenv("EXCHANGE_URL", "https://api.binance.com/api/v3")
EXCHANGE_WS_URL   = os.getenv("EXCHANGE_WS",  "wss://stream.binance.com:9443/ws")

# ── Assets & Timeframes ───────────────────────────────────────────────────
ASSETS     = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
TIMEFRAMES = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}
DEFAULT_ASSET = "BTCUSDT"
DEFAULT_TF    = "1h"

# ── Model ─────────────────────────────────────────────────────────────────
# Resolve model dir relative to this file so it works on Windows + any cwd
_HERE         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR     = os.getenv("MODEL_DIR", os.path.join(_HERE, "models"))
BASELINE_PATH = os.path.join(MODEL_DIR, "baseline.pkl")
LSTM_PATH     = os.path.join(MODEL_DIR, "lstm.npz")

# ── Feature Engineering ───────────────────────────────────────────────────
FEATURE_WINDOW      = 200   # bars needed before features are valid
LSTM_SEQ_LEN        = 60    # sequence length for LSTM
LABEL_HORIZON       = 5     # bars ahead for label
LABEL_THRESHOLD_PCT = 0.5   # min % move to be labelled

# ── Risk Management ───────────────────────────────────────────────────────
INITIAL_CAPITAL    = 10_000.0
MAX_RISK_PER_TRADE = 0.02
MAX_POSITIONS      = 3
STOP_ATR_MULT      = 2.0
TP_ATR_MULT        = 3.0
MIN_CONFIDENCE     = 0.60
COOLDOWN_BARS      = 3
MAX_BARS_HELD      = 50
COMMISSION_PCT     = 0.001
SLIPPAGE_PCT       = 0.0005

# ── Prediction ────────────────────────────────────────────────────────────
WEIGHT_BASELINE      = 0.35
WEIGHT_LSTM          = 0.65
REVERSAL_RSI_HIGH    = 72.0
REVERSAL_RSI_LOW     = 28.0
REVERSAL_BB_HIGH     = 0.88
REVERSAL_BB_LOW      = 0.12

# ── API ───────────────────────────────────────────────────────────────────
API_PORT       = int(os.getenv("PORT", "8080"))
API_HOST       = os.getenv("HOST", "0.0.0.0")
RATE_LIMIT_RPM = 60
LOG_LEVEL      = os.getenv("LOG_LEVEL", "INFO")
