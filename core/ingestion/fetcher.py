"""
Data Ingestion — fetches OHLCV candles from Binance REST API.
Falls back to synthetic data generation when offline / for testing.
"""
from __future__ import annotations
import time
import math
import random
import requests
from typing import List, Dict, Optional
from utils import config
from utils.logger import get_logger

log = get_logger(__name__)

# ── Types ─────────────────────────────────────────────────────────────────
Candle = Dict[str, float]   # {timestamp, open, high, low, close, volume}


# ── Binance REST fetcher ──────────────────────────────────────────────────

def fetch_candles(
    symbol: str,
    interval: str,
    start_ms: Optional[int] = None,
    limit: int = 500,
) -> List[Candle]:
    """Fetch up to `limit` OHLCV candles from Binance.

    Returns an empty list on network failure so callers can fall back gracefully.
    """
    endpoint = f"{config.EXCHANGE_BASE_URL}/klines"
    params: Dict = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms is not None:
        params["startTime"] = start_ms

    for attempt in range(1, config.MAX_RETRIES + 1 if hasattr(config, "MAX_RETRIES") else 4):
        try:
            resp = requests.get(endpoint, params=params, timeout=10)
            if resp.status_code == 200:
                return _parse_klines(resp.json(), symbol, interval)
            if resp.status_code == 429:
                sleep_s = min(2 ** attempt, 30)
                log.warning("Rate limited — sleeping %ds", sleep_s)
                time.sleep(sleep_s)
                continue
            log.error("Binance error %s for %s %s", resp.status_code, symbol, interval)
            return []
        except requests.RequestException as exc:
            log.warning("Request failed (attempt %d): %s", attempt, exc)
            time.sleep(2 ** attempt)

    return []


def _parse_klines(raw: list, symbol: str, interval: str) -> List[Candle]:
    candles = []
    for row in raw:
        candles.append({
            "timestamp": int(row[0]),
            "open":      float(row[1]),
            "high":      float(row[2]),
            "low":       float(row[3]),
            "close":     float(row[4]),
            "volume":    float(row[5]),
            "symbol":    symbol,
            "interval":  interval,
        })
    return candles


def fetch_full_history(
    symbol: str,
    interval: str,
    days: int = 365,
) -> List[Candle]:
    """Page through Binance API to build a full history."""
    interval_ms = config.TIMEFRAMES.get(interval, 3_600_000)
    end_ms      = int(time.time() * 1000)
    start_ms    = end_ms - days * 86_400_000
    all_candles: List[Candle] = []
    cursor = start_ms

    log.info("Fetching %d days of %s %s …", days, symbol, interval)
    while cursor < end_ms:
        batch = fetch_candles(symbol, interval, start_ms=cursor, limit=1000)
        if not batch:
            break
        all_candles.extend(batch)
        cursor = batch[-1]["timestamp"] + 1
        # Respect rate limit: ~1 req / 100 ms
        time.sleep(0.12)

    log.info("Fetched %d candles for %s %s", len(all_candles), symbol, interval)
    return all_candles


# ── Synthetic data (offline / testing) ───────────────────────────────────

def generate_synthetic(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    n: int = 500,
    base_price: float = 67_000.0,
    volatility: float = 0.015,
) -> List[Candle]:
    """Generate realistic-looking OHLCV data using geometric Brownian motion."""
    seed = hash(symbol) % (2 ** 31)
    rng  = random.Random(seed)
    candles: List[Candle] = []
    price = base_price
    ts    = int(time.time() * 1000) - n * config.TIMEFRAMES.get(interval, 3_600_000)

    for _ in range(n):
        ret   = rng.gauss(0.0002, volatility)
        open_ = price
        close = max(1.0, open_ * (1 + ret))
        high  = max(open_, close) * (1 + abs(rng.gauss(0, 0.004)))
        low   = min(open_, close) * (1 - abs(rng.gauss(0, 0.004)))
        vol   = abs(rng.gauss(800, 300)) * (base_price / 67_000)
        candles.append({
            "timestamp": ts,
            "open": round(open_, 4), "high": round(high, 4),
            "low":  round(low,  4),  "close": round(close, 4),
            "volume": round(vol, 4),
            "symbol": symbol, "interval": interval,
        })
        price = close
        ts   += config.TIMEFRAMES.get(interval, 3_600_000)

    return candles


def get_candles(
    symbol: str,
    interval: str,
    days: int = 365,
    use_synthetic: bool = False,
) -> List[Candle]:
    """High-level helper: tries live API, falls back to synthetic."""
    if use_synthetic:
        n = days * 24 if interval == "1h" else days
        return generate_synthetic(symbol, interval, n=min(n, 2000))

    candles = fetch_full_history(symbol, interval, days)
    if not candles:
        log.warning("Live fetch failed — using synthetic data")
        n = days * 24 if interval == "1h" else days
        candles = generate_synthetic(symbol, interval, n=min(n, 2000))
    return candles
