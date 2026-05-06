#!/usr/bin/env python3
"""
CLI script to train both models from scratch.
Usage:
    python scripts/train.py --synthetic
    python scripts/train.py --symbol ETHUSDT --interval 1h --days 365
"""
import sys
import os
import argparse

# Always resolve to project root regardless of where the script is called from
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import config
from utils.logger import get_logger
from core.ingestion.fetcher import get_candles
from core.pipeline.preprocessor import run as preprocess
from core.models.trainer import train_all

log = get_logger("train")


def main():
    p = argparse.ArgumentParser(description="Train CryptoOracle models")
    p.add_argument("--symbol",    default=config.DEFAULT_ASSET, help="e.g. BTCUSDT")
    p.add_argument("--interval",  default=config.DEFAULT_TF,    help="e.g. 1h")
    p.add_argument("--days",      type=int, default=365,         help="History days")
    p.add_argument("--synthetic", action="store_true",           help="Use synthetic data (no API key needed)")
    p.add_argument("--model-dir", default=config.MODEL_DIR,      help="Where to save models")
    args = p.parse_args()

    log.info("=== CryptoOracle Training Pipeline ===")
    log.info("Symbol:   %s", args.symbol)
    log.info("Interval: %s", args.interval)
    log.info("Days:     %d", args.days)
    log.info("Source:   %s", "synthetic" if args.synthetic else "live Binance API")

    log.info("Step 1/3 — Fetching data ...")
    candles = get_candles(args.symbol, args.interval, days=args.days, use_synthetic=args.synthetic)
    log.info("Got %d candles", len(candles))

    log.info("Step 2/3 — Preprocessing ...")
    split = preprocess(candles, args.interval)

    log.info("Step 3/3 — Training models ...")
    metrics = train_all(split, save_dir=args.model_dir)

    log.info("=== Training complete ===")
    log.info("Baseline -> accuracy: %.4f  AUC: %.4f",
             metrics["baseline"].get("accuracy", 0),
             metrics["baseline"].get("auc_roc", 0))
    if metrics.get("lstm"):
        log.info("LSTM     -> accuracy: %.4f  AUC: %.4f",
                 metrics["lstm"].get("accuracy", 0),
                 metrics["lstm"].get("auc_roc", 0))
    log.info("Models saved to: %s", args.model_dir)
    log.info("Next -> python scripts/backtest.py --synthetic")
    log.info("Next -> python api/server.py")


if __name__ == "__main__":
    main()
