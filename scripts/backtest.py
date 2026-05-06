#!/usr/bin/env python3
"""
CLI backtest script.
Usage:
    python scripts/backtest.py
    python scripts/backtest.py --symbol ETHUSDT --days 180 --synthetic
"""
import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils import config
from utils.logger import get_logger
from core.ingestion.fetcher import get_candles
from core.features.engineer import build as build_features
from core.prediction.predictor import PredictionEngine
from core.backtest.engine import run_backtest

log = get_logger("backtest")


def main():
    p = argparse.ArgumentParser(description="Run CryptoOracle backtest")
    p.add_argument("--symbol",    default=config.DEFAULT_ASSET)
    p.add_argument("--interval",  default=config.DEFAULT_TF)
    p.add_argument("--days",      type=int, default=90)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--output",    default=None, help="JSON output file")
    args = p.parse_args()

    log.info("Loading models from %s …", config.MODEL_DIR)
    try:
        engine = PredictionEngine.from_saved(config.MODEL_DIR)
    except Exception as e:
        log.error("Could not load models: %s — run train.py first", e)
        sys.exit(1)

    log.info("Fetching %d days of %s %s …", args.days, args.symbol, args.interval)
    candles = get_candles(args.symbol, args.interval, days=args.days, use_synthetic=args.synthetic)
    rows    = build_features(candles)
    log.info("Running backtest on %d feature rows …", len(rows))

    result = run_backtest(rows, engine)
    s = result["summary"]
    t = result["trades"]

    print("\n" + "="*52)
    print("  CryptoOracle Backtest Results")
    print("="*52)
    print(f"  Symbol      : {args.symbol} ({args.interval})")
    print(f"  Period      : {args.days} days")
    print(f"  Initial     : ${s['initial_capital']:,.2f}")
    print(f"  Final       : ${s['final_equity']:,.2f}")
    print(f"  Return      : {s['total_return_pct']:+.2f}%")
    print(f"  Ann. Return : {s.get('annual_return_pct',0):+.2f}%")
    print(f"  Max DD      : -{s['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe      : {s['sharpe_ratio']:.3f}")
    print(f"  Sortino     : {s['sortino_ratio']:.3f}")
    print(f"  Profit Fac. : {s['profit_factor']:.3f}")
    print(f"  Trades      : {t['total']}")
    print(f"  Win Rate    : {t['win_rate_pct']:.1f}%")
    print(f"  Avg Win     : ${t['avg_win']:.2f}")
    print(f"  Avg Loss    : ${t['avg_loss']:.2f}")
    print("="*52 + "\n")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        log.info("Full results saved → %s", args.output)


if __name__ == "__main__":
    main()
