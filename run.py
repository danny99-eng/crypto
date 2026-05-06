#!/usr/bin/env python3
"""
CryptoOracle — Root launcher.
Run this from anywhere:
    python run.py train --synthetic
    python run.py backtest --synthetic
    python run.py api
    python run.py test
"""
import sys
import os

# Pin project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py [train|backtest|api|test] [options]")
        print()
        print("  python run.py train --synthetic          # train with generated data")
        print("  python run.py train --days 365           # train with live Binance data")
        print("  python run.py backtest --synthetic       # run backtest")
        print("  python run.py api                        # start REST API on :8080")
        print("  python run.py test                       # run unit tests")
        sys.exit(0)

    command = sys.argv[1].lower()
    # Pass remaining args downstream
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "train":
        from scripts.train import main as _main
        _main()

    elif command == "backtest":
        from scripts.backtest import main as _main
        _main()

    elif command == "api":
        from api.server import app
        from utils import config
        print(f"\n  CryptoOracle API starting on http://localhost:{config.API_PORT}")
        print(f"  Open webapp/index.html and connect to http://localhost:{config.API_PORT}\n")
        app.run(host=config.API_HOST, port=config.API_PORT, debug=False)

    elif command == "test":
        import subprocess
        result = subprocess.run(
            [sys.executable, os.path.join(PROJECT_ROOT, "tests", "test_pipeline.py")],
            cwd=PROJECT_ROOT
        )
        sys.exit(result.returncode)

    else:
        print(f"Unknown command: {command}")
        print("Available: train, backtest, api, test")
        sys.exit(1)


if __name__ == "__main__":
    main()
