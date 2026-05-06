"""
Backtesting Engine — simulates strategy on historical feature rows.
Computes full performance metrics including Sharpe, Sortino, drawdown.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional
from utils import config
from utils.logger import get_logger
from core.prediction.predictor import PredictionEngine
from core.strategy.strategy import TradingStrategy

log = get_logger(__name__)


def run_backtest(
    feature_rows: List[Dict],
    engine:       PredictionEngine,
    initial_capital: float = config.INITIAL_CAPITAL,
) -> Dict:
    """
    Simulate strategy on feature_rows using engine for signals.
    Returns full performance report.
    """
    strategy     = TradingStrategy(initial_capital)
    equity_curve = [initial_capital]
    seq_len      = config.LSTM_SEQ_LEN

    # Pre-compute normalised feature matrix for LSTM sequences
    feat_cols = engine.feature_cols
    X_all = np.array([[r.get(c, 0.0) for c in feat_cols] for r in feature_rows], dtype=float)
    X_norm = (X_all - engine.mu) / (engine.sigma + 1e-8)

    log.info("Running backtest on %d bars …", len(feature_rows))

    for i, feat in enumerate(feature_rows):
        strategy.bar_index = i
        price = feat.get("close", 0.0)

        # Build LSTM sequence if available
        sequence = None
        if engine.lstm is not None and i >= seq_len:
            sequence = X_norm[i - seq_len: i]

        signal = engine.generate_signal(feat, sequence)

        # Check exits first
        strategy.check_exits(price, signal)

        # Update trailing stops
        atr = feat.get("atr_14", price * 0.01)
        for pos in strategy.open_positions:
            strategy.update_trailing_stop(pos, price, atr)

        # Enter if conditions met
        if strategy.should_enter(signal, feat):
            strategy.enter(signal, feat)

        equity_curve.append(strategy.mark_to_market(price))

    # Force-close remaining positions
    if feature_rows:
        last_price = feature_rows[-1].get("close", 0.0)
        for pos in list(strategy.open_positions):
            strategy._close(pos, last_price, "END_OF_TEST")

    return _compute_metrics(equity_curve, strategy.trade_history, initial_capital)


def _compute_metrics(
    equity_curve: List[float],
    trades: List[Dict],
    initial: float,
) -> Dict:
    eq  = np.array(equity_curve, dtype=float)
    final = float(eq[-1])
    total_ret = (final - initial) / initial * 100
    n_days    = len(eq)
    ann_ret   = ((final / initial) ** (365 / max(n_days, 1)) - 1) * 100

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    dd   = (peak - eq) / (peak + 1e-8) * 100
    max_dd = float(np.max(dd))

    # Sharpe & Sortino
    rets = np.diff(eq) / (eq[:-1] + 1e-8)
    rf   = config.RISK_FREE_RATE / 365 if hasattr(config, "RISK_FREE_RATE") else 0.05 / 365
    excess = rets - rf
    sharpe  = float(np.mean(excess) / (np.std(excess, ddof=1) + 1e-8) * np.sqrt(365))
    neg_r   = excess[excess < 0]
    downdev = float(np.std(neg_r, ddof=1)) if len(neg_r) > 1 else 1e-8
    sortino = float(np.mean(excess) / downdev * np.sqrt(365))

    # Trade stats
    n  = len(trades)
    if n == 0:
        return _empty_metrics(equity_curve, initial)

    winners = [t for t in trades if t["pnl"] > 0]
    losers  = [t for t in trades if t["pnl"] <= 0]
    wr      = len(winners) / n * 100
    avg_win = float(np.mean([t["pnl"] for t in winners])) if winners else 0.0
    avg_los = float(np.mean([t["pnl"] for t in losers]))  if losers  else 0.0
    gross_p = sum(t["pnl"] for t in winners)
    gross_l = abs(sum(t["pnl"] for t in losers))
    pf      = gross_p / (gross_l + 1e-8)
    expect  = (wr / 100 * avg_win) + ((1 - wr / 100) * avg_los)

    exit_reasons = {}
    for t in trades:
        r = t.get("exit_reason", "UNKNOWN")
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    return {
        "summary": {
            "initial_capital":  round(initial, 2),
            "final_equity":     round(final, 2),
            "total_return_pct": round(total_ret, 2),
            "annual_return_pct": round(ann_ret, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio":     round(sharpe, 3),
            "sortino_ratio":    round(sortino, 3),
            "profit_factor":    round(pf, 3),
            "expectancy":       round(expect, 2),
        },
        "trades": {
            "total":         n,
            "winners":       len(winners),
            "losers":        len(losers),
            "win_rate_pct":  round(wr, 2),
            "avg_win":       round(avg_win, 2),
            "avg_loss":      round(avg_los, 2),
            "gross_profit":  round(gross_p, 2),
            "gross_loss":    round(gross_l, 2),
            "exit_reasons":  exit_reasons,
        },
        "equity_curve": [round(v, 2) for v in equity_curve],
        "trade_log":    trades,
    }


def _empty_metrics(equity_curve: List[float], initial: float) -> Dict:
    return {
        "summary":     {"initial_capital": initial, "final_equity": equity_curve[-1],
                        "total_return_pct": 0, "max_drawdown_pct": 0,
                        "sharpe_ratio": 0, "sortino_ratio": 0},
        "trades":      {"total": 0, "win_rate_pct": 0},
        "equity_curve": equity_curve,
        "trade_log":   [],
    }


def walk_forward(
    all_rows: List[Dict],
    engine:   PredictionEngine,
    n_folds:  int   = 5,
    train_pct: float = 0.70,
    initial_capital: float = config.INITIAL_CAPITAL,
) -> Dict:
    """Walk-forward validation across n_folds to prevent overfitting."""
    fold_size = len(all_rows) // n_folds
    results   = []

    for fold in range(n_folds - 1):
        start = fold * fold_size
        split = start + int(fold_size * train_pct)
        end   = (fold + 1) * fold_size

        test_rows = all_rows[split:end]
        result    = run_backtest(test_rows, engine, initial_capital)
        result["fold"] = fold
        results.append(result)
        log.info("Fold %d: return=%.2f%%  sharpe=%.3f",
                 fold, result["summary"]["total_return_pct"],
                 result["summary"]["sharpe_ratio"])

    rets    = [r["summary"]["total_return_pct"] for r in results]
    sharpes = [r["summary"]["sharpe_ratio"]     for r in results]
    dds     = [r["summary"]["max_drawdown_pct"] for r in results]
    return {
        "folds":          results,
        "avg_return_pct": round(float(np.mean(rets)), 2),
        "avg_sharpe":     round(float(np.mean(sharpes)), 3),
        "avg_drawdown":   round(float(np.mean(dds)), 2),
        "consistency":    round(float(np.std(rets)), 2),
    }
