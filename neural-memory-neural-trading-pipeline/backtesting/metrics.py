from __future__ import annotations

import numpy as np


def compute_metrics(pnl: np.ndarray, trades: list[dict], bars_per_year: int = 365 * 24 * 4) -> dict[str, float]:
    cumulative = np.cumsum(pnl)
    total_return = float(cumulative[-1]) if len(cumulative) else 0.0
    returns = np.diff(cumulative, prepend=0.0)
    denom = np.std(returns) + 1e-12
    sharpe = float(np.sqrt(bars_per_year) * np.mean(returns) / denom)

    downside = returns[returns < 0]
    sortino = float(np.sqrt(bars_per_year) * np.mean(returns) / (np.std(downside) + 1e-12))

    cummax = np.maximum.accumulate(cumulative) if len(cumulative) else np.array([0.0])
    drawdown = (cumulative - cummax) / np.maximum(cummax, 1.0) if len(cumulative) else np.array([0.0])
    max_dd = float(np.min(drawdown))
    calmar = float(total_return / abs(max_dd)) if max_dd != 0 else 0.0

    wins = [t for t in trades if t.get("net_pnl", 0.0) > 0]
    losses = [t for t in trades if t.get("net_pnl", 0.0) <= 0]
    win_rate = float(len(wins) / len(trades)) if trades else 0.0
    avg_win = float(np.mean([t["net_pnl"] for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([t["net_pnl"] for t in losses])) if losses else 0.0
    total_win = sum(t["net_pnl"] for t in wins)
    total_loss = sum(t["net_pnl"] for t in losses)
    profit_factor = float(abs(total_win / total_loss)) if losses and total_loss != 0 else float("inf")
    exit_efficiency = float(np.mean([t.get("net_pnl", 0) / max(t.get("mfe", 0.01), 0.01) for t in trades])) if trades else 0.0

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_dd,
        "total_trades": float(len(trades)),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "exit_efficiency": exit_efficiency,
    }
