from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict


@dataclass
class BacktestResult:
    """Comprehensive backtest result data"""
    strategy_name: str
    hypothesis_id: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float
    alpha: float
    beta: float
    benchmark_return: float
    start_date: datetime
    end_date: datetime
    num_trades: int
    avg_trade_duration: timedelta
    equity_curve: List[float]
    drawdown_curve: List[float]
    monthly_returns: Dict[str, float]
    regime_performance: Dict[str, Dict[str, float]]
    risk_metrics: Dict[str, float]
    validation_status: str = "pending"


@dataclass
class WalkForwardResult:
    """Walk-forward analysis results"""
    strategy_name: str
    in_sample_results: List['BacktestResult']
    out_of_sample_results: List['BacktestResult']
    walk_forward_sharpe: float
    walk_forward_return: float
    overfitting_probability: float
    stability_score: float