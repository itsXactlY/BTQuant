"""
Backtesting Module

This module handles the validation of generated strategies using historical data
to assess performance and robustness.
"""

from .backtest_engine import BacktestEngine
from .metrics.performance_metrics import PerformanceMetrics
from .metrics.statistical_significance import StatisticalSignificance
from .validation.out_of_sample_validation import OutOfSampleValidation
from .optimization.walk_forward_optimization import WalkForwardOptimization
from .risk_management.risk_management import RiskManagement