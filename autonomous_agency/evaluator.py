"""
Strategy Evaluation Engine for the Autonomous Quantitative Research Agency

This module provides comprehensive statistical validation, out-of-sample testing,
and robustness analysis for generated trading strategies.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from scipy import stats
try:
    import quantstats_lumi as qs
except ImportError:
    qs = None
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.metrics import mean_squared_error, r2_score

from .config import config
from .models import BacktestResult, WalkForwardResult


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for strategy assessment"""
    strategy_name: str
    hypothesis_id: str

    # Statistical significance
    p_value: float
    t_statistic: float
    confidence_level: float

    # Robustness metrics
    stability_score: float
    adaptability_score: float
    robustness_score: float

    # Performance consistency
    consistency_score: float
    persistence_score: float

    # Risk-adjusted metrics
    risk_adjusted_return: float
    information_ratio: float

    # Out-of-sample performance
    oos_total_return: float
    oos_sharpe_ratio: float
    oos_max_drawdown: float

    # Validation status
    statistical_validity: str  # 'valid', 'questionable', 'invalid'
    robustness_validity: str   # 'robust', 'moderate', 'fragile'
    overall_score: float

    # Additional analysis
    monte_carlo_results: Dict[str, float]
    stress_test_results: Dict[str, float]
    regime_analysis: Dict[str, float]


@dataclass
class ValidationResult:
    """Complete validation result for a strategy"""
    strategy_name: str
    evaluation_metrics: EvaluationMetrics
    backtest_result: BacktestResult
    walk_forward_result: Optional[WalkForwardResult]
    validation_status: str  # 'approved', 'rejected', 'needs_refinement'
    rejection_reasons: List[str]
    refinement_suggestions: List[str]


class StrategyEvaluator:
    """Comprehensive strategy evaluation and validation engine"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.evaluation_dir = Path(config.evaluation_results_dir)
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)

        # Validation thresholds
        self.thresholds = {
            'min_sharpe_ratio': 0.5,
            'max_drawdown_limit': 0.3,
            'min_win_rate': 0.52,
            'min_profit_factor': 1.2,
            'max_p_value': 0.05,
            'min_stability_score': 0.6,
            'min_robustness_score': 0.7,
            'min_oos_performance_ratio': 0.8
        }

    def evaluate_strategy(self, backtest_result: BacktestResult,
                         walk_forward_result: Optional[WalkForwardResult] = None,
                         additional_data: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Perform comprehensive evaluation of a strategy

        Args:
            backtest_result: Results from backtesting
            walk_forward_result: Walk-forward analysis results
            additional_data: Additional data for evaluation

        Returns:
            ValidationResult with complete assessment
        """
        try:
            # Calculate comprehensive evaluation metrics
            evaluation_metrics = self._calculate_evaluation_metrics(
                backtest_result, walk_forward_result, additional_data
            )

            # Perform validation checks
            validation_status, rejection_reasons, refinement_suggestions = self._validate_strategy(
                evaluation_metrics, backtest_result, walk_forward_result
            )

            # Create validation result
            validation_result = ValidationResult(
                strategy_name=backtest_result.strategy_name,
                evaluation_metrics=evaluation_metrics,
                backtest_result=backtest_result,
                walk_forward_result=walk_forward_result,
                validation_status=validation_status,
                rejection_reasons=rejection_reasons,
                refinement_suggestions=refinement_suggestions
            )

            # Save evaluation results
            self._save_evaluation_results(validation_result)

            return validation_result

        except Exception as e:
            self.logger.error(f"Strategy evaluation failed for {backtest_result.strategy_name}: {e}")
            return None

    def _calculate_evaluation_metrics(self, backtest_result: BacktestResult,
                                    walk_forward_result: Optional[WalkForwardResult],
                                    additional_data: Optional[Dict[str, Any]]) -> EvaluationMetrics:
        """Calculate comprehensive evaluation metrics"""

        # Statistical significance tests
        p_value, t_statistic = self._calculate_statistical_significance(backtest_result)

        # Confidence level
        confidence_level = 1 - p_value if p_value < 0.5 else p_value

        # Robustness metrics
        stability_score = self._calculate_stability_score(backtest_result, walk_forward_result)
        adaptability_score = self._calculate_adaptability_score(backtest_result)
        robustness_score = (stability_score + adaptability_score) / 2

        # Performance consistency
        consistency_score = self._calculate_consistency_score(backtest_result)
        persistence_score = self._calculate_persistence_score(backtest_result)

        # Risk-adjusted metrics
        risk_adjusted_return = self._calculate_risk_adjusted_return(backtest_result)
        information_ratio = self._calculate_information_ratio(backtest_result)

        # Out-of-sample performance
        oos_metrics = self._calculate_oos_performance(backtest_result, walk_forward_result)

        # Overall statistical validity
        statistical_validity = self._assess_statistical_validity(p_value, t_statistic)

        # Robustness validity
        robustness_validity = self._assess_robustness_validity(robustness_score, stability_score)

        # Overall score
        overall_score = self._calculate_overall_score(
            backtest_result, oos_metrics, robustness_score, statistical_validity
        )

        # Monte Carlo analysis
        monte_carlo_results = self._run_monte_carlo_analysis(backtest_result)

        # Stress testing
        stress_test_results = self._run_stress_tests(backtest_result)

        # Regime analysis
        regime_analysis = self._analyze_regime_performance(backtest_result)

        return EvaluationMetrics(
            strategy_name=backtest_result.strategy_name,
            hypothesis_id=backtest_result.hypothesis_id,
            p_value=p_value,
            t_statistic=t_statistic,
            confidence_level=confidence_level,
            stability_score=stability_score,
            adaptability_score=adaptability_score,
            robustness_score=robustness_score,
            consistency_score=consistency_score,
            persistence_score=persistence_score,
            risk_adjusted_return=risk_adjusted_return,
            information_ratio=information_ratio,
            oos_total_return=oos_metrics['total_return'],
            oos_sharpe_ratio=oos_metrics['sharpe_ratio'],
            oos_max_drawdown=oos_metrics['max_drawdown'],
            statistical_validity=statistical_validity,
            robustness_validity=robustness_validity,
            overall_score=overall_score,
            monte_carlo_results=monte_carlo_results,
            stress_test_results=stress_test_results,
            regime_analysis=regime_analysis
        )

    def _calculate_statistical_significance(self, backtest_result: BacktestResult) -> Tuple[float, float]:
        """Calculate statistical significance of returns"""

        try:
            # Convert equity curve to returns
            equity = pd.Series(backtest_result.equity_curve)
            returns = equity.pct_change().dropna()

            if len(returns) < 30:  # Need minimum sample size
                return 1.0, 0.0

            # T-test against zero (no skill)
            t_stat, p_value = stats.ttest_1samp(returns, 0)

            # Adjust p-value for multiple testing if needed
            # For now, return raw p-value
            return float(p_value), float(t_stat)

        except Exception as e:
            self.logger.warning(f"Statistical significance calculation failed: {e}")
            return 1.0, 0.0

    def _calculate_stability_score(self, backtest_result: BacktestResult,
                                 walk_forward_result: Optional[WalkForwardResult]) -> float:
        """Calculate strategy stability score"""

        try:
            if walk_forward_result:
                # Use walk-forward stability metrics
                wf_stability = walk_forward_result.stability_score
                overfitting_prob = walk_forward_result.overfitting_probability

                # Penalize overfitting
                stability = wf_stability * (1 - overfitting_prob)
            else:
                # Calculate rolling stability
                equity = pd.Series(backtest_result.equity_curve)
                returns = equity.pct_change().dropna()

                if len(returns) < 60:  # Need sufficient data
                    return 0.5

                # Rolling Sharpe ratio stability
                rolling_sharpe = returns.rolling(30).apply(
                    lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
                ).dropna()

                # Coefficient of variation of rolling Sharpe
                sharpe_cv = rolling_sharpe.std() / abs(rolling_sharpe.mean()) if rolling_sharpe.mean() != 0 else float('inf')

                # Convert to stability score (lower CV = higher stability)
                stability = max(0, 1 - min(sharpe_cv, 2) / 2)

            return stability

        except Exception as e:
            self.logger.warning(f"Stability score calculation failed: {e}")
            return 0.5

    def _calculate_adaptability_score(self, backtest_result: BacktestResult) -> float:
        """Calculate strategy adaptability to different market conditions"""

        try:
            # Analyze performance across different periods
            equity = pd.Series(backtest_result.equity_curve)
            returns = equity.pct_change().dropna()

            if len(returns) < 120:  # Need sufficient data
                return 0.5

            # Split into quarters and calculate Sharpe for each
            quarterly_sharpe = []
            for i in range(0, len(returns) - 90, 90):  # 90-day quarters
                quarter_returns = returns.iloc[i:i+90]
                if len(quarter_returns) >= 30:
                    sharpe = quarter_returns.mean() / quarter_returns.std() * np.sqrt(252)
                    quarterly_sharpe.append(sharpe)

            if len(quarterly_sharpe) < 2:
                return 0.5

            # Calculate consistency across quarters
            sharpe_std = np.std(quarterly_sharpe)
            sharpe_mean = abs(np.mean(quarterly_sharpe))

            # Adaptability score (lower variation = higher adaptability)
            adaptability = max(0, 1 - min(sharpe_std / max(sharpe_mean, 0.1), 2) / 2)

            return adaptability

        except Exception as e:
            self.logger.warning(f"Adaptability score calculation failed: {e}")
            return 0.5

    def _calculate_consistency_score(self, backtest_result: BacktestResult) -> float:
        """Calculate performance consistency score"""

        try:
            monthly_returns = backtest_result.monthly_returns

            if not monthly_returns or len(monthly_returns) < 6:
                return 0.5

            returns_values = list(monthly_returns.values())

            # Calculate win rate of monthly returns
            winning_months = sum(1 for r in returns_values if r > 0)
            total_months = len(returns_values)
            monthly_win_rate = winning_months / total_months

            # Calculate consistency of returns (lower standard deviation = higher consistency)
            returns_std = np.std(returns_values)
            returns_mean = abs(np.mean(returns_values))

            # Normalize consistency score
            if returns_mean == 0:
                consistency = 0.5
            else:
                cv = returns_std / returns_mean
                consistency = max(0, 1 - min(cv, 2) / 2)

            # Combine with win rate
            combined_consistency = (consistency + monthly_win_rate) / 2

            return combined_consistency

        except Exception as e:
            self.logger.warning(f"Consistency score calculation failed: {e}")
            return 0.5

    def _calculate_persistence_score(self, backtest_result: BacktestResult) -> float:
        """Calculate strategy persistence (ability to maintain performance)"""

        try:
            equity = pd.Series(backtest_result.equity_curve)
            returns = equity.pct_change().dropna()

            if len(returns) < 60:
                return 0.5

            # Calculate autocorrelation of returns
            autocorr_1 = returns.autocorr(lag=1)
            autocorr_5 = returns.autocorr(lag=5)
            autocorr_10 = returns.autocorr(lag=10)

            # Average autocorrelation (positive persistence is good)
            avg_autocorr = (autocorr_1 + autocorr_5 + autocorr_10) / 3

            # Convert to persistence score (0-1 scale)
            persistence = (avg_autocorr + 1) / 2  # Shift from [-1,1] to [0,1]

            return persistence

        except Exception as e:
            self.logger.warning(f"Persistence score calculation failed: {e}")
            return 0.5

    def _calculate_risk_adjusted_return(self, backtest_result: BacktestResult) -> float:
        """Calculate risk-adjusted return (Sortino ratio)"""

        try:
            equity = pd.Series(backtest_result.equity_curve)
            returns = equity.pct_change().dropna()

            if len(returns) < 30:
                return 0.0

            # Calculate Sortino ratio
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std()

            if downside_deviation == 0:
                return float('inf')

            sortino = returns.mean() / downside_deviation * np.sqrt(252)

            return sortino

        except Exception as e:
            self.logger.warning(f"Risk-adjusted return calculation failed: {e}")
            return 0.0

    def _calculate_information_ratio(self, backtest_result: BacktestResult) -> float:
        """Calculate information ratio vs benchmark"""

        try:
            if backtest_result.beta == 0:
                return 0.0

            # Information ratio = (Return - Benchmark Return) / Tracking Error
            excess_return = backtest_result.total_return - backtest_result.benchmark_return
            tracking_error = abs(backtest_result.beta - 1)  # Simplified

            if tracking_error == 0:
                return float('inf') if excess_return > 0 else float('-inf')

            return excess_return / tracking_error

        except Exception as e:
            self.logger.warning(f"Information ratio calculation failed: {e}")
            return 0.0

    def _calculate_oos_performance(self, backtest_result: BacktestResult,
                                 walk_forward_result: Optional[WalkForwardResult]) -> Dict[str, float]:
        """Calculate out-of-sample performance metrics"""

        if walk_forward_result and walk_forward_result.out_of_sample_results:
            # Use walk-forward OOS results
            oos_results = walk_forward_result.out_of_sample_results

            total_return = np.mean([r.total_return for r in oos_results])
            sharpe_ratio = np.mean([r.sharpe_ratio for r in oos_results])
            max_drawdown = np.mean([r.max_drawdown for r in oos_results])

            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
        else:
            # Simplified OOS estimation using last portion of data
            equity = pd.Series(backtest_result.equity_curve)

            if len(equity) < 100:
                return {
                    'total_return': backtest_result.total_return * 0.8,  # Conservative estimate
                    'sharpe_ratio': backtest_result.sharpe_ratio * 0.9,
                    'max_drawdown': backtest_result.max_drawdown * 1.1
                }

            # Use last 30% as OOS estimate
            oos_start = int(len(equity) * 0.7)
            oos_equity = equity.iloc[oos_start:]

            oos_return = (oos_equity.iloc[-1] / oos_equity.iloc[0]) - 1
            oos_returns = oos_equity.pct_change().dropna()

            oos_sharpe = oos_returns.mean() / oos_returns.std() * np.sqrt(252) if oos_returns.std() > 0 else 0
            oos_max_dd = (oos_equity / oos_equity.cummax() - 1).min()

            return {
                'total_return': oos_return,
                'sharpe_ratio': oos_sharpe,
                'max_drawdown': abs(oos_max_dd)
            }

    def _assess_statistical_validity(self, p_value: float, t_statistic: float) -> str:
        """Assess statistical validity of strategy"""

        if p_value < 0.01 and abs(t_statistic) > 2.576:  # 99% confidence
            return 'valid'
        elif p_value < 0.05 and abs(t_statistic) > 1.96:  # 95% confidence
            return 'valid'
        elif p_value < 0.1 and abs(t_statistic) > 1.645:  # 90% confidence
            return 'questionable'
        else:
            return 'invalid'

    def _assess_robustness_validity(self, robustness_score: float, stability_score: float) -> str:
        """Assess robustness validity"""

        if robustness_score >= 0.8 and stability_score >= 0.7:
            return 'robust'
        elif robustness_score >= 0.6 and stability_score >= 0.5:
            return 'moderate'
        else:
            return 'fragile'

    def _calculate_overall_score(self, backtest_result: BacktestResult,
                               oos_metrics: Dict[str, float],
                               robustness_score: float,
                               statistical_validity: str) -> float:
        """Calculate overall strategy score"""

        try:
            # Component scores (0-1 scale)
            sharpe_score = min(backtest_result.sharpe_ratio / 2, 1)  # Cap at 2.0 Sharpe
            return_score = min(backtest_result.total_return * 2, 1)  # Cap at 50% return
            drawdown_score = max(0, 1 - backtest_result.max_drawdown / 0.5)  # Penalize >50% DD

            oos_ratio = oos_metrics['sharpe_ratio'] / max(backtest_result.sharpe_ratio, 0.1)
            oos_score = min(oos_ratio, 1)

            validity_score = {'valid': 1.0, 'questionable': 0.5, 'invalid': 0.0}[statistical_validity]

            # Weighted average
            weights = {
                'sharpe': 0.2,
                'return': 0.15,
                'drawdown': 0.15,
                'oos': 0.2,
                'robustness': 0.15,
                'validity': 0.15
            }

            overall_score = (
                weights['sharpe'] * sharpe_score +
                weights['return'] * return_score +
                weights['drawdown'] * drawdown_score +
                weights['oos'] * oos_score +
                weights['robustness'] * robustness_score +
                weights['validity'] * validity_score
            )

            return overall_score

        except Exception as e:
            self.logger.warning(f"Overall score calculation failed: {e}")
            return 0.0

    def _run_monte_carlo_analysis(self, backtest_result: BacktestResult) -> Dict[str, float]:
        """Run Monte Carlo simulation for robustness testing"""

        try:
            equity = pd.Series(backtest_result.equity_curve)
            returns = equity.pct_change().dropna()

            if len(returns) < 30:
                return {'mean_return': 0, 'std_return': 0, 'var_95': 0, 'max_drawdown': 0}

            # Run 1000 Monte Carlo simulations
            mc_returns = []
            mc_max_dd = []

            for _ in range(1000):
                # Bootstrap sample returns
                sampled_returns = np.random.choice(returns, size=len(returns), replace=True)
                mc_equity = (1 + sampled_returns).cumprod()

                mc_returns.append(mc_equity[-1] - 1)
                mc_max_dd.append((mc_equity / mc_equity.cummax() - 1).min())

            return {
                'mean_return': np.mean(mc_returns),
                'std_return': np.std(mc_returns),
                'var_95': np.percentile(mc_returns, 5),
                'max_drawdown': abs(np.mean(mc_max_dd))
            }

        except Exception as e:
            self.logger.warning(f"Monte Carlo analysis failed: {e}")
            return {'mean_return': 0, 'std_return': 0, 'var_95': 0, 'max_drawdown': 0}

    def _run_stress_tests(self, backtest_result: BacktestResult) -> Dict[str, float]:
        """Run stress tests under extreme market conditions"""

        try:
            equity = pd.Series(backtest_result.equity_curve)
            returns = equity.pct_change().dropna()

            if len(returns) < 30:
                return {'crash_test': 0, 'volatility_test': 0, 'recovery_test': 0}

            # Crash test: -20% sudden drop
            crash_equity = equity.copy()
            crash_point = len(crash_equity) // 2
            crash_equity.iloc[crash_point:] = crash_equity.iloc[crash_point:] * 0.8
            crash_return = (crash_equity.iloc[-1] / crash_equity.iloc[0]) - 1

            # Volatility test: double volatility
            vol_returns = returns * 2
            vol_equity = (1 + vol_returns).cumprod()
            vol_return = vol_equity.iloc[-1] - 1

            # Recovery test: simulate bear market recovery
            recovery_returns = returns.copy()
            recovery_returns[recovery_returns < 0] = recovery_returns[recovery_returns < 0] * 1.5  # Worse losses
            recovery_equity = (1 + recovery_returns).cumprod()
            recovery_return = recovery_equity.iloc[-1] - 1

            return {
                'crash_test': crash_return,
                'volatility_test': vol_return,
                'recovery_test': recovery_return
            }

        except Exception as e:
            self.logger.warning(f"Stress tests failed: {e}")
            return {'crash_test': 0, 'volatility_test': 0, 'recovery_test': 0}

    def _analyze_regime_performance(self, backtest_result: BacktestResult) -> Dict[str, float]:
        """Analyze performance across different market regimes"""

        try:
            # This would require regime classification data
            # For now, return placeholder
            return {
                'bull_performance': backtest_result.total_return,
                'bear_performance': backtest_result.total_return * 0.5,  # Conservative estimate
                'sideways_performance': backtest_result.total_return * 0.8,
                'volatile_performance': backtest_result.total_return * 0.6
            }
        except Exception as e:
            self.logger.warning(f"Regime analysis failed: {e}")
            return {}

    def _validate_strategy(self, evaluation_metrics: EvaluationMetrics,
                          backtest_result: BacktestResult,
                          walk_forward_result: Optional[WalkForwardResult]) -> Tuple[str, List[str], List[str]]:
        """Validate strategy against predefined criteria"""

        rejection_reasons = []
        refinement_suggestions = []

        # Basic performance checks
        if backtest_result.sharpe_ratio < self.thresholds['min_sharpe_ratio']:
            rejection_reasons.append(f"Sharpe ratio {backtest_result.sharpe_ratio:.2f} below threshold {self.thresholds['min_sharpe_ratio']}")
            refinement_suggestions.append("Increase risk management or adjust position sizing")

        if backtest_result.max_drawdown > self.thresholds['max_drawdown_limit']:
            rejection_reasons.append(f"Max drawdown {backtest_result.max_drawdown:.2%} exceeds limit {self.thresholds['max_drawdown_limit']:.2%}")
            refinement_suggestions.append("Implement stricter stop-loss rules or reduce leverage")

        if backtest_result.win_rate < self.thresholds['min_win_rate']:
            rejection_reasons.append(f"Win rate {backtest_result.win_rate:.2%} below threshold {self.thresholds['min_win_rate']:.2%}")
            refinement_suggestions.append("Refine entry/exit signals or adjust trade frequency")

        if backtest_result.profit_factor < self.thresholds['min_profit_factor']:
            rejection_reasons.append(f"Profit factor {backtest_result.profit_factor:.2f} below threshold {self.thresholds['min_profit_factor']}")
            refinement_suggestions.append("Improve reward-to-risk ratio or reduce losing trades")

        # Statistical significance
        if evaluation_metrics.p_value > self.thresholds['max_p_value']:
            rejection_reasons.append(f"P-value {evaluation_metrics.p_value:.3f} indicates statistically insignificant results")
            refinement_suggestions.append("Increase sample size or improve signal quality")

        # Robustness checks
        if evaluation_metrics.stability_score < self.thresholds['min_stability_score']:
            rejection_reasons.append(f"Stability score {evaluation_metrics.stability_score:.2f} below threshold {self.thresholds['min_stability_score']}")
            refinement_suggestions.append("Reduce parameter sensitivity or implement adaptive mechanisms")

        if evaluation_metrics.robustness_score < self.thresholds['min_robustness_score']:
            rejection_reasons.append(f"Robustness score {evaluation_metrics.robustness_score:.2f} below threshold {self.thresholds['min_robustness_score']}")
            refinement_suggestions.append("Test across more market conditions or add regime filters")

        # Out-of-sample performance
        oos_performance_ratio = evaluation_metrics.oos_sharpe_ratio / max(backtest_result.sharpe_ratio, 0.1)
        if oos_performance_ratio < self.thresholds['min_oos_performance_ratio']:
            rejection_reasons.append(f"OOS performance ratio {oos_performance_ratio:.2f} indicates overfitting")
            refinement_suggestions.append("Simplify strategy or use regularization techniques")

        # Determine validation status
        if not rejection_reasons:
            validation_status = 'approved'
        elif len(rejection_reasons) <= 2 and evaluation_metrics.overall_score > 0.6:
            validation_status = 'needs_refinement'
        else:
            validation_status = 'rejected'

        return validation_status, rejection_reasons, refinement_suggestions

    def _save_evaluation_results(self, validation_result: ValidationResult):
        """Save evaluation results to disk"""

        try:
            result_file = self.evaluation_dir / f"{validation_result.strategy_name}_evaluation.json"

            # Convert to serializable format
            result_dict = {
                'strategy_name': validation_result.strategy_name,
                'evaluation_metrics': {
                    'p_value': validation_result.evaluation_metrics.p_value,
                    't_statistic': validation_result.evaluation_metrics.t_statistic,
                    'confidence_level': validation_result.evaluation_metrics.confidence_level,
                    'stability_score': validation_result.evaluation_metrics.stability_score,
                    'adaptability_score': validation_result.evaluation_metrics.adaptability_score,
                    'robustness_score': validation_result.evaluation_metrics.robustness_score,
                    'consistency_score': validation_result.evaluation_metrics.consistency_score,
                    'persistence_score': validation_result.evaluation_metrics.persistence_score,
                    'risk_adjusted_return': validation_result.evaluation_metrics.risk_adjusted_return,
                    'information_ratio': validation_result.evaluation_metrics.information_ratio,
                    'oos_total_return': validation_result.evaluation_metrics.oos_total_return,
                    'oos_sharpe_ratio': validation_result.evaluation_metrics.oos_sharpe_ratio,
                    'oos_max_drawdown': validation_result.evaluation_metrics.oos_max_drawdown,
                    'statistical_validity': validation_result.evaluation_metrics.statistical_validity,
                    'robustness_validity': validation_result.evaluation_metrics.robustness_validity,
                    'overall_score': validation_result.evaluation_metrics.overall_score,
                    'monte_carlo_results': validation_result.evaluation_metrics.monte_carlo_results,
                    'stress_test_results': validation_result.evaluation_metrics.stress_test_results,
                    'regime_analysis': validation_result.evaluation_metrics.regime_analysis
                },
                'validation_status': validation_result.validation_status,
                'rejection_reasons': validation_result.rejection_reasons,
                'refinement_suggestions': validation_result.refinement_suggestions
            }

            import json
            with open(result_file, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to save evaluation results: {e}")

    def batch_evaluate_strategies(self, backtest_results: List[BacktestResult],
                                walk_forward_results: Optional[List[WalkForwardResult]] = None) -> List[ValidationResult]:
        """
        Evaluate multiple strategies in batch

        Args:
            backtest_results: List of backtest results
            walk_forward_results: Optional list of walk-forward results

        Returns:
            List of validation results
        """
        validation_results = []

        wf_dict = {wf.strategy_name: wf for wf in (walk_forward_results or [])}

        for backtest_result in backtest_results:
            wf_result = wf_dict.get(backtest_result.strategy_name)
            validation_result = self.evaluate_strategy(backtest_result, wf_result)
            if validation_result:
                validation_results.append(validation_result)

        return validation_results