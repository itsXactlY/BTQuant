"""
Out-of-Sample Validation Module

Framework for out-of-sample validation to test strategy robustness
and prevent overfitting to historical data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from datetime import datetime
from ..metrics.performance_metrics import PerformanceMetrics
from ..metrics.statistical_significance import StatisticalSignificance

class OutOfSampleValidation:
    """Class for out-of-sample validation of trading strategies"""
    
    def __init__(self):
        self.logger = logging.getLogger('OutOfSampleValidation')
        self.logger.info("OutOfSampleValidation initialized")
        
        # Initialize metrics calculators
        self.performance_metrics = PerformanceMetrics()
        self.statistical_significance = StatisticalSignificance()
    
    def split_data_in_sample_out_of_sample(self, data: pd.DataFrame, 
                                          split_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into in-sample and out-of-sample periods
        
        Args:
            data: Historical market data
            split_ratio: Ratio of data to use for in-sample (default: 0.7)
            
        Returns:
            Tuple of (in_sample_data, out_of_sample_data)
        """
        self.logger.info(f"Splitting data with {split_ratio*100}% in-sample ratio")
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
            
        if 'date' not in data.columns and data.index.name != 'date':
            raise ValueError("Data must have a date column or date index")
            
        # Ensure data is sorted by date
        if 'date' in data.columns:
            data = data.sort_values('date')
        else:
            data = data.sort_index()
            
        # Calculate split point
        split_idx = int(len(data) * split_ratio)
        
        # Split data
        in_sample = data.iloc[:split_idx]
        out_of_sample = data.iloc[split_idx:]
        
        self.logger.info(f"In-sample: {len(in_sample)} records, Out-of-sample: {len(out_of_sample)} records")
        
        return in_sample, out_of_sample
    
    def validate_strategy_out_of_sample(self, strategy: Dict[str, Any], 
                                       in_sample_data: pd.DataFrame, 
                                       out_of_sample_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate strategy performance on out-of-sample data
        
        Args:
            strategy: Strategy to validate
            in_sample_data: In-sample historical data
            out_of_sample_data: Out-of-sample historical data
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info(f"Validating strategy {strategy.get('id', 'unknown')} out-of-sample")
        
        # Run backtests on both datasets
        in_sample_results = self._run_backtest(strategy, in_sample_data)
        out_of_sample_results = self._run_backtest(strategy, out_of_sample_data)
        
        # Calculate performance metrics
        in_sample_metrics = self.performance_metrics.calculate_all_metrics(
            in_sample_results['returns']
        )
        out_of_sample_metrics = self.performance_metrics.calculate_all_metrics(
            out_of_sample_results['returns']
        )
        
        # Calculate statistical significance
        significance_results = self._calculate_significance(
            in_sample_results['returns'], 
            out_of_sample_results['returns']
        )
        
        # Check for overfitting
        overfitting_analysis = self._analyze_overfitting(
            in_sample_metrics, out_of_sample_metrics
        )
        
        return {
            'strategy_id': strategy.get('id', 'unknown'),
            'template': strategy.get('template', 'unknown'),
            'in_sample_performance': in_sample_metrics,
            'out_of_sample_performance': out_of_sample_metrics,
            'statistical_significance': significance_results,
            'overfitting_analysis': overfitting_analysis,
            'validation_passed': overfitting_analysis['overfitting_detected'] == False
        }
    
    def _run_backtest(self, strategy: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run a simplified backtest to get returns data
        
        Args:
            strategy: Strategy to backtest
            data: Historical data for backtesting
            
        Returns:
            Dictionary with backtest results including returns
        """
        # This is a simplified backtest - in a real implementation, this would
        # integrate with the actual backtesting engine
        
        # Generate random returns for demonstration
        # In practice, this would run the actual strategy
        n_periods = len(data)
        
        # Simulate strategy returns based on template
        if strategy.get('template') == 'mean_reversion':
            # Mean reversion strategy simulation
            returns = np.random.normal(0.001, 0.01, n_periods)
        elif strategy.get('template') == 'trend_following':
            # Trend following strategy simulation
            returns = np.random.normal(0.0015, 0.012, n_periods)
        else:
            # Default strategy simulation
            returns = np.random.normal(0.0008, 0.01, n_periods)
            
        return {
            'returns': returns,
            'sharpe_ratio': self.performance_metrics.calculate_sharpe_ratio(returns),
            'max_drawdown': self.performance_metrics.calculate_max_drawdown(returns)
        }
    
    def _calculate_significance(self, in_sample_returns: np.ndarray, 
                               out_of_sample_returns: np.ndarray) -> Dict[str, Any]:
        """
        Calculate statistical significance between in-sample and out-of-sample results
        
        Args:
            in_sample_returns: In-sample returns
            out_of_sample_returns: Out-of-sample returns
            
        Returns:
            Dictionary with significance test results
        """
        # Test if performance difference is statistically significant
        # Ensure arrays have compatible shapes
        min_length = min(len(out_of_sample_returns), len(in_sample_returns))
        if min_length > 0:
            out_of_sample_returns = out_of_sample_returns[:min_length]
            in_sample_returns = in_sample_returns[:min_length]
        
        p_value = self.statistical_significance.calculate_p_value(
            out_of_sample_returns, in_sample_returns
        )
        
        # Hypothesis test: is out-of-sample performance different from in-sample?
        # Use difference in means instead of array subtraction
        mean_diff = np.mean(out_of_sample_returns) - np.mean(in_sample_returns)
        hypothesis_test = self.statistical_significance.perform_hypothesis_test(
            out_of_sample_returns - in_sample_returns,
            null_hypothesis=0.0,
            alternative='two-sided'
        )
        
        return {
            'p_value': p_value,
            'hypothesis_test': hypothesis_test,
            'performance_degradation_significant': p_value < 0.05
        }
    
    def _analyze_overfitting(self, in_sample_metrics: Dict[str, float], 
                            out_of_sample_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze potential overfitting by comparing in-sample and out-of-sample performance
        
        Args:
            in_sample_metrics: In-sample performance metrics
            out_of_sample_metrics: Out-of-sample performance metrics
            
        Returns:
            Dictionary with overfitting analysis
        """
        # Calculate performance degradation
        sharpe_degradation = ((out_of_sample_metrics['sharpe_ratio'] - in_sample_metrics['sharpe_ratio']) /
                             (in_sample_metrics['sharpe_ratio'] + 1e-6)) * 100
        
        return_degradation = ((out_of_sample_metrics['total_return'] - in_sample_metrics['total_return']) /
                             (in_sample_metrics['total_return'] + 1e-6)) * 100
        
        drawdown_increase = ((out_of_sample_metrics['max_drawdown'] - in_sample_metrics['max_drawdown']) /
                           (abs(in_sample_metrics['max_drawdown']) + 1e-6)) * 100
        
        # Determine if overfitting is detected
        overfitting_detected = (
            sharpe_degradation < -20 or  # More than 20% Sharpe ratio degradation
            return_degradation < -30 or  # More than 30% return degradation
            drawdown_increase > 50       # More than 50% drawdown increase
        )
        
        return {
            'sharpe_degradation_percent': sharpe_degradation,
            'return_degradation_percent': return_degradation,
            'drawdown_increase_percent': drawdown_increase,
            'overfitting_detected': overfitting_detected,
            'overfitting_severity': self._calculate_overfitting_severity(
                sharpe_degradation, return_degradation, drawdown_increase
            )
        }
    
    def _calculate_overfitting_severity(self, sharpe_degradation: float, 
                                       return_degradation: float, 
                                       drawdown_increase: float) -> str:
        """
        Calculate severity of overfitting
        
        Args:
            sharpe_degradation: Percentage degradation in Sharpe ratio
            return_degradation: Percentage degradation in returns
            drawdown_increase: Percentage increase in drawdown
            
        Returns:
            Severity level as string
        """
        # Count severe degradation indicators
        severe_indicators = 0
        
        if sharpe_degradation < -30:
            severe_indicators += 1
        if return_degradation < -50:
            severe_indicators += 1
        if drawdown_increase > 100:
            severe_indicators += 1
            
        if severe_indicators >= 2:
            return 'severe'
        elif severe_indicators == 1:
            return 'moderate'
        elif sharpe_degradation < -10 or return_degradation < -20 or drawdown_increase > 30:
            return 'mild'
        else:
            return 'none'
    
    def perform_time_series_cross_validation(self, strategy: Dict[str, Any], 
                                            data: pd.DataFrame, 
                                            n_folds: int = 5, 
                                            fold_size_ratio: float = 0.2) -> Dict[str, Any]:
        """
        Perform time-series cross-validation (walk-forward validation)
        
        Args:
            strategy: Strategy to validate
            data: Historical market data
            n_folds: Number of validation folds
            fold_size_ratio: Size of each validation fold as ratio of total data
            
        Returns:
            Dictionary with cross-validation results
        """
        self.logger.info(f"Performing time-series cross-validation with {n_folds} folds")
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
            
        # Ensure data is sorted by date
        if 'date' in data.columns:
            data = data.sort_values('date')
        else:
            data = data.sort_index()
            
        # Calculate fold sizes
        total_size = len(data)
        fold_size = int(total_size * fold_size_ratio)
        
        # Initialize results storage
        fold_results = []
        
        for i in range(n_folds):
            self.logger.debug(f"Processing fold {i+1}/{n_folds}")
            
            # Calculate fold boundaries
            train_end = total_size - (n_folds - i) * fold_size
            val_end = total_size - (n_folds - i - 1) * fold_size
            
            # Split data
            train_data = data.iloc[:train_end]
            val_data = data.iloc[train_end:val_end]
            
            # Run backtest on validation fold
            val_results = self._run_backtest(strategy, val_data)
            
            # Calculate metrics
            metrics = self.performance_metrics.calculate_all_metrics(val_results['returns'])
            
            fold_results.append({
                'fold': i + 1,
                'train_size': len(train_data),
                'validation_size': len(val_data),
                'performance_metrics': metrics,
                'sharpe_ratio': metrics['sharpe_ratio'],
                'total_return': metrics['total_return']
            })
        
        # Calculate overall statistics
        sharpe_ratios = [r['sharpe_ratio'] for r in fold_results]
        total_returns = [r['total_return'] for r in fold_results]
        
        return {
            'strategy_id': strategy.get('id', 'unknown'),
            'template': strategy.get('template', 'unknown'),
            'folds': fold_results,
            'average_sharpe_ratio': np.mean(sharpe_ratios),
            'sharpe_ratio_std': np.std(sharpe_ratios),
            'average_total_return': np.mean(total_returns),
            'total_return_std': np.std(total_returns),
            'consistency_score': self._calculate_consistency_score(sharpe_ratios)
        }
    
    def _calculate_consistency_score(self, sharpe_ratios: List[float]) -> float:
        """
        Calculate consistency score based on Sharpe ratio variability
        
        Args:
            sharpe_ratios: List of Sharpe ratios from different folds
            
        Returns:
            Consistency score between 0 and 1
        """
        if len(sharpe_ratios) < 2:
            return 0.0
            
        # Calculate coefficient of variation
        mean_sharpe = np.mean(sharpe_ratios)
        std_sharpe = np.std(sharpe_ratios)
        
        if mean_sharpe == 0:
            return 0.0
            
        # Lower CV means higher consistency
        cv = std_sharpe / abs(mean_sharpe)
        
        # Convert to consistency score (0-1)
        consistency_score = max(0, 1 - cv)
        
        return consistency_score
    
    def analyze_robustness_across_market_conditions(self, strategy: Dict[str, Any], 
                                                   data: pd.DataFrame, 
                                                   market_conditions: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze strategy robustness across different market conditions
        
        Args:
            strategy: Strategy to analyze
            data: Full historical data
            market_conditions: Dictionary mapping condition names to filtered data
            
        Returns:
            Dictionary with robustness analysis results
        """
        self.logger.info("Analyzing strategy robustness across market conditions")
        
        condition_results = {}
        
        for condition_name, condition_data in market_conditions.items():
            if len(condition_data) == 0:
                continue
                
            # Run backtest on specific market condition
            results = self._run_backtest(strategy, condition_data)
            
            # Calculate metrics
            metrics = self.performance_metrics.calculate_all_metrics(results['returns'])
            
            condition_results[condition_name] = {
                'sample_size': len(condition_data),
                'performance_metrics': metrics,
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown']
            }
        
        # Calculate overall robustness metrics
        sharpe_ratios = [r['sharpe_ratio'] for r in condition_results.values()]
        max_drawdowns = [r['max_drawdown'] for r in condition_results.values()]
        
        return {
            'strategy_id': strategy.get('id', 'unknown'),
            'template': strategy.get('template', 'unknown'),
            'market_conditions': condition_results,
            'sharpe_ratio_range': max(sharpe_ratios) - min(sharpe_ratios),
            'max_drawdown_range': max(max_drawdowns) - min(max_drawdowns),
            'robustness_score': self._calculate_robustness_score(sharpe_ratios, max_drawdowns)
        }
    
    def _calculate_robustness_score(self, sharpe_ratios: List[float], 
                                   max_drawdowns: List[float]) -> float:
        """
        Calculate robustness score based on performance consistency
        
        Args:
            sharpe_ratios: List of Sharpe ratios across conditions
            max_drawdowns: List of max drawdowns across conditions
            
        Returns:
            Robustness score between 0 and 1
        """
        if len(sharpe_ratios) < 2:
            return 0.0
            
        # Calculate normalized scores
        sharpe_cv = np.std(sharpe_ratios) / (np.mean(sharpe_ratios) + 1e-6)
        drawdown_cv = np.std(max_drawdowns) / (np.mean(max_drawdowns) + 1e-6)
        
        # Combine into overall robustness score
        robustness_score = 1 - (sharpe_cv + drawdown_cv) / 2
        
        return max(0, min(1, robustness_score))