"""
Statistical Significance Testing Module

Framework for statistical significance testing of backtesting results.
Includes hypothesis testing, p-value calculations, and confidence intervals.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from scipy import stats
from scipy.stats import t, norm

class StatisticalSignificance:
    """Class for statistical significance testing of backtesting results"""
    
    def __init__(self):
        self.logger = logging.getLogger('StatisticalSignificance')
        self.logger.info("StatisticalSignificance initialized")
    
    def calculate_p_value(self, returns: Union[List[float], np.ndarray, pd.Series], 
                         benchmark_returns: Optional[Union[List[float], np.ndarray, pd.Series]] = None) -> float:
        """
        Calculate p-value for strategy returns using t-test
        
        Args:
            returns: Strategy returns
            benchmark_returns: Optional benchmark returns for comparison
            
        Returns:
            P-value indicating statistical significance
        """
        self.logger.info("Calculating p-value for strategy returns")
        
        returns_array = self._convert_to_array(returns)
        
        if benchmark_returns is not None:
            benchmark_array = self._convert_to_array(benchmark_returns)
            # Ensure arrays have the same length for paired t-test
            min_length = min(len(returns_array), len(benchmark_array))
            if min_length > 0:
                returns_array = returns_array[:min_length]
                benchmark_array = benchmark_array[:min_length]
            # Paired t-test against benchmark
            t_stat, p_value = stats.ttest_rel(returns_array, benchmark_array)
        else:
            # One-sample t-test against zero mean
            t_stat, p_value = stats.ttest_1samp(returns_array, 0)
            
        return p_value
    
    def calculate_confidence_intervals(self, returns: Union[List[float], np.ndarray, pd.Series], 
                                     confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate confidence intervals for mean return
        
        Args:
            returns: Strategy returns
            confidence_level: Confidence level (default: 0.95)
            
        Returns:
            Dictionary with lower and upper confidence bounds
        """
        self.logger.info(f"Calculating {confidence_level*100}% confidence intervals")
        
        returns_array = self._convert_to_array(returns)
        
        if len(returns_array) < 2:
            return {'lower': 0.0, 'upper': 0.0, 'mean': 0.0}
            
        # Calculate sample mean and standard error
        mean_return = np.mean(returns_array)
        std_error = stats.sem(returns_array)
        
        # Calculate critical value
        degrees_freedom = len(returns_array) - 1
        critical_value = t.ppf((1 + confidence_level) / 2, degrees_freedom)
        
        # Calculate margin of error
        margin_of_error = critical_value * std_error
        
        return {
            'mean': mean_return,
            'lower': mean_return - margin_of_error,
            'upper': mean_return + margin_of_error,
            'confidence_level': confidence_level
        }
    
    def perform_hypothesis_test(self, returns: Union[List[float], np.ndarray, pd.Series], 
                               null_hypothesis: float = 0.0, 
                               alternative: str = 'two-sided',
                               significance_level: float = 0.05) -> Dict[str, Any]:
        """
        Perform hypothesis test on strategy returns
        
        Args:
            returns: Strategy returns
            null_hypothesis: Null hypothesis value (default: 0.0)
            alternative: Alternative hypothesis ('two-sided', 'less', 'greater')
            significance_level: Significance level (default: 0.05)
            
        Returns:
            Dictionary with test results
        """
        self.logger.info(f"Performing hypothesis test: {alternative} alternative")
        
        returns_array = self._convert_to_array(returns)
        
        # Perform t-test
        t_stat, p_value = stats.ttest_1samp(returns_array, null_hypothesis)
        
        # Determine if we reject null hypothesis
        if alternative == 'two-sided':
            reject_null = p_value < significance_level
        elif alternative == 'less':
            reject_null = (p_value / 2) < significance_level and t_stat < 0
        elif alternative == 'greater':
            reject_null = (p_value / 2) < significance_level and t_stat > 0
        else:
            raise ValueError(f"Invalid alternative hypothesis: {alternative}")
            
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'reject_null_hypothesis': reject_null,
            'significance_level': significance_level,
            'null_hypothesis': null_hypothesis,
            'alternative': alternative
        }
    
    def calculate_sharpe_ratio_significance(self, returns: Union[List[float], np.ndarray, pd.Series], 
                                           risk_free_rate: float = 0.0, 
                                           periods_per_year: int = 252) -> Dict[str, float]:
        """
        Calculate statistical significance of Sharpe ratio
        
        Args:
            returns: Strategy returns
            risk_free_rate: Risk-free rate
            periods_per_year: Periods per year
            
        Returns:
            Dictionary with Sharpe ratio and significance metrics
        """
        self.logger.info("Calculating Sharpe ratio significance")
        
        returns_array = self._convert_to_array(returns)
        
        if len(returns_array) < 2:
            return {'sharpe_ratio': 0.0, 'p_value': 1.0, 'confidence_interval': {'lower': 0.0, 'upper': 0.0}}
            
        # Calculate excess returns
        excess_returns = returns_array - (risk_free_rate / periods_per_year)
        
        # Calculate Sharpe ratio
        mean_excess_return = np.mean(excess_returns)
        std_excess_return = np.std(excess_returns, ddof=1)
        
        if std_excess_return == 0:
            return {'sharpe_ratio': 0.0, 'p_value': 1.0, 'confidence_interval': {'lower': 0.0, 'upper': 0.0}}
            
        sharpe_ratio = mean_excess_return / std_excess_return * np.sqrt(periods_per_year)
        
        # Calculate standard error of Sharpe ratio
        n = len(returns_array)
        sharpe_se = np.sqrt((1 + sharpe_ratio**2 / 2) / (n - 1))
        
        # Calculate p-value (two-tailed test)
        p_value = 2 * (1 - norm.cdf(abs(sharpe_ratio) / sharpe_se))
        
        # Calculate confidence interval
        z_critical = norm.ppf(0.975)  # 95% confidence
        ci_lower = sharpe_ratio - z_critical * sharpe_se
        ci_upper = sharpe_ratio + z_critical * sharpe_se
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'p_value': p_value,
            'confidence_interval': {
                'lower': ci_lower,
                'upper': ci_upper
            }
        }
    
    def perform_multiple_comparisons(self, strategy_returns: List[Union[List[float], np.ndarray, pd.Series]], 
                                    control_returns: Union[List[float], np.ndarray, pd.Series],
                                    method: str = 'bonferroni') -> Dict[str, Any]:
        """
        Perform multiple comparisons with correction for multiple testing
        
        Args:
            strategy_returns: List of strategy returns to compare
            control_returns: Control/benchmark returns
            method: Correction method ('bonferroni', 'holm', 'fdr')
            
        Returns:
            Dictionary with comparison results
        """
        self.logger.info(f"Performing multiple comparisons with {method} correction")
        
        control_array = self._convert_to_array(control_returns)
        
        results = []
        p_values = []
        
        for i, returns in enumerate(strategy_returns):
            strategy_array = self._convert_to_array(returns)
            
            # Perform paired t-test
            t_stat, p_value = stats.ttest_rel(strategy_array, control_array)
            p_values.append(p_value)
            
            results.append({
                'strategy_id': f'strategy_{i}',
                't_statistic': t_stat,
                'p_value': p_value
            })
        
        # Apply multiple testing correction
        if method == 'bonferroni':
            adjusted_p_values = np.array(p_values) * len(p_values)
        elif method == 'holm':
            # Holm-Bonferroni method
            sorted_indices = np.argsort(p_values)
            adjusted_p_values = np.zeros(len(p_values))
            for i, idx in enumerate(sorted_indices):
                adjusted_p_values[idx] = p_values[idx] * (len(p_values) - i)
        elif method == 'fdr':
            # False Discovery Rate control
            sorted_p = np.sort(p_values)
            adjusted_p_values = np.zeros(len(p_values))
            for i in range(len(sorted_p)):
                adjusted_p_values[i] = sorted_p[i] * len(p_values) / (i + 1)
            # Reorder to match original p-values
            original_order = np.argsort(np.argsort(p_values))
            adjusted_p_values = adjusted_p_values[original_order]
        else:
            raise ValueError(f"Unknown correction method: {method}")
        
        # Add adjusted p-values to results
        for i, result in enumerate(results):
            result['adjusted_p_value'] = min(adjusted_p_values[i], 1.0)  # Cap at 1.0
            result['significant'] = result['adjusted_p_value'] < 0.05
        
        return {
            'comparisons': results,
            'correction_method': method,
            'num_comparisons': len(strategy_returns)
        }
    
    def calculate_autocorrelation(self, returns: Union[List[float], np.ndarray, pd.Series], 
                                 max_lag: int = 10) -> Dict[str, List[float]]:
        """
        Calculate autocorrelation of returns at different lags
        
        Args:
            returns: Strategy returns
            max_lag: Maximum lag to calculate
            
        Returns:
            Dictionary with autocorrelation values and p-values
        """
        self.logger.info(f"Calculating autocorrelation up to lag {max_lag}")
        
        returns_array = self._convert_to_array(returns)
        
        autocorrelations = []
        p_values = []
        
        for lag in range(1, max_lag + 1):
            # Calculate autocorrelation
            acf_value = self._calculate_acf(returns_array, lag)
            
            # Calculate p-value using Ljung-Box test approximation
            n = len(returns_array)
            if n > lag:
                # Approximate p-value for autocorrelation
                std_error = 1 / np.sqrt(n)
                z_score = acf_value / std_error
                p_value = 2 * (1 - norm.cdf(abs(z_score)))
            else:
                p_value = 1.0
                
            autocorrelations.append(acf_value)
            p_values.append(p_value)
        
        return {
            'lags': list(range(1, max_lag + 1)),
            'autocorrelations': autocorrelations,
            'p_values': p_values
        }
    
    def _calculate_acf(self, series: np.ndarray, lag: int) -> float:
        """
        Calculate autocorrelation function at specific lag
        
        Args:
            series: Time series data
            lag: Lag for autocorrelation
            
        Returns:
            Autocorrelation value
        """
        n = len(series)
        if n <= lag:
            return 0.0
            
        # Calculate mean
        mean = np.mean(series)
        
        # Calculate covariance and variance
        covariance = np.sum((series[lag:] - mean) * (series[:-lag] - mean)) / n
        variance = np.sum((series - mean) ** 2) / n
        
        if variance == 0:
            return 0.0
            
        return covariance / variance
    
    def perform_stationarity_test(self, returns: Union[List[float], np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Perform stationarity test (Augmented Dickey-Fuller test)
        
        Args:
            returns: Strategy returns
            
        Returns:
            Dictionary with test results
        """
        self.logger.info("Performing stationarity test (ADF)")
        
        returns_array = self._convert_to_array(returns)
        
        if len(returns_array) < 10:
            return {
                'test_statistic': 0.0,
                'p_value': 1.0,
                'critical_values': {'1%': 0.0, '5%': 0.0, '10%': 0.0},
                'stationary': False,
                'warning': 'Insufficient data for reliable test'
            }
        
        try:
            # Perform Augmented Dickey-Fuller test
            result = stats.adfuller(returns_array, autolag='AIC')
            
            return {
                'test_statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'stationary': result[1] < 0.05,  # Stationary if p < 0.05
                'lags_used': result[2]
            }
        except Exception as e:
            self.logger.error(f"ADF test failed: {str(e)}")
            return {
                'test_statistic': 0.0,
                'p_value': 1.0,
                'critical_values': {'1%': 0.0, '5%': 0.0, '10%': 0.0},
                'stationary': False,
                'error': str(e)
            }
    
    def _convert_to_array(self, data: Union[List[float], np.ndarray, pd.Series]) -> np.ndarray:
        """
        Convert input to numpy array
        
        Args:
            data: Input data
            
        Returns:
            Numpy array
        """
        if isinstance(data, pd.Series):
            return data.values
        elif isinstance(data, list):
            return np.array(data)
        elif isinstance(data, np.ndarray):
            return data
        else:
            self.logger.error(f"Unsupported data type: {type(data)}")
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def calculate_monte_carlo_significance(self, returns: Union[List[float], np.ndarray, pd.Series], 
                                          metric_func: callable, 
                                          n_simulations: int = 1000) -> Dict[str, float]:
        """
        Calculate statistical significance using Monte Carlo simulation
        
        Args:
            returns: Strategy returns
            metric_func: Function to calculate performance metric
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            Dictionary with significance results
        """
        self.logger.info(f"Performing Monte Carlo significance test with {n_simulations} simulations")
        
        returns_array = self._convert_to_array(returns)
        
        if len(returns_array) == 0:
            return {'p_value': 1.0, 'z_score': 0.0, 'significant': False}
            
        # Calculate observed metric
        observed_metric = metric_func(returns_array)
        
        # Generate random permutations
        simulated_metrics = []
        for _ in range(n_simulations):
            # Randomly shuffle returns
            shuffled_returns = np.random.permutation(returns_array)
            simulated_metric = metric_func(shuffled_returns)
            simulated_metrics.append(simulated_metric)
        
        # Calculate p-value
        if observed_metric > 0:
            # For positive metrics, count how many simulations exceed observed
            p_value = np.sum(np.array(simulated_metrics) >= observed_metric) / n_simulations
        else:
            # For negative metrics, count how many simulations are less than observed
            p_value = np.sum(np.array(simulated_metrics) <= observed_metric) / n_simulations
        
        # Calculate z-score
        mean_simulated = np.mean(simulated_metrics)
        std_simulated = np.std(simulated_metrics)
        
        if std_simulated == 0:
            z_score = 0.0
        else:
            z_score = (observed_metric - mean_simulated) / std_simulated
            
        return {
            'observed_metric': observed_metric,
            'mean_simulated': mean_simulated,
            'p_value': p_value,
            'z_score': z_score,
            'significant': p_value < 0.05
        }