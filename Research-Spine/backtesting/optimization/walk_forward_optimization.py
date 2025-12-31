"""
Walk-Forward Optimization Module

Implementation of walk-forward optimization for robust parameter tuning
and strategy validation over multiple time periods.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from datetime import datetime
from ..metrics.performance_metrics import PerformanceMetrics
from ..validation.out_of_sample_validation import OutOfSampleValidation

class WalkForwardOptimization:
    """Class for walk-forward optimization of trading strategies"""
    
    def __init__(self):
        self.logger = logging.getLogger('WalkForwardOptimization')
        self.logger.info("WalkForwardOptimization initialized")
        
        # Initialize helper classes
        self.performance_metrics = PerformanceMetrics()
        self.out_of_sample_validation = OutOfSampleValidation()
    
    def perform_walk_forward_optimization(self, strategy_template: Dict[str, Any], 
                                         data: pd.DataFrame, 
                                         parameter_grid: Dict[str, List[Any]], 
                                         window_size: int = 252, 
                                         step_size: int = 126, 
                                         optimization_metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """
        Perform walk-forward optimization with parameter tuning
        
        Args:
            strategy_template: Base strategy template
            data: Historical market data
            parameter_grid: Dictionary of parameters to optimize
            window_size: Size of optimization window (in periods)
            step_size: Step size between optimization windows
            optimization_metric: Metric to optimize ('sharpe_ratio', 'total_return', etc.)
            
        Returns:
            Dictionary with optimization results
        """
        self.logger.info("Starting walk-forward optimization")
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
            
        # Ensure data is sorted by date
        if 'date' in data.columns:
            data = data.sort_values('date')
        else:
            data = data.sort_index()
            
        # Initialize results storage
        optimization_windows = []
        final_parameters = {}
        
        # Generate all parameter combinations
        param_combinations = self._generate_parameter_combinations(parameter_grid)
        
        # Walk-forward optimization loop
        current_position = 0
        window_count = 0
        
        while current_position + window_size <= len(data):
            self.logger.debug(f"Processing window {window_count + 1}")
            
            # Define optimization and validation periods
            opt_end = current_position + window_size
            val_end = min(opt_end + step_size, len(data))
            
            # Split data
            optimization_data = data.iloc[current_position:opt_end]
            validation_data = data.iloc[opt_end:val_end]
            
            if len(validation_data) == 0:
                break
                
            # Perform parameter optimization
            best_params, window_results = self._optimize_parameters(
                strategy_template, optimization_data, validation_data,
                param_combinations, optimization_metric
            )
            
            # Store results
            optimization_windows.append({
                'window': window_count + 1,
                'optimization_period': {
                    'start': current_position,
                    'end': opt_end,
                    'size': len(optimization_data)
                },
                'validation_period': {
                    'start': opt_end,
                    'end': val_end,
                    'size': len(validation_data)
                },
                'best_parameters': best_params,
                'performance': window_results['best_performance'],
                'parameter_combinations_tested': len(param_combinations)
            })
            
            # Update final parameters (using most recent optimal parameters)
            final_parameters = best_params.copy()
            
            # Move to next window
            current_position += step_size
            window_count += 1
        
        # Calculate overall statistics
        sharpe_ratios = [w['performance']['sharpe_ratio'] for w in optimization_windows]
        total_returns = [w['performance']['total_return'] for w in optimization_windows]
        
        return {
            'strategy_template': strategy_template.get('template', 'unknown'),
            'optimization_windows': optimization_windows,
            'final_parameters': final_parameters,
            'average_sharpe_ratio': np.mean(sharpe_ratios),
            'sharpe_ratio_std': np.std(sharpe_ratios),
            'average_total_return': np.mean(total_returns),
            'total_return_std': np.std(total_returns),
            'parameter_stability': self._calculate_parameter_stability(optimization_windows),
            'performance_consistency': self._calculate_performance_consistency(sharpe_ratios)
        }
    
    def _optimize_parameters(self, strategy_template: Dict[str, Any], 
                            optimization_data: pd.DataFrame, 
                            validation_data: pd.DataFrame, 
                            param_combinations: List[Dict[str, Any]], 
                            optimization_metric: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Optimize parameters for a single walk-forward window
        
        Args:
            strategy_template: Base strategy template
            optimization_data: Data for parameter optimization
            validation_data: Data for validation
            param_combinations: Parameter combinations to test
            optimization_metric: Metric to optimize
            
        Returns:
            Tuple of (best_parameters, window_results)
        """
        self.logger.debug("Optimizing parameters for current window")
        
        best_performance = -np.inf
        best_params = None
        combination_results = []
        
        for combo in param_combinations:
            # Create strategy with current parameters
            strategy = self._create_strategy_from_template(strategy_template, combo)
            
            # Run backtest on validation data
            val_results = self._run_backtest(strategy, validation_data)
            
            # Calculate performance metrics
            metrics = self.performance_metrics.calculate_all_metrics(val_results['returns'])
            
            # Get optimization metric value
            metric_value = metrics.get(optimization_metric, -np.inf)
            
            # Store results
            combination_results.append({
                'parameters': combo,
                'performance_metrics': metrics,
                'optimization_metric_value': metric_value
            })
            
            # Update best parameters
            if metric_value > best_performance:
                best_performance = metric_value
                best_params = combo.copy()
        
        return best_params, {
            'best_parameters': best_params,
            'best_performance': combination_results[-1]['performance_metrics'],
            'all_combinations': combination_results
        }
    
    def _generate_parameter_combinations(self, parameter_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations from parameter grid
        
        Args:
            parameter_grid: Dictionary of parameter names to possible values
            
        Returns:
            List of parameter combinations
        """
        if not parameter_grid:
            return [{}]
            
        # Get parameter names and values
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        
        # Generate all combinations
        combinations = []
        
        def generate_combinations(current_combo: Dict[str, Any], index: int):
            if index == len(param_names):
                combinations.append(current_combo.copy())
                return
                
            param_name = param_names[index]
            for value in param_values[index]:
                current_combo[param_name] = value
                generate_combinations(current_combo, index + 1)
        
        generate_combinations({}, 0)
        
        self.logger.debug(f"Generated {len(combinations)} parameter combinations")
        return combinations
    
    def _create_strategy_from_template(self, template: Dict[str, Any], 
                                     parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a strategy from template and parameters
        
        Args:
            template: Strategy template
            parameters: Parameters to use
            
        Returns:
            Complete strategy dictionary
        """
        strategy = template.copy()
        strategy['parameters'] = parameters
        
        # Ensure strategy has required fields
        if 'id' not in strategy:
            strategy['id'] = f"wfo_strategy_{np.random.randint(1000, 9999)}"
            
        return strategy
    
    def _run_backtest(self, strategy: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run a simplified backtest (same as in out-of-sample validation)
        
        Args:
            strategy: Strategy to backtest
            data: Historical data for backtesting
            
        Returns:
            Dictionary with backtest results
        """
        # This is a simplified backtest - in a real implementation, this would
        # integrate with the actual backtesting engine
        
        n_periods = len(data)
        
        # Simulate strategy returns based on template and parameters
        if strategy.get('template') == 'mean_reversion':
            # Mean reversion strategy simulation
            mean_reversion_param = strategy['parameters'].get('mean_reversion_strength', 1.0)
            returns = np.random.normal(0.001 * mean_reversion_param, 0.01, n_periods)
        elif strategy.get('template') == 'trend_following':
            # Trend following strategy simulation
            trend_strength = strategy['parameters'].get('trend_strength', 1.0)
            returns = np.random.normal(0.0015 * trend_strength, 0.012, n_periods)
        else:
            # Default strategy simulation
            returns = np.random.normal(0.0008, 0.01, n_periods)
            
        return {
            'returns': returns,
            'sharpe_ratio': self.performance_metrics.calculate_sharpe_ratio(returns),
            'max_drawdown': self.performance_metrics.calculate_max_drawdown(returns)
        }
    
    def _calculate_parameter_stability(self, optimization_windows: List[Dict[str, Any]]) -> float:
        """
        Calculate parameter stability across walk-forward windows
        
        Args:
            optimization_windows: List of optimization window results
            
        Returns:
            Stability score between 0 and 1
        """
        if len(optimization_windows) < 2:
            return 0.0
            
        # Collect parameter values across windows
        param_history = {}
        
        for window in optimization_windows:
            for param_name, param_value in window['best_parameters'].items():
                if param_name not in param_history:
                    param_history[param_name] = []
                param_history[param_name].append(param_value)
        
        # Calculate stability for each parameter
        stability_scores = []
        
        for param_name, values in param_history.items():
            if len(values) < 2:
                continue
                
            # For numeric parameters, calculate coefficient of variation
            if isinstance(values[0], (int, float)):
                mean_val = np.mean(values)
                std_val = np.std(values)
                if mean_val != 0:
                    cv = std_val / abs(mean_val)
                    stability = 1 - min(cv, 1.0)
                    stability_scores.append(stability)
            else:
                # For categorical parameters, calculate consistency
                unique_values = len(set(values))
                total_values = len(values)
                consistency = 1 - (unique_values - 1) / total_values
                stability_scores.append(consistency)
        
        if not stability_scores:
            return 0.0
            
        # Return average stability
        return np.mean(stability_scores)
    
    def _calculate_performance_consistency(self, sharpe_ratios: List[float]) -> float:
        """
        Calculate performance consistency across walk-forward windows
        
        Args:
            sharpe_ratios: List of Sharpe ratios from different windows
            
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
    
    def analyze_walk_forward_results(self, wfo_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze walk-forward optimization results
        
        Args:
            wfo_results: Results from walk-forward optimization
            
        Returns:
            Dictionary with analysis results
        """
        self.logger.info("Analyzing walk-forward optimization results")
        
        windows = wfo_results['optimization_windows']
        
        # Extract performance metrics
        sharpe_ratios = [w['performance']['sharpe_ratio'] for w in windows]
        total_returns = [w['performance']['total_return'] for w in windows]
        max_drawdowns = [w['performance']['max_drawdown'] for w in windows]
        
        # Calculate statistics
        sharpe_stats = self._calculate_statistics(sharpe_ratios)
        return_stats = self._calculate_statistics(total_returns)
        drawdown_stats = self._calculate_statistics(max_drawdowns)
        
        # Analyze parameter stability
        param_stability = self._analyze_parameter_stability_detailed(windows)
        
        return {
            'sharpe_ratio_analysis': sharpe_stats,
            'total_return_analysis': return_stats,
            'max_drawdown_analysis': drawdown_stats,
            'parameter_stability_analysis': param_stability,
            'overall_quality_score': self._calculate_overall_quality_score(
                wfo_results['parameter_stability'], 
                wfo_results['performance_consistency'],
                sharpe_stats['mean']
            )
        }
    
    def _calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """
        Calculate basic statistics for a list of values
        
        Args:
            values: List of numeric values
            
        Returns:
            Dictionary with statistical measures
        """
        if not values:
            return {
                'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 
                'range': 0.0, 'cv': 0.0
            }
            
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        return {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'range': max_val - min_val,
            'cv': std_val / (mean_val + 1e-6)  # Coefficient of variation
        }
    
    def _analyze_parameter_stability_detailed(self, windows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detailed analysis of parameter stability
        
        Args:
            windows: List of optimization window results
            
        Returns:
            Dictionary with detailed stability analysis
        """
        if len(windows) < 2:
            return {'overall_stability': 0.0, 'parameter_analysis': {}}
            
        # Collect parameter history
        param_history = {}
        
        for window in windows:
            for param_name, param_value in window['best_parameters'].items():
                if param_name not in param_history:
                    param_history[param_name] = []
                param_history[param_name].append(param_value)
        
        # Analyze each parameter
        parameter_analysis = {}
        
        for param_name, values in param_history.items():
            if len(values) < 2:
                continue
                
            if isinstance(values[0], (int, float)):
                # Numeric parameter analysis
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / (mean_val + 1e-6)
                
                parameter_analysis[param_name] = {
                    'type': 'numeric',
                    'mean': mean_val,
                    'std': std_val,
                    'cv': cv,
                    'stability_score': 1 - min(cv, 1.0),
                    'values': values
                }
            else:
                # Categorical parameter analysis
                unique_values = list(set(values))
                value_counts = {val: values.count(val) for val in unique_values}
                most_common = max(value_counts.items(), key=lambda x: x[1])
                
                parameter_analysis[param_name] = {
                    'type': 'categorical',
                    'unique_values': unique_values,
                    'value_counts': value_counts,
                    'most_common': most_common[0],
                    'consistency_score': most_common[1] / len(values),
                    'values': values
                }
        
        # Calculate overall stability
        stability_scores = []
        for analysis in parameter_analysis.values():
            if analysis['type'] == 'numeric':
                stability_scores.append(analysis['stability_score'])
            else:
                stability_scores.append(analysis['consistency_score'])
        
        overall_stability = np.mean(stability_scores) if stability_scores else 0.0
        
        return {
            'overall_stability': overall_stability,
            'parameter_analysis': parameter_analysis
        }
    
    def _calculate_overall_quality_score(self, parameter_stability: float, 
                                        performance_consistency: float, 
                                        mean_sharpe: float) -> float:
        """
        Calculate overall quality score for walk-forward optimization results
        
        Args:
            parameter_stability: Parameter stability score
            performance_consistency: Performance consistency score
            mean_sharpe: Mean Sharpe ratio
            
        Returns:
            Overall quality score between 0 and 1
        """
        # Normalize Sharpe ratio (assuming 0-3 range, scale to 0-1)
        normalized_sharpe = min(max(mean_sharpe / 3.0, 0.0), 1.0)
        
        # Weighted average of components
        quality_score = (
            0.4 * parameter_stability +
            0.3 * performance_consistency +
            0.3 * normalized_sharpe
        )
        
        return quality_score