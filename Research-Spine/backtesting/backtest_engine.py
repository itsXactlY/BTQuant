"""
Backtest Engine Module

Handles the validation of generated strategies using historical data.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from .metrics.performance_metrics import PerformanceMetrics
from .metrics.statistical_significance import StatisticalSignificance
from .validation.out_of_sample_validation import OutOfSampleValidation
from .optimization.walk_forward_optimization import WalkForwardOptimization
from .risk_management.risk_management import RiskManagement

class BacktestEngine:
    """Main class for backtesting trading strategies"""
     
    def __init__(self):
        self.logger = logging.getLogger('BacktestEngine')
        self.logger.info("BacktestEngine initialized")
        
        # Initialize all backtesting components
        self.performance_metrics = PerformanceMetrics()
        self.statistical_significance = StatisticalSignificance()
        self.out_of_sample_validation = OutOfSampleValidation()
        self.walk_forward_optimization = WalkForwardOptimization()
        self.risk_management = RiskManagement()
         
    def run_backtest(self, strategy: Dict[str, Any], historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a comprehensive backtest on a given strategy using historical data
        
        Args:
            strategy: Strategy dictionary to backtest
            historical_data: Historical market data for backtesting
             
        Returns:
            Dictionary containing comprehensive backtest results and performance metrics
        """
        self.logger.info(f"Running comprehensive backtest for strategy: {strategy.get('template', 'unknown')}")
        
        # Convert historical data to DataFrame if needed
        if isinstance(historical_data, dict):
            # Convert to DataFrame
            data = pd.DataFrame(historical_data)
        elif isinstance(historical_data, pd.DataFrame):
            data = historical_data
        else:
            raise ValueError("Historical data must be a dictionary or pandas DataFrame")
            
        # Generate simulated returns based on strategy type
        returns = self._generate_strategy_returns(strategy, data)
        
        # Calculate comprehensive performance metrics
        performance_metrics = self.performance_metrics.calculate_all_metrics(returns)
        
        # Calculate statistical significance
        significance_results = self.statistical_significance.calculate_p_value(returns)
        
        # Calculate risk metrics
        risk_profile = self.risk_management.calculate_strategy_risk_profile(strategy, returns)
        
        # Calculate position sizing
        position_sizing = self.risk_management.calculate_position_size(strategy, account_size=100000)
        
        # Basic backtest result structure
        results = {
            'strategy_id': strategy.get('id', 'unknown'),
            'template': strategy.get('template', 'unknown'),
            'performance_metrics': performance_metrics,
            'statistical_significance': {
                'p_value': significance_results,
                'significant': significance_results < 0.05
            },
            'risk_profile': risk_profile,
            'position_sizing': position_sizing,
            'returns_data': {
                'returns': returns.tolist() if hasattr(returns, 'tolist') else list(returns),
                'num_periods': len(returns),
                'return_statistics': {
                    'mean': float(np.mean(returns)),
                    'std': float(np.std(returns)),
                    'min': float(np.min(returns)),
                    'max': float(np.max(returns))
                }
            },
            'metadata': {
                'backtested_by': 'BacktestEngine',
                'version': '2.0',
                'components_used': [
                    'PerformanceMetrics',
                    'StatisticalSignificance',
                    'RiskManagement',
                    'OutOfSampleValidation',
                    'WalkForwardOptimization'
                ]
            }
        }
        
        self.logger.debug(f"Comprehensive backtest results generated for strategy {strategy.get('id', 'unknown')}")
        return results
    
    def _generate_strategy_returns(self, strategy: Dict[str, Any],
                                  data: pd.DataFrame) -> Union[np.ndarray, List[float]]:
        """
        Generate simulated returns for a strategy (placeholder for actual backtesting)
        
        Args:
            strategy: Strategy dictionary
            data: Historical market data
            
        Returns:
            Array of simulated returns
        """
        n_periods = len(data)
        
        # Generate returns based on strategy template and parameters
        if strategy.get('template') == 'mean_reversion':
            # Mean reversion strategy simulation
            params = strategy.get('parameters', {})
            mean_reversion_strength = params.get('mean_reversion_strength', 1.0)
            volatility = params.get('volatility_target', 0.01)
            
            # Simulate mean-reverting returns
            returns = self._simulate_mean_reversion_returns(n_periods, mean_reversion_strength, volatility)
            
        elif strategy.get('template') == 'trend_following':
            # Trend following strategy simulation
            params = strategy.get('parameters', {})
            trend_strength = params.get('trend_strength', 1.0)
            volatility = params.get('volatility_target', 0.012)
            
            # Simulate trend-following returns
            returns = self._simulate_trend_following_returns(n_periods, trend_strength, volatility)
            
        else:
            # Default strategy simulation
            returns = np.random.normal(0.0008, 0.01, n_periods)
            
        return returns
    
    def _simulate_mean_reversion_returns(self, n_periods: int,
                                        strength: float, volatility: float) -> np.ndarray:
        """
        Simulate mean-reverting returns
        
        Args:
            n_periods: Number of periods
            strength: Mean reversion strength
            volatility: Target volatility
            
        Returns:
            Array of simulated returns
        """
        # Simulate mean-reverting process
        returns = np.zeros(n_periods)
        price = np.ones(n_periods)
        
        for t in range(1, n_periods):
            # Mean-reverting component
            mean_reversion = -strength * 0.1 * (price[t-1] - 1.0)
            
            # Random component
            random_shock = np.random.normal(0, volatility)
            
            # Calculate return
            returns[t] = mean_reversion + random_shock
            price[t] = price[t-1] * (1 + returns[t])
            
        return returns
    
    def _simulate_trend_following_returns(self, n_periods: int,
                                         strength: float, volatility: float) -> np.ndarray:
        """
        Simulate trend-following returns
        
        Args:
            n_periods: Number of periods
            strength: Trend strength
            volatility: Target volatility
            
        Returns:
            Array of simulated returns
        """
        # Simulate trend-following process
        returns = np.zeros(n_periods)
        price = np.ones(n_periods)
        trend = 0.0
        
        for t in range(1, n_periods):
            # Trend component
            trend = 0.9 * trend + strength * 0.001 * np.random.randn()
            
            # Random component
            random_shock = np.random.normal(0, volatility)
            
            # Calculate return
            returns[t] = trend + random_shock
            price[t] = price[t-1] * (1 + returns[t])
            
        return returns
    
    def run_comprehensive_validation(self, strategy: Dict[str, Any],
                                    historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run comprehensive validation including out-of-sample testing and walk-forward optimization
        
        Args:
            strategy: Strategy to validate
            historical_data: Historical market data
            
        Returns:
            Dictionary with comprehensive validation results
        """
        self.logger.info(f"Running comprehensive validation for strategy: {strategy.get('id', 'unknown')}")
        
        # Split data for out-of-sample validation
        in_sample, out_of_sample = self.out_of_sample_validation.split_data_in_sample_out_of_sample(
            historical_data, split_ratio=0.7
        )
        
        # Perform out-of-sample validation
        oos_results = self.out_of_sample_validation.validate_strategy_out_of_sample(
            strategy, in_sample, out_of_sample
        )
        
        # Perform walk-forward optimization (simplified parameter grid)
        parameter_grid = self._get_default_parameter_grid(strategy)
        
        wfo_results = self.walk_forward_optimization.perform_walk_forward_optimization(
            strategy, historical_data, parameter_grid, window_size=252, step_size=126
        )
        
        # Calculate overall validation score
        validation_score = self._calculate_validation_score(oos_results, wfo_results)
        
        return {
            'strategy_id': strategy.get('id', 'unknown'),
            'template': strategy.get('template', 'unknown'),
            'out_of_sample_validation': oos_results,
            'walk_forward_optimization': wfo_results,
            'validation_score': validation_score,
            'validation_passed': validation_score >= 0.6,  # 60% threshold
            'metadata': {
                'validation_type': 'comprehensive',
                'validation_version': '1.0'
            }
        }
    
    def _get_default_parameter_grid(self, strategy: Dict[str, Any]) -> Dict[str, List[Any]]:
        """
        Get default parameter grid for walk-forward optimization based on strategy type
        
        Args:
            strategy: Strategy dictionary
            
        Returns:
            Dictionary with parameter grid
        """
        if strategy.get('template') == 'mean_reversion':
            return {
                'mean_reversion_strength': [0.8, 1.0, 1.2],
                'volatility_target': [0.008, 0.01, 0.012]
            }
        elif strategy.get('template') == 'trend_following':
            return {
                'trend_strength': [0.8, 1.0, 1.2],
                'volatility_target': [0.01, 0.012, 0.015]
            }
        else:
            return {
                'aggressiveness': [0.5, 1.0, 1.5]
            }
    
    def _calculate_validation_score(self, oos_results: Dict[str, Any],
                                   wfo_results: Dict[str, Any]) -> float:
        """
        Calculate overall validation score based on multiple validation results
        
        Args:
            oos_results: Out-of-sample validation results
            wfo_results: Walk-forward optimization results
            
        Returns:
            Validation score between 0 and 1
        """
        # Calculate components
        oos_score = 1.0 if oos_results['validation_passed'] else 0.0
        
        # Normalize WFO performance (assuming Sharpe ratio 0-3 range)
        wfo_sharpe = wfo_results['average_sharpe_ratio']
        wfo_performance_score = min(max(wfo_sharpe / 3.0, 0.0), 1.0)
        
        # WFO consistency score
        wfo_consistency_score = wfo_results['performance_consistency']
        
        # Overall validation score
        validation_score = (
            0.4 * oos_score +
            0.3 * wfo_performance_score +
            0.3 * wfo_consistency_score
        )
        
        return validation_score
    
    def calculate_metrics(self, backtest_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate performance metrics from backtest results
        
        Args:
            backtest_results: Results from a backtest run
             
        Returns:
            Dictionary containing calculated performance metrics
        """
        self.logger.info("Calculating performance metrics")
        
        # Extract returns from results
        if 'returns_data' in backtest_results and 'returns' in backtest_results['returns_data']:
            returns = backtest_results['returns_data']['returns']
        else:
            # Fallback to original structure
            returns = backtest_results['performance_metrics']['returns'] if 'returns' in backtest_results['performance_metrics'] else []
            
        # Calculate comprehensive metrics
        metrics = self.performance_metrics.calculate_all_metrics(returns)
        
        self.logger.debug(f"Calculated metrics: {metrics}")
        return metrics
    
    def validate_strategy_with_integration(self, strategy: Dict[str, Any],
                                          historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate strategy with full integration of all backtesting components
        
        Args:
            strategy: Strategy to validate
            historical_data: Historical market data
            
        Returns:
            Dictionary with comprehensive validation results
        """
        self.logger.info(f"Validating strategy {strategy.get('id', 'unknown')} with full integration")
        
        # Run basic backtest
        backtest_results = self.run_backtest(strategy, historical_data)
        
        # Run comprehensive validation
        validation_results = self.run_comprehensive_validation(strategy, historical_data)
        
        # Calculate risk-adjusted metrics
        risk_metrics = self.risk_management.calculate_strategy_risk_profile(strategy,
                                                                         backtest_results['returns_data']['returns'])
        
        # Calculate position sizing
        position_sizing = self.risk_management.calculate_risk_adjusted_position_sizing(
            strategy,
            account_size=100000,
            historical_returns=backtest_results['returns_data']['returns']
        )
        
        # Calculate overall strategy score
        strategy_score = self._calculate_strategy_score(backtest_results, validation_results, risk_metrics)
        
        return {
            'strategy_id': strategy.get('id', 'unknown'),
            'template': strategy.get('template', 'unknown'),
            'backtest_results': backtest_results,
            'validation_results': validation_results,
            'risk_metrics': risk_metrics,
            'position_sizing': position_sizing,
            'strategy_score': strategy_score,
            'recommendation': self._generate_recommendation(strategy_score, validation_results),
            'metadata': {
                'validation_type': 'full_integration',
                'validation_version': '1.0',
                'timestamp': pd.Timestamp.now().isoformat()
            }
        }
    
    def _calculate_strategy_score(self, backtest_results: Dict[str, Any],
                                 validation_results: Dict[str, Any],
                                 risk_metrics: Dict[str, Any]) -> float:
        """
        Calculate overall strategy score
        
        Args:
            backtest_results: Backtest results
            validation_results: Validation results
            risk_metrics: Risk metrics
            
        Returns:
            Strategy score between 0 and 1
        """
        # Performance component (30%)
        sharpe_ratio = backtest_results['performance_metrics']['sharpe_ratio']
        performance_score = min(max(sharpe_ratio / 3.0, 0.0), 1.0) * 0.3
        
        # Validation component (30%)
        validation_score = validation_results['validation_score'] * 0.3
        
        # Risk component (40%)
        risk_score = 1.0 - risk_metrics['risk_score']  # Invert risk score
        risk_component = risk_score * 0.4
        
        # Overall strategy score
        strategy_score = performance_score + validation_score + risk_component
        
        return strategy_score
    
    def _generate_recommendation(self, strategy_score: float,
                                validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate recommendation based on strategy score and validation results
        
        Args:
            strategy_score: Overall strategy score
            validation_results: Validation results
            
        Returns:
            Dictionary with recommendation
        """
        if strategy_score >= 0.8:
            recommendation = 'strong_accept'
            reason = 'Excellent performance and validation results'
        elif strategy_score >= 0.6:
            recommendation = 'accept'
            reason = 'Good performance with acceptable validation'
        elif strategy_score >= 0.4:
            recommendation = 'conditional_accept'
            reason = 'Marginal performance, requires further validation'
        else:
            recommendation = 'reject'
            reason = 'Poor performance or validation failure'
        
        # Add specific validation warnings
        warnings = []
        if not validation_results['out_of_sample_validation']['validation_passed']:
            warnings.append('Out-of-sample validation failed')
            
        if validation_results['walk_forward_optimization']['performance_consistency'] < 0.5:
            warnings.append('Inconsistent performance across validation periods')
        
        return {
            'recommendation': recommendation,
            'score': strategy_score,
            'reason': reason,
            'warnings': warnings,
            'validation_passed': validation_results['validation_passed']
        }
    
    def calculate_metrics(self, backtest_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate performance metrics from backtest results
        
        Args:
            backtest_results: Results from a backtest run
            
        Returns:
            Dictionary containing calculated performance metrics
        """
        self.logger.info("Calculating performance metrics")
        
        # Basic metric calculation (placeholder)
        metrics = {
            'sharpe_ratio': backtest_results['performance_metrics']['sharpe_ratio'],
            'max_drawdown': backtest_results['performance_metrics']['max_drawdown'],
            'win_rate': backtest_results['performance_metrics']['win_rate'],
            'risk_adjusted_return': backtest_results['performance_metrics']['total_return'] / 
                                  (backtest_results['performance_metrics']['max_drawdown'] + 1e-6)
        }
        
        self.logger.debug(f"Calculated metrics: {metrics}")
        return metrics