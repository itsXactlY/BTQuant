"""
Multi-Objective Fitness Functions Module

Implements multiple fitness functions for evolutionary strategy selection.
Includes functions for different objectives like risk-adjusted returns, 
drawdown control, consistency, and robustness.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple

class MultiObjectiveFitness:
    """Class for calculating multi-objective fitness scores for trading strategies"""
    
    def __init__(self):
        self.logger = logging.getLogger('MultiObjectiveFitness')
        self.logger.info("MultiObjectiveFitness initialized")
    
    def calculate_fitness_scores(self, strategy: Dict[str, Any], backtest_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate multiple fitness scores for a strategy
        
        Args:
            strategy: Strategy dictionary
            backtest_results: Backtest results for the strategy
            
        Returns:
            Dictionary containing multiple fitness scores
        """
        self.logger.debug(f"Calculating multi-objective fitness for strategy: {strategy.get('template', 'unknown')}")
        
        metrics = backtest_results['performance_metrics']
        
        fitness_scores = {
            'risk_adjusted_fitness': self._calculate_risk_adjusted_fitness(metrics),
            'drawdown_control_fitness': self._calculate_drawdown_control_fitness(metrics),
            'consistency_fitness': self._calculate_consistency_fitness(metrics),
            'robustness_fitness': self._calculate_robustness_fitness(metrics),
            'profitability_fitness': self._calculate_profitability_fitness(metrics),
            'risk_control_fitness': self._calculate_risk_control_fitness(metrics)
        }
        
        self.logger.debug(f"Multi-objective fitness scores: {fitness_scores}")
        return fitness_scores
    
    def _calculate_risk_adjusted_fitness(self, metrics: Dict[str, float]) -> float:
        """
        Calculate risk-adjusted fitness score
        Focuses on Sharpe ratio, Sortino ratio, and Calmar ratio
        """
        # Normalize individual components
        sharpe_normalized = self._normalize_metric(metrics['sharpe_ratio'], 0, 5)
        sortino_normalized = self._normalize_metric(metrics['sortino_ratio'], 0, 5)
        calmar_normalized = self._normalize_metric(metrics['calmar_ratio'], 0, 10)
        
        # Weighted combination
        fitness = (
            0.5 * sharpe_normalized + 
            0.3 * sortino_normalized + 
            0.2 * calmar_normalized
        )
        
        return fitness
    
    def _calculate_drawdown_control_fitness(self, metrics: Dict[str, float]) -> float:
        """
        Calculate drawdown control fitness score
        Focuses on max drawdown, recovery time, and drawdown frequency
        """
        # Inverse relationship with drawdown (lower drawdown = higher fitness)
        max_drawdown_penalty = 1 - abs(metrics['max_drawdown'] / 100)  # Convert from percentage
        
        # Reward for better risk-adjusted returns
        risk_adjusted_reward = metrics['risk_adjusted_return'] / 100  # Convert from percentage
        
        # Combined fitness
        fitness = (
            0.7 * max_drawdown_penalty + 
            0.3 * risk_adjusted_reward
        )
        
        return max(0, fitness)
    
    def _calculate_consistency_fitness(self, metrics: Dict[str, float]) -> float:
        """
        Calculate consistency fitness score
        Focuses on win rate, volatility, and skewness
        """
        # Normalize components
        win_rate_normalized = self._normalize_metric(metrics['win_rate'], 0, 1)
        volatility_penalty = 1 - abs(metrics['volatility'] / 100)  # Convert from percentage
        
        # Reward positive skewness (fat right tail)
        skewness_reward = min(1, max(0, metrics['skewness'] / 2))
        
        # Combined fitness
        fitness = (
            0.4 * win_rate_normalized + 
            0.4 * volatility_penalty + 
            0.2 * skewness_reward
        )
        
        return fitness
    
    def _calculate_robustness_fitness(self, metrics: Dict[str, float]) -> float:
        """
        Calculate robustness fitness score
        Focuses on Omega ratio, kurtosis, and overall stability
        """
        # Normalize components
        omega_normalized = self._normalize_metric(metrics['omega_ratio'], 0, 5)
        
        # Penalize high kurtosis (fat tails = higher risk)
        kurtosis_penalty = 1 - abs(metrics['kurtosis'] / 10)
        
        # Reward for good risk-adjusted returns
        risk_adjusted_reward = metrics['risk_adjusted_return'] / 100
        
        # Combined fitness
        fitness = (
            0.5 * omega_normalized + 
            0.3 * kurtosis_penalty + 
            0.2 * risk_adjusted_reward
        )
        
        return fitness
    
    def _calculate_profitability_fitness(self, metrics: Dict[str, float]) -> float:
        """
        Calculate profitability fitness score
        Focuses on total return, annualized return, and win/loss ratio
        """
        # Normalize components
        total_return_normalized = self._normalize_metric(metrics['total_return'], 0, 200)
        annualized_return_normalized = self._normalize_metric(metrics['annualized_return'], 0, 100)
        
        # Calculate win/loss ratio
        if metrics['avg_loss'] == 0:
            win_loss_ratio = 1.0
        else:
            win_loss_ratio = abs(metrics['avg_win'] / metrics['avg_loss'])
        
        win_loss_normalized = self._normalize_metric(win_loss_ratio, 0, 5)
        
        # Combined fitness
        fitness = (
            0.4 * total_return_normalized + 
            0.4 * annualized_return_normalized + 
            0.2 * win_loss_normalized
        )
        
        return fitness
    
    def _calculate_risk_control_fitness(self, metrics: Dict[str, float]) -> float:
        """
        Calculate risk control fitness score
        Focuses on volatility, max drawdown, and risk-adjusted metrics
        """
        # Inverse relationship with volatility and drawdown
        volatility_penalty = 1 - abs(metrics['volatility'] / 100)
        drawdown_penalty = 1 - abs(metrics['max_drawdown'] / 100)
        
        # Reward for good risk-adjusted metrics
        risk_adjusted_reward = metrics['risk_adjusted_return'] / 100
        
        # Combined fitness
        fitness = (
            0.4 * volatility_penalty + 
            0.4 * drawdown_penalty + 
            0.2 * risk_adjusted_reward
        )
        
        return max(0, fitness)
    
    def _normalize_metric(self, value: float, min_val: float, max_val: float) -> float:
        """
        Normalize a metric value to [0, 1] range
        """
        if max_val == min_val:
            return 0.5  # Avoid division by zero
        
        # Clip values to avoid out-of-bounds
        clipped_value = max(min_val, min(max_val, value))
        
        # Normalize to [0, 1]
        normalized = (clipped_value - min_val) / (max_val - min_val)
        
        return normalized
    
    def calculate_composite_fitness(self, fitness_scores: Dict[str, float], 
                                   weights: Dict[str, float] = None) -> float:
        """
        Calculate composite fitness score from multiple objectives
        
        Args:
            fitness_scores: Dictionary of individual fitness scores
            weights: Optional custom weights for each objective
            
        Returns:
            Composite fitness score
        """
        # Default weights if not provided
        if weights is None:
            weights = {
                'risk_adjusted_fitness': 0.3,
                'drawdown_control_fitness': 0.2,
                'consistency_fitness': 0.2,
                'robustness_fitness': 0.1,
                'profitability_fitness': 0.1,
                'risk_control_fitness': 0.1
            }
        
        # Calculate weighted sum
        composite_score = 0.0
        total_weight = 0.0
        
        for objective, weight in weights.items():
            if objective in fitness_scores:
                composite_score += fitness_scores[objective] * weight
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            composite_score /= total_weight
        
        return composite_score