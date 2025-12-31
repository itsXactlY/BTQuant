"""
Risk Management and Position Sizing Module

Comprehensive risk management framework including position sizing,
portfolio risk analysis, and drawdown control.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from ..metrics.performance_metrics import PerformanceMetrics

class RiskManagement:
    """Class for risk management and position sizing"""
    
    def __init__(self):
        self.logger = logging.getLogger('RiskManagement')
        self.logger.info("RiskManagement initialized")
        
        # Initialize metrics calculator
        self.performance_metrics = PerformanceMetrics()
    
    def calculate_position_size(self, strategy: Dict[str, Any], 
                               account_size: float, 
                               risk_per_trade: float = 0.01, 
                               atr_period: int = 14) -> Dict[str, Any]:
        """
        Calculate position size based on risk management rules
        
        Args:
            strategy: Trading strategy
            account_size: Total account size
            risk_per_trade: Risk per trade as percentage of account (default: 1%)
            atr_period: ATR period for volatility calculation
            
        Returns:
            Dictionary with position sizing information
        """
        self.logger.info("Calculating position size")
        
        # Get strategy parameters
        params = strategy.get('parameters', {})
        
        # Calculate position size based on strategy type
        if strategy.get('template') == 'mean_reversion':
            # Mean reversion strategies typically use fixed fractional positioning
            position_size = account_size * risk_per_trade
            
        elif strategy.get('template') == 'trend_following':
            # Trend following strategies use volatility-based positioning
            # Simulate ATR calculation
            atr_value = self._simulate_atr_calculation(params.get('trend_strength', 1.0), atr_period)
            
            # Calculate position size based on ATR
            atr_multiple = params.get('atr_multiple', 2.0)
            stop_distance = atr_multiple * atr_value
            
            # Position size = (account_size * risk_per_trade) / stop_distance
            position_size = (account_size * risk_per_trade) / (stop_distance + 1e-6)
            
        else:
            # Default: fixed fractional positioning
            position_size = account_size * risk_per_trade
        
        return {
            'account_size': account_size,
            'risk_per_trade_percent': risk_per_trade * 100,
            'position_size': position_size,
            'max_position_size': account_size * 0.1,  # 10% of account max
            'leverage_ratio': position_size / (account_size + 1e-6),
            'risk_management_rules': self._get_risk_management_rules(strategy)
        }
    
    def _simulate_atr_calculation(self, volatility_factor: float, period: int) -> float:
        """
        Simulate ATR (Average True Range) calculation
        
        Args:
            volatility_factor: Volatility factor from strategy
            period: ATR period
            
        Returns:
            Simulated ATR value
        """
        # This simulates what would be calculated from actual price data
        base_atr = 0.01 * volatility_factor  # Base ATR value
        return base_atr * np.sqrt(period)
    
    def _get_risk_management_rules(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get risk management rules based on strategy type
        
        Args:
            strategy: Trading strategy
            
        Returns:
            Dictionary with risk management rules
        """
        rules = {
            'max_drawdown_limit': 0.2,  # 20% max drawdown
            'max_position_size': 0.1,   # 10% of account per position
            'max_sector_exposure': 0.2, # 20% per sector
            'stop_loss_required': True,
            'take_profit_required': True
        }
        
        # Strategy-specific rules
        if strategy.get('template') == 'mean_reversion':
            rules.update({
                'max_holding_period': '5 days',
                'volatility_scaling': True,
                'correlation_limits': {'max_correlation': 0.7}
            })
        elif strategy.get('template') == 'trend_following':
            rules.update({
                'max_holding_period': '30 days',
                'volatility_scaling': True,
                'trailing_stop_required': True
            })
        
        return rules
    
    def calculate_portfolio_risk(self, positions: List[Dict[str, Any]], 
                                correlation_matrix: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate portfolio-level risk metrics
        
        Args:
            positions: List of position dictionaries
            correlation_matrix: Optional correlation matrix between assets
            
        Returns:
            Dictionary with portfolio risk metrics
        """
        self.logger.info("Calculating portfolio risk metrics")
        
        if not positions:
            return {
                'portfolio_volatility': 0.0,
                'portfolio_var': 0.0,
                'portfolio_cvar': 0.0,
                'marginal_contributions': {}
            }
        
        # Extract position information
        weights = []
        volatilities = []
        position_ids = []
        
        for position in positions:
            weights.append(position.get('weight', 0.0))
            volatilities.append(position.get('volatility', 0.01))
            position_ids.append(position.get('id', 'unknown'))
        
        # Convert to numpy arrays
        weights_array = np.array(weights)
        volatilities_array = np.array(volatilities)
        
        # Normalize weights
        total_weight = np.sum(weights_array)
        if total_weight > 0:
            weights_array = weights_array / total_weight
        
        # Calculate portfolio volatility
        if correlation_matrix is not None and len(correlation_matrix) == len(positions):
            # Use correlation matrix for more accurate calculation
            portfolio_volatility = np.sqrt(
                np.dot(weights_array.T, 
                      np.dot(correlation_matrix * volatilities_array[:, np.newaxis] * volatilities_array[np.newaxis, :],
                             weights_array))
            )
        else:
            # Simple weighted average (assuming no correlation)
            portfolio_volatility = np.sqrt(np.sum((weights_array * volatilities_array) ** 2))
        
        # Calculate Value at Risk (VaR)
        portfolio_var = self._calculate_var(weights_array, volatilities_array, correlation_matrix)
        
        # Calculate Conditional VaR (CVaR)
        portfolio_cvar = self._calculate_cvar(weights_array, volatilities_array, correlation_matrix)
        
        # Calculate marginal risk contributions
        marginal_contributions = self._calculate_marginal_contributions(
            weights_array, volatilities_array, correlation_matrix, position_ids
        )
        
        return {
            'portfolio_volatility': portfolio_volatility * 100,  # Convert to percentage
            'portfolio_var': portfolio_var * 100,
            'portfolio_cvar': portfolio_cvar * 100,
            'marginal_contributions': marginal_contributions,
            'diversification_benefit': self._calculate_diversification_benefit(
                weights_array, volatilities_array, correlation_matrix
            )
        }
    
    def _calculate_var(self, weights: np.ndarray, volatilities: np.ndarray, 
                      correlation_matrix: Optional[np.ndarray] = None) -> float:
        """
        Calculate Value at Risk (VaR) at 95% confidence level
        
        Args:
            weights: Position weights
            volatilities: Position volatilities
            correlation_matrix: Correlation matrix
            
        Returns:
            VaR value
        """
        # For simplicity, we use parametric VaR based on portfolio volatility
        portfolio_vol = self._calculate_portfolio_volatility(weights, volatilities, correlation_matrix)
        
        # 95% VaR = 1.645 * portfolio_volatility
        var_95 = 1.645 * portfolio_vol
        
        return var_95
    
    def _calculate_cvar(self, weights: np.ndarray, volatilities: np.ndarray, 
                       correlation_matrix: Optional[np.ndarray] = None) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) at 95% confidence level
        
        Args:
            weights: Position weights
            volatilities: Position volatilities
            correlation_matrix: Correlation matrix
            
        Returns:
            CVaR value
        """
        # For simplicity, we use a multiple of VaR for CVaR
        var_95 = self._calculate_var(weights, volatilities, correlation_matrix)
        
        # CVaR is typically 1.3-1.5x VaR for normal distributions
        cvar_95 = 1.4 * var_95
        
        return cvar_95
    
    def _calculate_portfolio_volatility(self, weights: np.ndarray, volatilities: np.ndarray, 
                                       correlation_matrix: Optional[np.ndarray] = None) -> float:
        """
        Calculate portfolio volatility
        
        Args:
            weights: Position weights
            volatilities: Position volatilities
            correlation_matrix: Correlation matrix
            
        Returns:
            Portfolio volatility
        """
        if correlation_matrix is not None and len(correlation_matrix) == len(weights):
            # Use correlation matrix
            return np.sqrt(
                np.dot(weights.T, 
                      np.dot(correlation_matrix * volatilities[:, np.newaxis] * volatilities[np.newaxis, :],
                             weights))
            )
        else:
            # Simple weighted average
            return np.sqrt(np.sum((weights * volatilities) ** 2))
    
    def _calculate_marginal_contributions(self, weights: np.ndarray, volatilities: np.ndarray, 
                                         correlation_matrix: Optional[np.ndarray], 
                                         position_ids: List[str]) -> Dict[str, float]:
        """
        Calculate marginal risk contributions for each position
        
        Args:
            weights: Position weights
            volatilities: Position volatilities
            correlation_matrix: Correlation matrix
            position_ids: Position identifiers
            
        Returns:
            Dictionary with marginal contributions
        """
        marginal_contributions = {}
        
        if correlation_matrix is None:
            # Simple approach without correlation
            portfolio_vol = self._calculate_portfolio_volatility(weights, volatilities)
            for i, (weight, vol, pos_id) in enumerate(zip(weights, volatilities, position_ids)):
                # Marginal contribution = weight * volatility^2 / portfolio_volatility
                marginal_contributions[pos_id] = (weight * vol ** 2) / (portfolio_vol + 1e-6)
        else:
            # More accurate calculation with correlation
            portfolio_vol = self._calculate_portfolio_volatility(weights, volatilities, correlation_matrix)
            
            for i, pos_id in enumerate(position_ids):
                # Calculate partial derivative of portfolio variance with respect to weight i
                partial_derivative = 2 * np.dot(
                    correlation_matrix[i, :] * volatilities[i] * volatilities,
                    weights
                )
                
                marginal_contributions[pos_id] = (weights[i] * partial_derivative) / (2 * portfolio_vol + 1e-6)
        
        return marginal_contributions
    
    def _calculate_diversification_benefit(self, weights: np.ndarray, volatilities: np.ndarray, 
                                          correlation_matrix: Optional[np.ndarray] = None) -> float:
        """
        Calculate diversification benefit of the portfolio
        
        Args:
            weights: Position weights
            volatilities: Position volatilities
            correlation_matrix: Correlation matrix
            
        Returns:
            Diversification benefit score (0-1)
        """
        if len(weights) < 2:
            return 0.0
            
        # Calculate weighted average volatility (no diversification)
        weighted_avg_vol = np.sum(weights * volatilities)
        
        # Calculate actual portfolio volatility
        portfolio_vol = self._calculate_portfolio_volatility(weights, volatilities, correlation_matrix)
        
        # Diversification benefit = 1 - (portfolio_vol / weighted_avg_vol)
        if weighted_avg_vol == 0:
            return 0.0
            
        diversification_benefit = 1 - (portfolio_vol / weighted_avg_vol)
        
        return max(0, diversification_benefit)
    
    def implement_drawdown_control(self, strategy: Dict[str, Any], 
                                  historical_returns: Union[List[float], np.ndarray, pd.Series], 
                                  max_drawdown: float = 0.2) -> Dict[str, Any]:
        """
        Implement drawdown control mechanisms
        
        Args:
            strategy: Trading strategy
            historical_returns: Historical returns of the strategy
            max_drawdown: Maximum allowed drawdown (default: 20%)
            
        Returns:
            Dictionary with drawdown control recommendations
        """
        self.logger.info("Implementing drawdown control")
        
        returns_array = self._convert_to_array(historical_returns)
        
        # Calculate current drawdown metrics
        current_max_dd = self.performance_metrics.calculate_max_drawdown(returns_array)
        
        # Calculate rolling drawdowns
        rolling_dd = self._calculate_rolling_drawdowns(returns_array)
        
        # Determine drawdown control actions
        control_actions = []
        
        if current_max_dd > max_drawdown * 100:  # Convert percentage to absolute
            # Strategy has exceeded max drawdown
            control_actions.append({
                'action': 'reduce_position_size',
                'reduction_factor': 0.5,  # Reduce by 50%
                'trigger': f'Max drawdown {current_max_dd:.1f}% exceeds limit {max_drawdown*100:.1f}%'
            })
            
            control_actions.append({
                'action': 'increase_stop_loss',
                'new_stop_loss_multiplier': 1.5,
                'trigger': 'Drawdown limit exceeded'
            })
        
        # Check for drawdown trends
        recent_dd_trend = self._analyze_drawdown_trend(rolling_dd)
        
        if recent_dd_trend > 0.1:  # Increasing drawdown trend
            control_actions.append({
                'action': 'tighten_risk_parameters',
                'recommendation': 'Reduce volatility targets and increase stop loss sensitivity',
                'trigger': 'Increasing drawdown trend detected'
            })
        
        return {
            'current_max_drawdown': current_max_dd,
            'max_allowed_drawdown': max_drawdown * 100,
            'drawdown_control_actions': control_actions,
            'rolling_drawdown_analysis': {
                'mean_rolling_dd': np.mean(rolling_dd),
                'max_rolling_dd': np.max(rolling_dd),
                'trend': recent_dd_trend
            }
        }
    
    def _calculate_rolling_drawdowns(self, returns: np.ndarray, window: int = 30) -> np.ndarray:
        """
        Calculate rolling drawdowns
        
        Args:
            returns: Array of returns
            window: Rolling window size
            
        Returns:
            Array of rolling drawdowns
        """
        if len(returns) < window:
            return np.array([])
            
        rolling_drawdowns = []
        
        for i in range(window, len(returns) + 1):
            window_returns = returns[i-window:i]
            dd = self.performance_metrics.calculate_max_drawdown(window_returns)
            rolling_drawdowns.append(dd)
            
        return np.array(rolling_drawdowns)
    
    def _analyze_drawdown_trend(self, rolling_drawdowns: np.ndarray) -> float:
        """
        Analyze trend in rolling drawdowns
        
        Args:
            rolling_drawdowns: Array of rolling drawdowns
            
        Returns:
            Trend indicator (positive = increasing drawdowns)
        """
        if len(rolling_drawdowns) < 5:
            return 0.0
            
        # Use linear regression to detect trend
        x = np.arange(len(rolling_drawdowns))
        y = rolling_drawdowns
        
        # Calculate slope
        A = np.vstack([x, np.ones(len(x))]).T
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        
        return slope
    
    def calculate_risk_adjusted_position_sizing(self, strategy: Dict[str, Any], 
                                               account_size: float, 
                                               historical_returns: Union[List[float], np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Calculate risk-adjusted position sizing based on strategy performance
        
        Args:
            strategy: Trading strategy
            account_size: Account size
            historical_returns: Historical returns for risk assessment
            
        Returns:
            Dictionary with risk-adjusted position sizing
        """
        self.logger.info("Calculating risk-adjusted position sizing")
        
        returns_array = self._convert_to_array(historical_returns)
        
        # Calculate performance metrics
        metrics = self.performance_metrics.calculate_all_metrics(returns_array)
        
        # Calculate risk-adjusted position size
        sharpe_ratio = metrics['sharpe_ratio']
        max_drawdown = abs(metrics['max_drawdown'] / 100)  # Convert to absolute
        
        # Base position size (1% of account)
        base_position_size = account_size * 0.01
        
        # Adjust based on Sharpe ratio (higher Sharpe = larger positions)
        sharpe_adjustment = min(max(sharpe_ratio / 2.0, 0.5), 2.0)  # 0.5 to 2.0 multiplier
        
        # Adjust based on drawdown (higher drawdown = smaller positions)
        drawdown_adjustment = min(max(1.0 / (max_drawdown + 0.1), 0.5), 2.0)
        
        # Calculate final position size
        adjusted_position_size = base_position_size * sharpe_adjustment * drawdown_adjustment
        
        # Apply constraints
        min_position_size = account_size * 0.005  # 0.5% minimum
        max_position_size = account_size * 0.15  # 15% maximum
        
        final_position_size = max(min_position_size, min(adjusted_position_size, max_position_size))
        
        return {
            'account_size': account_size,
            'base_position_size': base_position_size,
            'sharpe_adjustment': sharpe_adjustment,
            'drawdown_adjustment': drawdown_adjustment,
            'adjusted_position_size': adjusted_position_size,
            'final_position_size': final_position_size,
            'position_size_percent': final_position_size / account_size,
            'risk_adjusted_leverage': final_position_size / (account_size * (max_drawdown + 0.01))
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
    
    def implement_volatility_targeting(self, strategy: Dict[str, Any], 
                                      historical_returns: Union[List[float], np.ndarray, pd.Series], 
                                      target_volatility: float = 0.15) -> Dict[str, Any]:
        """
        Implement volatility targeting for position sizing
        
        Args:
            strategy: Trading strategy
            historical_returns: Historical returns
            target_volatility: Target annualized volatility (default: 15%)
            
        Returns:
            Dictionary with volatility targeting parameters
        """
        self.logger.info("Implementing volatility targeting")
        
        returns_array = self._convert_to_array(historical_returns)
        
        # Calculate current volatility
        current_volatility = self.performance_metrics.calculate_volatility(returns_array) / 100  # Convert to absolute
        
        if current_volatility == 0:
            return {
                'current_volatility': 0.0,
                'target_volatility': target_volatility,
                'volatility_scaling_factor': 1.0,
                'warning': 'Cannot calculate scaling factor - zero volatility'
            }
        
        # Calculate volatility scaling factor
        scaling_factor = target_volatility / current_volatility
        
        # Apply constraints
        min_scaling = 0.5  # Minimum 50% of normal position size
        max_scaling = 2.0  # Maximum 200% of normal position size
        
        constrained_scaling = max(min_scaling, min(scaling_factor, max_scaling))
        
        return {
            'current_volatility': current_volatility,
            'target_volatility': target_volatility,
            'volatility_scaling_factor': scaling_factor,
            'constrained_scaling_factor': constrained_scaling,
            'recommended_position_adjustment': constrained_scaling,
            'volatility_targeting_active': abs(current_volatility - target_volatility) > 0.01
        }
    
    def calculate_strategy_risk_profile(self, strategy: Dict[str, Any], 
                                       historical_returns: Union[List[float], np.ndarray, pd.Series]) -> Dict[str, Any]:
        """
        Calculate comprehensive risk profile for a strategy
        
        Args:
            strategy: Trading strategy
            historical_returns: Historical returns
            
        Returns:
            Dictionary with comprehensive risk profile
        """
        self.logger.info("Calculating strategy risk profile")
        
        returns_array = self._convert_to_array(historical_returns)
        
        # Calculate basic metrics
        metrics = self.performance_metrics.calculate_all_metrics(returns_array)
        
        # Calculate risk metrics
        volatility = metrics['volatility']
        max_drawdown = abs(metrics['max_drawdown'])
        sharpe_ratio = metrics['sharpe_ratio']
        sortino_ratio = metrics['sortino_ratio']
        
        # Calculate tail risk metrics
        tail_risk = self._calculate_tail_risk(returns_array)
        
        # Calculate risk-adjusted return metrics
        risk_adjusted_metrics = {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': metrics['calmar_ratio'],
            'risk_adjusted_return': metrics['risk_adjusted_return']
        }
        
        # Determine risk category
        risk_category = self._determine_risk_category(volatility, max_drawdown, sharpe_ratio)
        
        return {
            'strategy_id': strategy.get('id', 'unknown'),
            'template': strategy.get('template', 'unknown'),
            'basic_risk_metrics': {
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'value_at_risk': tail_risk['var_95'],
                'conditional_var': tail_risk['cvar_95']
            },
            'risk_adjusted_metrics': risk_adjusted_metrics,
            'tail_risk_metrics': tail_risk,
            'risk_category': risk_category,
            'risk_score': self._calculate_risk_score(volatility, max_drawdown, sharpe_ratio),
            'recommended_risk_limits': self._get_recommended_risk_limits(risk_category)
        }
    
    def _calculate_tail_risk(self, returns: np.ndarray) -> Dict[str, float]:
        """
        Calculate tail risk metrics
        
        Args:
            returns: Array of returns
            
        Returns:
            Dictionary with tail risk metrics
        """
        if len(returns) < 10:
            return {
                'var_95': 0.0,
                'cvar_95': 0.0,
                'var_99': 0.0,
                'cvar_99': 0.0,
                'worst_1_percent': 0.0
            }
        
        # Calculate Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)  # 95% VaR
        var_99 = np.percentile(returns, 1)  # 99% VaR
        
        # Calculate Conditional VaR (CVaR)
        cvar_95 = np.mean(returns[returns <= var_95])
        cvar_99 = np.mean(returns[returns <= var_99])
        
        # Worst 1% of returns
        worst_1_percent = np.percentile(returns, 1)
        
        return {
            'var_95': var_95 * 100,  # Convert to percentage
            'cvar_95': cvar_95 * 100,
            'var_99': var_99 * 100,
            'cvar_99': cvar_99 * 100,
            'worst_1_percent': worst_1_percent * 100
        }
    
    def _determine_risk_category(self, volatility: float, max_drawdown: float, 
                                sharpe_ratio: float) -> str:
        """
        Determine risk category based on risk metrics
        
        Args:
            volatility: Annualized volatility
            max_drawdown: Maximum drawdown
            sharpe_ratio: Sharpe ratio
            
        Returns:
            Risk category string
        """
        # Normalize metrics for comparison
        vol_score = min(volatility / 30.0, 1.0)  # 30% volatility = max risk
        dd_score = min(max_drawdown / 50.0, 1.0)  # 50% drawdown = max risk
        sharpe_score = 1.0 - min(sharpe_ratio / 3.0, 1.0)  # Higher Sharpe = lower risk
        
        # Calculate overall risk score
        risk_score = (vol_score * 0.4 + dd_score * 0.4 + sharpe_score * 0.2)
        
        # Determine category
        if risk_score < 0.3:
            return 'low'
        elif risk_score < 0.6:
            return 'medium'
        elif risk_score < 0.8:
            return 'high'
        else:
            return 'very_high'
    
    def _calculate_risk_score(self, volatility: float, max_drawdown: float, 
                             sharpe_ratio: float) -> float:
        """
        Calculate quantitative risk score (0-1)
        
        Args:
            volatility: Annualized volatility
            max_drawdown: Maximum drawdown
            sharpe_ratio: Sharpe ratio
            
        Returns:
            Risk score between 0 and 1
        """
        # Normalize and weight components
        vol_component = min(volatility / 30.0, 1.0) * 0.4
        dd_component = min(max_drawdown / 50.0, 1.0) * 0.4
        sharpe_component = (1.0 - min(sharpe_ratio / 3.0, 1.0)) * 0.2
        
        risk_score = vol_component + dd_component + sharpe_component
        
        return risk_score
    
    def _get_recommended_risk_limits(self, risk_category: str) -> Dict[str, Any]:
        """
        Get recommended risk limits based on risk category
        
        Args:
            risk_category: Risk category
            
        Returns:
            Dictionary with recommended risk limits
        """
        if risk_category == 'low':
            return {
                'max_position_size': 0.15,  # 15% of account
                'max_sector_exposure': 0.25,  # 25% per sector
                'max_drawdown_limit': 0.25,  # 25% max drawdown
                'leverage_limit': 2.0
            }
        elif risk_category == 'medium':
            return {
                'max_position_size': 0.10,  # 10% of account
                'max_sector_exposure': 0.20,  # 20% per sector
                'max_drawdown_limit': 0.20,  # 20% max drawdown
                'leverage_limit': 1.5
            }
        elif risk_category == 'high':
            return {
                'max_position_size': 0.07,  # 7% of account
                'max_sector_exposure': 0.15,  # 15% per sector
                'max_drawdown_limit': 0.15,  # 15% max drawdown
                'leverage_limit': 1.0
            }
        else:  # very_high
            return {
                'max_position_size': 0.05,  # 5% of account
                'max_sector_exposure': 0.10,  # 10% per sector
                'max_drawdown_limit': 0.10,  # 10% max drawdown
                'leverage_limit': 0.5
            }