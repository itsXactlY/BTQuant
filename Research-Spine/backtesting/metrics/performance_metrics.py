"""
Performance Metrics Module

Comprehensive performance metrics calculation for backtesting results.
Includes Sharpe ratio, Sortino ratio, max drawdown, and other key metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
import logging
from scipy.stats import norm

class PerformanceMetrics:
    """Class for calculating comprehensive performance metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger('PerformanceMetrics')
        self.logger.info("PerformanceMetrics initialized")
    
    def calculate_all_metrics(self, returns: Union[List[float], np.ndarray, pd.Series], 
                            risk_free_rate: float = 0.0) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics
        
        Args:
            returns: Array of daily returns
            risk_free_rate: Annualized risk-free rate (default: 0.0)
            
        Returns:
            Dictionary containing all calculated metrics
        """
        self.logger.info("Calculating comprehensive performance metrics")
        
        # Convert to numpy array for easier calculations
        returns_array = self._convert_to_array(returns)
        
        # Basic metrics
        total_return = self.calculate_total_return(returns_array)
        annualized_return = self.calculate_annualized_return(returns_array)
        
        # Risk metrics
        volatility = self.calculate_volatility(returns_array)
        max_drawdown = self.calculate_max_drawdown(returns_array)
        
        # Risk-adjusted metrics
        sharpe_ratio = self.calculate_sharpe_ratio(returns_array, risk_free_rate)
        sortino_ratio = self.calculate_sortino_ratio(returns_array, risk_free_rate)
        
        # Trading metrics
        win_rate, avg_win, avg_loss = self.calculate_trading_metrics(returns_array)
        
        # Additional metrics
        calmar_ratio = self.calculate_calmar_ratio(returns_array, risk_free_rate)
        omega_ratio = self.calculate_omega_ratio(returns_array, risk_free_rate)
        
        # Statistical metrics
        skewness = self.calculate_skewness(returns_array)
        kurtosis = self.calculate_kurtosis(returns_array)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'calmar_ratio': calmar_ratio,
            'omega_ratio': omega_ratio,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'risk_adjusted_return': annualized_return / (volatility + 1e-6)
        }
    
    def calculate_total_return(self, returns: Union[List[float], np.ndarray, pd.Series]) -> float:
        """
        Calculate total return from a series of returns
        
        Args:
            returns: Array of returns
            
        Returns:
            Total return as a percentage
        """
        returns_array = self._convert_to_array(returns)
        cumulative_return = np.prod(1 + returns_array) - 1
        return cumulative_return * 100  # Convert to percentage
    
    def calculate_annualized_return(self, returns: Union[List[float], np.ndarray, pd.Series], 
                                   periods_per_year: int = 252) -> float:
        """
        Calculate annualized return
        
        Args:
            returns: Array of returns
            periods_per_year: Number of periods per year (default: 252 for daily)
            
        Returns:
            Annualized return as a percentage
        """
        returns_array = self._convert_to_array(returns)
        if len(returns_array) == 0:
            return 0.0
            
        # Calculate compound annual growth rate
        n_periods = len(returns_array)
        total_return = np.prod(1 + returns_array)
        annualized_return = (total_return ** (periods_per_year / n_periods)) - 1
        return annualized_return * 100  # Convert to percentage
    
    def calculate_volatility(self, returns: Union[List[float], np.ndarray, pd.Series], 
                           periods_per_year: int = 252) -> float:
        """
        Calculate annualized volatility
        
        Args:
            returns: Array of returns
            periods_per_year: Number of periods per year (default: 252 for daily)
            
        Returns:
            Annualized volatility as a percentage
        """
        returns_array = self._convert_to_array(returns)
        if len(returns_array) <= 1:
            return 0.0
            
        std_dev = np.std(returns_array, ddof=1)  # Sample standard deviation
        annualized_volatility = std_dev * np.sqrt(periods_per_year)
        return annualized_volatility * 100  # Convert to percentage
    
    def calculate_max_drawdown(self, returns: Union[List[float], np.ndarray, pd.Series]) -> float:
        """
        Calculate maximum drawdown
        
        Args:
            returns: Array of returns
            
        Returns:
            Maximum drawdown as a percentage
        """
        returns_array = self._convert_to_array(returns)
        if len(returns_array) == 0:
            return 0.0
            
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + returns_array)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdowns
        drawdowns = (cumulative_returns - running_max) / running_max
        
        # Return maximum drawdown as percentage
        max_drawdown = np.min(drawdowns)
        return max_drawdown * 100  # Convert to percentage
    
    def calculate_sharpe_ratio(self, returns: Union[List[float], np.ndarray, pd.Series], 
                             risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns: Array of returns
            risk_free_rate: Annualized risk-free rate (default: 0.0)
            periods_per_year: Number of periods per year (default: 252 for daily)
            
        Returns:
            Sharpe ratio
        """
        returns_array = self._convert_to_array(returns)
        if len(returns_array) <= 1:
            return 0.0
            
        # Calculate excess returns
        excess_returns = returns_array - (risk_free_rate / periods_per_year)
        
        # Calculate annualized mean excess return
        mean_excess_return = np.mean(excess_returns) * periods_per_year
        
        # Calculate annualized volatility
        volatility = np.std(excess_returns, ddof=1) * np.sqrt(periods_per_year)
        
        if volatility == 0:
            return 0.0
            
        sharpe_ratio = mean_excess_return / volatility
        return sharpe_ratio
    
    def calculate_sortino_ratio(self, returns: Union[List[float], np.ndarray, pd.Series], 
                              risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Calculate Sortino ratio (focus on downside deviation)
        
        Args:
            returns: Array of returns
            risk_free_rate: Annualized risk-free rate (default: 0.0)
            periods_per_year: Number of periods per year (default: 252 for daily)
            
        Returns:
            Sortino ratio
        """
        returns_array = self._convert_to_array(returns)
        if len(returns_array) <= 1:
            return 0.0
            
        # Calculate excess returns
        excess_returns = returns_array - (risk_free_rate / periods_per_year)
        
        # Calculate annualized mean excess return
        mean_excess_return = np.mean(excess_returns) * periods_per_year
        
        # Calculate downside deviation (only negative returns)
        negative_returns = excess_returns[excess_returns < 0]
        if len(negative_returns) == 0:
            return 0.0
            
        downside_deviation = np.std(negative_returns, ddof=1) * np.sqrt(periods_per_year)
        
        if downside_deviation == 0:
            return 0.0
            
        sortino_ratio = mean_excess_return / downside_deviation
        return sortino_ratio
    
    def calculate_trading_metrics(self, returns: Union[List[float], np.ndarray, pd.Series]) -> tuple:
        """
        Calculate trading metrics (win rate, average win, average loss)
        
        Args:
            returns: Array of returns
            
        Returns:
            Tuple of (win_rate, avg_win, avg_loss)
        """
        returns_array = self._convert_to_array(returns)
        if len(returns_array) == 0:
            return 0.0, 0.0, 0.0
            
        # Separate winning and losing trades
        winning_trades = returns_array[returns_array > 0]
        losing_trades = returns_array[returns_array < 0]
        
        # Calculate win rate
        total_trades = len(returns_array)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        
        # Calculate average win and loss
        avg_win = np.mean(winning_trades) * 100 if len(winning_trades) > 0 else 0.0
        avg_loss = np.mean(losing_trades) * 100 if len(losing_trades) > 0 else 0.0
        
        return win_rate, avg_win, avg_loss
    
    def calculate_calmar_ratio(self, returns: Union[List[float], np.ndarray, pd.Series], 
                             risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown)
        
        Args:
            returns: Array of returns
            risk_free_rate: Annualized risk-free rate (default: 0.0)
            periods_per_year: Number of periods per year (default: 252 for daily)
            
        Returns:
            Calmar ratio
        """
        returns_array = self._convert_to_array(returns)
        if len(returns_array) <= 1:
            return 0.0
            
        annualized_return = self.calculate_annualized_return(returns_array, periods_per_year) / 100
        max_drawdown = abs(self.calculate_max_drawdown(returns_array) / 100)
        
        if max_drawdown == 0:
            return 0.0
            
        calmar_ratio = annualized_return / max_drawdown
        return calmar_ratio
    
    def calculate_omega_ratio(self, returns: Union[List[float], np.ndarray, pd.Series], 
                            risk_free_rate: float = 0.0, threshold: float = 0.0) -> float:
        """
        Calculate Omega ratio (ratio of gains to losses relative to a threshold)
        
        Args:
            returns: Array of returns
            risk_free_rate: Annualized risk-free rate (default: 0.0)
            threshold: Return threshold for separating gains and losses
            
        Returns:
            Omega ratio
        """
        returns_array = self._convert_to_array(returns)
        if len(returns_array) == 0:
            return 0.0
            
        # Adjust returns by risk-free rate
        adjusted_returns = returns_array - (risk_free_rate / 252)
        
        # Separate gains and losses relative to threshold
        gains = adjusted_returns[adjusted_returns > threshold] - threshold
        losses = threshold - adjusted_returns[adjusted_returns < threshold]
        
        if len(gains) == 0 or len(losses) == 0:
            return 0.0
            
        # Calculate Omega ratio
        sum_gains = np.sum(gains)
        sum_losses = np.sum(losses)
        
        if sum_losses == 0:
            return 0.0
            
        omega_ratio = sum_gains / sum_losses
        return omega_ratio
    
    def calculate_skewness(self, returns: Union[List[float], np.ndarray, pd.Series]) -> float:
        """
        Calculate skewness of returns distribution
        
        Args:
            returns: Array of returns
            
        Returns:
            Skewness coefficient
        """
        returns_array = self._convert_to_array(returns)
        if len(returns_array) < 3:
            return 0.0
            
        skewness = pd.Series(returns_array).skew()
        return skewness
    
    def calculate_kurtosis(self, returns: Union[List[float], np.ndarray, pd.Series]) -> float:
        """
        Calculate kurtosis of returns distribution
        
        Args:
            returns: Array of returns
            
        Returns:
            Kurtosis coefficient
        """
        returns_array = self._convert_to_array(returns)
        if len(returns_array) < 4:
            return 0.0
            
        kurtosis = pd.Series(returns_array).kurtosis()
        return kurtosis
    
    def _convert_to_array(self, returns: Union[List[float], np.ndarray, pd.Series]) -> np.ndarray:
        """
        Convert input to numpy array
        
        Args:
            returns: Input returns data
            
        Returns:
            Numpy array of returns
        """
        if isinstance(returns, pd.Series):
            return returns.values
        elif isinstance(returns, list):
            return np.array(returns)
        elif isinstance(returns, np.ndarray):
            return returns
        else:
            self.logger.error(f"Unsupported returns type: {type(returns)}")
            raise ValueError(f"Unsupported returns type: {type(returns)}")
    
    def calculate_rolling_metrics(self, returns: Union[List[float], np.ndarray, pd.Series], 
                                window: int = 30) -> Dict[str, List[float]]:
        """
        Calculate rolling performance metrics
        
        Args:
            returns: Array of returns
            window: Rolling window size
            
        Returns:
            Dictionary of rolling metrics
        """
        returns_array = self._convert_to_array(returns)
        if len(returns_array) < window:
            return {}
            
        rolling_metrics = {
            'rolling_sharpe': [],
            'rolling_volatility': [],
            'rolling_drawdown': []
        }
        
        for i in range(window, len(returns_array) + 1):
            window_returns = returns_array[i-window:i]
            
            # Calculate rolling metrics
            sharpe = self.calculate_sharpe_ratio(window_returns)
            volatility = self.calculate_volatility(window_returns)
            drawdown = self.calculate_max_drawdown(window_returns)
            
            rolling_metrics['rolling_sharpe'].append(sharpe)
            rolling_metrics['rolling_volatility'].append(volatility)
            rolling_metrics['rolling_drawdown'].append(drawdown)
            
        return rolling_metrics