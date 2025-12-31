"""
Backtrader Integration Module

Implements integration with the backtrader framework for strategy execution.
"""

import logging
import backtrader as bt
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

class BacktraderStrategyFactory:
    """Factory for creating backtrader strategy classes from generated strategies"""
    
    def __init__(self):
        self.logger = logging.getLogger('BacktraderStrategyFactory')
        self.logger.info("BacktraderStrategyFactory initialized")
        
    def create_strategy_class(self, strategy: Dict[str, Any]) -> type:
        """
        Dynamically create a backtrader strategy class from a generated strategy
        
        Args:
            strategy: Generated strategy dictionary
            
        Returns:
            A backtrader Strategy class
        """
        template_name = strategy['template']
        parameters = strategy['parameters']
        
        self.logger.info(f"Creating backtrader strategy class for template: {template_name}")
        
        # Create a new strategy class based on the template
        if template_name == 'moving_average_crossover':
            return self._create_moving_average_crossover_strategy(parameters)
        elif template_name == 'rsi_mean_reversion':
            return self._create_rsi_mean_reversion_strategy(parameters)
        elif template_name == 'bollinger_bands_breakout':
            return self._create_bollinger_bands_breakout_strategy(parameters)
        else:
            self.logger.error(f"Unknown strategy template: {template_name}")
            raise ValueError(f"Unknown strategy template: {template_name}")
    
    def _create_moving_average_crossover_strategy(self, parameters: Dict[str, Any]) -> type:
        """Create a Moving Average Crossover strategy class"""
        
        class MovingAverageCrossoverStrategy(bt.Strategy):
            params = parameters
            
            def __init__(self):
                # Initialize indicators
                if self.p.ma_type == 'SMA':
                    self.fast_ma = bt.indicators.SimpleMovingAverage(
                        period=self.p.fast_period
                    )
                    self.slow_ma = bt.indicators.SimpleMovingAverage(
                        period=self.p.slow_period
                    )
                elif self.p.ma_type == 'EMA':
                    self.fast_ma = bt.indicators.ExponentialMovingAverage(
                        period=self.p.fast_period
                    )
                    self.slow_ma = bt.indicators.ExponentialMovingAverage(
                        period=self.p.slow_period
                    )
                elif self.p.ma_type == 'WMA':
                    self.fast_ma = bt.indicators.WeightedMovingAverage(
                        period=self.p.fast_period
                    )
                    self.slow_ma = bt.indicators.WeightedMovingAverage(
                        period=self.p.slow_period
                    )
                
                # Crossovers
                self.crossover_up = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
                self.crossover_down = bt.indicators.CrossOver(self.slow_ma, self.fast_ma)
                
                # Order tracking
                self.order = None
                self.stop_loss_price = None
                self.take_profit_price = None
            
            def next(self):
                # Check for open orders
                if self.order:
                    return
                
                # Check if we're in the market
                if not self.position:
                    # Buy signal: fast MA crosses above slow MA
                    if self.crossover_up > 0:
                        self.order = self.buy()
                        self.stop_loss_price = self.data.close[0] * (1 - self.p.stop_loss_pct)
                        self.take_profit_price = self.data.close[0] * (1 + self.p.take_profit_pct)
                        self.logger.debug(f"BUY at {self.data.close[0]}, SL: {self.stop_loss_price}, TP: {self.take_profit_price}")
                else:
                    # Sell signal: fast MA crosses below slow MA or hit stop loss/take profit
                    if (self.crossover_down > 0 or 
                        self.data.close[0] <= self.stop_loss_price or 
                        self.data.close[0] >= self.take_profit_price):
                        self.order = self.sell()
                        self.stop_loss_price = None
                        self.take_profit_price = None
                        self.logger.debug(f"SELL at {self.data.close[0]}")
            
            def notify_order(self, order):
                if order.status in [order.Submitted, order.Accepted]:
                    return
                
                if order.status in [order.Completed]:
                    if order.isbuy():
                        self.logger.debug(f"BUY EXECUTED at {order.executed.price}")
                    elif order.issell():
                        self.logger.debug(f"SELL EXECUTED at {order.executed.price}")
                
                self.order = None
        
        return MovingAverageCrossoverStrategy
    
    def _create_rsi_mean_reversion_strategy(self, parameters: Dict[str, Any]) -> type:
        """Create an RSI Mean Reversion strategy class"""
        
        class RSIMeanReversionStrategy(bt.Strategy):
            params = parameters
            
            def __init__(self):
                # Initialize RSI indicator
                self.rsi = bt.indicators.RSI(
                    period=self.p.rsi_period
                )
                
                # Order tracking
                self.order = None
                self.entry_price = None
                self.entry_bars = 0
            
            def next(self):
                # Check for open orders
                if self.order:
                    return
                
                # Check if we're in the market
                if not self.position:
                    # Buy signal: RSI crosses below oversold threshold
                    if (self.rsi[0] < self.p.oversold_threshold and 
                        self.rsi[-1] >= self.p.oversold_threshold):
                        size = self.broker.getvalue() * self.p.position_size_pct / self.data.close[0]
                        self.order = self.buy(size=size)
                        self.entry_price = self.data.close[0]
                        self.entry_bars = 0
                        self.logger.debug(f"BUY at {self.data.close[0]}, RSI: {self.rsi[0]:.2f}")
                else:
                    # Sell signal: RSI crosses above overbought threshold or max holding period reached
                    self.entry_bars += 1
                    sell_condition = (
                        (self.rsi[0] > self.p.overbought_threshold and
                         self.rsi[-1] <= self.p.overbought_threshold) or
                        self.entry_bars >= self.p.max_holding_period
                    )
                    if sell_condition:
                        self.order = self.sell()
                        self.entry_price = None
                        self.entry_bars = 0
                        self.logger.debug(f"SELL at {self.data.close[0]}, RSI: {self.rsi[0]:.2f}")
            
            def notify_order(self, order):
                if order.status in [order.Submitted, order.Accepted]:
                    return
                
                if order.status in [order.Completed]:
                    if order.isbuy():
                        self.logger.debug(f"BUY EXECUTED at {order.executed.price}")
                    elif order.issell():
                        self.logger.debug(f"SELL EXECUTED at {order.executed.price}")
                
                self.order = None
        
        return RSIMeanReversionStrategy
    
    def _create_bollinger_bands_breakout_strategy(self, parameters: Dict[str, Any]) -> type:
        """Create a Bollinger Bands Breakout strategy class"""
        
        class BollingerBandsBreakoutStrategy(bt.Strategy):
            params = parameters
            
            def __init__(self):
                # Initialize Bollinger Bands
                self.bb = bt.indicators.BollingerBands(
                    period=self.p.bb_period,
                    devfactor=self.p.bb_std_dev
                )
                
                # Initialize ATR for stop loss
                self.atr = bt.indicators.ATR(
                    period=self.p.atr_period
                )
                
                # Order tracking
                self.order = None
                self.stop_loss_price = None
                self.breakout_confirmed = False
            
            def next(self):
                # Check for open orders
                if self.order:
                    return
                
                # Check if we're in the market
                if not self.position:
                    # Buy signal: price breaks above upper band
                    if self.data.close[0] > self.bb.lines.top[0]:
                        if not self.p.breakout_confirmation or self.breakout_confirmed:
                            self.order = self.buy()
                            self.stop_loss_price = self.data.close[0] - self.atr[0] * self.p.atr_multiplier
                            self.breakout_confirmed = False
                            self.logger.debug(f"BUY at {self.data.close[0]}, BB Top: {self.bb.lines.top[0]:.2f}")
                        else:
                            self.breakout_confirmed = True
                else:
                    # Sell signal: price breaks below lower band or hits stop loss
                    if (self.data.close[0] < self.bb.lines.bot[0] or 
                        self.data.close[0] <= self.stop_loss_price):
                        self.order = self.sell()
                        self.stop_loss_price = None
                        self.breakout_confirmed = False
                        self.logger.debug(f"SELL at {self.data.close[0]}, BB Bot: {self.bb.lines.bot[0]:.2f}")
            
            def notify_order(self, order):
                if order.status in [order.Submitted, order.Accepted]:
                    return
                
                if order.status in [order.Completed]:
                    if order.isbuy():
                        self.logger.debug(f"BUY EXECUTED at {order.executed.price}")
                    elif order.issell():
                        self.logger.debug(f"SELL EXECUTED at {order.executed.price}")
                
                self.order = None
        
        return BollingerBandsBreakoutStrategy
    
    def run_backtest(self, strategy: Dict[str, Any], data: pd.DataFrame, 
                    initial_cash: float = 10000.0, **kwargs) -> Dict[str, Any]:
        """
        Run a backtest using backtrader framework
        
        Args:
            strategy: Generated strategy dictionary
            data: Pandas DataFrame with OHLCV data
            initial_cash: Initial cash for backtesting
            kwargs: Additional backtrader cerebro parameters
            
        Returns:
            Dictionary with backtest results and performance metrics
        """
        self.logger.info(f"Running backtest for strategy: {strategy['template']}")
        
        # Create cerebro engine
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(initial_cash)
        
        # Add strategy
        StrategyClass = self.create_strategy_class(strategy)
        cerebro.addstrategy(StrategyClass)
        
        # Add data feed
        data_feed = bt.feeds.PandasData(dataname=data)
        cerebro.adddata(data_feed)
        
        # Run backtest
        self.logger.debug("Starting backtest execution")
        results = cerebro.run(**kwargs)
        
        # Extract performance metrics
        strat = results[0]
        portfolio_value = cerebro.broker.getvalue()
        
        # Calculate performance metrics
        if len(strat) > 0:
            # Get trade analysis
            analyzer = strat.analyzers.getbyname('trade_analysis') if hasattr(strat, 'analyzers') else None
            
            performance_metrics = {
                'initial_cash': initial_cash,
                'final_value': portfolio_value,
                'total_return': (portfolio_value - initial_cash) / initial_cash,
                'sharpe_ratio': 0.0,  # Would need returns series to calculate properly
                'max_drawdown': 0.0,  # Would need equity curve to calculate properly
                'win_rate': 0.0,      # Would need trade analysis
                'total_trades': 0,    # Would need trade analysis
                'winning_trades': 0,  # Would need trade analysis
                'losing_trades': 0    # Would need trade analysis
            }
            
            # Basic metrics we can calculate
            if hasattr(strat, 'analyzers'):
                for analyzer in strat.analyzers:
                    if hasattr(analyzer, 'get_analysis'):
                        analysis = analyzer.get_analysis()
                        if 'sharperatio' in analysis:
                            performance_metrics['sharpe_ratio'] = analysis['sharperatio']
                        if 'max' in analysis and 'drawdown' in analysis['max']:
                            performance_metrics['max_drawdown'] = analysis['max']['drawdown']
        else:
            performance_metrics = {
                'initial_cash': initial_cash,
                'final_value': portfolio_value,
                'total_return': (portfolio_value - initial_cash) / initial_cash,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0
            }
        
        # Create result dictionary
        result = {
            'strategy_id': strategy.get('id', 'unknown'),
            'template': strategy['template'],
            'parameters': strategy['parameters'],
            'performance_metrics': performance_metrics,
            'backtest_metadata': {
                'initial_cash': initial_cash,
                'final_value': portfolio_value,
                'data_points': len(data),
                'backtrader_version': bt.__version__
            }
        }
        
        self.logger.info(f"Backtest completed. Final value: {portfolio_value:.2f}")
        return result
    
    def create_data_feed(self, data: pd.DataFrame) -> bt.feeds.PandasData:
        """
        Create a backtrader data feed from pandas DataFrame
        
        Args:
            data: Pandas DataFrame with OHLCV data
            
        Returns:
            Backtrader data feed
        """
        return bt.feeds.PandasData(dataname=data)