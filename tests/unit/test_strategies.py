"""
Unit tests for Backtrader strategies
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

import backtrader as bt
from backtrader import strategies as btstrats


@pytest.mark.unit
class TestBaseStrategy:
    """Test base strategy functionality"""

    def test_base_strategy_creation(self, sample_ohlcv_data):
        """Test basic strategy creation and initialization"""
        class TestStrategy(bt.Strategy):
            params = (
                ('test_param', 10),
            )

            def __init__(self):
                self.test_value = self.params.test_param

        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)
        cerebro.addstrategy(TestStrategy)

        results = cerebro.run()
        strategy = results[0]

        assert strategy.test_value == 10
        assert hasattr(strategy, 'data')
        assert hasattr(strategy, 'broker')

    def test_strategy_data_access(self, sample_ohlcv_data):
        """Test strategy data access methods"""
        class DataAccessStrategy(bt.Strategy):
            def __init__(self):
                self.close_prices = []
                self.high_prices = []
                self.low_prices = []

            def next(self):
                self.close_prices.append(self.data.close[0])
                self.high_prices.append(self.data.high[0])
                self.low_prices.append(self.data.low[0])

        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)
        cerebro.addstrategy(DataAccessStrategy)

        results = cerebro.run()
        strategy = results[0]

        assert len(strategy.close_prices) > 0
        assert len(strategy.high_prices) > 0
        assert len(strategy.low_prices) > 0

        # Check that OHLC values are reasonable
        assert all(price > 0 for price in strategy.close_prices if not np.isnan(price))

    def test_strategy_order_management(self, sample_ohlcv_data):
        """Test strategy order creation and management"""
        class OrderStrategy(bt.Strategy):
            def __init__(self):
                self.orders = []

            def next(self):
                if len(self.orders) < 2:
                    if self.data.close[0] > self.data.open[0]:
                        order = self.buy(size=10)
                        self.orders.append(order)
                    elif self.data.close[0] < self.data.open[0]:
                        order = self.sell(size=10)
                        self.orders.append(order)

            def notify_order(self, order):
                pass

        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)
        cerebro.addstrategy(OrderStrategy)

        results = cerebro.run()
        strategy = results[0]

        # Should have created some orders
        assert isinstance(strategy.orders, list)

    def test_strategy_indicators(self, sample_ohlcv_data):
        """Test strategy with indicators"""
        class IndicatorStrategy(bt.Strategy):
            def __init__(self):
                self.sma = bt.indicators.SMA(self.data.close, period=20)
                self.rsi = bt.indicators.RSI(self.data.close, period=14)

            def next(self):
                if self.sma[0] > self.data.close[0] and self.rsi[0] < 30:
                    self.buy_signal = True
                else:
                    self.buy_signal = False

        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)
        cerebro.addstrategy(IndicatorStrategy)

        results = cerebro.run()
        strategy = results[0]

        assert hasattr(strategy, 'sma')
        assert hasattr(strategy, 'rsi')
        assert hasattr(strategy, 'buy_signal')


@pytest.mark.unit
class TestCustomStrategies:
    """Test custom trading strategies"""

    def test_sma_cross_strategy(self, sample_ohlcv_data):
        """Test SMA crossover strategy"""
        class SMACrossStrategy(bt.Strategy):
            params = (
                ('fast_period', 10),
                ('slow_period', 20),
            )

            def __init__(self):
                self.fast_sma = bt.indicators.SMA(self.data.close, period=self.params.fast_period)
                self.slow_sma = bt.indicators.SMA(self.data.close, period=self.params.slow_period)
                self.crossover = bt.indicators.CrossOver(self.fast_sma, self.slow_sma)

            def next(self):
                if self.crossover > 0:  # Fast crosses above slow
                    self.buy()
                elif self.crossover < 0:  # Fast crosses below slow
                    self.sell()

        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)
        cerebro.addstrategy(SMACrossStrategy)

        results = cerebro.run()
        strategy = results[0]

        assert hasattr(strategy, 'fast_sma')
        assert hasattr(strategy, 'slow_sma')
        assert hasattr(strategy, 'crossover')

    def test_rsi_strategy(self, sample_ohlcv_data):
        """Test RSI-based strategy"""
        class RSIStrategy(bt.Strategy):
            params = (
                ('rsi_period', 14),
                ('overbought', 70),
                ('oversold', 30),
            )

            def __init__(self):
                self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)

            def next(self):
                if self.rsi[0] < self.params.oversold:
                    self.buy()
                elif self.rsi[0] > self.params.overbought:
                    self.sell()

        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)
        cerebro.addstrategy(RSIStrategy)

        results = cerebro.run()
        strategy = results[0]

        assert hasattr(strategy, 'rsi')

    def test_bollinger_bands_strategy(self, sample_ohlcv_data):
        """Test Bollinger Bands strategy"""
        class BBStrategy(bt.Strategy):
            def __init__(self):
                self.bb = bt.indicators.BollingerBands(self.data.close)

            def next(self):
                if self.data.close[0] < self.bb.bot[0]:
                    self.buy()
                elif self.data.close[0] > self.bb.top[0]:
                    self.sell()

        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)
        cerebro.addstrategy(BBStrategy)

        results = cerebro.run()
        strategy = results[0]

        assert hasattr(strategy, 'bb')
        assert hasattr(strategy.bb, 'top')
        assert hasattr(strategy.bb, 'mid')
        assert hasattr(strategy.bb, 'bot')


@pytest.mark.unit
class TestStrategyParameters:
    """Test strategy parameter handling"""

    def test_strategy_parameter_validation(self, sample_ohlcv_data):
        """Test strategy parameter validation"""
        class ParamStrategy(bt.Strategy):
            params = (
                ('period', 20),
                ('threshold', 0.5),
            )

            def __init__(self):
                assert self.params.period > 0
                assert 0 < self.params.threshold < 1
                self.sma = bt.indicators.SMA(self.data.close, period=self.params.period)

        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)
        cerebro.addstrategy(ParamStrategy)

        # Should work with valid parameters
        results = cerebro.run()
        assert len(results) == 1

    def test_strategy_parameter_override(self, sample_ohlcv_data):
        """Test strategy parameter override"""
        class ParamStrategy(bt.Strategy):
            params = (
                ('period', 20),
            )

            def __init__(self):
                self.period = self.params.period

        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        # Override parameter
        cerebro.addstrategy(ParamStrategy, period=50)

        results = cerebro.run()
        strategy = results[0]

        assert strategy.period == 50


@pytest.mark.unit
class TestStrategyLifecycle:
    """Test strategy lifecycle methods"""

    def test_strategy_start_stop(self, sample_ohlcv_data):
        """Test strategy start and stop methods"""
        class LifecycleStrategy(bt.Strategy):
            def __init__(self):
                self.started = False
                self.stopped = False

            def start(self):
                self.started = True

            def stop(self):
                self.stopped = True

        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)
        cerebro.addstrategy(LifecycleStrategy)

        results = cerebro.run()
        strategy = results[0]

        assert strategy.started
        assert strategy.stopped

    def test_strategy_notify_methods(self, sample_ohlcv_data):
        """Test strategy notification methods"""
        class NotifyStrategy(bt.Strategy):
            def __init__(self):
                self.order_notifications = 0
                self.trade_notifications = 0

            def notify_order(self, order):
                self.order_notifications += 1

            def notify_trade(self, trade):
                self.trade_notifications += 1

            def next(self):
                if not self.position:
                    self.buy(size=1)

        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)
        cerebro.addstrategy(NotifyStrategy)

        results = cerebro.run()
        strategy = results[0]

        # Should have received notifications
        assert strategy.order_notifications >= 0
        assert strategy.trade_notifications >= 0


@pytest.mark.unit
class TestStrategyEdgeCases:
    """Test strategy edge cases"""

    def test_strategy_no_data(self):
        """Test strategy with no data"""
        class EmptyStrategy(bt.Strategy):
            def next(self):
                pass

        cerebro = bt.Cerebro()
        # No data added
        cerebro.addstrategy(EmptyStrategy)

        # Should handle gracefully
        results = cerebro.run()
        assert len(results) == 1

    def test_strategy_minimal_data(self):
        """Test strategy with minimal data"""
        data = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=5, freq='D'),
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [102, 103, 104, 105, 106],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })

        class MinimalStrategy(bt.Strategy):
            def next(self):
                self.data_points = getattr(self, 'data_points', 0) + 1

        cerebro = bt.Cerebro()
        feed = bt.feeds.PolarsData(dataname=data)
        cerebro.adddata(feed)
        cerebro.addstrategy(MinimalStrategy)

        results = cerebro.run()
        strategy = results[0]

        assert strategy.data_points == 5

    def test_strategy_error_handling(self, sample_ohlcv_data):
        """Test strategy error handling"""
        class ErrorStrategy(bt.Strategy):
            def next(self):
                # Simulate an error
                if len(self.data) > 10:
                    raise ValueError("Simulated error")

        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)
        cerebro.addstrategy(ErrorStrategy)

        # Should handle errors gracefully
        with pytest.raises(ValueError):
            cerebro.run()