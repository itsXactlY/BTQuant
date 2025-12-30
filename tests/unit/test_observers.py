"""
Unit tests for Backtrader observers
"""

import pytest
import numpy as np
import pandas as pd

import backtrader as bt
from backtrader import observers as btobs


@pytest.mark.unit
class TestPortfolioObservers:
    """Test portfolio monitoring observers"""

    def test_broker_observer(self, sample_ohlcv_data, mock_strategy):
        """Test Broker observer"""
        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        # Add broker observer
        cerebro.addobserver(btobs.Broker, _name='broker')
        cerebro.addstrategy(mock_strategy)

        results = cerebro.run()
        strategy = results[0]

        # Check that observer was added
        assert 'broker' in strategy.observers
        broker_obs = strategy.observers.broker

        # Broker observer should track cash and value
        assert hasattr(broker_obs, 'cash')
        assert hasattr(broker_obs, 'value')

    def test_drawdown_observer(self, sample_ohlcv_data, mock_strategy):
        """Test Drawdown observer"""
        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        # Add drawdown observer
        cerebro.addobserver(btobs.DrawDown, _name='drawdown')
        cerebro.addstrategy(mock_strategy)

        results = cerebro.run()
        strategy = results[0]

        assert 'drawdown' in strategy.observers
        dd_obs = strategy.observers.drawdown

        # Drawdown observer should track drawdown metrics
        assert hasattr(dd_obs, 'drawdown')
        assert hasattr(dd_obs, 'maxdrawdown')

    def test_trades_observer(self, sample_ohlcv_data):
        """Test Trades observer"""
        class TradingStrategy(bt.Strategy):
            def __init__(self):
                self.trade_count = 0

            def next(self):
                if self.trade_count < 3:  # Limit trades
                    if self.data.close[0] > self.data.open[0]:
                        self.buy(size=10)
                        self.trade_count += 1

        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        # Add trades observer
        cerebro.addobserver(btobs.Trades, _name='trades')
        cerebro.addstrategy(TradingStrategy)

        results = cerebro.run()
        strategy = results[0]

        assert 'trades' in strategy.observers
        trades_obs = strategy.observers.trades

        # Trades observer should track trade information
        assert hasattr(trades_obs, 'trades')


@pytest.mark.unit
class TestPerformanceObservers:
    """Test performance tracking observers"""

    def test_logreturns_observer(self, sample_ohlcv_data, mock_strategy):
        """Test LogReturns observer"""
        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        # Add log returns observer
        cerebro.addobserver(btobs.LogReturns, _name='logreturns')
        cerebro.addstrategy(mock_strategy)

        results = cerebro.run()
        strategy = results[0]

        assert 'logreturns' in strategy.observers
        lr_obs = strategy.observers.logreturns

        assert hasattr(lr_obs, 'logreturns')

    def test_time_return_observer(self, sample_ohlcv_data, mock_strategy):
        """Test TimeReturn observer"""
        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        # Add time return observer
        cerebro.addobserver(btobs.TimeReturn, _name='timereturn', timeframe=bt.TimeFrame.Days)
        cerebro.addstrategy(mock_strategy)

        results = cerebro.run()
        strategy = results[0]

        assert 'timereturn' in strategy.observers
        tr_obs = strategy.observers.timereturn

        assert hasattr(tr_obs, 'timereturn')


@pytest.mark.unit
class TestSignalObservers:
    """Test signal and action observers"""

    def test_buysell_observer(self, sample_ohlcv_data):
        """Test BuySell observer"""
        class SignalStrategy(bt.Strategy):
            def next(self):
                if self.data.close[0] > self.data.open[0]:
                    self.buy()
                elif self.data.close[0] < self.data.open[0]:
                    self.sell()

        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        # Add buy/sell observer
        cerebro.addobserver(btobs.BuySell, _name='buysell')
        cerebro.addstrategy(SignalStrategy)

        results = cerebro.run()
        strategy = results[0]

        assert 'buysell' in strategy.observers
        bs_obs = strategy.observers.buysell

        # BuySell observer should track buy/sell signals
        assert hasattr(bs_obs, 'buy')
        assert hasattr(bs_obs, 'sell')


@pytest.mark.unit
class TestBenchmarkObservers:
    """Test benchmark comparison observers"""

    def test_benchmark_observer(self, sample_ohlcv_data, mock_strategy):
        """Test Benchmark observer"""
        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        # Add benchmark observer
        cerebro.addobserver(btobs.Benchmark, _name='benchmark', data=data)
        cerebro.addstrategy(mock_strategy)

        results = cerebro.run()
        strategy = results[0]

        assert 'benchmark' in strategy.observers
        bench_obs = strategy.observers.benchmark

        # Benchmark observer should track benchmark performance
        assert hasattr(bench_obs, 'benchmark')


@pytest.mark.unit
class TestObserverEdgeCases:
    """Test observers with edge cases"""

    def test_multiple_observers_same_type(self, sample_ohlcv_data, mock_strategy):
        """Test multiple observers of the same type"""
        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        # Add multiple drawdown observers with different names
        cerebro.addobserver(btobs.DrawDown, _name='drawdown1')
        cerebro.addobserver(btobs.DrawDown, _name='drawdown2')
        cerebro.addstrategy(mock_strategy)

        results = cerebro.run()
        strategy = results[0]

        # Both observers should exist
        assert 'drawdown1' in strategy.observers
        assert 'drawdown2' in strategy.observers

        dd1 = strategy.observers.drawdown1
        dd2 = strategy.observers.drawdown2

        # They should have the same structure
        assert hasattr(dd1, 'drawdown')
        assert hasattr(dd2, 'drawdown')

    def test_observers_with_no_activity(self, sample_ohlcv_data):
        """Test observers when strategy has no activity"""
        class InactiveStrategy(bt.Strategy):
            def next(self):
                pass  # No actions

        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        # Add various observers
        observers = [
            (btobs.Broker, 'broker'),
            (btobs.DrawDown, 'drawdown'),
            (btobs.Trades, 'trades'),
            (btobs.BuySell, 'buysell')
        ]

        for obs_class, name in observers:
            cerebro.addobserver(obs_class, _name=name)

        cerebro.addstrategy(InactiveStrategy)
        results = cerebro.run()
        strategy = results[0]

        # All observers should still exist and not crash
        for _, name in observers:
            assert name in strategy.observers

    def test_observer_with_minimal_data(self):
        """Test observers with minimal data"""
        # Create minimal dataset
        data = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=5, freq='D'),
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [102, 103, 104, 105, 106],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })

        cerebro = bt.Cerebro()
        feed = bt.feeds.PolarsData(dataname=data)
        cerebro.adddata(feed)

        # Add observers
        cerebro.addobserver(btobs.Broker, _name='broker')
        cerebro.addobserver(btobs.DrawDown, _name='drawdown')

        cerebro.addstrategy(bt.Strategy)  # Empty strategy
        results = cerebro.run()
        strategy = results[0]

        # Should not crash with minimal data
        assert 'broker' in strategy.observers
        assert 'drawdown' in strategy.observers

    def test_observer_initialization_parameters(self, sample_ohlcv_data, mock_strategy):
        """Test observer initialization with different parameters"""
        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        # Add time return observer with different timeframes
        cerebro.addobserver(btobs.TimeReturn, _name='daily', timeframe=bt.TimeFrame.Days)
        cerebro.addobserver(btobs.TimeReturn, _name='weekly', timeframe=bt.TimeFrame.Weeks)
        cerebro.addstrategy(mock_strategy)

        results = cerebro.run()
        strategy = results[0]

        # Both observers should exist
        assert 'daily' in strategy.observers
        assert 'weekly' in strategy.observers

        daily_obs = strategy.observers.daily
        weekly_obs = strategy.observers.weekly

        # Both should have timereturn attribute
        assert hasattr(daily_obs, 'timereturn')
        assert hasattr(weekly_obs, 'timereturn')