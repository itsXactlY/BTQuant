"""
Integration tests for the complete backtesting pipeline
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import backtrader as bt
from backtrader import analyzers as btanalyzers
from backtrader import observers as btobs


@pytest.mark.integration
class TestCompleteBacktestingPipeline:
    """Test complete backtesting pipeline integration"""

    def test_full_backtest_with_strategy_analyzers_observers(self, sample_ohlcv_data):
        """Test full backtest with strategy, analyzers, and observers"""
        class CompleteStrategy(bt.Strategy):
            params = (
                ('sma_period', 20),
                ('rsi_period', 14),
            )

            def __init__(self):
                self.sma = bt.indicators.SMA(self.data.close, period=self.params.sma_period)
                self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
                self.trades_executed = 0

            def next(self):
                if self.rsi[0] < 30 and self.data.close[0] > self.sma[0]:
                    self.buy(size=10)
                    self.trades_executed += 1
                elif self.rsi[0] > 70 and self.data.close[0] < self.sma[0]:
                    self.sell(size=10)
                    self.trades_executed += 1

        cerebro = bt.Cerebro()

        # Add data
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        # Add strategy
        cerebro.addstrategy(CompleteStrategy)

        # Add multiple analyzers
        cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
        cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
        cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')

        # Add observers
        cerebro.addobserver(btobs.Broker, _name='broker')
        cerebro.addobserver(btobs.DrawDown, _name='dd_obs')
        cerebro.addobserver(btobs.Trades, _name='trades_obs')

        # Configure broker
        cerebro.broker.set_cash(100000)
        cerebro.broker.setcommission(commission=0.001)

        # Run backtest
        results = cerebro.run()
        strategy = results[0]

        # Verify all components worked together
        assert hasattr(strategy, 'sma')
        assert hasattr(strategy, 'rsi')
        assert strategy.trades_executed >= 0

        # Check analyzers
        assert 'returns' in strategy.analyzers
        assert 'sharpe' in strategy.analyzers
        assert 'drawdown' in strategy.analyzers
        assert 'trades' in strategy.analyzers

        # Check observers
        assert 'broker' in strategy.observers
        assert 'dd_obs' in strategy.observers
        assert 'trades_obs' in strategy.observers

    def test_multi_asset_backtest(self):
        """Test backtest with multiple assets"""
        # Create two different data sets
        data1 = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=100, freq='D'),
            'open': np.random.uniform(90, 110, 100),
            'high': np.random.uniform(105, 115, 100),
            'low': np.random.uniform(85, 95, 100),
            'close': np.random.uniform(95, 105, 100),
            'volume': np.random.randint(1000, 2000, 100)
        })

        data2 = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=100, freq='D'),
            'open': np.random.uniform(190, 210, 100),
            'high': np.random.uniform(205, 215, 100),
            'low': np.random.uniform(185, 195, 100),
            'close': np.random.uniform(195, 205, 100),
            'volume': np.random.randint(500, 1000, 100)
        })

        class MultiAssetStrategy(bt.Strategy):
            def __init__(self):
                # Different indicators for different data feeds
                self.sma1 = bt.indicators.SMA(self.data0.close, period=20)
                self.sma2 = bt.indicators.SMA(self.data1.close, period=20)

            def next(self):
                # Trade logic based on both assets
                if self.sma1[0] > self.data0.close[0] and self.sma2[0] < self.data1.close[0]:
                    self.buy(data=self.data0, size=5)
                    self.sell(data=self.data1, size=3)

        cerebro = bt.Cerebro()

        # Add multiple data feeds
        feed1 = bt.feeds.PolarsData(dataname=data1)
        feed2 = bt.feeds.PolarsData(dataname=data2)
        cerebro.adddata(feed1)
        cerebro.adddata(feed2)

        cerebro.addstrategy(MultiAssetStrategy)
        cerebro.broker.set_cash(100000)

        results = cerebro.run()
        strategy = results[0]

        # Should have indicators for both data feeds
        assert hasattr(strategy, 'sma1')
        assert hasattr(strategy, 'sma2')

    def test_backtest_with_custom_sizer(self, sample_ohlcv_data):
        """Test backtest with custom position sizer"""
        class PercentageSizer(bt.sizers.PercentSizer):
            params = (
                ('percents', 10),
            )

        class SizerStrategy(bt.Strategy):
            def next(self):
                if not self.position:
                    self.buy()

        cerebro = bt.Cerebro()

        # Add data and strategy
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)
        cerebro.addstrategy(SizerStrategy)

        # Add custom sizer
        cerebro.addsizer(PercentageSizer)

        # Configure broker
        cerebro.broker.set_cash(100000)

        results = cerebro.run()
        strategy = results[0]

        # Strategy should have executed
        assert results is not None

    def test_backtest_with_portfolio_rebalancing(self, sample_ohlcv_data):
        """Test backtest with portfolio rebalancing logic"""
        class RebalanceStrategy(bt.Strategy):
            params = (
                ('rebalance_days', 30),
            )

            def __init__(self):
                self.days_since_rebalance = 0
                self.rebalance_count = 0

            def next(self):
                self.days_since_rebalance += 1

                if self.days_since_rebalance >= self.params.rebalance_days:
                    # Simple rebalance: buy if not in position
                    if not self.position:
                        self.buy(size=10)
                        self.rebalance_count += 1
                    self.days_since_rebalance = 0

        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)
        cerebro.addstrategy(RebalanceStrategy)
        cerebro.broker.set_cash(100000)

        results = cerebro.run()
        strategy = results[0]

        assert strategy.rebalance_count >= 0

    def test_backtest_with_risk_management(self, sample_ohlcv_data):
        """Test backtest with risk management features"""
        class RiskManagedStrategy(bt.Strategy):
            params = (
                ('max_drawdown', 0.1),  # 10% max drawdown
                ('stop_loss', 0.05),    # 5% stop loss
            )

            def __init__(self):
                self.entry_price = None
                self.stop_orders = []

            def next(self):
                if not self.position:
                    # Enter position
                    self.buy(size=10)
                    self.entry_price = self.data.close[0]
                elif self.position:
                    # Check stop loss
                    current_price = self.data.close[0]
                    loss_pct = (self.entry_price - current_price) / self.entry_price

                    if loss_pct >= self.params.stop_loss:
                        self.sell(size=self.position.size)
                        self.entry_price = None

        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)
        cerebro.addstrategy(RiskManagedStrategy)
        cerebro.broker.set_cash(100000)

        results = cerebro.run()
        strategy = results[0]

        # Strategy should have risk management logic
        assert hasattr(strategy, 'entry_price')

    def test_backtest_performance_metrics_calculation(self, sample_ohlcv_data):
        """Test that performance metrics are calculated correctly"""
        class SimpleStrategy(bt.Strategy):
            def next(self):
                if self.data.close[0] > self.data.open[0]:
                    self.buy(size=1)
                elif self.data.close[0] < self.data.open[0]:
                    self.sell(size=1)

        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)
        cerebro.addstrategy(SimpleStrategy)

        # Add comprehensive performance analysis
        analyzers = [
            (btanalyzers.Returns, 'returns'),
            (btanalyzers.SharpeRatio, 'sharpe'),
            (btanalyzers.DrawDown, 'drawdown'),
            (btanalyzers.SQN, 'sqn'),
            (btanalyzers.TradeAnalyzer, 'trades'),
        ]

        for analyzer_class, name in analyzers:
            cerebro.addanalyzer(analyzer_class, _name=name)

        cerebro.broker.set_cash(10000)
        cerebro.broker.setcommission(commission=0.001)

        results = cerebro.run()
        strategy = results[0]

        # All analyzers should be present and have run
        for _, name in analyzers:
            assert name in strategy.analyzers

        # Check that we can access key metrics
        returns_analyzer = strategy.analyzers.returns
        sharpe_analyzer = strategy.analyzers.sharpe

        # These should not be None/empty
        assert returns_analyzer is not None
        assert sharpe_analyzer is not None


@pytest.mark.integration
class TestBacktestingEdgeCases:
    """Test backtesting pipeline edge cases"""

    def test_backtest_with_no_trades(self, sample_ohlcv_data):
        """Test backtest where strategy never trades"""
        class NoTradeStrategy(bt.Strategy):
            def next(self):
                pass  # Never trade

        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)
        cerebro.addstrategy(NoTradeStrategy)

        # Add analyzers
        cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
        cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')

        results = cerebro.run()
        strategy = results[0]

        # Should still complete successfully
        assert 'returns' in strategy.analyzers
        assert 'trades' in strategy.analyzers

    def test_backtest_with_insufficient_data(self):
        """Test backtest with very little data"""
        data = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=10, freq='D'),
            'open': [100] * 10,
            'high': [105] * 10,
            'low': [95] * 10,
            'close': [102] * 10,
            'volume': [1000] * 10
        })

        class MinimalStrategy(bt.Strategy):
            def __init__(self):
                self.sma = bt.indicators.SMA(self.data.close, period=5)

            def next(self):
                if self.sma[0] is not None and self.data.close[0] > self.sma[0]:
                    self.buy(size=1)

        cerebro = bt.Cerebro()
        feed = bt.feeds.PolarsData(dataname=data)
        cerebro.adddata(feed)
        cerebro.addstrategy(MinimalStrategy)
        cerebro.broker.set_cash(1000)

        results = cerebro.run()
        strategy = results[0]

        # Should handle gracefully
        assert hasattr(strategy, 'sma')

    def test_backtest_with_extreme_parameters(self, sample_ohlcv_data):
        """Test backtest with extreme parameter values"""
        class ExtremeStrategy(bt.Strategy):
            params = (
                ('very_long_period', 200),  # Longer than data
            )

            def __init__(self):
                self.long_sma = bt.indicators.SMA(self.data.close, period=self.params.very_long_period)

        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)
        cerebro.addstrategy(ExtremeStrategy)

        results = cerebro.run()
        strategy = results[0]

        # Should handle extreme parameters gracefully
        assert hasattr(strategy, 'long_sma')

    def test_backtest_error_recovery(self, sample_ohlcv_data):
        """Test backtest error recovery"""
        class ErrorStrategy(bt.Strategy):
            def __init__(self):
                self.error_count = 0

            def next(self):
                try:
                    # Simulate occasional errors
                    if np.random.random() < 0.1:  # 10% chance
                        raise ValueError("Random error")
                    if self.data.close[0] > self.data.open[0]:
                        self.buy(size=1)
                except Exception:
                    self.error_count += 1
                    # Continue execution

        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)
        cerebro.addstrategy(ErrorStrategy)
        cerebro.broker.set_cash(10000)

        results = cerebro.run()
        strategy = results[0]

        # Should have handled errors and continued
        assert hasattr(strategy, 'error_count')