"""
Performance tests for backtesting speed and memory usage
"""

import pytest
import time
import psutil
import os
import numpy as np
import pandas as pd
from datetime import datetime

import backtrader as bt
from backtrader import analyzers as btanalyzers
from backtrader import indicators as btind


@pytest.mark.performance
class TestBacktestingSpeed:
    """Test backtesting execution speed"""

    def test_simple_strategy_performance(self, benchmark):
        """Benchmark simple strategy execution speed"""
        # Create large dataset
        np.random.seed(42)
        n_points = 10000
        dates = pd.date_range('2020-01-01', periods=n_points, freq='5min')

        # Generate realistic price data
        prices = [100]
        for _ in range(n_points - 1):
            change = np.random.normal(0, 0.001)  # Small changes for 5min data
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)

        data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': [p * 1.001 for p in prices],
            'low': [p * 0.999 for p in prices],
            'close': prices,
            'volume': np.random.randint(100, 1000, n_points)
        })

        class SimpleStrategy(bt.Strategy):
            def next(self):
                if self.data.close[0] > self.data.open[0]:
                    self.buy(size=1)
                elif self.data.close[0] < self.data.open[0]:
                    self.sell(size=1)

        def run_backtest():
            cerebro = bt.Cerebro()
            feed = bt.feeds.PolarsData(dataname=data)
            cerebro.adddata(feed)
            cerebro.addstrategy(SimpleStrategy)
            cerebro.broker.set_cash(10000)
            return cerebro.run()

        # Benchmark the execution
        result = benchmark(run_backtest)
        assert result is not None

    def test_indicator_heavy_strategy_performance(self, benchmark):
        """Benchmark strategy with many indicators"""
        # Create dataset
        np.random.seed(42)
        n_points = 5000
        dates = pd.date_range('2020-01-01', periods=n_points, freq='1H')

        prices = [100]
        for _ in range(n_points - 1):
            change = np.random.normal(0, 0.002)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)

        data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': [p * 1.002 for p in prices],
            'low': [p * 0.998 for p in prices],
            'close': prices,
            'volume': np.random.randint(500, 2000, n_points)
        })

        class IndicatorHeavyStrategy(bt.Strategy):
            def __init__(self):
                # Many indicators
                self.sma5 = btind.SMA(self.data.close, period=5)
                self.sma10 = btind.SMA(self.data.close, period=10)
                self.sma20 = btind.SMA(self.data.close, period=20)
                self.sma50 = btind.SMA(self.data.close, period=50)
                self.ema12 = btind.EMA(self.data.close, period=12)
                self.ema26 = btind.EMA(self.data.close, period=26)
                self.rsi = btind.RSI(self.data.close, period=14)
                self.macd = btind.MACD(self.data.close)
                self.bb = btind.BollingerBands(self.data.close)
                self.stoch = btind.Stochastic(self.data)
                self.cci = btind.CCI(self.data, period=20)
                self.williams = btind.WilliamsR(self.data)

            def next(self):
                # Complex logic using all indicators
                if (self.rsi[0] < 30 and
                    self.macd.macd[0] > self.macd.signal[0] and
                    self.data.close[0] < self.bb.bot[0]):
                    self.buy(size=10)
                elif (self.rsi[0] > 70 or
                      self.data.close[0] > self.bb.top[0]):
                    if self.position:
                        self.sell(size=self.position.size)

        def run_heavy_backtest():
            cerebro = bt.Cerebro()
            feed = bt.feeds.PolarsData(dataname=data)
            cerebro.adddata(feed)
            cerebro.addstrategy(IndicatorHeavyStrategy)
            cerebro.broker.set_cash(10000)
            return cerebro.run()

        result = benchmark(run_heavy_backtest)
        assert result is not None

    def test_multi_asset_portfolio_performance(self, benchmark):
        """Benchmark multi-asset portfolio performance"""
        np.random.seed(42)
        n_points = 2000
        dates = pd.date_range('2020-01-01', periods=n_points, freq='1D')

        # Create 5 correlated assets
        base_trend = np.linspace(0, 100, n_points)
        assets = []

        for i in range(5):
            correlation = 0.7  # 70% correlation
            noise = np.random.normal(0, 10, n_points)
            prices = 100 + base_trend * (0.5 + i * 0.1) + noise * (1 - correlation) + base_trend * correlation
            assets.append(prices)

        data_frames = []
        for i, prices in enumerate(assets):
            df = pd.DataFrame({
                'datetime': dates,
                'open': prices,
                'high': prices * 1.01,
                'low': prices * 0.99,
                'close': prices,
                'volume': np.random.randint(1000, 5000, n_points)
            })
            data_frames.append(df)

        class MultiAssetStrategy(bt.Strategy):
            def __init__(self):
                # Indicators for each asset
                self.smas = []
                for i in range(5):
                    sma = btind.SMA(getattr(self, f'data{i}').close, period=20)
                    self.smas.append(sma)

            def next(self):
                # Allocate to assets with best momentum
                momentums = []
                for i in range(5):
                    data = getattr(self, f'data{i}')
                    if len(data) > 10:
                        momentum = data.close[0] / data.close[-10]
                        momentums.append((i, momentum))
                    else:
                        momentums.append((i, 1.0))

                # Sort by momentum
                momentums.sort(key=lambda x: x[1], reverse=True)

                # Allocate to top 3 assets
                total_allocation = 0
                for asset_idx, _ in momentums[:3]:
                    allocation = 0.3 / 3  # Equal weight among top 3
                    self.order_target_percent(
                        data=getattr(self, f'data{asset_idx}'),
                        target=allocation
                    )
                    total_allocation += allocation

                # Reduce allocation to bottom assets
                for asset_idx, _ in momentums[3:]:
                    self.order_target_percent(
                        data=getattr(self, f'data{asset_idx}'),
                        target=0.05  # Minimal allocation
                    )

        def run_multi_asset_backtest():
            cerebro = bt.Cerebro()
            for i, df in enumerate(data_frames):
                feed = bt.feeds.PolarsData(dataname=df)
                cerebro.adddata(feed, name=f'data{i}')

            cerebro.addstrategy(MultiAssetStrategy)
            cerebro.broker.set_cash(100000)
            return cerebro.run()

        result = benchmark(run_multi_asset_backtest)
        assert result is not None


@pytest.mark.performance
class TestMemoryUsage:
    """Test memory usage during backtesting"""

    def test_memory_usage_scaling(self):
        """Test how memory usage scales with data size"""
        memory_usage = []

        for n_points in [1000, 5000, 10000]:
            # Create dataset of varying sizes
            dates = pd.date_range('2020-01-01', periods=n_points, freq='1H')
            prices = np.random.normal(100, 10, n_points).cumsum()

            data = pd.DataFrame({
                'datetime': dates,
                'open': prices,
                'high': prices * 1.002,
                'low': prices * 0.998,
                'close': prices,
                'volume': np.random.randint(1000, 5000, n_points)
            })

            class MemoryTestStrategy(bt.Strategy):
                def __init__(self):
                    # Many indicators to test memory usage
                    self.smas = [btind.SMA(self.data.close, period=i) for i in range(5, 51, 5)]
                    self.emas = [btind.EMA(self.data.close, period=i) for i in range(5, 51, 5)]

                def next(self):
                    pass

            # Measure memory before
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB

            cerebro = bt.Cerebro()
            feed = bt.feeds.PolarsData(dataname=data)
            cerebro.adddata(feed)
            cerebro.addstrategy(MemoryTestStrategy)
            cerebro.run()

            # Measure memory after
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage.append((n_points, mem_after - mem_before))

        # Memory usage should scale reasonably with data size
        assert len(memory_usage) == 3
        # Basic check that memory usage increased
        assert all(mem_diff >= 0 for _, mem_diff in memory_usage)

    def test_memory_cleanup_after_backtest(self):
        """Test that memory is properly cleaned up after backtesting"""
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Run multiple backtests
        for i in range(5):
            dates = pd.date_range('2020-01-01', periods=1000, freq='1D')
            prices = np.random.normal(100, 5, 1000).cumsum()

            data = pd.DataFrame({
                'datetime': dates,
                'open': prices,
                'high': prices * 1.01,
                'low': prices * 0.99,
                'close': prices,
                'volume': np.random.randint(1000, 3000, 1000)
            })

            class CleanupTestStrategy(bt.Strategy):
                def __init__(self):
                    # Create many objects
                    self.indicators = [btind.SMA(self.data.close, period=j) for j in range(10, 50)]

            cerebro = bt.Cerebro()
            feed = bt.feeds.PolarsData(dataname=data)
            cerebro.adddata(feed)
            cerebro.addstrategy(CleanupTestStrategy)
            cerebro.run()

        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = mem_after - mem_before

        # Memory increase should be reasonable (less than 500MB for this test)
        assert mem_increase < 500, f"Memory increase too high: {mem_increase}MB"


@pytest.mark.performance
class TestIndicatorPerformance:
    """Test performance of individual indicators"""

    @pytest.mark.parametrize("period", [5, 20, 50, 100])
    def test_sma_performance_scaling(self, benchmark, period):
        """Test SMA performance with different periods"""
        n_points = 5000
        dates = pd.date_range('2020-01-01', periods=n_points, freq='1H')
        prices = np.random.normal(100, 5, n_points).cumsum()

        data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices,
            'volume': np.random.randint(1000, 3000, n_points)
        })

        def run_sma_test():
            cerebro = bt.Cerebro()
            feed = bt.feeds.PolarsData(dataname=data)
            cerebro.adddata(feed)

            class SMATestStrategy(bt.Strategy):
                def __init__(self):
                    self.sma = btind.SMA(self.data.close, period=period)

                def next(self):
                    pass  # Just create the indicator

            cerebro.addstrategy(SMATestStrategy)
            return cerebro.run()

        result = benchmark(run_sma_test)
        assert result is not None

    def test_complex_indicator_performance(self, benchmark):
        """Test performance of complex indicators"""
        n_points = 3000
        dates = pd.date_range('2020-01-01', periods=n_points, freq='1H')
        prices = np.random.normal(100, 5, n_points).cumsum()

        data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices,
            'volume': np.random.randint(1000, 3000, n_points)
        })

        def run_complex_indicators():
            cerebro = bt.Cerebro()
            feed = bt.feeds.PolarsData(dataname=data)
            cerebro.adddata(feed)

            class ComplexIndicatorStrategy(bt.Strategy):
                def __init__(self):
                    # Complex indicators that require more computation
                    self.ichimoku = btind.Ichimoku(self.data)
                    self.vortex = btind.Vortex(self.data)
                    self.mesa = btind.MesaAdaptiveMovingAverage(self.data.close)

                def next(self):
                    pass

            cerebro.addstrategy(ComplexIndicatorStrategy)
            return cerebro.run()

        result = benchmark(run_complex_indicators)
        assert result is not None


@pytest.mark.performance
class TestOptimizationPerformance:
    """Test performance of optimization runs"""

    def test_parameter_optimization_performance(self, benchmark):
        """Test performance of parameter optimization"""
        n_points = 1000
        dates = pd.date_range('2020-01-01', periods=n_points, freq='1D')
        prices = np.random.normal(100, 5, n_points).cumsum()

        data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(1000, 3000, n_points)
        })

        class OptimizableStrategy(bt.Strategy):
            params = (
                ('sma_period', 20),
                ('rsi_period', 14),
            )

            def __init__(self):
                self.sma = btind.SMA(self.data.close, period=self.params.sma_period)
                self.rsi = btind.RSI(self.data.close, period=self.params.rsi_period)

            def next(self):
                if self.rsi[0] < 30 and self.data.close[0] > self.sma[0]:
                    self.buy(size=10)
                elif self.rsi[0] > 70:
                    if self.position:
                        self.sell(size=self.position.size)

        def run_optimization():
            cerebro = bt.Cerebro()
            feed = bt.feeds.PolarsData(dataname=data)
            cerebro.adddata(feed)

            # Optimize parameters
            cerebro.optstrategy(
                OptimizableStrategy,
                sma_period=range(10, 31, 5),
                rsi_period=range(10, 21, 2)
            )

            cerebro.broker.set_cash(10000)
            cerebro.addanalyzer(btanalyzers.Returns, _name='returns')

            return cerebro.run()

        result = benchmark(run_optimization)
        assert result is not None
        assert len(result) > 1  # Should have multiple optimization runs


@pytest.mark.performance
class TestDataLoadingPerformance:
    """Test performance of data loading and processing"""

    def test_large_dataset_loading(self, benchmark):
        """Test loading and processing large datasets"""
        n_points = 50000  # Large dataset
        dates = pd.date_range('2015-01-01', periods=n_points, freq='5min')

        # Generate realistic high-frequency data
        prices = [100]
        for _ in range(n_points - 1):
            change = np.random.normal(0, 0.0005)  # Very small changes for 5min data
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)

        data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': [p * 1.0002 for p in prices],
            'low': [p * 0.9998 for p in prices],
            'close': prices,
            'volume': np.random.randint(10, 100, n_points)
        })

        def load_large_dataset():
            cerebro = bt.Cerebro()
            feed = bt.feeds.PolarsData(dataname=data)
            cerebro.adddata(feed)

            class MinimalStrategy(bt.Strategy):
                def next(self):
                    pass

            cerebro.addstrategy(MinimalStrategy)
            return cerebro.run()

        result = benchmark(load_large_dataset)
        assert result is not None

    def test_data_resampling_performance(self, benchmark):
        """Test performance of data resampling"""
        # Create high-frequency data
        n_points = 10000
        dates = pd.date_range('2020-01-01', periods=n_points, freq='1min')

        prices = [100]
        for _ in range(n_points - 1):
            change = np.random.normal(0, 0.0002)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)

        data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': [p * 1.0001 for p in prices],
            'low': [p * 0.9999 for p in prices],
            'close': prices,
            'volume': np.random.randint(1, 50, n_points)
        })

        def resample_data():
            cerebro = bt.Cerebro()
            feed = bt.feeds.PolarsData(dataname=data)
            cerebro.adddata(feed)

            # Add resampling filter
            cerebro.resampledata(feed, timeframe=bt.TimeFrame.Minutes, compression=5)  # 5min bars

            class ResampleStrategy(bt.Strategy):
                def next(self):
                    pass

            cerebro.addstrategy(ResampleStrategy)
            return cerebro.run()

        result = benchmark(resample_data)
        assert result is not None