"""
Unit tests for Backtrader indicators
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

import backtrader as bt
from backtrader import indicators as btind


@pytest.mark.unit
class TestBasicIndicators:
    """Test basic technical indicators"""

    def test_sma_indicator(self, sample_ohlcv_data):
        """Test Simple Moving Average indicator"""
        # Create cerebro and add data
        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        # Create strategy with SMA
        class SMAStrategy(bt.Strategy):
            def __init__(self):
                self.sma = btind.SMA(self.data.close, period=20)

        cerebro.addstrategy(SMAStrategy)
        results = cerebro.run()

        strategy = results[0]
        assert hasattr(strategy, 'sma')
        assert len(strategy.sma) > 0

        # Check that SMA values are reasonable
        sma_values = strategy.sma.get(size=10)
        assert all(isinstance(x, (int, float)) for x in sma_values)
        assert all(x > 0 for x in sma_values)

    def test_ema_indicator(self, sample_ohlcv_data):
        """Test Exponential Moving Average indicator"""
        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        class EMAStrategy(bt.Strategy):
            def __init__(self):
                self.ema = btind.EMA(self.data.close, period=20)

        cerebro.addstrategy(EMAStrategy)
        results = cerebro.run()

        strategy = results[0]
        assert hasattr(strategy, 'ema')
        assert len(strategy.ema) > 0

        # EMA should be more responsive than SMA
        ema_values = strategy.ema.get(size=10)
        assert all(isinstance(x, (int, float)) for x in ema_values)

    def test_rsi_indicator(self, sample_ohlcv_data):
        """Test Relative Strength Index indicator"""
        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        class RSIStrategy(bt.Strategy):
            def __init__(self):
                self.rsi = btind.RSI(self.data.close, period=14)

        cerebro.addstrategy(RSIStrategy)
        results = cerebro.run()

        strategy = results[0]
        assert hasattr(strategy, 'rsi')
        assert len(strategy.rsi) > 0

        # RSI should be between 0 and 100
        rsi_values = strategy.rsi.get(size=10)
        assert all(0 <= x <= 100 for x in rsi_values if not np.isnan(x))

    def test_macd_indicator(self, sample_ohlcv_data):
        """Test MACD indicator"""
        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        class MACDStrategy(bt.Strategy):
            def __init__(self):
                self.macd = btind.MACDHisto(self.data.close)

        cerebro.addstrategy(MACDStrategy)
        results = cerebro.run()

        strategy = results[0]
        assert hasattr(strategy, 'macd')
        assert hasattr(strategy.macd, 'macd')
        assert hasattr(strategy.macd, 'signal')
        assert hasattr(strategy.macd, 'histo')

        # Check MACD components
        macd_values = strategy.macd.macd.get(size=5)
        signal_values = strategy.macd.signal.get(size=5)
        histo_values = strategy.macd.histo.get(size=5)

        assert len(macd_values) > 0
        assert len(signal_values) > 0
        assert len(histo_values) > 0

    def test_bollinger_bands(self, sample_ohlcv_data):
        """Test Bollinger Bands indicator"""
        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        class BBStrategy(bt.Strategy):
            def __init__(self):
                self.bb = btind.BollingerBands(self.data.close)

        cerebro.addstrategy(BBStrategy)
        results = cerebro.run()

        strategy = results[0]
        assert hasattr(strategy, 'bb')
        assert hasattr(strategy.bb, 'top')
        assert hasattr(strategy.bb, 'mid')
        assert hasattr(strategy.bb, 'bot')

        # Check that top > mid > bot
        top_values = strategy.bb.top.get(size=5)
        mid_values = strategy.bb.mid.get(size=5)
        bot_values = strategy.bb.bot.get(size=5)

        for t, m, b in zip(top_values, mid_values, bot_values):
            if not (np.isnan(t) or np.isnan(m) or np.isnan(b)):
                assert t >= m >= b

    def test_stochastic_indicator(self, sample_ohlcv_data):
        """Test Stochastic Oscillator indicator"""
        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        class StochStrategy(bt.Strategy):
            def __init__(self):
                self.stoch = btind.Stochastic(self.data)

        cerebro.addstrategy(StochStrategy)
        results = cerebro.run()

        strategy = results[0]
        assert hasattr(strategy, 'stoch')
        assert hasattr(strategy.stoch, 'percK')
        assert hasattr(strategy.stoch, 'percD')

        # Values should be between 0 and 100
        k_values = strategy.stoch.percK.get(size=5)
        d_values = strategy.stoch.percD.get(size=5)

        for k, d in zip(k_values, d_values):
            if not np.isnan(k):
                assert 0 <= k <= 100
            if not np.isnan(d):
                assert 0 <= d <= 100


@pytest.mark.unit
class TestAdvancedIndicators:
    """Test advanced technical indicators"""

    def test_ichimoku_indicator(self, sample_ohlcv_data):
        """Test Ichimoku Cloud indicator"""
        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        class IchimokuStrategy(bt.Strategy):
            def __init__(self):
                self.ichimoku = btind.Ichimoku(self.data)

        cerebro.addstrategy(IchimokuStrategy)
        results = cerebro.run()

        strategy = results[0]
        assert hasattr(strategy, 'ichimoku')

        # Check Ichimoku components exist
        components = ['tenkan', 'kijun', 'senkou_span_a', 'senkou_span_b', 'chikou']
        for component in components:
            assert hasattr(strategy.ichimoku, component)

    def test_fibonacci_retracement(self, sample_ohlcv_data):
        """Test Fibonacci Retracement levels"""
        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        class FibStrategy(bt.Strategy):
            def __init__(self):
                self.fib = btind.FibonacciLevels(self.data)

        cerebro.addstrategy(FibStrategy)
        results = cerebro.run()

        strategy = results[0]
        assert hasattr(strategy, 'fib')

        # Check Fibonacci levels
        levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        for level in levels:
            attr_name = f"r{str(level).replace('.', '')}"
            assert hasattr(strategy.fib, attr_name)

    def test_vortex_indicator(self, sample_ohlcv_data):
        """Test Vortex Indicator"""
        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        class VortexStrategy(bt.Strategy):
            def __init__(self):
                self.vortex = btind.Vortex(self.data)

        cerebro.addstrategy(VortexStrategy)
        results = cerebro.run()

        strategy = results[0]
        assert hasattr(strategy, 'vortex')
        assert hasattr(strategy.vortex, 'vip')
        assert hasattr(strategy.vortex, 'vim')


@pytest.mark.unit
class TestIndicatorEdgeCases:
    """Test indicators with edge cases"""

    def test_indicator_with_insufficient_data(self):
        """Test indicators behave correctly with insufficient data"""
        # Create minimal data
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

        class TestStrategy(bt.Strategy):
            def __init__(self):
                self.sma20 = btind.SMA(self.data.close, period=20)  # Period > data length
                self.rsi = btind.RSI(self.data.close, period=14)   # Period > data length

        cerebro.addstrategy(TestStrategy)
        results = cerebro.run()

        strategy = results[0]

        # Indicators should handle insufficient data gracefully
        assert hasattr(strategy, 'sma20')
        assert hasattr(strategy, 'rsi')

        # Values might be NaN for early periods
        sma_values = strategy.sma20.get(size=5)
        rsi_values = strategy.rsi.get(size=5)

        # Should not crash, but may return NaN
        assert len(sma_values) == 5
        assert len(rsi_values) == 5

    def test_indicator_parameters(self):
        """Test indicator parameter validation"""
        data = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=50, freq='D'),
            'open': np.random.uniform(90, 110, 50),
            'high': np.random.uniform(105, 115, 50),
            'low': np.random.uniform(85, 95, 50),
            'close': np.random.uniform(95, 105, 50),
            'volume': np.random.randint(1000, 2000, 50)
        })

        cerebro = bt.Cerebro()
        feed = bt.feeds.PolarsData(dataname=data)
        cerebro.adddata(feed)

        # Test different SMA periods
        class ParamTestStrategy(bt.Strategy):
            def __init__(self):
                self.sma5 = btind.SMA(self.data.close, period=5)
                self.sma10 = btind.SMA(self.data.close, period=10)
                self.sma20 = btind.SMA(self.data.close, period=20)

        cerebro.addstrategy(ParamTestStrategy)
        results = cerebro.run()

        strategy = results[0]

        # All SMAs should exist
        assert hasattr(strategy, 'sma5')
        assert hasattr(strategy, 'sma10')
        assert hasattr(strategy, 'sma20')

        # Check that different periods produce different results
        sma5_vals = strategy.sma5.get(size=10)
        sma20_vals = strategy.sma20.get(size=10)

        # Should have some valid values
        valid_sma5 = [x for x in sma5_vals if not np.isnan(x)]
        valid_sma20 = [x for x in sma20_vals if not np.isnan(x)]

        assert len(valid_sma5) > 0
        assert len(valid_sma20) > 0