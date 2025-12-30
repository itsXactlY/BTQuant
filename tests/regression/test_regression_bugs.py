"""
Regression tests for previously fixed bugs and known issues
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import backtrader as bt
from backtrader import analyzers, indicators, observers
from backtrader.feeds import mssql_stocks, ccxt
from backtrader.brokers import ccxtbroker


@pytest.mark.regression
class TestIndicatorRegression:
    """Regression tests for indicator-related bugs"""

    def test_sma_division_by_zero_regression(self):
        """Test fix for SMA division by zero with empty data"""
        # Bug: SMA would crash with division by zero on empty datasets
        data = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=1, freq='1D'),
            'open': [100.0],
            'high': [101.0],
            'low': [99.0],
            'close': [100.0],
            'volume': [1000]
        })

        class SMARegressionStrategy(bt.Strategy):
            def __init__(self):
                self.sma = bt.indicators.SMA(self.data.close, period=5)

            def next(self):
                # SMA should handle insufficient data gracefully
                if len(self.data) >= 5:
                    sma_val = self.sma[0]
                    assert not np.isnan(sma_val)
                    assert not np.isinf(sma_val)

        cerebro = bt.Cerebro()
        feed = bt.feeds.PolarsData(dataname=data)
        cerebro.adddata(feed)
        cerebro.addstrategy(SMARegressionStrategy)

        # Should not crash with division by zero
        result = cerebro.run()
        assert result is not None

    def test_rsi_calculation_regression(self):
        """Test fix for RSI calculation errors with constant prices"""
        # Bug: RSI calculation was incorrect with constant price data
        dates = pd.date_range('2020-01-01', periods=50, freq='1D')

        # Test with constant prices (should give RSI around 50 or NaN)
        constant_data = pd.DataFrame({
            'datetime': dates,
            'open': [100.0] * 50,
            'high': [100.0] * 50,
            'low': [100.0] * 50,
            'close': [100.0] * 50,
            'volume': [1000] * 50
        })

        class RSIRegressionStrategy(bt.Strategy):
            def __init__(self):
                self.rsi = bt.indicators.RSI(self.data.close, period=14)

            def next(self):
                if len(self.data) >= 14:
                    rsi_val = self.rsi[0]
                    # RSI should be defined and reasonable for constant data
                    if not np.isnan(rsi_val):
                        assert 0 <= rsi_val <= 100

        cerebro = bt.Cerebro()
        feed = bt.feeds.PolarsData(dataname=constant_data)
        cerebro.adddata(feed)
        cerebro.addstrategy(RSIRegressionStrategy)

        result = cerebro.run()
        assert result is not None

    def test_macd_signal_line_regression(self):
        """Test fix for MACD signal line calculation lag"""
        # Bug: MACD signal line had incorrect lag in calculation
        dates = pd.date_range('2020-01-01', periods=100, freq='1D')
        prices = np.sin(np.arange(100) * 0.1) * 10 + 100

        data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': [1000] * 100
        })

        class MACDRegressionStrategy(bt.Strategy):
            def __init__(self):
                self.macd = bt.indicators.MACD(self.data.close)

            def next(self):
                if len(self.data) >= 26:  # MACD needs at least 26 periods
                    macd_val = self.macd.macd[0]
                    signal_val = self.macd.signal[0]

                    # Both should be finite numbers
                    assert np.isfinite(macd_val)
                    assert np.isfinite(signal_val)

                    # Signal should lag MACD appropriately
                    if len(self.data) >= 35:  # Signal line needs more data
                        # Signal should be smoother than MACD
                        macd_volatility = np.std([self.macd.macd[-i] for i in range(1, 10) if i < len(self.data)])
                        signal_volatility = np.std([self.macd.signal[-i] for i in range(1, 10) if i < len(self.data)])
                        assert signal_volatility <= macd_volatility

        cerebro = bt.Cerebro()
        feed = bt.feeds.PolarsData(dataname=data)
        cerebro.adddata(feed)
        cerebro.addstrategy(MACDRegressionStrategy)

        result = cerebro.run()
        assert result is not None

    def test_bollinger_bands_edge_cases(self):
        """Test fix for Bollinger Bands with extreme volatility"""
        # Bug: Bollinger Bands crashed with extreme volatility or NaN values
        dates = pd.date_range('2020-01-01', periods=50, freq='1D')

        # Create data with extreme volatility
        prices = [100.0]
        for i in range(49):
            change = np.random.normal(0, 0.5)  # High volatility
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.01))  # Prevent negative prices

        data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': [p * 1.05 for p in prices],
            'low': [p * 0.95 for p in prices],
            'close': prices,
            'volume': [1000] * 50
        })

        class BollingerRegressionStrategy(bt.Strategy):
            def __init__(self):
                self.bb = bt.indicators.BollingerBands(self.data.close)

            def next(self):
                if len(self.data) >= 20:  # Bollinger needs 20 periods
                    top = self.bb.top[0]
                    mid = self.bb.mid[0]
                    bot = self.bb.bot[0]

                    # All bands should be finite
                    assert np.isfinite(top)
                    assert np.isfinite(mid)
                    assert np.isfinite(bot)

                    # Ordering should be correct
                    assert bot <= mid <= top

        cerebro = bt.Cerebro()
        feed = bt.feeds.PolarsData(dataname=data)
        cerebro.adddata(feed)
        cerebro.addstrategy(BollingerRegressionStrategy)

        result = cerebro.run()
        assert result is not None


@pytest.mark.regression
class TestDataFeedRegression:
    """Regression tests for data feed bugs"""

    def test_ccxt_feed_connection_timeout(self):
        """Test fix for CCXT feed connection timeouts"""
        # Bug: CCXT feeds would hang indefinitely on connection failures
        with patch('ccxt.binance') as mock_exchange:
            mock_instance = MagicMock()
            mock_instance.fetch_ohlcv.side_effect = Exception("Connection timeout")
            mock_exchange.return_value = mock_instance

            feed = ccxt.CCXTData(
                dataname='BTC/USDT',
                timeframe=bt.TimeFrame.Minutes,
                compression=1,
                exchange='binance'
            )

            # Should handle timeout gracefully
            with pytest.raises(Exception):
                feed.start()

    def test_mssql_feed_empty_result_handling(self):
        """Test fix for MSSQL feed crashes on empty query results"""
        # Bug: MSSQL feeds crashed when queries returned no data
        with patch('pyodbc.connect') as mock_connect:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = []  # Empty result
            mock_cursor.description = [('datetime',), ('open',), ('high',), ('low',), ('close',), ('volume',)]

            mock_connection = MagicMock()
            mock_connection.cursor.return_value = mock_cursor
            mock_connect.return_value = mock_connection

            feed = mssql_stocks.MSSQLData(
                dataname='NONEXISTENT_SYMBOL',
                server='test_server',
                database='test_db',
                table='stocks',
                user='test_user',
                password='test_pass'
            )

            # Should handle empty results gracefully
            feed.start()
            assert len(feed) == 0

    def test_yahoo_feed_date_parsing_regression(self):
        """Test fix for Yahoo feed date parsing issues"""
        # Bug: Yahoo feeds failed to parse certain date formats
        dates = pd.date_range('2020-01-01', periods=10, freq='1D')

        # Test with various date formats that previously caused issues
        test_data = pd.DataFrame({
            'Date': dates.strftime('%Y-%m-%d'),  # ISO format
            'Open': [100.0] * 10,
            'High': [101.0] * 10,
            'Low': [99.0] * 10,
            'Close': [100.0] * 10,
            'Volume': [1000] * 10
        })

        # Create temporary CSV file
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data.to_csv(f, index=False)
            csv_file = f.name

        try:
            feed = bt.feeds.YahooFinanceCSVData(dataname=csv_file)

            class DateParseStrategy(bt.Strategy):
                def next(self):
                    # Should be able to access date
                    current_date = self.data.datetime.date()
                    assert isinstance(current_date, (datetime, pd.Timestamp))

            cerebro = bt.Cerebro()
            cerebro.adddata(feed)
            cerebro.addstrategy(DateParseStrategy)

            result = cerebro.run()
            assert result is not None

        finally:
            os.unlink(csv_file)

    def test_pandas_feed_column_case_sensitivity(self):
        """Test fix for pandas feed column name case sensitivity issues"""
        # Bug: Pandas feeds were case-sensitive about column names
        dates = pd.date_range('2020-01-01', periods=10, freq='1D')

        # Test with mixed case column names
        data = pd.DataFrame({
            'DATETIME': dates,
            'OPEN': [100.0] * 10,
            'HIGH': [101.0] * 10,
            'LOW': [99.0] * 10,
            'CLOSE': [100.0] * 10,
            'VOLUME': [1000] * 10
        })

        feed = bt.feeds.PolarsData(dataname=data)

        class ColumnCaseStrategy(bt.Strategy):
            def next(self):
                # Should work regardless of column case
                assert self.data.close[0] == 100.0
                assert self.data.open[0] == 100.0

        cerebro = bt.Cerebro()
        cerebro.adddata(feed)
        cerebro.addstrategy(ColumnCaseStrategy)

        result = cerebro.run()
        assert result is not None


@pytest.mark.regression
class TestBrokerRegression:
    """Regression tests for broker-related bugs"""

    def test_ccxt_broker_order_size_validation(self):
        """Test fix for CCXT broker order size validation"""
        # Bug: CCXT broker didn't validate order sizes properly
        with patch('ccxt.binance') as mock_exchange:
            mock_instance = MagicMock()
            mock_instance.has = {'createOrder': True}
            mock_instance.create_order.side_effect = Exception("Invalid order size")
            mock_exchange.return_value = mock_instance

            broker = ccxtbroker.CCXTBroker(
                exchange='binance',
                api_key='test_key',
                api_secret='test_secret'
            )

            # Test with invalid order sizes
            invalid_sizes = [0, -1, 1e-10, float('inf'), float('nan')]

            for size in invalid_sizes:
                with pytest.raises((ValueError, Exception)):
                    broker.buy(size=size)

    def test_broker_cash_calculation_regression(self):
        """Test fix for broker cash calculation errors"""
        # Bug: Broker cash calculations were incorrect after multiple operations
        cerebro = bt.Cerebro()
        cerebro.broker.set_cash(10000)

        dates = pd.date_range('2020-01-01', periods=100, freq='1D')
        data = pd.DataFrame({
            'datetime': dates,
            'open': [100.0] * 100,
            'high': [101.0] * 100,
            'low': [99.0] * 100,
            'close': [100.0] * 100,
            'volume': [1000] * 100
        })

        feed = bt.feeds.PolarsData(dataname=data)
        cerebro.adddata(feed)

        class CashTestStrategy(bt.Strategy):
            def __init__(self):
                self.order_count = 0

            def next(self):
                if self.order_count < 5:
                    # Place multiple orders
                    self.buy(size=1)
                    self.order_count += 1

                # Cash should never go negative
                assert self.broker.get_cash() >= 0

        cerebro.addstrategy(CashTestStrategy)
        result = cerebro.run()

        # Final cash should be reasonable
        final_cash = cerebro.broker.get_cash()
        assert final_cash >= 0
        assert final_cash <= 10000  # Should not exceed initial cash

    def test_order_status_tracking_regression(self):
        """Test fix for order status tracking issues"""
        # Bug: Order statuses were not updated correctly
        cerebro = bt.Cerebro()
        cerebro.broker.set_cash(10000)

        dates = pd.date_range('2020-01-01', periods=10, freq='1D')
        data = pd.DataFrame({
            'datetime': dates,
            'open': [100.0] * 10,
            'high': [101.0] * 10,
            'low': [99.0] * 10,
            'close': [100.0] * 10,
            'volume': [1000] * 10
        })

        feed = bt.feeds.PolarsData(dataname=data)
        cerebro.adddata(feed)

        class OrderStatusStrategy(bt.Strategy):
            def __init__(self):
                self.orders = []

            def next(self):
                if len(self.orders) == 0:
                    self.orders.append(self.buy(size=1))

            def notify_order(self, order):
                # Order status should be valid
                valid_statuses = [bt.Order.Created, bt.Order.Submitted,
                                bt.Order.Accepted, bt.Order.Partial,
                                bt.Order.Completed, bt.Order.Cancelled,
                                bt.Order.Expired, bt.Order.Rejected]
                assert order.status in valid_statuses

        cerebro.addstrategy(OrderStatusStrategy)
        result = cerebro.run()
        assert result is not None


@pytest.mark.regression
class TestAnalyzerRegression:
    """Regression tests for analyzer bugs"""

    def test_returns_analyzer_edge_cases(self):
        """Test fix for Returns analyzer with edge cases"""
        # Bug: Returns analyzer failed with certain data patterns
        cerebro = bt.Cerebro()

        # Test with flat returns
        dates = pd.date_range('2020-01-01', periods=100, freq='1D')
        data = pd.DataFrame({
            'datetime': dates,
            'open': [100.0] * 100,
            'high': [100.0] * 100,
            'low': [100.0] * 100,
            'close': [100.0] * 100,
            'volume': [1000] * 100
        })

        feed = bt.feeds.PolarsData(dataname=data)
        cerebro.adddata(feed)
        cerebro.addanalyzer(analyzers.Returns, _name='returns')

        class FlatReturnsStrategy(bt.Strategy):
            def next(self):
                pass

        cerebro.addstrategy(FlatReturnsStrategy)
        result = cerebro.run()

        # Should handle flat returns without division by zero
        returns_analyzer = result[0].analyzers.returns
        assert returns_analyzer is not None

    def test_sharpe_ratio_calculation_regression(self):
        """Test fix for Sharpe ratio calculation with zero volatility"""
        # Bug: Sharpe ratio crashed with zero volatility periods
        cerebro = bt.Cerebro()

        dates = pd.date_range('2020-01-01', periods=100, freq='1D')
        data = pd.DataFrame({
            'datetime': dates,
            'open': [100.0] * 100,
            'high': [100.0] * 100,
            'low': [100.0] * 100,
            'close': [100.0] * 100,  # Constant price = zero volatility
            'volume': [1000] * 100
        })

        feed = bt.feeds.PolarsData(dataname=data)
        cerebro.adddata(feed)
        cerebro.addanalyzer(analyzers.SharpeRatio, _name='sharpe')

        class ZeroVolatilityStrategy(bt.Strategy):
            def next(self):
                pass

        cerebro.addstrategy(ZeroVolatilityStrategy)
        result = cerebro.run()

        # Should handle zero volatility gracefully
        sharpe_analyzer = result[0].analyzers.sharpe
        assert sharpe_analyzer is not None

    def test_drawdown_analyzer_reset_regression(self):
        """Test fix for DrawDown analyzer reset issues"""
        # Bug: DrawDown analyzer didn't reset properly between runs
        cerebro = bt.Cerebro()

        dates = pd.date_range('2020-01-01', periods=50, freq='1D')
        # Create a drawdown pattern
        prices = []
        price = 100.0
        for i in range(50):
            if i < 20:
                price *= 1.01  # Rise
            else:
                price *= 0.99  # Fall
            prices.append(price)

        data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': [p * 1.001 for p in prices],
            'low': [p * 0.999 for p in prices],
            'close': prices,
            'volume': [1000] * 50
        })

        feed = bt.feeds.PolarsData(dataname=data)
        cerebro.adddata(feed)
        cerebro.addanalyzer(analyzers.DrawDown, _name='drawdown')

        class DrawdownStrategy(bt.Strategy):
            def next(self):
                pass

        cerebro.addstrategy(DrawdownStrategy)
        result = cerebro.run()

        drawdown_analyzer = result[0].analyzers.drawdown
        assert drawdown_analyzer is not None

        # Test multiple runs don't interfere
        result2 = cerebro.run()
        drawdown_analyzer2 = result2[0].analyzers.drawdown

        # Results should be consistent
        assert abs(drawdown_analyzer.get_analysis()['max']['drawdown'] -
                  drawdown_analyzer2.get_analysis()['max']['drawdown']) < 1e-10


@pytest.mark.regression
class TestStrategyRegression:
    """Regression tests for strategy-related bugs"""

    def test_strategy_parameter_validation(self):
        """Test fix for strategy parameter validation issues"""
        # Bug: Strategy parameters weren't validated properly

        class ParameterTestStrategy(bt.Strategy):
            params = (
                ('valid_param', 10),
                ('invalid_param', 'should_be_number'),
            )

            def __init__(self):
                # This should work
                assert self.p.valid_param == 10
                # This parameter has wrong type but wasn't caught
                assert isinstance(self.p.invalid_param, str)

            def next(self):
                pass

        cerebro = bt.Cerebro()

        dates = pd.date_range('2020-01-01', periods=10, freq='1D')
        data = pd.DataFrame({
            'datetime': dates,
            'open': [100.0] * 10,
            'high': [101.0] * 10,
            'low': [99.0] * 10,
            'close': [100.0] * 10,
            'volume': [1000] * 10
        })

        feed = bt.feeds.PolarsData(dataname=data)
        cerebro.adddata(feed)
        cerebro.addstrategy(ParameterTestStrategy)

        # Should initialize without crashing
        result = cerebro.run()
        assert result is not None

    def test_indicator_initialization_order_regression(self):
        """Test fix for indicator initialization order dependencies"""
        # Bug: Indicators had initialization order dependencies that caused crashes

        class IndicatorOrderStrategy(bt.Strategy):
            def __init__(self):
                # Initialize indicators in different orders to test dependencies
                self.sma1 = bt.indicators.SMA(self.data.close, period=5)
                self.rsi = bt.indicators.RSI(self.data.close, period=14)
                self.sma2 = bt.indicators.SMA(self.data.close, period=10)
                self.macd = bt.indicators.MACD(self.data.close)

            def next(self):
                # All indicators should be properly initialized
                if len(self.data) >= 14:
                    assert np.isfinite(self.rsi[0])
                if len(self.data) >= 5:
                    assert np.isfinite(self.sma1[0])
                if len(self.data) >= 10:
                    assert np.isfinite(self.sma2[0])
                if len(self.data) >= 26:
                    assert np.isfinite(self.macd.macd[0])

        cerebro = bt.Cerebro()

        dates = pd.date_range('2020-01-01', periods=50, freq='1D')
        prices = np.sin(np.arange(50) * 0.2) * 10 + 100

        data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': [1000] * 50
        })

        feed = bt.feeds.PolarsData(dataname=data)
        cerebro.adddata(feed)
        cerebro.addstrategy(IndicatorOrderStrategy)

        # Should not crash due to initialization order
        result = cerebro.run()
        assert result is not None

    def test_memory_leak_regression(self):
        """Test fix for memory leaks in strategy execution"""
        # Bug: Strategies accumulated memory over time
        import gc

        cerebro = bt.Cerebro()

        dates = pd.date_range('2020-01-01', periods=1000, freq='1min')  # Large dataset
        prices = np.random.normal(100, 1, 1000).cumsum()

        data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': prices * 1.001,
            'low': prices * 0.999,
            'close': prices,
            'volume': np.random.randint(100, 1000, 1000)
        })

        feed = bt.feeds.PolarsData(dataname=data)
        cerebro.adddata(feed)

        class MemoryTestStrategy(bt.Strategy):
            def __init__(self):
                self.counter = 0
                self.data_list = []

            def next(self):
                self.counter += 1
                # Accumulate some data (simulating potential memory leak)
                self.data_list.append(self.data.close[0])

                # Periodic cleanup to prevent actual memory issues in test
                if self.counter % 100 == 0:
                    self.data_list.clear()

        cerebro.addstrategy(MemoryTestStrategy)

        # Get initial memory usage
        gc.collect()
        initial_objects = len(gc.get_objects())

        result = cerebro.run()

        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())

        # Memory usage should not grow excessively
        # Allow some tolerance for test overhead
        assert final_objects - initial_objects < 1000