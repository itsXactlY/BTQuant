"""
Unit tests for Backtrader data feeds
"""

import pytest
import pandas as pd
import polars as pl
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

import backtrader as bt
from backtrader import feeds as btfeeds


@pytest.mark.unit
class TestBasicDataFeeds:
    """Test basic data feed functionality"""

    def test_polars_data_feed_creation(self, sample_ohlcv_data):
        """Test PolarsData feed creation and basic functionality"""
        # Create PolarsData feed
        data = btfeeds.PolarsData(dataname=sample_ohlcv_data)

        # Check basic attributes
        assert hasattr(data, 'lines')
        assert hasattr(data, 'open')
        assert hasattr(data, 'high')
        assert hasattr(data, 'low')
        assert hasattr(data, 'close')
        assert hasattr(data, 'volume')

        # Test data loading
        cerebro = bt.Cerebro()
        cerebro.adddata(data)

        # Should be able to run without errors
        results = cerebro.run()
        assert len(results) == 1

    def test_polars_data_with_pandas_conversion(self):
        """Test PolarsData feed with pandas DataFrame conversion"""
        # Create pandas DataFrame
        pandas_data = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=100, freq='D'),
            'open': np.random.uniform(90, 110, 100),
            'high': np.random.uniform(105, 115, 100),
            'low': np.random.uniform(85, 95, 100),
            'close': np.random.uniform(95, 105, 100),
            'volume': np.random.randint(1000, 2000, 100)
        })

        # Convert to polars
        polars_data = pl.from_pandas(pandas_data)

        # Create feed
        data = btfeeds.PolarsData(dataname=polars_data)

        cerebro = bt.Cerebro()
        cerebro.adddata(data)

        results = cerebro.run()
        assert len(results) == 1

    def test_data_feed_iteration(self, sample_ohlcv_data):
        """Test data feed iteration and data access"""
        data = btfeeds.PolarsData(dataname=sample_ohlcv_data)

        cerebro = bt.Cerebro()
        cerebro.adddata(data)

        class DataTestStrategy(bt.Strategy):
            def __init__(self):
                self.data_points = 0
                self.close_prices = []

            def next(self):
                self.data_points += 1
                self.close_prices.append(self.data.close[0])

        cerebro.addstrategy(DataTestStrategy)
        results = cerebro.run()

        strategy = results[0]
        assert strategy.data_points > 0
        assert len(strategy.close_prices) > 0
        assert all(isinstance(price, (int, float)) for price in strategy.close_prices)


@pytest.mark.unit
class TestExchangeDataFeeds:
    """Test exchange-specific data feeds"""

    @patch('ccxt.binance')
    def test_ccxt_feed_creation(self, mock_ccxt):
        """Test CCXT feed creation with mocked exchange"""
        # Mock CCXT exchange
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.return_value = [
            [1640995200000, 50000, 51000, 49000, 50500, 1000],
            [1641081600000, 50500, 52000, 50000, 51500, 1200],
        ]
        mock_ccxt.return_value = mock_exchange

        # Create CCXT feed
        data = btfeeds.CCXT(
            exchange='binance',
            symbol='BTC/USDT',
            timeframe=bt.TimeFrame.Minutes,
            compression=1,
            fromdate=datetime(2022, 1, 1),
            todate=datetime(2022, 1, 2)
        )

        assert data is not None
        assert hasattr(data, 'exchange')
        assert data.exchange == 'binance'

    @patch('yfinance.download')
    def test_yahoo_feed_creation(self, mock_yfinance):
        """Test Yahoo Finance feed creation"""
        # Mock yfinance response
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [102, 103, 104],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2023-01-01', periods=3))
        mock_yfinance.return_value = mock_data

        # Create Yahoo feed
        data = btfeeds.YahooFinance(
            symbol='AAPL',
            fromdate=datetime(2023, 1, 1),
            todate=datetime(2023, 1, 3)
        )

        assert data is not None


@pytest.mark.unit
class TestDatabaseDataFeeds:
    """Test database-backed data feeds"""

    @patch('pyodbc.connect')
    def test_mssql_crypto_feed(self, mock_connect):
        """Test MSSQL crypto feed with mocked database"""
        # Mock database connection and cursor
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            (datetime(2023, 1, 1), 50000, 51000, 49000, 50500, 1000),
            (datetime(2023, 1, 2), 50500, 52000, 50000, 51500, 1200),
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Create MSSQL crypto feed
        data = btfeeds.MSSQLCrypto(
            symbol='BTC/USDT',
            fromdate=datetime(2023, 1, 1),
            todate=datetime(2023, 1, 2)
        )

        assert data is not None

    @patch('pyodbc.connect')
    def test_mssql_stocks_feed(self, mock_connect):
        """Test MSSQL stocks feed with mocked database"""
        # Mock database connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            (datetime(2023, 1, 1), 100, 105, 95, 102, 10000),
            (datetime(2023, 1, 2), 102, 108, 98, 105, 12000),
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Create MSSQL stocks feed
        data = btfeeds.MSSQLStocks(
            symbol='AAPL',
            fromdate=datetime(2023, 1, 1),
            todate=datetime(2023, 1, 2)
        )

        assert data is not None


@pytest.mark.unit
class TestHotSpineDataFeeds:
    """Test HotSpine data feeds"""

    @patch('backtrader.hotspine.reader.HotSpineReader')
    def test_hotspine_feed_creation(self, mock_reader_class):
        """Test HotSpine feed creation"""
        # Mock HotSpine reader
        mock_reader = Mock()
        mock_reader.is_healthy.return_value = True
        mock_reader.poll_trade.return_value = None
        mock_reader_class.return_value = mock_reader

        # Create HotSpine feed
        data = btfeeds.HotSpineData(
            symbol_id=123,
            shm_name="/test_hotspine"
        )

        assert data is not None
        assert hasattr(data, 'symbol_id')
        assert data.symbol_id == 123
        assert hasattr(data, 'p')
        assert hasattr(data.p, 'shm_name')

    @patch('backtrader.hotspine.reader.HotSpineReader')
    def test_hotspine_feed_with_trades(self, mock_reader_class):
        """Test HotSpine feed with mock trade data"""
        # Create mock trade
        mock_trade = Mock()
        mock_trade.ts_exchange = 1640995200000000  # microseconds
        mock_trade.ts_local = 1640995200000000
        mock_trade.price = 50000.0
        mock_trade.size = 1.0
        mock_trade.symbol_id = 123
        mock_trade.side = 0  # BUY

        # Mock reader
        mock_reader = Mock()
        mock_reader.is_healthy.return_value = True
        mock_reader.poll_trade.side_effect = [mock_trade, None]  # Return trade then None
        mock_reader_class.return_value = mock_reader

        # Create HotSpine feed
        data = btfeeds.HotSpineData(
            symbol_id=123,
            shm_name="/test_hotspine"
        )

        # Test in cerebro
        cerebro = bt.Cerebro()

        class TradeCaptureStrategy(bt.Strategy):
            def __init__(self):
                self.trades_received = []

            def next(self):
                # Capture trade data
                trade_data = {
                    'datetime': self.data.datetime[0],
                    'open': self.data.open[0],
                    'high': self.data.high[0],
                    'low': self.data.low[0],
                    'close': self.data.close[0],
                    'volume': self.data.volume[0]
                }
                self.trades_received.append(trade_data)

        cerebro.adddata(data)
        cerebro.addstrategy(TradeCaptureStrategy)

        results = cerebro.run()
        strategy = results[0]

        # Should have received trade data
        assert len(strategy.trades_received) > 0


@pytest.mark.unit
class TestDataFeedEdgeCases:
    """Test data feeds with edge cases"""

    def test_empty_data_feed(self):
        """Test behavior with empty data"""
        # Create empty polars DataFrame
        empty_data = pl.DataFrame({
            'datetime': [],
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': []
        })

        data = btfeeds.PolarsData(dataname=empty_data)

        cerebro = bt.Cerebro()
        cerebro.adddata(data)

        # Should handle empty data gracefully
        results = cerebro.run()
        assert len(results) == 1

    def test_data_feed_with_invalid_data(self):
        """Test data feed with invalid/missing data"""
        # Create data with NaN values
        invalid_data = pl.DataFrame({
            'datetime': [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            'open': [100, np.nan],
            'high': [105, 106],
            'low': [95, np.nan],
            'close': [np.nan, 103],
            'volume': [1000, 1100]
        })

        data = btfeeds.PolarsData(dataname=invalid_data)

        cerebro = bt.Cerebro()
        cerebro.adddata(data)

        # Should handle NaN values gracefully
        results = cerebro.run()
        assert len(results) == 1

    def test_data_feed_date_filtering(self, sample_ohlcv_data):
        """Test data feed with date filtering"""
        # Filter data to specific date range
        start_date = datetime(2023, 6, 1)
        end_date = datetime(2023, 12, 31)

        data = btfeeds.PolarsData(
            dataname=sample_ohlcv_data,
            fromdate=start_date,
            todate=end_date
        )

        cerebro = bt.Cerebro()
        cerebro.adddata(data)

        class DateCheckStrategy(bt.Strategy):
            def __init__(self):
                self.dates = []

            def next(self):
                self.dates.append(self.data.datetime.date())

        cerebro.addstrategy(DateCheckStrategy)
        results = cerebro.run()
        strategy = results[0]

        # Check that dates are within range
        if strategy.dates:
            min_date = min(strategy.dates)
            max_date = max(strategy.dates)
            assert min_date >= start_date.date()
            assert max_date <= end_date.date()

    @patch('ccxt.binance')
    def test_ccxt_feed_error_handling(self, mock_ccxt):
        """Test CCXT feed error handling"""
        # Mock exchange to raise exception
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.side_effect = Exception("API Error")
        mock_ccxt.return_value = mock_exchange

        # Should handle API errors gracefully
        with pytest.raises(Exception):
            data = btfeeds.CCXT(
                exchange='binance',
                symbol='BTC/USDT',
                timeframe=bt.TimeFrame.Minutes,
                compression=1
            )
            # Try to load data
            cerebro = bt.Cerebro()
            cerebro.adddata(data)
            cerebro.run()