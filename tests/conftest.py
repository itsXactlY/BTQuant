"""
Pytest configuration and shared fixtures for BTQuant testing suite
"""

import pytest
import sys
import os
from unittest.mock import Mock, MagicMock
import pandas as pd
import polars as pl
import numpy as np
from datetime import datetime, timedelta

# Import backtrader components from installed package
import backtrader as bt
from backtrader import feeds, indicators, analyzers, observers


@pytest.fixture(scope="session")
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)  # For reproducible tests

    # Generate 1000 data points
    n_points = 1000
    start_date = datetime(2023, 1, 1)

    # Generate realistic price data
    base_price = 50000.0
    prices = []
    current_price = base_price

    for i in range(n_points):
        # Random walk with some volatility
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        current_price *= (1 + change)
        prices.append(current_price)

    # Create OHLCV data
    data = []
    for i, close in enumerate(prices):
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else close
        volume = np.random.randint(1000, 10000)

        data.append({
            'datetime': start_date + timedelta(days=i),
            'open': open_price,
            'high': max(open_price, high),
            'low': min(open_price, low),
            'close': close,
            'volume': volume
        })

    return pl.DataFrame(data)


@pytest.fixture
def mock_strategy():
    """Create a mock strategy for testing"""
    class MockStrategy(bt.Strategy):
        params = (
            ('test_param', 10),
        )

        def __init__(self):
            self.orders = []
            self.trades = []
            self.buy_signals = 0
            self.sell_signals = 0

        def next(self):
            # Simple mock logic
            if self.data.close[0] > self.data.open[0]:
                self.buy_signals += 1
            else:
                self.sell_signals += 1

        def notify_order(self, order):
            self.orders.append(order)

        def notify_trade(self, trade):
            self.trades.append(trade)

    return MockStrategy


@pytest.fixture
def cerebro_instance():
    """Create a Cerebro instance for testing"""
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(100000)
    cerebro.broker.setcommission(commission=0.001)
    return cerebro


@pytest.fixture
def mock_data_feed(sample_ohlcv_data):
    """Create a mock data feed"""
    return bt.feeds.PolarsData(dataname=sample_ohlcv_data)


@pytest.fixture
def mock_broker():
    """Create a mock broker for testing"""
    broker = Mock()
    broker.get_cash.return_value = 100000
    broker.get_value.return_value = 100000
    broker.get_position.return_value = Mock(size=0, price=0)
    return broker


@pytest.fixture
def mock_exchange_api():
    """Mock exchange API responses"""
    api = Mock()
    api.fetch_ohlcv.return_value = [
        [1640995200000, 50000, 51000, 49000, 50500, 1000],
        [1641081600000, 50500, 52000, 50000, 51500, 1200],
        [1641168000000, 51500, 53000, 51000, 52500, 1100],
    ]
    api.fetch_ticker.return_value = {
        'symbol': 'BTC/USDT',
        'last': 52500,
        'bid': 52400,
        'ask': 52600,
        'volume': 1000
    }
    return api


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data"""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture
def mock_database_connection():
    """Mock database connection for testing"""
    conn = Mock()
    conn.cursor.return_value = Mock()
    conn.commit.return_value = None
    conn.close.return_value = None
    return conn


# Custom pytest marks
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "system: marks tests as system tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "security: marks tests as security tests")
    config.addinivalue_line("markers", "regression: marks tests as regression tests")


# Test utilities
class TestUtils:
    """Utility functions for tests"""

    @staticmethod
    def create_test_data(length=100, start_price=100.0, volatility=0.02):
        """Create synthetic OHLCV test data"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=length, freq='D')
        prices = [start_price]

        for _ in range(length - 1):
            change = np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)

        data = []
        for i, close in enumerate(prices):
            high = close * (1 + abs(np.random.normal(0, 0.01)))
            low = close * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] if i > 0 else close
            volume = np.random.randint(100, 1000)

            data.append({
                'datetime': dates[i],
                'open': open_price,
                'high': max(open_price, high),
                'low': min(open_price, low),
                'close': close,
                'volume': volume
            })

        return pl.DataFrame(data)

    @staticmethod
    def assert_strategy_signals(strategy, expected_buys=0, expected_sells=0):
        """Assert that strategy generated expected signals"""
        assert strategy.buy_signals == expected_buys, f"Expected {expected_buys} buy signals, got {strategy.buy_signals}"
        assert strategy.sell_signals == expected_sells, f"Expected {expected_sells} sell signals, got {strategy.sell_signals}"

    @staticmethod
    def run_backtest(strategy_class, data, **kwargs):
        """Helper to run a simple backtest"""
        cerebro = bt.Cerebro()
        cerebro.addstrategy(strategy_class, **kwargs)
        cerebro.adddata(data)
        cerebro.broker.set_cash(10000)

        results = cerebro.run()
        return results[0], cerebro


# Make TestUtils available as a fixture
@pytest.fixture
def test_utils():
    return TestUtils()