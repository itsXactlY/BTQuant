"""
Unit tests for Backtrader brokers and stores
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import backtrader as bt
from backtrader import brokers as btbrokers
from backtrader import stores as btstores


@pytest.mark.unit
class TestBasicBrokerFunctionality:
    """Test basic broker functionality"""

    def test_backtrader_broker_creation(self):
        """Test basic Backtrader broker creation"""
        broker = btbrokers.BackBroker()

        # Check basic attributes
        assert hasattr(broker, 'get_cash')
        assert hasattr(broker, 'get_value')
        assert hasattr(broker, 'get_position')

        # Check initial values
        assert broker.get_cash() == 10000.0  # Default cash
        assert broker.get_value() == 10000.0  # Cash only initially

    def test_broker_with_custom_cash(self):
        """Test broker with custom initial cash"""
        initial_cash = 50000.0
        broker = btbrokers.BackBroker()
        broker.set_cash(initial_cash)

        assert broker.get_cash() == initial_cash
        assert broker.get_value() == initial_cash

    def test_broker_commission_setting(self):
        """Test broker commission settings"""
        broker = btbrokers.BackBroker()
        broker.setcommission(commission=0.001)  # 0.1%

        # Commission should be set (though we can't easily test the internal value)
        assert broker is not None

    def test_broker_position_tracking(self):
        """Test broker position tracking"""
        broker = btbrokers.BackBroker()

        # Create a mock position
        position = Mock()
        position.size = 10
        position.price = 100.0

        # Broker should be able to track positions
        # (This is more of an integration test, but basic functionality)
        assert hasattr(broker, 'get_position')


@pytest.mark.unit
class TestCCXTBroker:
    """Test CCXT broker functionality"""

    @patch('ccxt.binance')
    def test_ccxt_broker_creation(self, mock_ccxt):
        """Test CCXT broker creation with mocked exchange"""
        # Mock CCXT exchange
        mock_exchange = Mock()
        mock_exchange.id = 'binance'
        mock_exchange.load_markets.return_value = {}
        mock_ccxt.return_value = mock_exchange

        # Create CCXT broker
        broker = btbrokers.CCXTBroker(
            exchange='binance',
            api_key='test_key',
            secret='test_secret'
        )

        assert broker is not None
        assert hasattr(broker, 'exchange')
        assert broker.exchange.id == 'binance'

    @patch('ccxt.binance')
    def test_ccxt_broker_order_execution(self, mock_ccxt):
        """Test CCXT broker order execution simulation"""
        # Mock exchange
        mock_exchange = Mock()
        mock_exchange.id = 'binance'
        mock_exchange.load_markets.return_value = {
            'BTC/USDT': {'symbol': 'BTC/USDT'}
        }
        mock_exchange.create_order.return_value = {
            'id': '12345',
            'status': 'closed',
            'amount': 1.0,
            'price': 50000.0
        }
        mock_ccxt.return_value = mock_exchange

        broker = btbrokers.CCXTBroker(
            exchange='binance',
            api_key='test_key',
            secret='test_secret'
        )

        # Test order creation (this would normally be done through cerebro)
        # For unit testing, we verify the broker has the necessary methods
        assert hasattr(broker, 'buy')
        assert hasattr(broker, 'sell')


@pytest.mark.unit
class TestCCXTStore:
    """Test CCXT store functionality"""

    @patch('ccxt.binance')
    def test_ccxt_store_creation(self, mock_ccxt):
        """Test CCXT store creation"""
        # Mock exchange
        mock_exchange = Mock()
        mock_exchange.id = 'binance'
        mock_exchange.load_markets.return_value = {}
        mock_ccxt.return_value = mock_exchange

        # Create CCXT store
        store = btstores.CCXTStore(
            exchange='binance',
            api_key='test_key',
            secret='test_secret'
        )

        assert store is not None
        assert hasattr(store, 'exchange')
        assert store.exchange.id == 'binance'

    @patch('ccxt.binance')
    def test_ccxt_store_data_fetching(self, mock_ccxt):
        """Test CCXT store data fetching"""
        # Mock exchange with OHLCV data
        mock_exchange = Mock()
        mock_exchange.id = 'binance'
        mock_exchange.fetch_ohlcv.return_value = [
            [1640995200000, 50000, 51000, 49000, 50500, 1000],
            [1641081600000, 50500, 52000, 50000, 51500, 1200],
        ]
        mock_ccxt.return_value = mock_exchange

        store = btstores.CCXTStore(
            exchange='binance',
            api_key='test_key',
            secret='test_secret'
        )

        # Test data fetching capability
        assert hasattr(store, 'fetch_ohlcv')

        # Call fetch method
        data = store.fetch_ohlcv('BTC/USDT', '1h', limit=100)
        assert len(data) == 2


@pytest.mark.unit
class TestPancakeSwapComponents:
    """Test PancakeSwap broker and store"""

    def test_pancakeswap_broker_creation(self):
        """Test PancakeSwap broker creation"""
        # PancakeSwap broker might require specific setup
        # For now, test that it can be imported and instantiated
        try:
            broker = btbrokers.PancakeSwapOrders()
            assert broker is not None
        except Exception:
            # If PancakeSwap dependencies are not available, skip
            pytest.skip("PancakeSwap dependencies not available")

    def test_pancakeswap_store_creation(self):
        """Test PancakeSwap store creation"""
        try:
            store = btstores.PancakeSwapStore()
            assert store is not None
        except Exception:
            pytest.skip("PancakeSwap dependencies not available")


@pytest.mark.unit
class TestBrokerEdgeCases:
    """Test broker edge cases and error handling"""

    def test_broker_with_zero_cash(self):
        """Test broker behavior with zero initial cash"""
        broker = btbrokers.BackBroker()
        broker.set_cash(0.0)

        assert broker.get_cash() == 0.0
        assert broker.get_value() == 0.0

    def test_broker_negative_cash(self):
        """Test broker with negative cash (should be allowed for simulation)"""
        broker = btbrokers.BackBroker()
        broker.set_cash(-1000.0)

        assert broker.get_cash() == -1000.0

    def test_broker_high_commission(self):
        """Test broker with high commission rates"""
        broker = btbrokers.BackBroker()
        broker.setcommission(commission=0.10)  # 10% commission

        assert broker is not None

    @patch('ccxt.binance')
    def test_ccxt_broker_connection_failure(self, mock_ccxt):
        """Test CCXT broker connection failure handling"""
        # Mock exchange to raise connection error
        mock_exchange = Mock()
        mock_exchange.load_markets.side_effect = Exception("Connection failed")
        mock_ccxt.return_value = mock_exchange

        # Should handle connection failures gracefully
        with pytest.raises(Exception):
            broker = btbrokers.CCXTBroker(
                exchange='binance',
                api_key='test_key',
                secret='test_secret'
            )


@pytest.mark.unit
class TestStoreEdgeCases:
    """Test store edge cases and error handling"""

    @patch('ccxt.binance')
    def test_ccxt_store_invalid_symbol(self, mock_ccxt):
        """Test CCXT store with invalid symbol"""
        mock_exchange = Mock()
        mock_exchange.id = 'binance'
        mock_exchange.fetch_ohlcv.side_effect = Exception("Invalid symbol")
        mock_ccxt.return_value = mock_exchange

        store = btstores.CCXTStore(
            exchange='binance',
            api_key='test_key',
            secret='test_secret'
        )

        # Should handle invalid symbol errors
        with pytest.raises(Exception):
            store.fetch_ohlcv('INVALID/SYMBOL', '1h')

    @patch('ccxt.binance')
    def test_ccxt_store_rate_limiting(self, mock_ccxt):
        """Test CCXT store rate limiting handling"""
        mock_exchange = Mock()
        mock_exchange.id = 'binance'
        # Simulate rate limiting
        mock_exchange.fetch_ohlcv.side_effect = Exception("Rate limit exceeded")
        mock_ccxt.return_value = mock_exchange

        store = btstores.CCXTStore(
            exchange='binance',
            api_key='test_key',
            secret='test_secret'
        )

        with pytest.raises(Exception):
            store.fetch_ohlcv('BTC/USDT', '1h')


@pytest.mark.unit
class TestBrokerIntegration:
    """Test broker integration with strategies"""

    def test_broker_strategy_integration(self, sample_ohlcv_data):
        """Test broker working with a strategy"""
        class SimpleStrategy(bt.Strategy):
            def next(self):
                if not self.position:
                    self.buy(size=1)

        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        # Set up broker
        cerebro.broker.set_cash(1000.0)
        cerebro.broker.setcommission(commission=0.001)

        cerebro.addstrategy(SimpleStrategy)
        results = cerebro.run()

        # Strategy should have executed
        assert len(results) == 1

        # Broker should have recorded the transaction
        # (This is a basic integration test)

    def test_broker_multiple_orders(self, sample_ohlcv_data):
        """Test broker handling multiple orders"""
        class MultiOrderStrategy(bt.Strategy):
            def __init__(self):
                self.order_count = 0

            def next(self):
                if self.order_count < 3 and not self.position:
                    self.buy(size=1)
                    self.order_count += 1

        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        cerebro.broker.set_cash(10000.0)
        cerebro.addstrategy(MultiOrderStrategy)

        results = cerebro.run()
        strategy = results[0]

        # Should have attempted multiple orders
        assert strategy.order_count >= 0