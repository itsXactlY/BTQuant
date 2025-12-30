"""
Unit tests for Backtrader analyzers
"""

import pytest
import numpy as np
import pandas as pd

import backtrader as bt
from backtrader import analyzers as btanalyzers


@pytest.mark.unit
class TestPerformanceAnalyzers:
    """Test performance analysis analyzers"""

    def test_returns_analyzer(self, sample_ohlcv_data, mock_strategy):
        """Test Returns analyzer"""
        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        # Add returns analyzer
        cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
        cerebro.addstrategy(mock_strategy)

        results = cerebro.run()
        strategy = results[0]

        # Check that analyzer was added
        assert 'returns' in strategy.analyzers
        returns_analyzer = strategy.analyzers.returns

        # Check analyzer has expected attributes
        assert hasattr(returns_analyzer, 'rets')

    def test_sharpe_ratio_analyzer(self, sample_ohlcv_data, mock_strategy):
        """Test Sharpe Ratio analyzer"""
        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        # Add Sharpe ratio analyzer
        cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
        cerebro.addstrategy(mock_strategy)

        results = cerebro.run()
        strategy = results[0]

        assert 'sharpe' in strategy.analyzers
        sharpe_analyzer = strategy.analyzers.sharpe

        # Sharpe ratio should be a number
        assert hasattr(sharpe_analyzer, 'sharpe')

    def test_drawdown_analyzer(self, sample_ohlcv_data, mock_strategy):
        """Test Drawdown analyzer"""
        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        # Add drawdown analyzer
        cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
        cerebro.addstrategy(mock_strategy)

        results = cerebro.run()
        strategy = results[0]

        assert 'drawdown' in strategy.analyzers
        dd_analyzer = strategy.analyzers.drawdown

        # Check drawdown attributes
        assert hasattr(dd_analyzer, 'max')
        assert hasattr(dd_analyzer, 'drawdown')

    def test_trade_analyzer(self, sample_ohlcv_data):
        """Test Trade Analyzer"""
        # Create a strategy that makes trades
        class TradingStrategy(bt.Strategy):
            def __init__(self):
                self.order_count = 0

            def next(self):
                if self.order_count < 5:  # Limit trades for testing
                    if self.data.close[0] > self.data.open[0]:
                        self.buy(size=10)
                        self.order_count += 1
                    elif self.data.close[0] < self.data.open[0]:
                        self.sell(size=10)
                        self.order_count += 1

        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        # Add trade analyzer
        cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')
        cerebro.addstrategy(TradingStrategy)

        results = cerebro.run()
        strategy = results[0]

        assert 'trades' in strategy.analyzers
        trade_analyzer = strategy.analyzers.trades

        # Check trade analyzer structure
        assert hasattr(trade_analyzer, 'trades')

    def test_sq_analyzer(self, sample_ohlcv_data, mock_strategy):
        """Test SQN (System Quality Number) analyzer"""
        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        # Add SQN analyzer
        cerebro.addanalyzer(btanalyzers.SQN, _name='sqn')
        cerebro.addstrategy(mock_strategy)

        results = cerebro.run()
        strategy = results[0]

        assert 'sqn' in strategy.analyzers
        sqn_analyzer = strategy.analyzers.sqn

        # SQN should have sqn attribute
        assert hasattr(sqn_analyzer, 'sqn')


@pytest.mark.unit
class TestRiskAnalyzers:
    """Test risk analysis analyzers"""

    def test_calmar_ratio_analyzer(self, sample_ohlcv_data, mock_strategy):
        """Test Calmar Ratio analyzer"""
        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        # Add Calmar ratio analyzer
        cerebro.addanalyzer(btanalyzers.Calmar, _name='calmar')
        cerebro.addstrategy(mock_strategy)

        results = cerebro.run()
        strategy = results[0]

        assert 'calmar' in strategy.analyzers
        calmar_analyzer = strategy.analyzers.calmar

        assert hasattr(calmar_analyzer, 'calmar')

    def test_leverage_analyzer(self, sample_ohlcv_data, mock_strategy):
        """Test Leverage analyzer"""
        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        # Add leverage analyzer
        cerebro.addanalyzer(btanalyzers.Leverage, _name='leverage')
        cerebro.addstrategy(mock_strategy)

        results = cerebro.run()
        strategy = results[0]

        assert 'leverage' in strategy.analyzers
        leverage_analyzer = strategy.analyzers.leverage

        assert hasattr(leverage_analyzer, 'leverage')


@pytest.mark.unit
class TestReportingAnalyzers:
    """Test reporting and data export analyzers"""

    def test_positions_analyzer(self, sample_ohlcv_data, mock_strategy):
        """Test Positions analyzer"""
        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        # Add positions analyzer
        cerebro.addanalyzer(btanalyzers.Positions, _name='positions')
        cerebro.addstrategy(mock_strategy)

        results = cerebro.run()
        strategy = results[0]

        assert 'positions' in strategy.analyzers
        pos_analyzer = strategy.analyzers.positions

        assert hasattr(pos_analyzer, 'positions')

    def test_transactions_analyzer(self, sample_ohlcv_data, mock_strategy):
        """Test Transactions analyzer"""
        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        # Add transactions analyzer
        cerebro.addanalyzer(btanalyzers.Transactions, _name='transactions')
        cerebro.addstrategy(mock_strategy)

        results = cerebro.run()
        strategy = results[0]

        assert 'transactions' in strategy.analyzers
        tx_analyzer = strategy.analyzers.transactions

        assert hasattr(tx_analyzer, 'transactions')

    def test_time_return_analyzer(self, sample_ohlcv_data, mock_strategy):
        """Test Time Return analyzer"""
        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        # Add time return analyzer
        cerebro.addanalyzer(btanalyzers.TimeReturn, _name='timereturn', timeframe=bt.TimeFrame.Days)
        cerebro.addstrategy(mock_strategy)

        results = cerebro.run()
        strategy = results[0]

        assert 'timereturn' in strategy.analyzers
        tr_analyzer = strategy.analyzers.timereturn

        assert hasattr(tr_analyzer, 'rets')


@pytest.mark.unit
class TestAnalyzerEdgeCases:
    """Test analyzers with edge cases"""

    def test_analyzer_with_no_trades(self, sample_ohlcv_data):
        """Test analyzers when no trades are made"""
        class NoTradeStrategy(bt.Strategy):
            def next(self):
                pass  # No trading

        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        # Add multiple analyzers
        analyzers = [
            (btanalyzers.TradeAnalyzer, 'trades'),
            (btanalyzers.SharpeRatio, 'sharpe'),
            (btanalyzers.DrawDown, 'drawdown'),
            (btanalyzers.SQN, 'sqn')
        ]

        for analyzer_class, name in analyzers:
            cerebro.addanalyzer(analyzer_class, _name=name)

        cerebro.addstrategy(NoTradeStrategy)
        results = cerebro.run()
        strategy = results[0]

        # All analyzers should still exist and not crash
        for _, name in analyzers:
            assert name in strategy.analyzers

    def test_analyzer_with_minimal_data(self):
        """Test analyzers with minimal data"""
        # Create minimal dataset
        data = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=10, freq='D'),
            'open': [100] * 10,
            'high': [105] * 10,
            'low': [95] * 10,
            'close': [102] * 10,
            'volume': [1000] * 10
        })

        cerebro = bt.Cerebro()
        feed = bt.feeds.PolarsData(dataname=data)
        cerebro.adddata(feed)

        # Add analyzers
        cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
        cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe')

        cerebro.addstrategy(bt.Strategy)  # Empty strategy
        results = cerebro.run()
        strategy = results[0]

        # Should not crash with minimal data
        assert 'returns' in strategy.analyzers
        assert 'sharpe' in strategy.analyzers

    def test_multiple_analyzers_same_type(self, sample_ohlcv_data, mock_strategy):
        """Test multiple analyzers of the same type with different parameters"""
        cerebro = bt.Cerebro()
        data = bt.feeds.PolarsData(dataname=sample_ohlcv_data)
        cerebro.adddata(data)

        # Add multiple Sharpe ratio analyzers with different risk-free rates
        cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe_0', riskfreerate=0.0)
        cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe_2', riskfreerate=0.02)
        cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe_5', riskfreerate=0.05)

        cerebro.addstrategy(mock_strategy)
        results = cerebro.run()
        strategy = results[0]

        # All analyzers should exist
        assert 'sharpe_0' in strategy.analyzers
        assert 'sharpe_2' in strategy.analyzers
        assert 'sharpe_5' in strategy.analyzers

        # Different risk-free rates should potentially give different results
        sharpe_0 = strategy.analyzers.sharpe_0
        sharpe_2 = strategy.analyzers.sharpe_2
        sharpe_5 = strategy.analyzers.sharpe_5

        assert hasattr(sharpe_0, 'sharpe')
        assert hasattr(sharpe_2, 'sharpe')
        assert hasattr(sharpe_5, 'sharpe')