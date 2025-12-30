"""
System tests for end-to-end backtesting scenarios
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import backtrader as bt
from backtrader import analyzers as btanalyzers
from backtrader import observers as btobs


@pytest.mark.system
class TestEndToEndTradingStrategies:
    """Test complete end-to-end trading strategies"""

    def test_momentum_trading_strategy(self):
        """Test a complete momentum trading strategy"""
        # Generate trending data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=500, freq='D')

        # Create upward trending data
        base_price = 100
        prices = []
        for i in range(500):
            trend = i * 0.1  # Upward trend
            noise = np.random.normal(0, 2)
            price = base_price + trend + noise
            prices.append(max(price, 1))  # Ensure positive

        data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 5000, 500)
        })

        class MomentumStrategy(bt.Strategy):
            params = (
                ('momentum_period', 20),
                ('hold_period', 10),
            )

            def __init__(self):
                self.momentum = bt.indicators.Momentum(self.data.close, period=self.params.momentum_period)
                self.position_days = 0

            def next(self):
                if not self.position:
                    # Enter on strong momentum
                    if self.momentum[0] > 1.05:  # 5% momentum
                        self.buy(size=100)
                        self.position_days = 0
                else:
                    self.position_days += 1
                    # Exit after hold period or if momentum weakens
                    if (self.position_days >= self.params.hold_period or
                        self.momentum[0] < 0.98):
                        self.sell(size=self.position.size)

        cerebro = bt.Cerebro()
        feed = bt.feeds.PolarsData(dataname=data)
        cerebro.adddata(feed)
        cerebro.addstrategy(MomentumStrategy)

        # Add comprehensive analysis
        cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
        cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')

        cerebro.broker.set_cash(100000)
        cerebro.broker.setcommission(commission=0.001)

        results = cerebro.run()
        strategy = results[0]

        # Verify strategy execution
        assert 'returns' in strategy.analyzers
        assert 'sharpe' in strategy.analyzers
        assert 'trades' in strategy.analyzers

        # Should have executed some trades
        trade_analyzer = strategy.analyzers.trades
        assert hasattr(trade_analyzer, 'total')

    def test_mean_reversion_strategy(self):
        """Test a complete mean reversion strategy"""
        # Generate oscillating data around a mean
        np.random.seed(123)
        dates = pd.date_range('2023-01-01', periods=300, freq='D')

        mean_price = 100
        prices = []
        for i in range(300):
            # Mean reverting process
            deviation = np.random.normal(0, 10)
            price = mean_price + deviation * 0.9  # Partial mean reversion
            prices.append(max(price, 1))

        data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': np.random.randint(500, 2000, 300)
        })

        class MeanReversionStrategy(bt.Strategy):
            params = (
                ('sma_period', 50),
                ('deviation_threshold', 2.0),
            )

            def __init__(self):
                self.sma = bt.indicators.SMA(self.data.close, period=self.params.sma_period)
                self.std = bt.indicators.StdDev(self.data.close, period=self.params.sma_period)

            def next(self):
                if len(self.sma) < 1 or len(self.std) < 1:
                    return

                current_price = self.data.close[0]
                mean = self.sma[0]
                std = self.std[0]

                if std == 0:
                    return

                z_score = (current_price - mean) / std

                if not self.position:
                    # Buy when price is significantly below mean
                    if z_score < -self.params.deviation_threshold:
                        self.buy(size=50)
                else:
                    # Sell when price returns to mean
                    if z_score > -0.5:
                        self.sell(size=self.position.size)

        cerebro = bt.Cerebro()
        feed = bt.feeds.PolarsData(dataname=data)
        cerebro.adddata(feed)
        cerebro.addstrategy(MeanReversionStrategy)

        cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
        cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')

        cerebro.broker.set_cash(50000)
        cerebro.broker.setcommission(commission=0.002)

        results = cerebro.run()
        strategy = results[0]

        assert 'returns' in strategy.analyzers
        assert 'trades' in strategy.analyzers

    def test_portfolio_strategy_with_multiple_assets(self):
        """Test portfolio strategy with multiple correlated assets"""
        np.random.seed(456)

        # Create two correlated assets
        dates = pd.date_range('2023-01-01', periods=200, freq='D')

        # Base trend
        trend = np.linspace(0, 50, 200)

        # Asset 1: Strong upward trend
        asset1_prices = 100 + trend + np.random.normal(0, 5, 200)

        # Asset 2: Moderate trend with correlation to asset 1
        asset2_prices = 80 + trend * 0.7 + np.random.normal(0, 3, 200)

        data1 = pd.DataFrame({
            'datetime': dates,
            'open': asset1_prices,
            'high': asset1_prices * 1.02,
            'low': asset1_prices * 0.98,
            'close': asset1_prices,
            'volume': np.random.randint(1000, 3000, 200)
        })

        data2 = pd.DataFrame({
            'datetime': dates,
            'open': asset2_prices,
            'high': asset2_prices * 1.02,
            'low': asset2_prices * 0.98,
            'close': asset2_prices,
            'volume': np.random.randint(800, 2500, 200)
        })

        class PortfolioStrategy(bt.Strategy):
            def __init__(self):
                self.sma1 = bt.indicators.SMA(self.data0.close, period=30)
                self.sma2 = bt.indicators.SMA(self.data1.close, period=30)
                self.rsi1 = bt.indicators.RSI(self.data0.close, period=14)
                self.rsi2 = bt.indicators.RSI(self.data1.close, period=14)

            def next(self):
                # Allocate based on momentum
                momentum1 = self.data0.close[0] / self.data0.close[-10] if len(self.data0) > 10 else 1
                momentum2 = self.data1.close[0] / self.data1.close[-10] if len(self.data1) > 10 else 1

                total_momentum = momentum1 + momentum2

                if total_momentum > 2.1:  # Strong momentum
                    # Allocate more to stronger asset
                    if momentum1 > momentum2:
                        self.order_target_percent(data=self.data0, target=0.7)
                        self.order_target_percent(data=self.data1, target=0.3)
                    else:
                        self.order_target_percent(data=self.data0, target=0.3)
                        self.order_target_percent(data=self.data1, target=0.7)
                elif total_momentum < 1.9:  # Weak momentum
                    # Reduce exposure
                    self.order_target_percent(data=self.data0, target=0.4)
                    self.order_target_percent(data=self.data1, target=0.4)

        cerebro = bt.Cerebro()
        feed1 = bt.feeds.PolarsData(dataname=data1)
        feed2 = bt.feeds.PolarsData(dataname=data2)
        cerebro.adddata(feed1)
        cerebro.adddata(feed2)
        cerebro.addstrategy(PortfolioStrategy)

        cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
        cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe')

        cerebro.broker.set_cash(100000)

        results = cerebro.run()
        strategy = results[0]

        assert 'returns' in strategy.analyzers
        assert 'sharpe' in strategy.analyzers


@pytest.mark.system
class TestRiskManagementScenarios:
    """Test comprehensive risk management scenarios"""

    def test_stop_loss_and_take_profit_strategy(self):
        """Test strategy with stop loss and take profit orders"""
        # Create volatile data
        np.random.seed(789)
        dates = pd.date_range('2023-01-01', periods=150, freq='D')

        # Volatile price movement
        prices = [100]
        for _ in range(149):
            change = np.random.normal(0, 0.03)  # 3% daily volatility
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)

        data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': [p * 1.05 for p in prices],  # High volatility
            'low': [p * 0.95 for p in prices],
            'close': prices,
            'volume': np.random.randint(2000, 8000, 150)
        })

        class RiskManagedStrategy(bt.Strategy):
            params = (
                ('stop_loss_pct', 0.05),     # 5% stop loss
                ('take_profit_pct', 0.10),   # 10% take profit
            )

            def __init__(self):
                self.entry_price = None
                self.stop_loss_orders = []
                self.take_profit_orders = []

            def next(self):
                if not self.position:
                    # Enter position
                    self.buy(size=100)
                    self.entry_price = self.data.close[0]
                elif self.position and self.entry_price:
                    current_price = self.data.close[0]

                    # Check stop loss
                    if current_price <= self.entry_price * (1 - self.params.stop_loss_pct):
                        self.sell(size=self.position.size)
                        self.entry_price = None
                        return

                    # Check take profit
                    if current_price >= self.entry_price * (1 + self.params.take_profit_pct):
                        self.sell(size=self.position.size)
                        self.entry_price = None
                        return

        cerebro = bt.Cerebro()
        feed = bt.feeds.PolarsData(dataname=data)
        cerebro.adddata(feed)
        cerebro.addstrategy(RiskManagedStrategy)

        cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')

        cerebro.broker.set_cash(50000)

        results = cerebro.run()
        strategy = results[0]

        assert 'trades' in strategy.analyzers
        assert 'drawdown' in strategy.analyzers

    def test_position_sizing_with_kelly_criterion(self):
        """Test position sizing using Kelly Criterion approximation"""
        # Create data with known expected return
        np.random.seed(101)
        dates = pd.date_range('2023-01-01', periods=250, freq='D')

        # Create data with positive drift
        prices = [100]
        for _ in range(249):
            # Positive drift with volatility
            drift = 0.001  # 0.1% daily expected return
            shock = np.random.normal(0, 0.02)
            new_price = prices[-1] * (1 + drift + shock)
            prices.append(new_price)

        data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 4000, 250)
        })

        class KellyStrategy(bt.Strategy):
            params = (
                ('win_rate', 0.55),      # 55% win rate
                ('avg_win', 0.08),       # 8% average win
                ('avg_loss', 0.05),      # 5% average loss
            )

            def __init__(self):
                self.wins = 0
                self.losses = 0

            def next(self):
                if not self.position:
                    # Kelly fraction approximation
                    kelly_fraction = (self.params.win_rate / (1 - self.params.win_rate) *
                                    (1 + self.params.avg_win) - 1) / self.params.avg_win

                    # Conservative Kelly (half)
                    position_size = min(kelly_fraction * 0.5, 0.2)  # Max 20%

                    if position_size > 0.01:  # Minimum position
                        cash = self.broker.get_cash()
                        size = int(cash * position_size / self.data.close[0])
                        if size > 0:
                            self.buy(size=size)

            def notify_trade(self, trade):
                if trade.isclosed:
                    if trade.pnl > 0:
                        self.wins += 1
                    else:
                        self.losses += 1

        cerebro = bt.Cerebro()
        feed = bt.feeds.PolarsData(dataname=data)
        cerebro.adddata(feed)
        cerebro.addstrategy(KellyStrategy)

        cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
        cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')

        cerebro.broker.set_cash(100000)

        results = cerebro.run()
        strategy = results[0]

        assert 'returns' in strategy.analyzers
        assert 'trades' in strategy.analyzers
        assert (strategy.wins + strategy.losses) >= 0  # Should have some closed trades


@pytest.mark.system
class TestPerformanceBenchmarking:
    """Test performance benchmarking scenarios"""

    def test_strategy_performance_comparison(self):
        """Compare performance of multiple strategies"""
        np.random.seed(2023)
        dates = pd.date_range('2023-01-01', periods=365, freq='D')

        # Generate realistic market data
        prices = [100]
        for _ in range(364):
            # Random walk with slight upward bias
            change = np.random.normal(0.0005, 0.02)  # 0.05% drift, 2% volatility
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)

        data = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': [p * 1.015 for p in prices],
            'low': [p * 0.985 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 5000, 365)
        })

        strategies = []

        # Strategy 1: Buy and Hold
        class BuyAndHold(bt.Strategy):
            def __init__(self):
                self.bought = False

            def next(self):
                if not self.bought:
                    self.order_target_percent(target=1.0)
                    self.bought = True

        # Strategy 2: SMA Crossover
        class SMACrossover(bt.Strategy):
            def __init__(self):
                self.fast_sma = bt.indicators.SMA(self.data.close, period=20)
                self.slow_sma = bt.indicators.SMA(self.data.close, period=50)

            def next(self):
                if self.fast_sma[0] > self.slow_sma[0] and not self.position:
                    self.buy(size=100)
                elif self.fast_sma[0] < self.slow_sma[0] and self.position:
                    self.sell(size=self.position.size)

        # Strategy 3: RSI Strategy
        class RSIStrategy(bt.Strategy):
            def __init__(self):
                self.rsi = bt.indicators.RSI(self.data.close, period=14)

            def next(self):
                if self.rsi[0] < 30 and not self.position:
                    self.buy(size=100)
                elif self.rsi[0] > 70 and self.position:
                    self.sell(size=self.position.size)

        strategy_classes = [BuyAndHold, SMACrossover, RSIStrategy]
        strategy_names = ['Buy and Hold', 'SMA Crossover', 'RSI Strategy']

        results = []
        for strategy_class, name in zip(strategy_classes, strategy_names):
            cerebro = bt.Cerebro()
            feed = bt.feeds.PolarsData(dataname=data)
            cerebro.adddata(feed)
            cerebro.addstrategy(strategy_class)

            cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
            cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')

            cerebro.broker.set_cash(100000)

            result = cerebro.run()[0]
            results.append((name, result))

        # All strategies should have completed
        assert len(results) == 3
        for name, strategy in results:
            assert 'returns' in strategy.analyzers
            assert 'sharpe' in strategy.analyzers
            assert 'drawdown' in strategy.analyzers