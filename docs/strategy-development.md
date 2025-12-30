# BTQuant Strategy Development Guide

## Overview

This guide provides comprehensive information on developing trading strategies with BTQuant. You'll learn how to create custom strategies, implement indicators, manage risk, and optimize performance.

## Strategy Architecture

### Strategy Inheritance Options

BTQuant offers two main approaches for strategy development:

#### 1. Direct Backtrader Strategy
```python
import backtrader as bt

class MyStrategy(bt.Strategy):
    params = (
        ('period', 20),
        ('printlog', False),
    )
    
    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(
            self.data, period=self.params.period)
    
    def next(self):
        if self.data.close[0] > self.sma[0]:
            self.buy()
        elif self.data.close[0] < self.sma[0]:
            self.sell()
```

#### 2. BaseStrategy (Recommended)
```python
from backtrader.strategies.base import BaseStrategy

class MyAdvancedStrategy(BaseStrategy):
    params = (
        ('period', 20),
        ('take_profit', 2.0),
        ('stop_loss', 1.0),
    )
    
    def __init__(self):
        super().__init__()
        self.sma = bt.indicators.SimpleMovingAverage(
            self.data, period=self.params.period)
    
    def buy_or_short_condition(self):
        if not self.buy_executed and self.data.close[0] > self.sma[0]:
            self.create_order('BUY')
            return True
        return False
    
    def sell_or_cover_condition(self):
        if self.buy_executed and self.data.close[0] < self.sma[0]:
            self.close_all_positions()
            return True
        return False
```

### Why Use BaseStrategy?

The `BaseStrategy` provides several advantages:

1. **Built-in Position Management**: Automatic tracking of entry prices, position sizes, and P&L
2. **DCA Support**: Built-in Dollar-Cost Averaging functionality
3. **Risk Management**: Take-profit and stop-loss automation
4. **Live Trading Ready**: Seamless transition from backtesting to live trading
5. **Order Tracking**: Automatic persistence of active orders
6. **Alert System**: Built-in support for Discord and Telegram notifications

## Creating Custom Indicators

### Simple Custom Indicator

```python
import backtrader as bt

class CustomRSI(bt.Indicator):
    lines = ('custom_rsi',)
    params = (('period', 14),)
    
    def __init__(self):
        # Calculate price changes
        delta = self.data - self.data(-1)
        
        # Calculate gains and losses
        gain = bt.If(delta > 0, delta, 0)
        loss = bt.If(delta < 0, abs(delta), 0)
        
        # Calculate average gains and losses
        avg_gain = bt.indicators.MovingAverageSimple(gain, period=self.params.period)
        avg_loss = bt.indicators.MovingAverageSimple(loss, period=self.params.period)
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        self.lines.custom_rsi = 100 - (100 / (1 + rs))
```

### Advanced Indicator with Multiple Lines

```python
class BollingerBandsCustom(bt.Indicator):
    lines = ('mid', 'upper', 'lower', 'bandwidth')
    params = (('period', 20), ('devfactor', 2.0))
    
    def __init__(self):
        self.lines.mid = bt.indicators.SimpleMovingAverage(
            self.data, period=self.params.period)
        
        std_dev = bt.indicators.StandardDeviation(
            self.data, period=self.params.period)
        
        self.lines.upper = self.lines.mid + (std_dev * self.params.devfactor)
        self.lines.lower = self.lines.mid - (std_dev * self.params.devfactor)
        self.lines.bandwidth = (self.lines.upper - self.lines.lower) / self.lines.mid
```

### Using Custom Indicators in Strategies

```python
class StrategyWithCustomIndicator(BaseStrategy):
    params = (
        ('bb_period', 20),
        ('bb_dev', 2.0),
        ('rsi_period', 14),
    )
    
    def __init__(self):
        super().__init__()
        
        # Use custom indicators
        self.bb = BollingerBandsCustom(
            self.data,
            period=self.params.bb_period,
            devfactor=self.params.bb_dev
        )
        
        self.rsi = CustomRSI(
            self.data,
            period=self.params.rsi_period
        )
    
    def buy_or_short_condition(self):
        # Buy when price touches lower Bollinger Band and RSI is oversold
        if (not self.buy_executed and 
            self.data.close[0] <= self.bb.lines.lower[0] and
            self.rsi.lines.custom_rsi[0] < 30):
            
            self.create_order('BUY')
            return True
        return False
    
    def sell_or_cover_condition(self):
        # Sell when price touches upper Bollinger Band or RSI is overbought
        if (self.buy_executed and 
            (self.data.close[0] >= self.bb.lines.upper[0] or
             self.rsi.lines.custom_rsi[0] > 70)):
            
            self.close_all_positions()
            return True
        return False
```

## Risk Management

### Position Sizing

```python
class RiskManagedStrategy(BaseStrategy):
    params = (
        ('risk_per_trade', 0.01),  # 1% risk per trade
        ('max_position_size', 0.1),  # Max 10% of portfolio
    )
    
    def _calculate_position_size(self):
        """Calculate position size based on risk management"""
        if self.p.backtest:
            available_cash = self.broker.getcash()
        else:
            # For live trading, use actual balance
            available_cash = self.broker.getcash()
        
        # Calculate position size based on risk per trade
        risk_amount = available_cash * self.params.risk_per_trade
        
        # Calculate stop loss distance (example: 2% below entry)
        stop_loss_distance = self.data.close[0] * 0.02
        
        # Calculate position size
        position_size = risk_amount / stop_loss_distance
        
        # Apply maximum position size constraint
        max_size = available_cash * self.params.max_position_size / self.data.close[0]
        position_size = min(position_size, max_size)
        
        return max(0, position_size)
    
    def buy_or_short_condition(self):
        if not self.buy_executed:
            # Calculate dynamic position size
            size = self._calculate_position_size()
            
            if size > 0:
                self.create_order('BUY', size=size)
                return True
        return False
```

### Stop Loss and Take Profit

```python
class StrategyWithRiskManagement(BaseStrategy):
    params = (
        ('stop_loss_pct', 2.0),  # 2% stop loss
        ('take_profit_pct', 5.0),  # 5% take profit
    )
    
    def check_stop_loss(self):
        """Check if stop loss is hit"""
        if self.buy_executed and self.average_entry_price:
            current_price = self.data.close[0]
            stop_loss_price = self.average_entry_price * (1 - self.params.stop_loss_pct / 100)
            
            if current_price <= stop_loss_price:
                self.close_all_positions()
                return True
        return False
    
    def check_take_profit(self):
        """Check if take profit is hit"""
        if self.buy_executed and self.average_entry_price:
            current_price = self.data.close[0]
            take_profit_price = self.average_entry_price * (1 + self.params.take_profit_pct / 100)
            
            if current_price >= take_profit_price:
                self.close_all_positions()
                return True
        return False
    
    def next(self):
        # Check risk management first
        if self.check_stop_loss() or self.check_take_profit():
            return
        
        # Then check strategy conditions
        super().next()
```

### DCA (Dollar-Cost Averaging)

```python
class DCAStrategy(BaseStrategy):
    params = (
        ('dca_levels', [2.0, 5.0, 10.0]),  # DCA at 2%, 5%, 10% drops
        ('dca_amounts', [1.0, 2.0, 4.0]),  # Multipliers for each DCA level
        ('max_dca_levels', 3),
    )
    
    def __init__(self):
        super().__init__()
        self.DCA = True  # Enable DCA mode
        self.dca_level = 0
    
    def dca_or_short_condition(self):
        """Implement DCA logic"""
        if not self.buy_executed or self.dca_level >= self.params.max_dca_levels:
            return False
        
        current_price = self.data.close[0]
        entry_price = self.average_entry_price or self.first_entry_price
        
        # Check each DCA level
        for i, (drop_pct, multiplier) in enumerate(zip(self.params.dca_levels, self.params.dca_amounts)):
            if i <= self.dca_level:
                continue
                
            target_price = entry_price * (1 - drop_pct / 100)
            
            if current_price <= target_price:
                # Calculate DCA size
                base_size = self.sizes[0] if self.sizes else 1.0
                dca_size = base_size * multiplier
                
                self.create_order('BUY', size=dca_size)
                self.dca_level = i + 1
                return True
        
        return False
```

## Strategy Parameters and Optimization

### Parameter Definition

```python
class OptimizableStrategy(BaseStrategy):
    params = (
        ('sma_fast_period', 10),
        ('sma_slow_period', 30),
        ('rsi_period', 14),
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
        ('risk_reward_ratio', 2.0),
        ('max_trades_per_day', 3),
    )
    
    def __init__(self):
        super().__init__()
        
        self.sma_fast = bt.indicators.SimpleMovingAverage(
            self.data, period=self.params.sma_fast_period)
        self.sma_slow = bt.indicators.SimpleMovingAverage(
            self.data, period=self.params.sma_slow_period)
        self.rsi = bt.indicators.RelativeStrengthIndex(
            self.data, period=self.params.rsi_period)
```

### Strategy Optimization

```python
from backtrader.utils.backtest import optimize_backtest

def optimize_strategy():
    # Load data
    data = get_crypto_data('BTC/USDT', '2024-01-01', '2024-03-01', '1h', 'binance')
    
    # Define parameter ranges for optimization
    results = optimize_backtest(
        strategy=OptimizableStrategy,
        data=data,
        init_cash=10000,
        sma_fast_period=[5, 10, 15, 20],
        sma_slow_period=[30, 50, 100],
        rsi_period=[10, 14, 20],
        rsi_oversold=[20, 25, 30, 35],
        rsi_overbought=[65, 70, 75, 80],
        max_workers=4,
        show_progress=True
    )
    
    # Print best results
    best_result = max(results, key=lambda x: x['final_value'])
    print(f"Best parameters: {best_result['params']}")
    print(f"Best result: ${best_result['final_value']:.2f}")
    
    return results
```

### Multi-Objective Optimization

```python
def multi_objective_optimization():
    data = get_crypto_data('BTC/USDT', '2024-01-01', '2024-03-01', '1h', 'binance')
    
    results = optimize_backtest(
        strategy=OptimizableStrategy,
        data=data,
        init_cash=10000,
        sma_fast_period=[10, 15, 20],
        sma_slow_period=[30, 50, 100],
        rsi_period=[14, 20],
        rsi_oversold=[25, 30, 35],
        rsi_overbought=[65, 70, 75],
        max_workers=4
    )
    
    # Analyze results
    for result in results:
        final_value = result['final_value']
        total_trades = result['total_trades']
        win_rate = result['win_rate']
        
        # Calculate risk-adjusted metrics
        sharpe_ratio = result.get('sharpe_ratio', 0)
        max_drawdown = result.get('max_drawdown', 0)
        
        print(f"Params: {result['params']}")
        print(f"  Final Value: ${final_value:.2f}")
        print(f"  Total Trades: {total_trades}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {max_drawdown:.2f}%")
        print()
```

## Advanced Strategy Patterns

### Multi-Timeframe Analysis

```python
class MultiTimeframeStrategy(BaseStrategy):
    params = (
        ('slow_tf', '1h'),
        ('fast_tf', '15m'),
        ('trend_threshold', 0.01),
    )
    
    def __init__(self):
        super().__init__()
        
        # Add resampled data
        self.slow_data = self.datas[0].resample('1h')
        self.fast_data = self.datas[0].resample('15m')
        
        # Indicators on different timeframes
        self.slow_sma = bt.indicators.SimpleMovingAverage(
            self.slow_data, period=50)
        self.fast_sma = bt.indicators.SimpleMovingAverage(
            self.fast_data, period=20)
    
    def buy_or_short_condition(self):
        # Check trend on slow timeframe
        slow_trend = (self.data.close[0] / self.slow_sma[0]) - 1
        
        if slow_trend > self.params.trend_threshold:
            # Uptrend - look for buy signals on fast timeframe
            if (not self.buy_executed and 
                self.fast_data.close[0] > self.fast_sma[0]):
                self.create_order('BUY')
                return True
        elif slow_trend < -self.params.trend_threshold:
            # Downtrend - look for sell signals
            if (not self.buy_executed and 
                self.fast_data.close[0] < self.fast_sma[0]):
                self.create_order('SELL')
                return True
        
        return False
```

### Machine Learning Integration

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class MLStrategy(BaseStrategy):
    params = (
        ('lookback_period', 50),
        ('model_update_frequency', 100),
    )
    
    def __init__(self):
        super().__init__()
        
        # Technical indicators for features
        self.rsi = bt.indicators.RelativeStrengthIndex(self.data, period=14)
        self.macd = bt.indicators.MACD(self.data)
        self.bollinger = bt.indicators.BollingerBands(self.data)
        
        self.model = None
        self.feature_history = []
        self.target_history = []
        self.bar_count = 0
    
    def _extract_features(self):
        """Extract features for ML model"""
        features = [
            self.data.close[0] / self.data.close[-1] - 1,  # Price change
            self.rsi[0],  # RSI
            (self.data.close[0] - self.bollinger.lines.bot[0]) / 
            (self.bollinger.lines.top[0] - self.bollinger.lines.bot[0]),  # Bollinger position
            self.macd.lines.macd[0],  # MACD
            self.data.volume[0] / max(1, self.data.volume[-10:].mean()),  # Volume ratio
        ]
        return np.array(features).reshape(1, -1)
    
    def _train_model(self):
        """Train the ML model"""
        if len(self.feature_history) < 100:
            return
        
        X = np.array(self.feature_history)
        y = np.array(self.target_history)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
    
    def _predict_signal(self):
        """Predict trading signal"""
        if self.model is None:
            return 0
        
        features = self._extract_features()
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        return prediction, max(probability)
    
    def next(self):
        self.bar_count += 1
        
        # Collect training data
        if len(self.feature_history) < 1000:  # Limit history size
            features = self._extract_features().flatten()
            # Create target: 1 if price goes up in next 5 bars, 0 otherwise
            future_return = (self.data.close[5] / self.data.close[0]) - 1 if len(self.data) > 5 else 0
            target = 1 if future_return > 0.001 else 0
            
            self.feature_history.append(features)
            self.target_history.append(target)
        
        # Update model periodically
        if self.bar_count % self.params.model_update_frequency == 0:
            self._train_model()
        
        # Make predictions and trade
        if self.model is not None:
            prediction, confidence = self._predict_signal()
            
            if prediction == 1 and confidence > 0.6:  # Buy signal with high confidence
                if not self.buy_executed:
                    self.create_order('BUY')
            elif prediction == 0 and confidence > 0.6:  # Sell signal with high confidence
                if self.buy_executed:
                    self.close_all_positions()
        
        super().next()
```

### Event-Driven Strategy

```python
class EventDrivenStrategy(BaseStrategy):
    params = (
        ('event_threshold', 0.02),  # 2% price move triggers event
        ('event_timeout', 10),      # 10 bars to act on event
    )
    
    def __init__(self):
        super().__init__()
        self.event_detected = False
        self.event_price = 0
        self.event_timer = 0
    
    def detect_event(self):
        """Detect significant price movements"""
        if len(self.data) < 2:
            return False
        
        price_change = abs(self.data.close[0] / self.data.close[-1] - 1)
        
        if price_change > self.params.event_threshold:
            return True
        return False
    
    def handle_event(self):
        """Handle detected events"""
        if self.event_detected:
            self.event_timer += 1
            
            # Reset if timeout reached
            if self.event_timer > self.params.event_timeout:
                self.event_detected = False
                self.event_timer = 0
                return
            
            # Implement event-based trading logic
            current_price = self.data.close[0]
            
            if self.event_price < current_price:  # Upward event
                if not self.buy_executed:
                    self.create_order('BUY')
            else:  # Downward event
                if self.buy_executed:
                    self.close_all_positions()
        else:
            # Check for new events
            if self.detect_event():
                self.event_detected = True
                self.event_price = self.data.close[0]
                self.event_timer = 0
    
    def next(self):
        self.handle_event()
        super().next()
```

## Strategy Testing and Validation

### Unit Testing Strategies

```python
import unittest
from backtrader import Cerebro
from backtrader.feeds.polarfeed import PolarsData

class TestStrategy(unittest.TestCase):
    def setUp(self):
        self.cerebro = Cerebro()
        self.strategy = None
    
    def test_simple_ma_strategy(self):
        # Create test data
        import pandas as pd
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        prices = [100 + i * 0.1 for i in range(100)]  # Upward trend
        df = pd.DataFrame({
            'datetime': dates,
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000] * 100
        })
        
        data = PolarsData(dataname=df)
        self.cerebro.adddata(data)
        
        # Add strategy
        self.cerebro.addstrategy(SimpleMAStrategy, fast_period=10, slow_period=30)
        
        # Run backtest
        results = self.cerebro.run()
        
        # Check results
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].analyzers.getbyname('trade_analyzer').get_analysis()['total']['total'] > 0)
    
    def test_strategy_parameters(self):
        # Test different parameter combinations
        for fast_period in [5, 10, 15]:
            for slow_period in [20, 30, 50]:
                with self.subTest(fast_period=fast_period, slow_period=slow_period):
                    # Test strategy with these parameters
                    self.cerebro.addstrategy(
                        SimpleMAStrategy, 
                        fast_period=fast_period, 
                        slow_period=slow_period
                    )
                    results = self.cerebro.run()
                    self.assertIsNotNone(results)
```

### Performance Metrics

```python
def analyze_strategy_performance(results):
    """Analyze strategy performance metrics"""
    
    # Extract key metrics
    analyzer = results[0].analyzers.getbyname('trade_analyzer')
    analysis = analyzer.get_analysis()
    
    # Basic metrics
    total_trades = analysis['total']['total']
    won_trades = analysis['won']['total']
    lost_trades = analysis['lost']['total']
    
    # Calculate derived metrics
    win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
    profit_factor = (analysis['won']['pnl']['total'] / abs(analysis['lost']['pnl']['total'])) if analysis['lost']['pnl']['total'] != 0 else float('inf')
    
    # Risk metrics
    drawdown = results[0].analyzers.getbyname('drawdown').get_analysis()
    max_drawdown = drawdown['max']['drawdown']
    
    # Return metrics
    returns = results[0].analyzers.getbyname('returns').get_analysis()
    total_return = returns['rtot']
    annualized_return = returns['ravg'] * 252  # Assuming daily data
    
    # Print comprehensive report
    print("=== Strategy Performance Report ===")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2f}%")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Annualized Return: {annualized_return:.2f}%")
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'total_return': total_return,
        'annualized_return': annualized_return
    }
```

## Best Practices

### 1. Code Organization
- Use meaningful variable names
- Document your strategy logic
- Separate concerns (indicators, signals, risk management)
- Use version control for strategy iterations

### 2. Risk Management
- Always implement stop-loss mechanisms
- Use position sizing based on risk, not fixed amounts
- Monitor maximum drawdown
- Diversify across multiple strategies if possible

### 3. Backtesting Quality
- Use high-quality historical data
- Account for slippage and commissions
- Test on out-of-sample data
- Consider market regime changes

### 4. Live Trading Preparation
- Paper trade before going live
- Monitor strategy performance regularly
- Have a plan for strategy failure
- Implement proper logging and alerting

### 5. Performance Optimization
- Use efficient data structures
- Minimize calculations in `next()` method
- Consider using compiled indicators when possible
- Profile your strategy for bottlenecks

## Next Steps

1. **Explore Examples**: Check `Examples/` directory for complete strategy examples
2. **Learn Advanced Features**: Study machine learning integration and multi-timeframe analysis
3. **Optimization**: Learn about parameter optimization and walk-forward analysis
4. **Risk Management**: Implement advanced risk management techniques
5. **Live Trading**: Prepare your strategy for live deployment

This guide provides the foundation for developing sophisticated trading strategies with BTQuant. Remember that successful algorithmic trading requires continuous learning, testing, and adaptation to changing market conditions.