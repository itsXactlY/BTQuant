
# Cheatcode Strategy Implementation Guide

## Overview
This is a complete Backtrader implementation of the TradingView Cheatcode ZERO and Cheatcode A strategies. The strategy combines multiple technical indicators and signal confirmation systems for robust trading decisions.

## Installation Requirements

```bash
pip install backtrader
pip install pandas
pip install numpy
```

## Basic Usage

```python
import backtrader as bt
from cheatcode_strategy import CheatcodeStrategy
from datetime import datetime

# Create cerebro instance
cerebro = bt.Cerebro()

# Add strategy with custom parameters
cerebro.addstrategy(
    CheatcodeStrategy,
    # RSI Parameters
    rsi_period=14,
    rsi_overbought=70,
    rsi_oversold=30,

    # MACD Parameters  
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,

    # GRaB System
    ema_grab_period=34,

    # Trading Parameters
    min_signals_for_trade=2,
    position_size=95,  # 95% of available cash
    stop_loss_pct=2.0,
    take_profit_pct=4.0
)

# Add data feed
data = bt.feeds.YahooFinanceData(
    dataname='FILUSDT',  # Or your preferred symbol
    fromdate=datetime(2023, 1, 1),
    todate=datetime(2024, 1, 1),
    timeframe=bt.TimeFrame.Minutes,
    compression=1
)
cerebro.adddata(data)

# Set broker parameters
cerebro.broker.set_cash(10000)
cerebro.broker.setcommission(commission=0.001)  # 0.1% commission

# Add analyzers
cerebro.addanalyzer(bt.analyzers.SharpeRatio)
cerebro.addanalyzer(bt.analyzers.Returns)
cerebro.addanalyzer(bt.analyzers.DrawDown)

# Run backtest
results = cerebro.run()

# Print results
print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')
cerebro.plot()
```

## Strategy Components

### 1. Camarilla Pivot Points
- Calculates H1-H6 resistance levels and L1-L6 support levels
- Uses previous day's High, Low, Close data
- Provides key reversal zones for entries and exits

### 2. RSI/MACD Combination
- Custom RSI/MACD signal combination from the original Pine script
- Generates long signals when RSI > 51 and Signal < MACD
- Generates short signals when RSI < 49 and Signal > MACD

### 3. GRaB EMA System  
- 34-period EMA system on High, Low, Close
- Bullish when price breaks above EMA High
- Bearish when price breaks below EMA Low

### 4. TKE Oscillator
- Total Kinetic Energy composite oscillator
- Combines RSI, Momentum, and Williams %R
- Oversold/Overbought conditions trigger signals

### 5. Hull Moving Average
- Advanced trend-following indicator
- Uses weighted moving averages for reduced lag
- Provides trend direction confirmation

## Signal Confirmation System

The strategy requires a minimum number of confirming signals before entering trades:

**Long Entry Signals:**
- RSI/MACD combo bullish
- GRaB system bullish  
- TKE oversold bounce
- Hull MA bullish trend
- Pivot level support bounce

**Short Entry Signals:**
- RSI/MACD combo bearish
- GRaB system bearish
- TKE overbought rejection  
- Hull MA bearish trend
- Pivot level resistance rejection

## Risk Management

### Position Sizing
- Configurable position size as percentage of available cash
- Default: 95% of available cash per trade

### Stop Loss / Take Profit
- Configurable stop loss percentage (default: 2%)
- Configurable take profit percentage (default: 4%)
- Automatic position closure when levels are hit

## Parameter Optimization

Key parameters that can be optimized:

```python
# Add parameter optimization
cerebro.optstrategy(
    CheatcodeStrategy,
    min_signals_for_trade=range(1, 4),
    stop_loss_pct=[1.0, 1.5, 2.0, 2.5],
    take_profit_pct=[2.0, 3.0, 4.0, 5.0]
)
```

## Custom Data Integration

For crypto data or custom timeframes:

```python
# For crypto data
import ccxt
import pandas as pd

# Fetch data from exchange
exchange = ccxt.binance()
ohlcv = exchange.fetch_ohlcv('FIL/USDT', '1m', limit=1000)

# Convert to pandas DataFrame
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

# Create custom data feed
class CustomDataFeed(bt.feeds.PandasData):
    params = (
        ('datetime', 0),
        ('open', 1),
        ('high', 2), 
        ('low', 3),
        ('close', 4),
        ('volume', 5),
    )

data = CustomDataFeed(dataname=df)
cerebro.adddata(data)
```

## Monitoring and Logging

The strategy includes comprehensive logging:
- Entry/exit signals with reasons
- Price levels and position sizes
- Risk management actions
- Performance metrics

## Advanced Usage

### Multiple Timeframes
```python
# Add multiple timeframe data
data_1m = bt.feeds.YahooFinanceData(dataname='FILUSDT', timeframe=bt.TimeFrame.Minutes, compression=1)
data_5m = bt.feeds.YahooFinanceData(dataname='FILUSDT', timeframe=bt.TimeFrame.Minutes, compression=5)

cerebro.adddata(data_1m, name='1m')
cerebro.adddata(data_5m, name='5m')

# Access in strategy with self.datas[0] (1m) and self.datas[1] (5m)
```

### Live Trading Integration
The strategy can be adapted for live trading by:
1. Replacing data feeds with live data sources
2. Implementing order execution through broker APIs
3. Adding real-time monitoring and alerting

## Performance Tips

1. **Lookback Periods**: Ensure sufficient historical data for all indicators
2. **Commission Settings**: Include realistic commission/slippage costs
3. **Parameter Optimization**: Use walk-forward analysis for robust parameters
4. **Risk Management**: Always include stop losses and position sizing
5. **Signal Filtering**: Adjust `min_signals_for_trade` based on market conditions

## Troubleshooting

Common issues and solutions:

1. **Insufficient Data**: Ensure enough historical data for longest indicator period
2. **No Signals Generated**: Check parameter settings and signal thresholds  
3. **Poor Performance**: Optimize parameters or adjust signal confirmation requirements
4. **Memory Issues**: Use limited lookback periods for large datasets

This implementation provides a solid foundation for the Cheatcode strategy in Backtrader while maintaining the core logic from the original Pine scripts.
