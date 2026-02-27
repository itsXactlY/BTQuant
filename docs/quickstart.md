# BTQuant Quick Start Guide

## Overview

This guide will help you get started with BTQuant quickly. You'll learn how to create your first strategy, run a backtest, set up real-time market monitoring, and detect manipulation patterns.

## Prerequisites

Before starting, ensure you have:
- [x] BTQuant installed (see [Installation Guide](installation.md))
- [x] Python 3.12+ and C++17 environments set up
- [x] Basic understanding of Python programming
- [x] C++ market data collector running (for real-time features)

## Your First Strategy

Let's create a simple moving average crossover strategy as your first example.

### Step 1: Create a Strategy File

Create a new file called `simple_ma_strategy.py`:

```python
import backtrader as bt

class SimpleMAStrategy(bt.Strategy):
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
        ('printlog', False),
    )

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.data_close = self.datas[0].close
        
        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add a MovingAverageSimple indicator
        self.sma_fast = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.fast_period)
        self.sma_slow = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.slow_period)

        # Indicators for the plotting show
        bt.indicators.ExponentialMovingAverage(self.datas[0], period=25)
        bt.indicators.WeightedMovingAverage(self.datas[0], period=25,
                                          subplot=True)
        bt.indicators.StochasticSlow(self.datas[0])
        bt.indicators.MACDHisto(self.datas[0])
        rsi = bt.indicators.RSI(self.datas[0])
        bt.indicators.SmoothedMovingAverage(rsi, period=10)
        bt.indicators.ATR(self.datas[0], plot=False)

    def log(self, txt, dt=None, doprint=False):
        ''' Logging function for this strategy'''
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.data_close[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

            # Not yet ... we MIGHT BUY if ...
            if self.sma_fast[0] > self.sma_slow[0]:
                # BUY, BUY, BUY!!! (with default parameters)
                self.log('BUY CREATE, %.2f' % self.data_close[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy()

        else:

            if self.sma_fast[0] < self.sma_slow[0]:
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.data_close[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()

    def stop(self):
        self.log('(MA Period fast %2d, slow %2d) Ending Value %.2f' %
                 (self.params.fast_period, self.params.slow_period, self.broker.getvalue()), doprint=True)
```

### Step 2: Create a Backtest Script

Create a file called `run_backtest.py`:

```python
from backtrader import backtest
from simple_ma_strategy import SimpleMAStrategy
from backtrader.utils.ccxt_data import get_crypto_data

def main():
    # Get data for backtesting
    print("Fetching data...")
    data = get_crypto_data(
        asset='BTC/USDT',
        start_date='2024-01-01',
        end_date='2024-01-31',
        timeframe='1h',
        exchange='binance'
    )
    
    if data is None:
        print("Failed to fetch data. Please check your internet connection and exchange availability.")
        return
    
    print(f"Data fetched successfully. Shape: {data.shape}")
    
    # Run backtest
    print("Running backtest...")
    result = backtest(
        strategy=SimpleMAStrategy,
        data=data,
        init_cash=10000,  # Starting capital
        backtest=True,
        plot=True,        # Generate plot
        quantstats=True,  # Generate QuantStats report
        asset_name='BTC/USDT'
    )
    
    print(f"Backtest completed!")
    print(f"Final portfolio value: ${result:.2f}")

if __name__ == '__main__':
    main()
```

### Step 3: Run Your First Backtest

```bash
# Activate your virtual environment
source .btq/bin/activate

# Run the backtest
python run_backtest.py
```

You should see output similar to:
```
Fetching data...
Data fetched successfully. Shape: (744, 6)
Running backtest...
Close, 42345.12
Close, 42456.78
...
Backtest completed!
Final portfolio value: $10542.34
```

## Understanding the Output

### Console Output
- **Data fetching**: Shows progress of downloading market data
- **Strategy logs**: Each bar's closing price and trade executions
- **Trade notifications**: Buy/sell orders, prices, and commissions
- **Final results**: Ending portfolio value

### Generated Files
- **Plot**: A candlestick chart with your strategy's buy/sell signals
- **QuantStats report**: `QuantStats/BTC_USDT_2024-01-01_12-00-00.html` with detailed performance metrics

## Real-Time Market Monitoring

Once you have the C++ market data collector running, you can monitor live markets for manipulation patterns.

### Start the Market Data Collector

First, ensure your market data collector is running:

```bash
# Terminal 1: Start market data collection
cd dependencies/ccapi/example/build/src/market_data_collector
./market_data_collector
```

### Run the Manipulation Detector

In another terminal, start the real-time detector:

```bash
# Terminal 2: Start manipulation detection
cd tests/new/build
./manipulation_monitor
```

### Expected Output

You should see real-time detection alerts:

```
🚨 StopHunt(symbol=BTC-USDT, exchange=binance, deviation=-1.2%, signal=LONG)
💰 Arbitrage(buy=kraken@42150, sell=binance@42250, profit=65bps)
🐋 WhaleDetected(symbol=ETH-USDT, size=$250000, lagging=3 exchanges)
```

### Understanding Detection Signals

- **🚨 Stop Hunt**: Fake wicks designed to trigger stop-loss orders
- **💰 Arbitrage**: Cross-exchange price discrepancies
- **🐋 Whale Front-Run**: Large trades that may move markets
- **🏦 Liquidity Imbalance**: Thin orderbooks signaling manipulation targets
- **🎭 Spoofing**: Fake orders to manipulate market perception

## Using the BaseStrategy

For more advanced features, you can use BTQuant's `BaseStrategy`:

```python
from backtrader.strategies.base import BaseStrategy

class AdvancedStrategy(BaseStrategy):
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
        ('take_profit', 2.0),  # 2% take profit
        ('percent_sizer', 0.1),  # Use 10% of capital per trade
    )

    def __init__(self):
        super().__init__()
        self.sma_fast = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.fast_period)
        self.sma_slow = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.slow_period)

    def buy_or_short_condition(self):
        """Override to implement entry logic"""
        if not self.buy_executed and self.sma_fast[0] > self.sma_slow[0]:
            self.create_order('BUY')
            return True
        return False

    def sell_or_cover_condition(self):
        """Override to implement exit logic"""
        if self.buy_executed and self.sma_fast[0] < self.sma_slow[0]:
            # Close position
            for order_tracker in self.active_orders[:]:
                self.close_order(order_tracker)
            return True
        return False

    def check_stop_loss(self):
        """Override to implement custom stop loss"""
        if self.buy_executed and self.average_entry_price:
            current_price = self.data.close[0]
            stop_loss_price = self.average_entry_price * 0.95  # 5% stop loss
            
            if current_price <= stop_loss_price:
                for order_tracker in self.active_orders[:]:
                    self.close_order(order_tracker)
                return True
        return False
```

## Quick Examples

### 1. Simple EMA Crossover

```python
class EMACrossover(BaseStrategy):
    params = (
        ('fast_ema', 12),
        ('slow_ema', 26),
    )

    def __init__(self):
        super().__init__()
        self.ema_fast = bt.indicators.ExponentialMovingAverage(
            self.data, period=self.params.fast_ema)
        self.ema_slow = bt.indicators.ExponentialMovingAverage(
            self.data, period=self.params.slow_ema)

    def buy_or_short_condition(self):
        if not self.buy_executed and self.ema_fast[0] > self.ema_slow[0]:
            self.create_order('BUY')
            return True
        return False

    def sell_or_cover_condition(self):
        if self.buy_executed and self.ema_fast[0] < self.ema_slow[0]:
            self.close_all_positions()
            return True
        return False
```

### 2. RSI Strategy

```python
class RSIStrategy(BaseStrategy):
    params = (
        ('rsi_period', 14),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30),
    )

    def __init__(self):
        super().__init__()
        self.rsi = bt.indicators.RelativeStrengthIndex(
            self.data, period=self.params.rsi_period)

    def buy_or_short_condition(self):
        if not self.buy_executed and self.rsi[0] < self.params.rsi_oversold:
            self.create_order('BUY')
            return True
        return False

    def sell_or_cover_condition(self):
        if self.buy_executed and self.rsi[0] > self.params.rsi_overbought:
            self.close_all_positions()
            return True
        return False
```

## Data Sources

### CCXT Data (Recommended for beginners)
```python
from backtrader.utils.ccxt_data import get_crypto_data

data = get_crypto_data(
    asset='BTC/USDT',
    start_date='2024-01-01',
    end_date='2024-01-31',
    timeframe='1h',
    exchange='binance'
)
```

### SQL Server Data (For advanced users)
```python
from backtrader.feeds.mssql_crypto import get_database_data

data = get_database_data(
    ticker='BTC',
    start_date='2024-01-01',
    end_date='2024-01-31',
    time_resolution='1h',
    pair='USDT'
)
```

### CSV Data
```python
import pandas as pd
from backtrader.feeds.polarfeed import PolarsData

df = pd.read_csv('your_data.csv')
data = PolarsData(dataname=df)
```

## Running Different Types of Backtests

### 1. Basic Backtest
```python
result = backtest(
    strategy=YourStrategy,
    data=data,
    init_cash=10000,
    backtest=True
)
```

### 2. With Optimization
```python
from backtrader.utils.backtest import optimize_backtest

results = optimize_backtest(
    strategy=YourStrategy,
    data=data,
    init_cash=10000,
    fast_period=[10, 20, 30],
    slow_period=[50, 100, 150],
    max_workers=4
)
```

### 3. Bulk Backtest (Multiple Assets)
```python
from backtrader.utils.backtest import bulk_backtest

coins = ['BTC', 'ETH', 'ADA', 'SOL']
results = bulk_backtest(
    strategy=YourStrategy,
    coins=coins,
    start_date='2024-01-01',
    end_date='2024-01-31',
    interval='1h',
    init_cash=10000,
    max_workers=4
)
```

## Next Steps

1. **Explore Examples**: Check the `Examples/` directory for more complete examples
2. **Set up Live Monitoring**: Configure your exchange connections and start real-time detection
3. **Learn Strategy Development**: Read [Strategy Development Guide](user-guide/strategies.md)
4. **Understand Configuration**: Review [Configuration Guide](technical/configuration.md)
5. **Advanced Features**: Explore [API Reference](technical/api-reference.md)
6. **Launch Dashboard**: Use the QuantStats dashboard for performance analysis

## Common Issues and Solutions

### Issue: "No data available"
**Solution**: Check your internet connection and ensure the exchange is accessible. Try a different exchange or timeframe.

### Issue: "Shared memory not found"
**Solution**: Ensure the C++ market data collector is running first. Check `/dev/shm/btquant_hotspine` exists.

### Issue: "Module not found"
**Solution**: Ensure you're in the correct virtual environment and BTQuant is properly installed.

### Issue: "Permission denied"
**Solution**: Check file permissions and ensure you have write access to the output directory.

### Issue: "No detection signals"
**Solution**: Check that multiple exchanges are configured and market data is flowing. Adjust detector thresholds if needed.

## Getting Help

- **Documentation**: This quick start guide covers the basics
- **Examples**: See `Examples/` directory for working code
- **Manipulation Detection**: Check [Detection Quick Start](../tests/new/QUICKSTART.md)
- **Troubleshooting**: Check [Troubleshooting Guide](troubleshooting.md)
- **Community**: Join the BTQuant community for support

You're now ready to start building your own trading strategies with BTQuant! The framework provides powerful tools for both historical backtesting and real-time market analysis with manipulation detection.