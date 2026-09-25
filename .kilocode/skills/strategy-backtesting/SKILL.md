---
name: strategy-backtesting
description: Comprehensive backtesting workflows and best practices for BTQuant strategies
---

# BTQuant Backtesting Best Practices

When creating and executing backtests for strategy evaluation:

## Backtest Setup

### Data Sourcing

```python
from btquant import Backtest, DataSource

# From MSSQL (recommended)
data_source = DataSource.from_mssql(
    symbol='BTCUSDT',
    exchange='binance',
    timeframe='1h',
    start_date='2023-01-01',
    end_date='2024-01-01'
)

# From CSV
data_source = DataSource.from_csv(filepath='data/BTCUSDT_1h.csv')

# From CCXT exchange
data_source = DataSource.from_ccxt(
    exchange='binance',
    symbol='BTC/USDT',
    timeframe='1h'
)
```

## Backtest Configuration

- Set realistic commission rates (0.1% per exchange)
- Configure slippage (0.01% - 0.05% depending on liquidity)
- Use proper timeframes (avoid sub-minute)
- Set starting capital appropriately
- Configure reinvestment vs fixed capital mode

## Performance Metrics

Always calculate and report:
- **Return metrics**: Total return, CAGR, monthly returns
- **Risk metrics**: Sharpe ratio, Sortino ratio, max drawdown
- **Trade metrics**: Win rate, profit factor, expectancy
- **Efficiency**: Calmar ratio, recovery factor

## Out-of-Sample Testing

- Use at least 20% of data as out-of-sample
- Time-series split (not random split)
- Test on different market regimes
- Validate on multiple time periods

## Common Pitfalls

- Look-ahead bias (using future data)
- Overfitting to historical data
- Not accounting for transaction costs
- Ignoring slippage and partial fills
- Cherry-picking date ranges
- Insufficient trading activity
