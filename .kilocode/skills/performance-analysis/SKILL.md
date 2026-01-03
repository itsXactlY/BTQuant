---
name: performance-analysis
description: Advanced performance analysis and reporting for trading strategies
---

# BTQuant Performance Analysis

When analyzing strategy performance:

## Return Analysis

### Return Decomposition

```python
import pandas as pd

# Daily returns
returns = equity_curve.pct_change()

# Cumulative returns
cumulative = (1 + returns).cumprod() - 1

# Rolling return periods
rolling_1m = (equity_curve / equity_curve.shift(21)) - 1  # Monthly
rolling_1y = (equity_curve / equity_curve.shift(252)) - 1  # Yearly

# Return percentiles
percentiles = returns.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
```

## Risk Metrics

### Volatility Calculation

```python
# Annualized volatility
returns_daily = equity_curve.pct_change()
volatility = returns_daily.std() * np.sqrt(252)

# Rolling volatility
rolling_vol = returns_daily.rolling(20).std() * np.sqrt(252)
```

### Drawdown Analysis

```python
def calculate_drawdown(equity):
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    return drawdown

drawdown = calculate_drawdown(equity_curve)
max_dd = drawdown.min()
```

## Risk-Adjusted Returns

### Sharpe Ratio

```python
def sharpe_ratio(returns, risk_free_rate=0.02):
    excess_return = returns.mean() - risk_free_rate / 252
    return (excess_return * 252) / (returns.std() * np.sqrt(252))
```

### Sortino Ratio

```python
def sortino_ratio(returns, risk_free_rate=0.02):
    excess_return = returns.mean() - risk_free_rate / 252
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    return (excess_return * 252) / downside_vol
```

## Trade Analysis

### Win Rate and Profit Factor

```python
def analyze_trades(trades):
    trades_df = pd.DataFrame(trades)
    
    # Win rate
    winning_trades = (trades_df['pnl'] > 0).sum()
    win_rate = winning_trades / len(trades_df)
    
    # Profit factor
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss
    
    return {
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'expectancy': trades_df['pnl'].mean()
    }
```

## Visualization

### Equity Curve

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

ax1.plot(equity_curve, label='Equity')
ax1.set_ylabel('Equity ($)')
ax1.legend()
ax1.grid(True)

drawdown = calculate_drawdown(equity_curve)
ax2.fill_between(equity_curve.index, drawdown, alpha=0.3, color='red')
ax2.set_ylabel('Drawdown (%)')

plt.tight_layout()
plt.show()
```

### Monthly Returns Heatmap

```python
import seaborn as sns

monthly_returns = equity_curve.resample('M').last().pct_change()
heatmap_data = monthly_returns.groupby(
    [monthly_returns.index.year, monthly_returns.index.month]
).sum().unstack()

plt.figure(figsize=(14, 6))
sns.heatmap(heatmap_data, annot=True, fmt='.1%', cmap='RdYlGn')
plt.title('Monthly Returns (%)')
plt.show()
```
