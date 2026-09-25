---
name: documentation
description: Documentation standards for BTQuant strategy projects
---

# BTQuant Documentation Standards

When creating strategy documentation:

## README Structure

### README.md Template

```markdown
# Strategy Name

Brief one-sentence description.

## Overview

Detailed explanation (2-3 paragraphs):
- What the strategy does
- Market conditions it targets
- Key indicators/logic

## Strategy Logic

### Entry Signals
- List entry conditions
- Example: Fast MA crosses above slow MA with volume

### Exit Signals
- List exit conditions

### Position Sizing
- How position size is calculated
- Risk management approach

## Parameters

| Parameter | Type | Default | Min | Max | Description |
|-----------|------|---------|-----|-----|-------------|
| fast_period | int | 12 | 5 | 30 | Fast MA period |
| slow_period | int | 26 | 20 | 100 | Slow MA period |

## Backtest Results

### Summary Metrics
- Total Return: 125%
- Sharpe Ratio: 1.8
- Max Drawdown: 22%
- Win Rate: 52%
- Profit Factor: 1.8

### Backtest Parameters
- Period: 2023-01-01 to 2024-01-01
- Asset: BTCUSDT
- Timeframe: 1h
- Initial Capital: $10,000

## Usage

```python
from strategies.ma_crossover import MovingAverageCrossoverStrategy

strategy = MovingAverageCrossoverStrategy(
    fast_period=12,
    slow_period=26,
    risk_percent=2.0
)

backtest = Backtest(strategy, data)
results = backtest.run()
```

## Installation

1. Copy strategy file to strategies/
2. Install dependencies
3. Configure parameters
4. Run backtest

## Risk Considerations

- May underperform in low volatility
- Max recommended position: 5%
- Not suitable for pairs trading

## Improvements and Variants

- Add volume confirmation
- Test with volatility adjustment
- Implement dynamic parameters
```

## Change Log Format

```markdown
## Changelog

### [1.1.0] - 2024-01-15

#### Added
- Volume confirmation filter
- Dynamic parameter adjustment

#### Changed
- Updated MA periods based on optimization
- Improved exit logic

#### Fixed
- Bug in position sizing
- Incorrect risk calculation

### [1.0.0] - 2024-01-01
- Initial release
```

## API Documentation

```python
def calculate_position_size(self, risk_amount: float) -> float:
    """Calculate position size based on risk.
    
    Parameters
    ----------
    risk_amount : float
        Dollar amount to risk per trade
        
    Returns
    -------
    float
        Number of contracts/coins to trade
        
    Raises
    ------
    ValueError
        If risk_amount is negative
        
    Examples
    --------
    >>> size = strategy.calculate_position_size(100)
    >>> print(size)
    0.5
    """
```

## Performance Report Template

```markdown
# Strategy Performance Report

## Executive Summary

[1-2 paragraph summary]

## Key Metrics

### Returns
- Total Return: XX%
- CAGR: XX%
- Monthly Return Range: XX% to XX%

### Risk
- Volatility: XX%
- Maximum Drawdown: XX%
- Sharpe Ratio: X.XX
- Sortino Ratio: X.XX

### Trade Statistics
- Total Trades: XXX
- Win Rate: XX%
- Profit Factor: X.XX
- Average Win/Loss: XX

## Detailed Analysis

[Include equity curve, monthly heatmap, drawdown analysis]

## Sensitivity Analysis

[Include sensitivity to parameter changes]

## Conclusion

[Summary and recommendations]
```
