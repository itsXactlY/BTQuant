---
name: strategy-development
description: BTQuant strategy development best practices and patterns for autonomous agents
---

# BTQuant Strategy Development Guidelines

When developing new trading strategies for BTQuant, follow these patterns and conventions:

## Strategy Structure

All strategies should inherit from BTQuant's base strategy classes:

```python
from btquant import Strategy
import pandas as pd

class MyStrategy(Strategy):
    def __init__(self, name, exchange, symbol, timeframe, **kwargs):
        super().__init__(name, exchange, symbol, timeframe, **kwargs)
        self.fast_ma_period = kwargs.get('fast_ma_period', 12)
        self.slow_ma_period = kwargs.get('slow_ma_period', 26)
        self.risk_percent = kwargs.get('risk_percent', 2.0)
        
    def on_candle(self, candle):
        # Implement signal generation logic
        pass
```

## Parameter Management

- Use configuration dictionaries for all strategy parameters
- Expose parameters for agent-driven optimization
- Document parameter ranges and sensitivities
- Group related parameters using namespaces

## Signal Generation

- Keep indicators and signal logic separated
- Cache expensive calculations
- Use vectorized operations with Polars/Pandas
- Emit clear signal events with metadata

## Risk Management

- Always implement position sizing based on risk percent
- Include stop-loss calculations
- Use Kelly Criterion or similar for optimal sizing
- Log all risk decisions
