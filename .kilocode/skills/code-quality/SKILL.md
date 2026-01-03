---
name: code-quality
description: Code quality standards and testing for BTQuant strategy development
---

# BTQuant Code Quality Standards

When developing and reviewing strategy code:

## Style and Formatting

Follow PEP 8:

```python
# Good: Clear, descriptive names
class MovingAverageCrossoverStrategy(Strategy):
    def __init__(self, fast_period=12, slow_period=26):
        self.fast_period = fast_period
        self.slow_period = slow_period
```

## Import Organization

```python
# Standard library
import sys
from datetime import datetime

# Third party
import pandas as pd
import polars as pl
import numpy as np

# Local
from btquant import Strategy
from btquant.indicators import SMA, EMA
```

## Documentation

### Docstring Format

```python
class MovingAverageCrossoverStrategy(Strategy):
    """Simple moving average crossover strategy.
    
    Generates buy signals when fast MA crosses above slow MA.
    
    Parameters
    ----------
    fast_period : int, default=12
        Period for fast moving average
    slow_period : int, default=26
        Period for slow moving average
    """
```

## Testing

### Unit Tests

```python
import unittest

class TestMAStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = MovingAverageCrossoverStrategy()
        
    def test_initialization(self):
        self.assertEqual(self.strategy.fast_period, 12)
        
    def test_signal_generation(self):
        df = pd.read_csv('fixtures/sample_data.csv')
        signals = self.strategy.generate_signals(df)
        self.assertEqual(len(signals), len(df))
```

## Error Handling

```python
class StrategyException(Exception):
    pass

try:
    strategy.validate(df)
except StrategyException as e:
    logger.error(f"Validation failed: {e}")
```

## Security

### Credential Management

```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('BINANCE_API_KEY')
assert api_key, "Missing BINANCE_API_KEY"
```
