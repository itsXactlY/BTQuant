---
name: advanced-debugging
description: Advanced debugging techniques for BTQuant strategy development (code mode)
---

# Advanced Debugging for BTQuant

When debugging complex strategy behavior:

## Signal Tracing

### Instrument Strategy for Debugging

```python
class DebugStrategy(MovingAverageCrossoverStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_signals = []
        self.debug_positions = []
        
    def on_candle(self, candle):
        self.debug_signals.append({
            'timestamp': candle['timestamp'],
            'close': candle['close'],
            'fast_ma': self.fast_ma,
            'slow_ma': self.slow_ma,
            'signal': self.current_signal
        })
        super().on_candle(candle)
        
    def log_debug_info(self):
        for signal in self.debug_signals:
            print(f"{signal['timestamp']}: Close={signal['close']:.2f}, "
                  f"Fast={signal['fast_ma']:.2f}, Slow={signal['slow_ma']:.2f}")
```

## Performance Profiling

### Identify Bottlenecks

```python
import cProfile
import pstats
from io import StringIO

profile = cProfile.Profile()
profile.enable()

results = backtest.run()

profile.disable()
s = StringIO()
ps = pstats.Stats(profile, stream=s).sort_stats('cumulative')
ps.print_stats(20)
print(s.getvalue())
```

## Memory Usage

### Track Memory Consumption

```python
import tracemalloc

tracemalloc.start()
results = backtest.run()
current, peak = tracemalloc.get_traced_memory()

print(f"Current: {current / 1024 / 1024:.2f} MB")
print(f"Peak: {peak / 1024 / 1024:.2f} MB")
```

## Logging Strategy

### Structured Logging

```python
import logging

logger = logging.getLogger('btquant.strategy')
logger.setLevel(logging.DEBUG)

logger.debug(f"Signal generated: {signal}")
logger.info(f"Position opened: {position}")
logger.warning(f"Unusual drawdown: {drawdown:.2%}")
logger.error(f"Strategy validation failed: {error}")
```
