---
name: strategy-optimization
description: Parameter optimization and hyperparameter tuning for BTQuant strategies
---

# BTQuant Strategy Optimization

When optimizing strategy parameters:

## Optimization Methods

### Grid Search

```python
from btquant import Optimizer

params_grid = {
    'fast_ma_period': range(5, 20, 2),
    'slow_ma_period': range(20, 50, 5),
    'risk_percent': [1.0, 1.5, 2.0, 2.5]
}

optimizer = Optimizer.grid_search(
    strategy=MyStrategy,
    params=params_grid,
    data_source=backtest_data,
    metric='sharpe_ratio',
    workers=4
)
```

### Bayesian Optimization

```python
params_dist = {
    'fast_ma_period': {'type': 'int', 'min': 5, 'max': 20},
    'slow_ma_period': {'type': 'int', 'min': 20, 'max': 50}
}

optimizer = Optimizer.bayesian(
    strategy=MyStrategy,
    params=params_dist,
    data_source=backtest_data,
    metric='sharpe_ratio',
    n_trials=50
)
```

## Optimization Strategy

1. **Coarse tuning**: Wide ranges, quick evaluations
2. **Fine tuning**: Narrow around best performers
3. **Validation**: Test on held-out data
4. **Sensitivity analysis**: Check robustness

## Robustness Testing

- Test on different market regimes
- Use walk-forward optimization
- Test on out-of-sample data
- Vary start/end dates
- Check across symbols and timeframes

## Objective Functions

- **Sharpe Ratio**: Risk-adjusted returns (recommended)
- **Sortino Ratio**: Downside-deviation adjusted
- **Profit Factor**: Gross profit / gross loss
- **Custom**: Weighted combinations for multi-metric

## Parameter Constraints

- Enforce logical relationships (fast < slow)
- Set realistic bounds
- Use penalty functions for invalid combinations
- Document why constraints exist
