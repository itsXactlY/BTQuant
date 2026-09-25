---
name: autonomous-workflows
description: Autonomous agent workflows for continuous strategy development
---

# Autonomous Workflow Patterns for BTQuant

When setting up autonomous agents to manage strategy development:

## Multi-Stage Development Pipeline

### Stage 1: Strategy Generation

Agent task: Generate new strategy variants

```
Create 3 new trading strategy variations based on:
1. Moving average crossover with dynamic periods
2. RSI mean reversion with volatility adjustment
3. MACD with trend confirmation

For each:
- Write to strategies/ directory
- Include parameter config in JSON
- Add docstring with logic explanation
- Make testable with sample data
```

### Stage 2: Strategy Validation

Agent task: Validate new strategies

```
For each new strategy file:
1. Check syntax and imports
2. Run linter (pylint/flake8)
3. Verify required methods exist
4. Test instantiation with dummy data
5. Check for obvious errors

Generate validation report
```

### Stage 3: Backtesting

Agent task: Run comprehensive backtest suite

```
For each validated strategy:
1. Test on 1-year BTCUSDT 1h candles
2. Test on 1-year ETHUSDT 1h candles
3. Grid search with 5 best param combos
4. Generate backtest report for each
5. Extract metrics to summary CSV
```

### Stage 4: Optimization

Agent task: Parameter optimization

```
For best strategies from Stage 3:
1. Run Bayesian optimization on parameters
2. Use 80% data for optimization
3. Validate on 20% out-of-sample
4. Check for overfitting
5. Generate optimization report
```

### Stage 5: Evaluation

Agent task: Multi-metric evaluation

```
For optimized strategies:
1. Cross-validate on multiple periods
2. Test on different assets
3. Stress test with higher slippage
4. Test different market regimes
5. Compare with baselines

Evaluation criteria:
- Sharpe > 1.5?
- Win rate > 45%?
- Profit factor > 1.5?
```

### Stage 6: Documentation

Agent task: Create strategy documentation

```
For approved strategies:
1. Generate markdown documentation
2. Include strategy logic explanation
3. List parameters and optimal values
4. Provide usage examples
5. Include performance summary statistics
```

## Continuous Monitoring

### Performance Tracking

```
Daily:
1. Fetch latest trades from broker
2. Calculate daily returns and Sharpe
3. Compare with backtest expectations
4. Check for regime changes
5. Alert if metrics degrade > 20%

Weekly:
1. Generate performance report
2. Update strategy rankings
3. Identify underperformers
4. Recommend optimizations
```

### Automated Retraining

```
When detected:
1. Performance degradation > 25%
2. Market regime change
3. New data window (weekly)
4. Parameter drift

Action:
1. Run mini-optimization on recent data
2. Compare new vs old parameters
3. Backtest new parameters
4. Decide: update or alert
```

## Success Criteria for Autonomous Agents

- **Fully automated**: Zero human intervention
- **Robust**: Handles errors gracefully
- **Observable**: Clear logging
- **Reproducible**: Same inputs = same outputs
- **Efficient**: Completes in reasonable time
- **Documented**: Clear output for review
- **Safe**: Cannot execute trades without approval
