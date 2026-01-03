---
name: strategy-evaluation
description: Comprehensive strategy evaluation frameworks and metrics
---

# BTQuant Strategy Evaluation Framework

When evaluating strategy performance:

## Evaluation Metrics

### Return Metrics
- **Total Return**: `(final_balance - initial_balance) / initial_balance`
- **CAGR**: Compound Annual Growth Rate
- **Monthly Returns**: Distribution and consistency
- **Return Per Trade**: Average profit/loss per trade

### Risk Metrics
- **Volatility**: Standard deviation of returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Drawdown Duration**: Recovery time
- **Value at Risk (VaR)**: 95th percentile loss

### Risk-Adjusted Returns
- **Sharpe Ratio** (preferred): Return / Volatility
- **Sortino Ratio**: Return / Downside Volatility
- **Calmar Ratio**: CAGR / Max Drawdown
- **Recovery Factor**: Total Return / Max Drawdown

### Trade Statistics
- **Total Trades**: Sample size
- **Win Rate**: % profitable
- **Profit Factor**: Gross Profit / Gross Loss (>1.5 good)
- **Expectancy**: Average profit per trade
- **Consecutive Wins/Losses**: Risk of ruin

## Evaluation Process

### Phase 1: In-Sample Analysis

```python
results = backtest.run(strategy, data_in_sample)
evaluator = Evaluator(results)
report = evaluator.generate_report()

print(f"Sharpe: {report['sharpe_ratio']:.2f}")
print(f"Max DD: {report['max_drawdown']:.2%}")
print(f"Win Rate: {report['win_rate']:.2%}")
```

### Phase 2: Out-of-Sample Validation

- Test on 20-30% held-out data
- Different date ranges
- Check if metrics degrade > 20%

### Phase 3: Robustness Testing

- Parameter sensitivity: ±10% variation
- Market regime testing: Bull, bear, sideways
- Symbol variation
- Timeframe variation

## Minimum Thresholds for Production

- **Sharpe Ratio**: > 1.5 (good), > 2.0 (excellent)
- **Win Rate**: > 45%
- **Profit Factor**: > 1.5
- **Max Drawdown**: < 30%
- **In-Sample/Out-Sample Gap**: < 20%
- **Minimum Trades**: > 100

## Red Flags

- Gap between in-sample and out-of-sample > 20%
- Win rate near 100% (overfitting)
- All profits from few trades
- Performance degradation over time
