# Backtesting Guide

This comprehensive guide covers backtesting in BTQuant, including basic backtests, optimization, analysis, and best practices for robust strategy validation.

## Table of Contents

- [Basic Backtesting](#basic-backtesting)
- [Optimization](#optimization)
- [Performance Analysis](#performance-analysis)
- [Walk-Forward Analysis](#walk-forward-analysis)
- [Risk Metrics](#risk-metrics)
- [Benchmarking](#benchmarking)
- [Common Pitfalls](#common-pitfalls)
- [Best Practices](#best-practices)

## Basic Backtesting

### Simple Backtest

```python
from backtrader import backtest
from backtrader.strategies.Vumanchu_A import VuManchCipher_A
from backtrader.utils.ccxt_data import get_crypto_data

# Get data
data = get_crypto_data('BTC/USDT', '2024-01-01', '2024-12-31', '1h', 'binance')

# Run backtest
result = backtest(
    strategy=VuManchCipher_A,
    data=data,
    init_cash=10000,
    commission=0.001,  # 0.1% commission
    backtest=True
)

print(f"Final Portfolio Value: ${result:.2f}")
```

### Advanced Backtest Configuration

```python
result = backtest(
    strategy=MyStrategy,
    data=data,
    init_cash=100000,
    commission=0.00075,  # Binance spot commission
    margin=1.0,          # No leverage
    stake=100,           # Fixed stake per trade
    quantstats=True,     # Generate QuantStats report
    plot=True,           # Generate plots
    asset_name='BTC/USDT'
)
```

### Multi-Asset Backtesting

```python
from backtrader.utils.backtest import bulk_backtest

# Test strategy on multiple assets
coins = ['BTC', 'ETH', 'ADA', 'SOL', 'DOT']
results = bulk_backtest(
    strategy=MyStrategy,
    coins=coins,
    start_date='2024-01-01',
    end_date='2024-12-31',
    interval='1d',
    init_cash=10000,
    max_workers=4  # Parallel processing
)

# Analyze results
for coin, result in results.items():
    print(f"{coin}: ${result:.2f}")
```

### Custom Commission Models

```python
import backtrader as bt

# Custom commission for different markets
class CustomCommission(bt.CommInfoBase):
    def _getcommission(self, size, price, pseudoexec):
        # Volume-based commission
        if size * price > 10000:  # Large trades
            return abs(size) * price * 0.0005  # 0.05%
        else:
            return abs(size) * price * 0.001   # 0.1%

# Use in backtest
result = backtest(
    strategy=MyStrategy,
    data=data,
    init_cash=10000,
    commission=CustomCommission()
)
```

## Optimization

### Parameter Optimization

```python
from backtrader.utils.backtest import optimize_backtest

# Optimize strategy parameters
results = optimize_backtest(
    strategy=RSIStrategy,
    data=data,
    init_cash=10000,
    rsi_period=[10, 14, 21, 28],
    rsi_overbought=[65, 70, 75],
    rsi_oversold=[25, 30, 35],
    take_profit=[1.0, 2.0, 3.0],
    stop_loss=[3.0, 5.0, 7.0],
    max_workers=8
)

# Get best result
best_result = results[0]
print(f"Best parameters: {best_result['params']}")
print(f"Best return: {best_result['return']:.2f}%")
print(f"Best Sharpe: {best_result['sharpe']:.2f}")
```

### Advanced Optimization

```python
# Multi-objective optimization
results = optimize_backtest(
    strategy=MyStrategy,
    data=data,
    init_cash=10000,
    # Parameter ranges
    fast_period=range(5, 25, 5),
    slow_period=range(20, 50, 10),
    # Optimization criteria
    opt_criteria='sharpe',  # Maximize Sharpe ratio
    opt_direction='max',
    # Constraints
    max_drawdown_limit=0.2,  # Max 20% drawdown
    min_trades=10           # Minimum 10 trades
)
```

### Optimization Analysis

```python
import pandas as pd

# Convert results to DataFrame
df_results = pd.DataFrame(results)

# Analyze parameter sensitivity
import seaborn as sns
import matplotlib.pyplot as plt

# Heatmap of returns by parameters
pivot_table = df_results.pivot_table(
    values='return',
    index='fast_period',
    columns='slow_period',
    aggfunc='mean'
)

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn')
plt.title('Parameter Optimization Heatmap')
plt.show()
```

### Genetic Algorithm Optimization

```python
from deap import base, creator, tools, algorithms
import random

def genetic_optimize(strategy_class, data, generations=50, population_size=100):
    """Genetic algorithm optimization"""

    # Define fitness function
    def evaluate(individual):
        params = {
            'fast_period': individual[0],
            'slow_period': individual[1],
            'rsi_period': individual[2]
        }

        result = backtest(strategy_class, data, init_cash=10000, **params)
        return result['sharpe'],  # Maximize Sharpe

    # Genetic algorithm setup
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("fast_period", random.randint, 5, 25)
    toolbox.register("slow_period", random.randint, 20, 50)
    toolbox.register("rsi_period", random.randint, 10, 30)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                    (toolbox.fast_period, toolbox.slow_period, toolbox.rsi_period))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Run optimization
    population = toolbox.population(n=population_size)
    algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2,
                       ngen=generations, verbose=True)

    # Get best individual
    best_ind = tools.selBest(population, 1)[0]
    return {
        'fast_period': best_ind[0],
        'slow_period': best_ind[1],
        'rsi_period': best_ind[2]
    }
```

## Performance Analysis

### QuantStats Integration

```python
# Generate comprehensive performance report
result = backtest(
    strategy=MyStrategy,
    data=data,
    init_cash=10000,
    quantstats=True,
    benchmark='BTC'  # Compare against BTC benchmark
)

# The report will be saved as: QuantStats/MyStrategy_BTC_2024-01-01_12-00-00.html
```

### Custom Performance Metrics

```python
def calculate_advanced_metrics(result):
    """Calculate additional performance metrics"""

    returns = result['returns']
    drawdowns = result['drawdowns']

    # Calmar Ratio (annual return / max drawdown)
    annual_return = returns.mean() * 252
    max_drawdown = drawdowns.max()
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0

    # Sortino Ratio (similar to Sharpe but only penalizes downside volatility)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino_ratio = annual_return / downside_std if downside_std > 0 else 0

    # Win Rate
    winning_trades = len([t for t in result['trades'] if t['pnl'] > 0])
    total_trades = len(result['trades'])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    # Profit Factor
    gross_profit = sum([t['pnl'] for t in result['trades'] if t['pnl'] > 0])
    gross_loss = abs(sum([t['pnl'] for t in result['trades'] if t['pnl'] < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return {
        'calmar_ratio': calmar_ratio,
        'sortino_ratio': sortino_ratio,
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }
```

### Risk-Adjusted Returns

```python
import numpy as np

def calculate_risk_metrics(returns, risk_free_rate=0.02):
    """Calculate comprehensive risk metrics"""

    # Basic metrics
    total_return = (1 + returns).prod() - 1
    annual_return = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)

    # Sharpe Ratio
    excess_returns = returns - risk_free_rate/252
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = drawdowns.min()

    # Value at Risk (95% confidence)
    var_95 = np.percentile(returns, 5)

    # Expected Shortfall (CVaR)
    cvar_95 = returns[returns <= var_95].mean()

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'cvar_95': cvar_95
    }
```

## Walk-Forward Analysis

### Basic Walk-Forward

```python
def walk_forward_analysis(strategy_class, data, train_window=252, test_window=21):
    """Perform walk-forward analysis"""

    results = []
    n_periods = len(data)

    for i in range(train_window, n_periods - test_window, test_window):
        # Training data
        train_data = data.iloc[i-train_window:i]

        # Test data
        test_data = data.iloc[i:i+test_window]

        # Optimize on training data
        opt_results = optimize_backtest(
            strategy=strategy_class,
            data=train_data,
            init_cash=10000,
            fast_period=[10, 15, 20],
            slow_period=[30, 40, 50]
        )

        # Get best parameters
        best_params = opt_results[0]['params']

        # Test on out-of-sample data
        test_result = backtest(
            strategy=strategy_class,
            data=test_data,
            init_cash=10000,
            **best_params
        )

        results.append({
            'train_end': data.index[i-1],
            'test_start': data.index[i],
            'test_end': data.index[i+test_window-1],
            'params': best_params,
            'return': test_result['return'],
            'sharpe': test_result['sharpe']
        })

    return results
```

### Anchored Walk-Forward

```python
def anchored_walk_forward(strategy_class, data, anchor_date, test_window=63):
    """Anchored walk-forward analysis"""

    results = []
    anchor_idx = data.index.get_loc(anchor_date)

    # Start from anchor point
    for i in range(anchor_idx + test_window, len(data), test_window):
        # Training data: from start to anchor
        train_data = data.iloc[:anchor_idx+1]

        # Test data: anchor + test_window
        test_start = i - test_window
        test_end = i
        test_data = data.iloc[test_start:test_end]

        # Optimize and test
        opt_results = optimize_backtest(strategy_class, train_data)
        best_params = opt_results[0]['params']

        test_result = backtest(strategy_class, test_data, **best_params)

        results.append({
            'anchor_date': anchor_date,
            'test_period': f"{data.index[test_start]} to {data.index[test_end-1]}",
            'params': best_params,
            'return': test_result['return']
        })

    return results
```

### Walk-Forward Efficiency

```python
def calculate_walk_forward_efficiency(wf_results):
    """Calculate walk-forward efficiency metrics"""

    returns = [r['return'] for r in wf_results]

    # Annualized return
    total_return = np.prod([1 + r for r in returns]) - 1
    years = len(wf_results) * (21/252)  # Assuming 21 trading days per test period
    annualized_return = (1 + total_return) ** (1/years) - 1

    # Sharpe ratio of walk-forward returns
    wf_sharpe = np.mean(returns) / np.std(returns) * np.sqrt(12)  # Monthly Sharpe

    # Consistency ratio (percentage of positive periods)
    positive_periods = sum(1 for r in returns if r > 0)
    consistency_ratio = positive_periods / len(returns)

    return {
        'annualized_return': annualized_return,
        'walk_forward_sharpe': wf_sharpe,
        'consistency_ratio': consistency_ratio,
        'total_periods': len(wf_results)
    }
```

## Risk Metrics

### Portfolio Risk Analysis

```python
def portfolio_risk_analysis(returns, confidence_level=0.95):
    """Comprehensive portfolio risk analysis"""

    # Historical VaR
    var_historical = np.percentile(returns, (1 - confidence_level) * 100)

    # Parametric VaR (assuming normal distribution)
    mean_return = returns.mean()
    std_return = returns.std()
    var_parametric = mean_return + std_return * stats.norm.ppf(1 - confidence_level)

    # Monte Carlo VaR
    n_simulations = 10000
    mc_returns = np.random.normal(mean_return, std_return, n_simulations)
    var_monte_carlo = np.percentile(mc_returns, (1 - confidence_level) * 100)

    # Expected Shortfall (CVaR)
    tail_returns = returns[returns <= var_historical]
    expected_shortfall = tail_returns.mean()

    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = drawdowns.min()

    # Stress Testing
    stress_scenarios = {
        'mild_stress': returns * 1.5,      # 50% increase in volatility
        'severe_stress': returns * 2.0,    # 100% increase in volatility
        'crash_scenario': returns - 0.1   # 10% instant loss
    }

    stress_results = {}
    for scenario, stress_returns in stress_scenarios.items():
        stress_var = np.percentile(stress_returns, (1 - confidence_level) * 100)
        stress_results[scenario] = stress_var

    return {
        'var_historical': var_historical,
        'var_parametric': var_parametric,
        'var_monte_carlo': var_monte_carlo,
        'expected_shortfall': expected_shortfall,
        'max_drawdown': max_drawdown,
        'stress_testing': stress_results
    }
```

### Scenario Analysis

```python
def scenario_analysis(strategy_class, data, scenarios):
    """Test strategy under different market scenarios"""

    results = {}

    for scenario_name, scenario_data in scenarios.items():
        result = backtest(strategy_class, scenario_data, init_cash=10000)

        # Calculate metrics
        metrics = calculate_risk_metrics(result['returns'])

        results[scenario_name] = {
            'return': result['return'],
            'sharpe': result['sharpe'],
            'max_drawdown': result['max_drawdown'],
            'risk_metrics': metrics
        }

    return results

# Define scenarios
scenarios = {
    'bull_market': data[data['close'] > data['close'].shift(1)],  # Only up days
    'bear_market': data[data['close'] < data['close'].shift(1)],  # Only down days
    'high_volatility': data[data['close'].pct_change().abs() > 0.05],  # High vol days
    'low_volatility': data[data['close'].pct_change().abs() < 0.01],   # Low vol days
}

scenario_results = scenario_analysis(MyStrategy, data, scenarios)
```

## Benchmarking

### Benchmark Comparison

```python
def benchmark_comparison(strategy_returns, benchmark_returns):
    """Compare strategy performance against benchmark"""

    # Calculate metrics for both
    strategy_metrics = calculate_risk_metrics(strategy_returns)
    benchmark_metrics = calculate_risk_metrics(benchmark_returns)

    # Alpha (excess return)
    alpha = strategy_metrics['annual_return'] - benchmark_metrics['annual_return']

    # Beta (market sensitivity)
    covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)
    beta = covariance / benchmark_variance

    # Information Ratio
    tracking_error = np.std(strategy_returns - benchmark_returns)
    information_ratio = alpha / tracking_error if tracking_error > 0 else 0

    # Up/Down Market Capture
    up_market = benchmark_returns > 0
    down_market = benchmark_returns < 0

    up_capture = (strategy_returns[up_market].mean() /
                 benchmark_returns[up_market].mean()) if up_market.any() else 0

    down_capture = (strategy_returns[down_market].mean() /
                   benchmark_returns[down_market].mean()) if down_market.any() else 0

    return {
        'alpha': alpha,
        'beta': beta,
        'information_ratio': information_ratio,
        'up_market_capture': up_capture,
        'down_market_capture': down_capture,
        'strategy_metrics': strategy_metrics,
        'benchmark_metrics': benchmark_metrics
    }
```

### Rolling Performance Analysis

```python
def rolling_performance_analysis(returns, window=252):
    """Analyze rolling performance metrics"""

    rolling_metrics = []

    for i in range(window, len(returns)):
        window_returns = returns[i-window:i]

        metrics = calculate_risk_metrics(window_returns)
        metrics['date'] = returns.index[i-1]

        rolling_metrics.append(metrics)

    return pd.DataFrame(rolling_metrics)
```

## Common Pitfalls

### 1. Overfitting
```python
# Avoid: Too many parameters
# Good: Limit parameters and use walk-forward analysis

def check_overfitting(results_df, n_params=3):
    """Check for overfitting indicators"""

    # Degrees of freedom ratio
    n_observations = len(results_df)
    dof_ratio = n_params / n_observations

    # Parameter stability
    param_stability = results_df.groupby('fast_period')['return'].std().mean()

    # Out-of-sample performance
    in_sample_best = results_df['return'].max()
    oos_performance = results_df['oos_return'].mean()

    return {
        'dof_ratio': dof_ratio,
        'param_stability': param_stability,
        'oos_vs_is_ratio': oos_performance / in_sample_best if in_sample_best > 0 else 0
    }
```

### 2. Look-Ahead Bias
```python
# Avoid: Using future data
# Good: Ensure all calculations use only past data

def check_look_ahead_bias(strategy_func, data):
    """Test for look-ahead bias"""

    # Run strategy normally
    normal_result = backtest(strategy_func, data)

    # Run with shifted data (simulate future leak)
    shifted_data = data.shift(1)  # Shift by one period
    shifted_result = backtest(strategy_func, shifted_data)

    # Compare results - should be very different if no look-ahead bias
    difference = abs(normal_result['return'] - shifted_result['return'])

    return difference < 0.01  # Should be False for good strategies
```

### 3. Survivorship Bias
```python
# Include delisted assets in backtests
# Test on assets that existed throughout the period

def survivorship_bias_test(strategy_func, all_assets_data, active_assets_data):
    """Test for survivorship bias"""

    # Test on only currently active assets
    active_result = bulk_backtest(strategy_func, active_assets_data)

    # Test on all assets (including delisted)
    all_result = bulk_backtest(strategy_func, all_assets_data)

    # Compare performance
    active_return = np.mean(list(active_result.values()))
    all_return = np.mean(list(all_result.values()))

    survivorship_bias = active_return - all_return

    return {
        'active_only_return': active_return,
        'all_assets_return': all_return,
        'survivorship_bias': survivorship_bias
    }
```

## Best Practices

### 1. Data Quality
- Use clean, gap-free data
- Handle missing data appropriately
- Validate data integrity before backtesting

### 2. Walk-Forward Analysis
- Always use walk-forward optimization
- Test on out-of-sample data
- Use anchored walk-forward for stability

### 3. Risk Management
- Calculate appropriate risk metrics
- Test under various market conditions
- Include transaction costs and slippage

### 4. Performance Evaluation
- Use multiple metrics (Sharpe, Sortino, Calmar)
- Compare against appropriate benchmarks
- Analyze drawdowns and recovery time

### 5. Robustness Testing
- Test parameter sensitivity
- Use Monte Carlo simulation
- Validate on different markets/timeframes

### 6. Documentation
- Document all assumptions
- Record parameter ranges tested
- Save optimization results

### 7. Automation
- Automate backtesting pipelines
- Use version control for strategies
- Implement continuous testing

This comprehensive backtesting guide provides the foundation for robust, reliable strategy validation in BTQuant.