---
name: testing-strategies
description: Comprehensive testing frameworks for trading strategies (code mode)
---

# Strategy Testing Framework (Code Mode)

When implementing and testing strategy code:

## Unit Test Template

```python
import pytest
import pandas as pd

class TestMovingAverageCrossoverStrategy:
    @pytest.fixture
    def strategy(self):
        return MovingAverageCrossoverStrategy(
            fast_period=12,
            slow_period=26,
            risk_percent=2.0
        )
    
    @pytest.fixture
    def sample_data(self):
        return pd.read_csv('fixtures/sample_ohlcv.csv')
    
    def test_initialization(self, strategy):
        assert strategy.fast_period == 12
        assert strategy.slow_period == 26
        assert strategy.risk_percent == 2.0
    
    def test_minimum_data_requirement(self, strategy, sample_data):
        min_required = strategy.slow_period + 1
        assert len(sample_data) >= min_required
    
    def test_signal_generation(self, strategy, sample_data):
        signals = strategy.generate_signals(sample_data)
        assert len(signals) == len(sample_data)
        assert signals.isin([0, 1, -1]).all()
    
    def test_position_sizing(self, strategy):
        size = strategy.calculate_position_size(risk_amount=100)
        assert size > 0
        assert isinstance(size, float)
```

## Integration Test Template

```python
def test_full_backtest_cycle():
    data = load_test_data('fixtures/full_year_data.csv')
    assert len(data) > 250
    
    strategy = MovingAverageCrossoverStrategy()
    backtest = Backtest(strategy, data, initial_capital=10000)
    results = backtest.run()
    
    assert results['total_trades'] > 0
    assert results['ending_balance'] > 0
    assert results['sharpe_ratio'] is not None
    assert 'equity_curve' in results
    assert -1 <= results['max_drawdown'] <= 0
    assert 0 <= results['win_rate'] <= 1
```

## Edge Case Testing

```python
def test_insufficient_data():
    strategy = MovingAverageCrossoverStrategy(slow_period=100)
    df_short = pd.read_csv('fixtures/sample.csv').head(50)
    assert strategy.validate(df_short) == False

def test_missing_values():
    df = pd.read_csv('fixtures/sample.csv')
    df.loc[0:10, 'close'] = None
    assert strategy.detect_anomaly(df)
```
