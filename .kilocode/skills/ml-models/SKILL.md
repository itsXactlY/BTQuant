---
name: ml-models
description: Machine learning model integration for BTQuant strategy development
---

# BTQuant ML Model Integration

When developing ML-based trading strategies:

## Model Types

### Classification Models

```python
from sklearn.ensemble import RandomForestClassifier

X = df[['sma_20', 'sma_50', 'rsi', 'macd']]
y = (df['close'].shift(-1) > df['close']).astype(int)

model = RandomForestClassifier(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)

pred = model.predict(X_test)
prob = model.predict_proba(X_test)
```

### Regression Models

```python
from sklearn.ensemble import GradientBoostingRegressor

y = df['close'].pct_change().shift(-1)

model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1
)
model.fit(X_train, y_train)
```

## Feature Engineering

### Technical Indicators

```python
import ta

df['rsi'] = ta.momentum.rsi(df['close'], window=14)
df['macd'] = ta.trend.macd_diff(df['close'])
df['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
```

### Statistical Features

```python
df['returns'] = df['close'].pct_change()
df['volatility'] = df['returns'].rolling(20).std()
df['skewness'] = df['returns'].rolling(20).skew()
df['kurtosis'] = df['returns'].rolling(20).kurt()
```

## Training Best Practices

### Train/Validation/Test Split

```python
# Time series split (NOT random)
train_size = int(0.6 * len(df))
val_size = int(0.2 * len(df))

X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]
X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]
```

## Avoiding Data Leakage

- Only use data available at prediction time
- Apply feature scaling on train, transform on test
- Avoid future information in features
- Use expanding windows for training data
- Separate train/val/test chronologically

## Integration with BTQuant

```python
class MLStrategy(Strategy):
    def __init__(self, model, feature_cols):
        self.model = model
        self.feature_cols = feature_cols
        
    def on_candle(self, candle):
        features = self.calculate_features()
        signal = self.model.predict(features)[0]
        
        if signal and not self.position:
            self.buy()
        elif not signal and self.position:
            self.sell()
```
