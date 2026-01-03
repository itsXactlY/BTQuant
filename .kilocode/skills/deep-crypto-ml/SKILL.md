---
name: deep-crypto-ml
description: Use modern deep learning for crypto alpha in BTQuant, beyond naive OHLCV LSTMs.
---

# Deep Crypto ML – BTQuant

This skill guides agents to design **deep learning models** for crypto trading that:

- Use multiple data sources (orderbook, trades, altdata),
- Respect time-series structure,
- Integrate properly with BTQuant.

---

## 1. Data Modalities

Agents should consider at least:

- **Orderbook snapshots (L2)**:
  - Price/size at multiple levels
- **Trades**:
  - Aggressor side, size, price, venue
- **OHLCV**:
  - Higher timeframe aggregates
- **Altdata** (if available):
  - Sentiment/flow, funding, on-chain metrics

These should be aligned into **multichannel sequences**.

---

## 2. Architectures

### 2.1 Temporal ConvNets / TCN

Use 1D convolutions or TCNs for sequences of:

- Orderbook imbalance,
- Trade imbalance,
- Realized volatility features.

Advantages:

- Good for local patterns,
- Efficient at high frequency.

### 2.2 Transformers / Attention

Use attention mechanisms for:

- Longer-horizon dependencies,
- Combining different feature groups:
  - Microstructure
  - Altdata
  - Regime features

Example conceptual structure:

- Encode orderflow with TCN
- Encode altdata with separate encoder
- Fuse via Transformer / attention layer

---

## 3. Targets & Losses

Examples:

- Predict **distribution** of short-horizon returns:
  - Output mean/variance or quantiles; trade using distribution-aware rules
- Predict **probability of large move** (> threshold) up/down
- Predict **value of taking specific action** (Q-values for RL)

Losses:

- Cross-entropy for classification
- Quantile loss for VaR-style targets
- Policy gradient or actor–critic for RL

---

## 4. Training Practices

Agents must:

- Use **time-based splits**:
  - Train → validation → test in chronological order
- Avoid leakage:
  - No future info in features
  - Scaling/normalization fit only on training data
- Use **walk-forward retraining** in production designs.

Cross-asset validation is strongly encouraged:
- Model should not only work on 1 symbol over 1 period.

---

## 5. Integration into BTQuant

Models should be:

- Saved/loaded via a clear interface in BTQuant (e.g. `models/` directory),
- Wrapped in strategy classes that:
  - Build features,
  - Call `model.predict(...)`,
  - Translate predictions into **risk-aware trade instructions** (not blind buy/sell).

Agents must consider:

- Inference latency,
- Feature-engineering cost,
- Robustness when models see out-of-distribution data.

---

## 6. Non-Stationarity & Lifespan

Agents must encode:

- **model versioning** and **deployment windows**,
- logic for:
  - rolling retraining,
  - performance monitoring,
  - automatic decommissioning of models whose live performance deviates heavily from backtests.
