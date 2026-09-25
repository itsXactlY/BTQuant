---
name: regime-detection-meta
description: Build regime-aware meta-strategies and controllers for BTQuant that adapt which strategies run and how they are risked.
---

# Regime Detection & Meta-Strategies – BTQuant

This skill describes how agents should build **regime detectors** and **meta-controllers** that decide:

- Which strategies are active,
- How much risk each gets,
- When to shut down or shift style.

The goal is **regime-aware allocation**, not tuning static indicator parameters.

---

## 1. Regime Features

Agents should design feature sets that capture:

### 1.1 Volatility & Trend Regimes

- Realized volatility (short, medium, long horizons)
- Directional bias:
  - Rolling returns over multiple windows
- Volatility-of-volatility

### 1.2 Liquidity & Microstructure Regimes

- Average spread per symbol/venue
- Orderbook depth at top levels
- Trade size distribution, presence of "toxic flow"
- Occurrence of outlier moves, gaps, and spikes

### 1.3 Structural Crypto Regimes

- Funding rates (sign, magnitude, persistence)
- Perp–spot basis levels and volatility
- Cross-venue spread stability
- Market-wide stress signals (e.g. correlated crashes, risk-off behavior)

---

## 2. Regime Labeling

Agents should:

- Either:
  - Define **rule-based** regimes (e.g. "high-vol + wide spread"),
  - Or learn **unsupervised clusters** (e.g. k-means on regime features).
- Typical labels:
  - `TRENDING_HIGH_VOL`
  - `MEAN_REVERT_LOW_VOL`
  - `ILLQUID_WIDE_SPREAD`
  - `CARRY_FRIENDLY` (stable basis + funding)

Labeling must be performed with **causal information only** (no future leakage).

---

## 3. Meta-Controller Logic

Instead of "one strategy for all seasons", agents should:

- Maintain a **set of sub-strategies**:
  - Microstructure alpha
  - Cross-venue arb
  - Carry/funding strategies
  - Longer-horizon directional / factor strategies
- For each regime:
  - Define **which strategies are active**,
  - Assign **risk budgets** (position limits, leverage, capital allocation).

Example:

- `TRENDING_HIGH_VOL`:
  - Downweight mean-reversion
  - Upweight trend-following and market-making with wider spreads
- `ILLQUID_WIDE_SPREAD`:
  - Strongly reduce participation
  - Only run safest carry/arb, or even go flat

---

## 4. RL / Bandit Approaches

Advanced agents may implement:

- Contextual bandits:
  - Input: regime features
  - Output: which strategy mix to run
- RL:
  - Policy that chooses actions like "increase risk on microstructure alpha by X%"
  - Reward: risk-adjusted PnL over evaluation window

These must be trained with **realistic transaction costs and constraints**.

---

## 5. BTQuant Integration Pattern

Conceptual BTQuant pattern:

```python
class RegimeManager:
    def __init__(self, feature_builder, classifier_or_policy):
        self.feature_builder = feature_builder
        self.policy = classifier_or_policy

    def on_timer(self, now):
        features = self.feature_builder.build(now)
        regime = self.policy.predict_regime(features)
        self.apply_regime(regime)

    def apply_regime(self, regime):
        # Adjust which BTQuant strategies are enabled
        # and their risk budgets.
        ...
```

Agents should:
- Keep regime logic in a separate module,
- Make regime decisions relatively infrequently (e.g. every N minutes/hours),
- Encode explicit hysteresis to avoid over-switching.

---

## 6. Guardrails

Regime-aware control must:
- Never disable all safety mechanisms,
- Respect hard portfolio limits even in "favorable" regimes,
- Detect when data/labels are stale and fall back to conservative defaults.
