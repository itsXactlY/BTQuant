---
name: microstructure-alpha
description: Design BTQuant strategies that extract alpha from orderbook microstructure and flow, not legacy TA.
---

# Microstructure Alpha – BTQuant

This skill describes how agents should think about **short-horizon alpha** for BTQuant using modern
market microstructure and orderflow, especially for crypto CEXs and perps.

The goal is **NOT** "MA crossover"; it is to exploit **asymmetries in the limit order book and trade flow**.

---

## 1. Data & Features

### 1.1 Core Microstructure Inputs

Agents should rely primarily on:

- Level 2 orderbook snapshots (bid/ask ladders)
- Trade prints (aggressor side, size, price)
- Spreads and depth at multiple levels
- Cancel/replace events (when available from BTQuant's tickdata)

Typical BTQuant data abstractions:

- `orderbook[t]` per symbol/venue: levels of (price, size) on bid/ask
- `trades[t]` stream: `(timestamp, price, size, side, venue)`
- `top_of_book[t]`: best bid/ask, spread, mid

### 1.2 Example Feature Families

Agents should design features like:

- **Queue imbalance** at top N levels:
  - `QI = (BidSize1..N - AskSize1..N) / (BidSize1..N + AskSize1..N)`
- **Book slope / convexity**
  - Depth distribution away from mid on each side
- **Spread & micro-spread dynamics**
  - Spread widening/narrowing rates
- **Aggressive flow imbalance**
  - Net volume of aggressor buys vs sells over last X ms
- **Cancellation bursts**
  - Rapid depletion of one side of the book, especially at/near the best price
- **Short-horizon realized volatility**
  - `RV` over last N ticks or last M seconds

Agents should treat these as **feature vectors over short horizons**, not as standalone indicators.

---

## 2. Targets & Alpha Formulation

### 2.1 Predictive Targets

For short-horizon microstructure strategies, natural targets include:

- **Signed mid-price move** over next N ticks or milliseconds:
  - `target = sign(mid[t+Δ] - mid[t])`
- **Return bucket**:
  - `target ∈ {-2, -1, 0, +1, +2}` (strong down → strong up)
- **Spread regime / micro-vol regime**:
  - classify whether spread is about to widen or narrow materially

### 2.2 From Prediction to Trading Logic

Agents should map predictions to actions via:

- **Threshold policies**:
  - Only act when predicted edge exceeds a cost-adjusted threshold
- **Risk-budgeted position sizing**:
  - Position size is a function of predicted edge, volatility, and risk limits
- **Maker vs taker choice**:
  - For "mild" predicted moves → try to **provide liquidity** at best bid/ask or slightly inside
  - For "strong + urgent" predictions → **take liquidity** if EV > cost

---

## 3. Strategy Structure in BTQuant

### 3.1 Example Skeleton (Conceptual)

```python
# Pseudocode, not tied to exact BTQuant API signatures

class MicrostructureAlphaStrategy(BTQStrategy):
    def __init__(self, venues, symbol, horizon_ms=200, risk_budget=0.01, model=None):
        super().__init__(venues, symbol)
        self.horizon_ms = horizon_ms
        self.risk_budget = risk_budget
        self.model = model  # could be ML, could be handcrafted score
        self.feature_buffer = FeatureBuffer(window_ms=2000)

    def on_orderbook(self, venue, book_snapshot):
        # Update feature buffer with latest L2
        self.feature_buffer.update_orderbook(venue, book_snapshot)

    def on_trade(self, venue, trade):
        # Update feature buffer with flow and aggressor info
        self.feature_buffer.update_trade(venue, trade)

    def on_timer(self, now):
        # Called at high frequency (e.g. every 50ms)
        features = self.feature_buffer.build_features(now, horizon_ms=self.horizon_ms)
        edge = self.model.predict_edge(features)  # positive = upward edge, negative = downward

        action = self.decide_action(edge, now)
        if action is not None:
            self.execute(action, now)
```

Agents should adhere to event-driven architecture and keep:
- Feature computation in separate helpers,
- Model inference separate from execution policy.

---

## 4. Execution & Risk

### 4.1 Latency- and Cost-Aware

Microstructure alpha is usually small; agents must:
- Encode explicit estimates of:
  - Fees (maker/taker)
  - Slippage
  - Latency (time from signal to order on book)
- Only trade when ExpectedValue > costs + safety margin.

### 4.2 Inventory & Exposure Limits

Agents must enforce:
- Max position per symbol
- Max notional exposure across venues
- Time-based inventory decay (flatten if edge disappears)

---

## 5. What NOT To Do

Agents must avoid:
- Treating simple moving averages as the primary signal
- Ignoring orderbook state and flow in high-frequency setups
- Using unrealistic fill assumptions (must respect queue priority, slippage, latency)
- Ignoring exchange-specific tick size / lot size / fee model

This skill is about short-horizon microstructure edge in BTQuant, not retro indicator trading.
