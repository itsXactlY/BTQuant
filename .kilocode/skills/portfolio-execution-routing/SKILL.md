---
name: portfolio-execution-routing
description: Design smart execution and routing for BTQuant across venues, minimizing slippage and risk.
---

# Portfolio Execution & Routing – BTQuant

This skill explains how agents should design execution and routing logic that:

- Splits orders across venues,
- Minimizes slippage + fees,
- Respects inventory and risk constraints.

---

## 1. Execution Objectives

Agents must consider:

- **Total cost** = price impact + fees + spread + opportunity cost
- **Risk**:
  - inventory and exposure limits,
  - latency and failure risk per venue.

---

## 2. Smart Routing Across Venues

### 2.1 Cost Model

For each venue:

- Estimate:
  - effective spread,
  - fee schedule (maker/taker),
  - expected slippage at given size,
  - latency.

Agents should maintain a **cost model** that can answer:

> If I want to buy X units now, what is expected all-in cost at each venue?

### 2.2 Splitting Logic

- Favor venues with:
  - deeper books,
  - lower fees,
  - lower latency,
  - more reliable connectivity.
- Possibly mix:
  - Passive orders on deep venues,
  - Aggressive orders on faster venues when needed.

---

## 3. Execution Algorithms

### 3.1 TWAP / VWAP Variants

Agents can:

- Implement time-sliced execution (TWAP),
- Volume-weighted execution (VWAP),
- But adjust slices based on **microstructure signals** (from `microstructure-alpha`):
  - If toxic flow detected → slow down, reduce passive exposure
  - If favorable flow → accelerate fills

### 3.2 Opportunistic Passive Execution

- Place passive orders where microstructure alpha suggests positive EV,
- Cancel when:
  - queue position is poor,
  - adverse flow emerges,
  - regime changes.

---

## 4. Inventory & Risk Management

Agents must:

- Track inventory per symbol, per venue, and aggregate.
- Enforce:
  - maximum position size,
  - maximum net exposure,
  - diversification constraints.

Execution layer should be **aware of**:

- Portfolio context:
  - not just single order, but overall risk.

---

## 5. BTQuant Integration Pattern

Conceptual pattern:

```python
class ExecutionRouter:
    def __init__(self, venues, cost_model, risk_manager):
        self.venues = venues
        self.cost_model = cost_model
        self.risk_manager = risk_manager

    def execute_target_delta(self, symbol, target_delta, urgency, context):
        # 1. Consult risk manager
        if not self.risk_manager.allows(symbol, target_delta):
            return

        # 2. Decide routing and algo
        plan = self.cost_model.build_plan(symbol, target_delta, urgency, context)

        # 3. Submit and manage child orders on multiple venues
        self._submit_plan(plan)
```

Strategies should:
- Request target delta or inventory, not micro-manage orders.
- Let execution router decide order placement and routing.

---

## 6. Failure Handling

Agents must design:
- Retry logic with backoff,
- Venue blacklisting when error rate exceeds threshold,
- Fallback behaviors:
  - flatten positions,
  - reduce trading intensity,
  - temporarily disable aggressive strategies.
- Execution errors must never escalate into unbounded risk.
