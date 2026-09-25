---
name: btquant-architecture-patterns
description: Use BTQuant-native architecture patterns for modular strategies, risk, and execution.
---

# BTQuant Architecture Patterns

This skill focuses on how agents should **structure code and components** in BTQuant for advanced
strategies, ensuring modularity, composability, and safety.

---

## 1. Separation of Concerns

Agents should keep:

- **Alpha modules**:
  - compute signals (microstructure, arb, ML, etc.)
- **Risk modules**:
  - enforce limits, caps, portfolio constraints.
- **Execution modules**:
  - handle order placement, routing, monitoring.

No mixing of these concerns in a single monolithic class.

---

## 2. Event-Driven Design

BTQuant strategies should be built around events:

- `on_orderbook(venue, snapshot)`
- `on_trade(venue, trade)`
- `on_timer(now)`
- `on_risk_event(event)`
- etc.

Agents should:

- Avoid polling-style logic,
- Use asynchronous streaming interfaces where available.

---

## 3. Config-Driven Strategies

Strategies should:

- Take their configuration from **external config** (YAML/JSON),
- Avoid hard-coded parameters in code.

Configs include:

- Symbols, venues,
- Risk budgets,
- Model/file paths,
- Regime thresholds.

This enables:

- Automated generation & tuning by agents,
- Easier deployment and rollback.

---

## 4. Multi-Strategy Systems

Agents should be comfortable designing:

- A **top-level controller** that manages:
  - multiple alpha strategies (microstructure, arb, carry, ML, etc.),
  - per-strategy risk budgets,
  - cross-strategy conflict resolution.

---

## 5. Production Readiness

BTQuant code must:

- Log all critical events,
- Expose metrics for monitoring (PnL, risk, latency, errors),
- Support graceful shutdown and recovery.

Agents should propose code that can be **run in real systems**, not just toy scripts.
