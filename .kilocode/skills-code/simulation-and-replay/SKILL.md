---
name: simulation-and-replay
description: Build and enhance discrete-event simulators and replay frameworks for BTQuant (code mode).
---

# Simulation & Replay – BTQuant (Code Mode)

This skill describes how agents should design **simulation and replay tools**:

- Discrete-event simulation,
- Scenario testing,
- Regression testing.

---

## 1. Discrete-Event Simulation

Agents can build simulators that:

- Step through event queues:
  - orderbook updates,
  - trades,
  - user orders,
  - exchange responses.
- Evaluate strategy performance under:
  - different latency assumptions,
  - different fee schedules,
  - artificial "what-if" environments.

---

## 2. Scenario Frameworks

Agents should define scenarios like:

- Venue outage,
- Latency spike on one venue,
- Liquidity drying up on specific symbol,
- Extreme volatility bursts.

Strategies must be tested across such scenarios before being considered robust.

---

## 3. Integration with BTQuant

Simulation frameworks should:

- Use the same core strategy code as production,
- Provide hooks to:
  - override latency,
  - override book state,
  - inject errors.

This enables **realistic testing without touching production connectors**.
