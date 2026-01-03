---
name: hft-debugging-microstructure
description: Advanced debugging patterns for microstructure and HFT-style BTQuant strategies (code mode).
---

# HFT & Microstructure Debugging – BTQuant (Code Mode)

This skill covers debugging patterns for **microsecond–millisecond-level** logic:

- Order placement vs book state,
- Queue position,
- Fill patterns,
- Slippage and markout.

---

## 1. Tick Replay

Agents should:

- Build replay utilities that:
  - load recorded orderbook/trade streams from BTQuant's data store,
  - step through them deterministically,
  - run the strategy against historical streams.

This enables:

- Exact reproducibility of bugs,
- Fast testing of microstructure ideas.

---

## 2. Order Timeline Inspection

For each order:

- Track:
  - submit time,
  - ACK time,
  - when it appeared in local book model,
  - fill or cancel time,
  - realized slippage and markout.

Agents should create tools to:

- Print summarized timelines,
- Group order behavior by venue, symbol, strategy.

---

## 3. Queue Position & Priority

Agents should:

- Estimate where their passive orders sit in the queue,
- Debug cases where expected fills did not occur.

Patterns:

- Compare expected vs actual fill rates,
- Identify systematic misestimation of queue position.

---

## 4. Diagnostics & Logging

Agents must:

- Instrument critical paths with **structured logs**:
  - order decision,
  - features used,
  - model outputs,
  - risk state.

This should be configuration-driven so it can be enabled for debugging and disabled in production.
