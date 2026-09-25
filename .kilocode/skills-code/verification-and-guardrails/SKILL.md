---
name: verification-and-guardrails
description: Implement tests and safety guardrails for BTQuant strategies and systems (code mode).
---

# Verification & Guardrails – BTQuant (Code Mode)

This skill focuses on **tests and safety mechanisms**:

- Unit/integration tests for risk logic,
- Kill-switches,
- Sanity checks.

---

## 1. Risk Logic Tests

Agents should write tests that verify:

- Position limits are enforced,
- Margin and leverage constraints are respected,
- No position sizes above defined caps can be opened,
- Strategies cannot bypass risk checks.

---

## 2. Kill Switch & Safe State

Tests should:

- Simulate conditions where kill-switch should trigger:
  - extreme drawdown,
  - connectivity failure,
  - repeated order rejects.
- Verify that:
  - no further orders are sent,
  - positions are flattened where possible,
  - system enters a safe state.

---

## 3. Sanity & Consistency Checks

Agents should enforce:

- No negative balances,
- PnL accounting consistency across modules,
- No orphaned positions or unhedged legs in arb strategies.

---

## 4. CI Integration

These tests should run in CI/CD so that:

- Any change to strategies or risk modules must pass safety tests,
- Bad modifications are caught early.
