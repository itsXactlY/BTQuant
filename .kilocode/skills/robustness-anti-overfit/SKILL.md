---
name: robustness-anti-overfit
description: Enforce anti-overfitting, stress tests, and hard risk gates for BTQuant strategies.
---

# Robustness & Anti-Overfitting – BTQuant

This skill defines how agents must **validate** and **stress test** strategies so they don't
optimize themselves into extinction.

---

## 1. Evaluation Principles

Agents must ensure:

- No look-ahead bias,
- Proper time-based splits,
- Sufficient trade/sample count,
- Sensible capacity assumptions.

---

## 2. Time-Based Validation

### 2.1 Splits

- Use chronological splits:
  - Train: early period
  - Validation: middle
  - Test: latest
- For more complex setups:
  - Walk-forward (rolling) optimization and testing.

### 2.2 Cross-Asset Validation

- Strategies should be tested on:
  - multiple symbols,
  - multiple exchanges,
  where applicable.

If a strategy only works on one coin over one narrow period, treat as suspicious.

---

## 3. Stress Testing

Agents must simulate:

- Higher slippage (e.g. 2x–05x baseline),
- Wider spreads,
- Latency spikes,
- Reduced fill rates,
- Exchange outages or rejects on one leg of an arb.

Strategies that fail catastrophically under mild stress conditions should not be deployed.

---

## 4. Risk Gates & Thresholds

Examples of minimal production criteria:

- Sharpe (out-of-sample) above a threshold,
- Max drawdown below some absolute limit,
- Profit factor > 1.5,
- Sufficient trade count (e.g. 200+ trades over test period),
- Reasonable turnover and capacity.

Agents should encode such **gates** in BTQuant evaluation workflows.

---

## 5. Live vs Backtest Drift

Agents must design monitoring:

- Compare live performance vs backtest expectation:
  - risk-adjusted returns,
  - hit ratios,
  - slippage vs model assumptions.
- If drift exceeds thresholds (e.g. > 30% degradation):
  - reduce risk,
  - trigger re-evaluation,
  - possibly disable strategy.

---

## 6. Documentation of Robustness

Every strategy an agent proposes should have:

- A clear description of:
  - validation setup,
  - stress tests performed,
  - production gates,
- Explanation of where the strategy might **break** and how that is mitigated.
