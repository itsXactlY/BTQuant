---
name: altdata-sentiment-flow
description: Integrate sentiment and alternative data as context for BTQuant strategies, not as standalone toy signals.
---

# Altdata, Sentiment & Flow – BTQuant

This skill teaches agents to use **alternative data** (news, social, on-chain, funding) as
**context** to modulate strategies, not as simple "buy when Twitter bullish" rules.

---

## 1. Data Types

Potential sources:

- News streams (headlines, categories, sentiment scores),
- Social data (e.g. X/Telegram-derived sentiment),
- On-chain activity (addresses, flows, DeFi metrics),
- Exchange-level flow (funding changes, open interest spikes).

Agents must treat these as **additional features**, not guaranteed alpha.

---

## 2. Use Cases

### 2.1 Volatility & Event Flags

- Use altdata to predict:
  - probability of volatility spike,
  - elevated risk of gaps and gappy orderbooks.

Strategies may:

- Reduce size before expected event risk,
- Tighten risk limits when sentiment indicates potential stress.

### 2.2 Regime & Context Features

- Feed aggregated sentiment/altdata into:
  - `regime-detection-meta` as extra regime features,
  - `deep-crypto-ml` models as contextual embeddings.

---

## 3. Integration Patterns

Agents should:

- Create **separate data ingestion and preprocessing** modules in BTQuant,
- Align altdata timestamps to trading data:
  - handle delays,
  - avoid future leakage (no using news headlines released after the trade time).

Feature examples:

- Rolling sentiment scores per asset,
- Counts of "extreme sentiment" events,
- Funding rate changes labeled as positive/negative regime shifts.

---

## 4. Risk Considerations

Altdata is noisy and adversarial:

- Treat as **weak signal**,
- Use primarily to modulate:
  - aggressiveness,
  - exposure,
  - stop levels,
  instead of direct trade triggers.

Agents must document how altdata is used and what happens when it becomes unavailable.
