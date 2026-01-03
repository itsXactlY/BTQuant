---
name: cross-venue-arbitrage
description: Design BTQuant strategies for cross-venue and cross-instrument arbitrage, basis, and funding edges.
---

# Cross-Venue Arbitrage – BTQuant

This skill explains how agents should design **cross-exchange and cross-instrument arbitrage**
strategies on top of BTQuant, including:

- Latency arbitrage
- Perp–spot–futures basis trades
- Funding-rate arbitrage
- Cross-venue spread and synthetic pairs

---

## 1. Core Concepts

### 1.1 Instruments & Venues

Typical crypto arb setup in BTQuant:

- Venues: Binance, OKX, Bybit, MEXC, etc.
- Instruments:
  - Spot pairs (e.g. BTC/USDT)
  - Perpetual futures (e.g. BTCUSDT perp)
  - Delivery futures (quarterly, etc.)

Agents should maintain:

- A **unified symbol map** (BTQuant's mapping between venue-specific tickers)
- Per-venue **fees, tick sizes, lot sizes, and latency characteristics**

---

## 2. Latency Arbitrage

### 2.1 Quote Staleness

Agents should:

- Track **best bid/ask** per venue and estimate:
  - Age of the quote (last update timestamp)
  - Discrepancies across venues

Idea: if venue A's quote is stale vs venue B's more recent move, there may be a short-lived arb.

### 2.2 Execution Logic

- Only attempt latency arb if:
  - BTQuant latency + order round-trip < expected decay time of mispricing
  - Size is appropriately small vs book depth
- Execution decisions:
  - Take liquidity on "lagging" venue, hedge on leading venue
  - Enforce strict **max holding time** before forced flatten

---

## 3. Basis & Funding Arbitrage

### 3.1 Perp–Spot Basis

Define:

- `basis = perp_price - spot_price`
- Normalize vs spot: `basis_pct = (perp - spot) / spot`

Agents should:

- Monitor basis over time per venue
- Integrate funding rate schedule for perps
- Combine:
  - Current basis
  - Expected funding payments
  - Borrow/shorting costs (if any)

### 3.2 Strategy Skeleton

- If **basis is rich** (perp overpriced):
  - Short perp, long spot (or basket)
- If **basis is cheap** (perp underpriced):
  - Long perp, short spot

Constraints:

- Borrow capacity / margin limits
- Maximum leverage per venue
- Liquidation & funding risk controls

---

## 4. Cross-Venue Spread Trading

Agents should consider:

- Price discrepancies in the **same perp** across exchanges
- Liquidity differences:
  - Use more liquid venue as "anchor"
- Patterns:
  - Enter long on cheaper venue, short on more expensive venue
  - Flatten when spread mean-reverts or when risk/latency conditions change

BTQuant responsibilities:

- Maintain synchronized prices across venues
- Provide stable connectivity and tickdata for spread calculations

---

## 5. Risk, Capacity & Guardrails

### 5.1 Inventory & Exposure

Strategies must:

- Limit net exposure per asset and per venue
- Track hedged vs unhedged leg risk:
  - What happens if one venue rejects orders or goes down?

### 5.2 Slippage & Fee Accounting

- Use conservative estimates for:
  - Markout after a trade
  - Fees (maker/taker)
  - Borrow rates and funding

### 5.3 Failure Modes

Agents must consider:

- Venue halts / delistings
- Drastic changes to fee schedules or funding formulas
- Liquidity evaporation in stress scenarios

Arb strategies are not free money; this skill ensures agents encode realistic assumptions in BTQuant.
