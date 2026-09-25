# BTQuant Kilo Code Skills – Next-Gen Suite

This document describes the **next-generation skills suite** for Kilo Code agents working on BTQuant.

The focus is on **microstructure, cross-venue arbitrage, deep learning, regime-aware control,
and robustness**, not simple 1970s-style TA.

---

## Skills Overview

### Core Skills (`.kilocode/skills/`)

#### `microstructure-alpha`
Design short-horizon alpha from orderbook and trade flow.
- Queue imbalance, book slope, aggressive flow analysis
- Threshold-based execution policies
- Realistic latency and cost assumptions

#### `cross-venue-arbitrage`
Latency arb, perp–spot basis, funding arb, and cross-exchange spreads.
- Quote staleness tracking
- Basis and funding calculations
- Risk-aware hedging logic

#### `regime-detection-meta`
Regime features and meta-controllers for strategy selection and risk allocation.
- Volatility, liquidity, and structural crypto regimes
- Rule-based and unsupervised classification
- Adaptive risk budgeting per regime

#### `deep-crypto-ml`
Modern deep learning architectures for crypto (TCN, Transformers, multi-modal).
- Temporal ConvNets for orderflow patterns
- Attention-based fusion of microstructure + altdata
- Walk-forward validation and live monitoring

#### `portfolio-execution-routing`
Smart execution and routing across venues, with risk-aware cost minimization.
- Cost models per venue
- TWAP/VWAP with microstructure awareness
- Opportunistic passive execution

#### `robustness-anti-overfit`
Anti-overfitting, stress tests, and production gates.
- Time-based splits and cross-asset validation
- Stress scenarios (latency spikes, outages, liquidity drying)
- Live drift monitoring vs backtest

#### `altdata-sentiment-flow`
Use altdata as context (sentiment, on-chain, funding), not toy signals.
- Volatility and event flagging
- Regime features and contextual embeddings
- Failure handling for data unavailability

#### `btquant-architecture-patterns`
Architecture patterns for modular, production-grade BTQuant systems.
- Separation of concerns (alpha, risk, execution)
- Event-driven design and config-driven strategies
- Multi-strategy systems and production readiness

---

### Code Mode Skills (`.kilocode/skills-code/`)

#### `hft-debugging-microstructure`
Debugging microstructure strategies (queue position, fills, markout).
- Tick replay utilities
- Order timeline inspection
- Queue position estimation

#### `simulation-and-replay`
Building replay and simulation frameworks to test strategies.
- Discrete-event simulation
- Scenario frameworks (outages, latency spikes, etc.)
- Integration with BTQuant strategy code

#### `verification-and-guardrails`
Tests and safety mechanisms (risk limits, kill switches, consistency checks).
- Risk logic unit/integration tests
- Kill switch and safe state verification
- CI/CD integration

---

## Getting Started

1. **Install Kilo Code**: `npm install -g @kilocode/cli`
2. **Use in BTQuant**: `cd /path/to/BTQuant && kilocode`
3. **Reference Skills**: Include skill names in your requests, e.g.
   ```
   > Design a microstructure alpha strategy using microstructure-alpha skill...
   ```

---

## Design Philosophy

- **Event-driven**: Strategies respond to orderbook and trade events, not static timers
- **Cost-aware**: Every decision encodes realistic fees, slippage, and latency
- **Regime-adaptive**: Single "one-size-fits-all" strategy is not realistic
- **Robustly tested**: Walk-forward validation, stress tests, and live monitoring
- **Modular**: Alpha, risk, and execution are separate concerns
- **Config-driven**: Easy to tune and deploy without code changes

Agents using this suite build trading systems that reflect the reality of crypto markets:
high-frequency, fragmented, regime-dependent, and adversarial.

---

## Next Steps

See `docs/kilocode-agents.md` for an extensive **How-To Guide** with workflows and examples.
