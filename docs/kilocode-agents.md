# Using Kilo Code Agents with BTQuant – Advanced How-To

This guide explains how to use the **new Kilo Code skills suite** with BTQuant to build
**bleeding-edge trading strategies**, not basic TA toys.

It assumes:

- You already have BTQuant and its dependencies installed,
- You are comfortable with git, Python, and crypto market structure.

---

## 1. Setup

### 1.1 Install Kilo Code CLI

```bash
npm install -g @kilocode/cli
```

Verify installation:

```bash
kilocode --version
```

### 1.2 Clone BTQuant & Checkout Skills Branch

```bash
git clone https://github.com/itsXactlY/BTQuant.git
cd BTQuant
git checkout add-kilo-agent-skills  # or your updated branch
```

Ensure `.kilocode/` exists with the new skills structure:

```bash
ls -la .kilocode/skills/
# Should show: microstructure-alpha/, cross-venue-arbitrage/, regime-detection-meta/, etc.

ls -la .kilocode/skills-code/
# Should show: hft-debugging-microstructure/, simulation-and-replay/, verification-and-guardrails/
```

---

## 2. How Kilo Skills Work (Conceptual)

Kilo Code's skills system lets you provide domain-specific knowledge to the agent.

Each skill is a `.kilocode/skills/<name>/SKILL.md` (or `skills-code/` for code mode).

When you run `kilocode` in the BTQuant repo, the agent:

1. Reads your codebase,
2. Loads SKILL.md content as instructions and patterns,
3. Uses them when planning changes.

This is how you **"teach" the agent** to think in terms of:
- Microstructure and orderflow,
- Cross-venue arbitrage,
- Reinforcement learning and meta-controllers,
- Deep learning models,
- Robust evaluation,
- Production-grade architecture.

Instead of naive indicators and copy-paste indicators.

---

## 3. Typical Workflows

### 3.1 Designing a New Microstructure Alpha Strategy

#### Start Kilo in BTQuant

```bash
cd /path/to/BTQuant
kilocode
```

#### Describe Your Task

```text
> Design and implement a new BTCUSDT microstructure alpha strategy for Binance, using tickdata
> and orderbook L2, integrated with BTQuant. Use the microstructure-alpha skill. It should
> produce short-horizon signals (50-200ms) and send risk-aware target deltas to the execution layer.
> Include queue imbalance, book slope, and aggressive flow features.
```

#### What the Agent Will Do

The agent will:

1. Read `microstructure-alpha/SKILL.md`,
2. Understand that you want event-driven features, not simple indicators,
3. Propose:
   - Where to put the new strategy class (e.g., `strategies/microstructure/binance_btcusdt.py`),
   - How to structure feature buffers and on-demand computation,
   - Latency and cost assumptions (Binance-specific maker/taker fees, typical spreads),
   - Risk budgets and position sizing,
4. Generate skeleton code with:
   - Event handlers (`on_orderbook()`, `on_trade()`, `on_timer()`),
   - Feature computation helpers,
   - Model inference wrapper (if ML-based) or handcrafted scoring,
   - Action decision and risk checks.

#### Review and Iterate

Use `/mode debug` if needed:

```text
> /mode debug
> Add detailed logging to the feature buffer and order submission paths.
> Ensure we can reproduce exact fill patterns in replay mode.
```

Ask for further refinement:

```text
> Integrate this with our CCAPI feed for real-time orderbook snapshots.
> Use the portfolio-execution-routing skill to handle order placement across Binance spot and futures.
```

---

### 3.2 Building a Cross-Venue Arbitrage Strategy

#### Start in Architect Mode

```bash
cd /path/to/BTQuant
kilocode --mode architect
```

#### Describe the Task

```text
> Implement a cross-venue basis and funding arbitrage engine between Binance and OKX perps and
> spot in BTQuant. Use cross-venue-arbitrage, portfolio-execution-routing, and robustness-anti-overfit
> skills. It must respect inventory and margin limits per venue. Expected latency between fills
> should be <500ms.
```

#### Agent's Approach

The agent will:

1. Use `cross-venue-arbitrage/SKILL.md` to:
   - Define basis calculation (`perp_price - spot_price`) for each venue pair,
   - Integrate funding rates and borrow costs,
   - Design entry/exit rules (rich basis → short perp, long spot; cheap basis → opposite).

2. Use `portfolio-execution-routing/SKILL.md` to:
   - Build cost models per venue (fees, typical slippage, latency),
   - Split orders intelligently (more volume on lower-fee venue),
   - Manage hedging order routing (submit spot/perp legs with coordination).

3. Use `robustness-anti-overfit/SKILL.md` to:
   - Design backtest with proper time-based splits (train on 2023, validate on 2024, test on 2025),
   - Stress scenarios (OKX-specific outage, 2x latency spikes, funding rate gaps),
   - Live monitoring of basis and realized P&L vs backtest expectations.

#### Implementation Structure

Expect code like:

```python
# btquant/strategies/cross_venue_arb/basis_arbitrage.py
class BasisArbitrageStrategy(BTQStrategy):
    def __init__(self, venues, spot_symbol, perp_symbol, config):
        self.spot_venue = ...  # e.g. 'binance'
        self.perp_venue = ...  # e.g. 'okx'
        self.basis_threshold = config['basis_threshold_bps']
        self.cost_model = CostModel(venues)
        self.risk_manager = InventoryManager(config['max_notional'])

    def on_price_update(self, venue, symbol, price, bid_depth, ask_depth):
        # Recalculate basis, check if it exceeds threshold
        basis = self._calc_basis()
        if abs(basis) > self.basis_threshold:
            self._maybe_trade(basis)

    def _maybe_trade(self, basis):
        if not self.risk_manager.allows_trade():
            return
        # Use portfolio-execution-routing to decide order placement
        routing_plan = self.cost_model.build_plan(...)
        self._execute(routing_plan)
```

---

### 3.3 Regime-Aware Meta-Controller

#### Task

```text
> Build a RegimeManager in BTQuant that uses regime-detection-meta and deep-crypto-ml skills to
> switch between microstructure alpha, cross-venue arb, and carry strategies. It should be fully
> config-driven and have clear production gates. Update risk budgets every 5 minutes based on
> current regime (HIGH_VOL, MEAN_REVERT, ILLIQUID, CARRY_FRIENDLY).
```

#### Skill Application

The agent will:

1. Use `regime-detection-meta/SKILL.md` to:
   - Define regime features (realized vol, spread, trade size distribution, funding persistence),
   - Implement rule-based or ML classifier (`detect_regime()` function),
   - Map regimes to strategy allocations (e.g., HIGH_VOL → reduce mean-reversion, increase trend).

2. Use `deep-crypto-ml/SKILL.md` to:
   - Optionally enhance regime classification with a trained TCN or Transformer,
   - Use as a contextual input to strategy risk multipliers.

#### Code Pattern

```python
# btquant/strategies/meta/regime_manager.py
class RegimeManager:
    def __init__(self, strategies, config):
        self.strategies = strategies  # {name: strategy_instance}
        self.regime_detector = RegimeDetector(config)
        self.regimes = ['HIGH_VOL', 'MEAN_REVERT', 'ILLIQUID', 'CARRY_FRIENDLY']
        self.regime_allocations = config['regime_allocations']  # regime -> risk budget mapping

    def on_timer(self, now):
        regime = self.regime_detector.detect(now)
        self._apply_regime(regime)

    def _apply_regime(self, regime):
        allocation = self.regime_allocations[regime]
        for strategy_name, budget_fraction in allocation.items():
            strategy = self.strategies[strategy_name]
            strategy.set_risk_budget(budget_fraction)
```

Example config (YAML):

```yaml
regime_allocations:
  HIGH_VOL:
    microstructure_alpha: 0.4
    carry_strategy: 0.1
    cross_venue_arb: 0.0  # disabled
  MEAN_REVERT:
    microstructure_alpha: 0.6
    carry_strategy: 0.4
    cross_venue_arb: 0.2
  ILLIQUID:
    microstructure_alpha: 0.0
    carry_strategy: 0.2
    cross_venue_arb: 0.1
  CARRY_FRIENDLY:
    microstructure_alpha: 0.2
    carry_strategy: 0.6
    cross_venue_arb: 0.4
```

---

### 3.4 Hardening & Guardrails

#### Task

```text
> Using robustness-anti-overfit and verification-and-guardrails, add evaluation pipelines and
> tests that ensure the new microstructure strategy has no look-ahead bias, passes walk-forward
> validation, and respects all risk limits. Integrate these tests into BTQuant's CI configuration.
```

#### Agent's Approach

The agent will:

1. Use `robustness-anti-overfit/SKILL.md` to:
   - Design a walk-forward backtest (train on rolling 3-month windows, test on next 1 month),
   - Add stress scenarios (2x latency, 50% reduced liquidity, funding rate gaps),
   - Define production gates (Sharpe > 1.0, max drawdown < 20%, profit factor > 1.5).

2. Use `verification-and-guardrails/SKILL.md` to:
   - Write unit tests for position limits and risk checks,
   - Test kill-switch triggers (drawdown threshold, repeated order rejects),
   - Add CI/CD checks (run tests on every commit to strategies/).

#### Generated Test Files

```python
# tests/test_risk_limits.py
def test_position_limit_enforced():
    strategy = MicrostructureAlphaStrategy(...)
    strategy.set_position_limit(10.0)  # 10 BTC max
    # Try to open 15 BTC position
    result = strategy.decide_action(huge_upside_signal)
    assert result is None  # Rejected

def test_max_notional_enforced():
    strategy = StrategyWithMultipleSymbols(...)
    strategy.set_max_notional(100000)  # 100k USDT
    # Try to open positions exceeding this
    # Should reject or reduce sizes

# tests/test_kill_switch.py
def test_kill_switch_on_drawdown():
    strategy = ...
    strategy.set_max_drawdown(0.15)  # 15% DD limit
    # Simulate losses
    # Verify kill-switch activates and no more orders are sent
```

---

## 4. Autonomous Mode in CI/CD

You can run Kilo Code in autonomous mode to perform tasks without interaction:

```bash
echo "Run full backtest and robustness evaluation for all microstructure strategies" \
  | kilocode --auto --timeout 1200
```

With the skills in place, the agent knows:

- Which metrics to compute (Sharpe, max DD, profit factor, etc.),
- How to perform time-based splits,
- What stress tests to include,
- When a strategy passes production gates.

Autonomous mode is powerful; ensure your auto-approval config **only allows safe actions**:

```bash
# .kilocode/config.json
{
  "auto_approve": {
    "allowed_commands": ["git", "pytest", "python", "bash"],
    "forbidden_patterns": ["rm -rf", "drop"],
    "require_approval_for": ["merge_to_main", "deploy_to_production"]
  }
}
```

---

## 5. Best Practices for Working with This Suite

### 5.1 Always Specify Relevant Skills

When asking for work, explicitly mention which skills to use:

```text
❌ "Build a trading strategy."
✅ "Build a BTCUSDT microstructure alpha using the microstructure-alpha skill."
```

### 5.2 Keep Strategies Modular

Build small, composable components:

- **Alpha module**: computes signals
- **Risk module**: enforces limits
- **Execution module**: handles order placement
- **Regime module**: decides which strategies are active

Do NOT mix these concerns in one class.

### 5.3 Demand Robust Evaluation

Ask the agent explicitly for:

```text
> Use walk-forward validation with 3-month rolling windows.
> Add stress tests for 2x latency and 50% liquidity reduction.
> Validate on multiple symbols and exchanges.
> Report live drift metrics (expected vs actual P&L, hit rates).
```

### 5.4 Treat ML as One Component, Not Magic

When using deep learning (from `deep-crypto-ml` skill):

```text
> Build a TCN model that predicts orderbook imbalance from the last 50 milliseconds of data.
> Use it alongside handcrafted microstructure features.
> Report feature importance and out-of-sample performance.
> Do NOT rely solely on the model; use it as one signal in a broader framework.
```

### 5.5 Review Diffs Like You Would Any PR

The agent is powerful but not omniscient. Use your own expertise:

```text
> Review the generated basis arbitrage code.
> Spot-check: Are margin requirements correct for OKX? Are liquidation risks considered?
> Is the latency assumption (500ms) realistic for our infra?
```

### 5.6 Test in Simulation Before Live

Use the `simulation-and-replay` skill:

```text
> Use discrete-event simulation to test the strategy under venue outage (Binance down for 10 seconds).
> What happens to unhedged positions?
```

---

## 6. Extending the Skills

If you want to encode new ideas as skills:

### 6.1 Create a New Skill Directory

```bash
mkdir -p .kilocode/skills/your-skill-name
```

### 6.2 Write SKILL.md

```markdown
---
name: your-skill-name
description: Brief description of what agents should think about when using this skill.
---

# Your Skill Name

## 1. Core Concepts
...

## 2. BTQuant Integration
...

## 3. Example Patterns
...
```

### 6.3 Reference in Tasks

```text
> Use your-skill-name skill to design ...
```

Over time, BTQuant becomes an environment where agents natively think in your preferred patterns:
**microstructure, cross-venue, RL, deep crypto ML, robust risk, and nothing stuck in 1970s TA.**

---

## 7. Troubleshooting

### Skill Not Found

Ensure `.kilocode/skills/your-skill-name/SKILL.md` exists:

```bash
find .kilocode -name "SKILL.md" | sort
```

### Agent Ignoring Skills

Make sure the skill directory structure is correct:

```
.kilocode/
├── skills/
│   ├── microstructure-alpha/
│   │   └── SKILL.md
│   └── ...
└── skills-code/
    └── ...
```

### Questions During Agent Planning

Be specific in your task description:

❌ "Build a strategy."
✅ "Build a BTCUSDT microstructure strategy targeting the top 1 level of Binance L2, with 200ms horizon."

---

## 8. Real-World Example: Full Workflow

### Step 1: Initialize Kilo

```bash
cd /path/to/BTQuant
kilocode
```

### Step 2: Design Microstructure Strategy

```text
> Using microstructure-alpha skill, design a BTCUSDT microstructure strategy for Binance.
> Signals should be based on queue imbalance and aggressive flow over 100-200ms horizons.
> Output a skeleton Python class in strategies/microstructure/.
> The class should integrate with BTQuant's orderbook and trade feeds.
```

### Step 3: Add Execution Integration

```text
> Using portfolio-execution-routing skill, extend the strategy to send target deltas
> to the execution layer instead of micro-managing orders. Include latency and cost estimates.
```

### Step 4: Harden with Evaluation

```text
> Using robustness-anti-overfit and verification-and-guardrails skills, add:
> - Walk-forward backtest (rolling 3-month windows)
> - Stress tests (2x latency, 50% less liquidity)
> - Production gates (Sharpe > 1.0, max DD < 20%)
> - Unit tests for risk limits and kill switches
```

### Step 5: Run Autonomous Backtest

```bash
echo "Run full backtest suite for microstructure strategies" | kilocode --auto
```

### Step 6: Deploy

If all tests pass, deploy to production with monitoring:

```bash
kilocode
> Deploy the microstructure strategy with live performance tracking.
> Compare realized P&L and hit rates to backtest expectations.
> Alert if drift exceeds 30%.
```

---

## Summary

With this skills suite, you:

- **Think at the right level**: Microstructure, arb, regimes, not indicators,
- **Let agents do the grunt work**: Code generation, testing, tuning,
- **Stay in control**: Review code, spot-check logic, veto bad ideas,
- **Build production systems**: Robust evaluation, risk limits, graceful degradation.

Good luck building the next generation of BTQuant trading systems!
