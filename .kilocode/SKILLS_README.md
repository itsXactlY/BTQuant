# BTQuant Kilo Code Skills Suite

Comprehensive agent skills for autonomous strategy development, testing, and evaluation using Kilo Code CLI.

## Overview

This skills suite empowers autonomous AI agents to manage the complete trading strategy development lifecycle in BTQuant.

## Generic Skills (All Modes)

Available in all Kilo Code modes:

### 1. **strategy-development**
- Strategy architecture patterns
- Parameter management
- Signal generation best practices
- Risk management implementation
- Multi-exchange support

**Use case**: When creating new trading strategies from scratch

### 2. **strategy-backtesting**
- Backtest setup and configuration
- Data sourcing from MSSQL, CCXT, CSV
- Performance metrics calculation
- Out-of-sample validation
- Pitfall avoidance

**Use case**: When running comprehensive strategy backtests

### 3. **strategy-optimization**
- Grid search optimization
- Bayesian optimization
- Random search
- Robustness testing
- Parameter constraints

**Use case**: When tuning strategy parameters for optimal performance

### 4. **strategy-evaluation**
- Comprehensive evaluation metrics
- Return and risk analysis
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Trade statistics
- Minimum production thresholds

**Use case**: When assessing strategy viability for deployment

### 5. **data-management**
- Data sourcing from multiple sources
- OHLCV validation
- Data preparation and resampling
- Feature engineering
- Storage and caching

**Use case**: When working with trading data

### 6. **exchange-integration**
- CCXT integration patterns
- API credential management
- Order management (market, limit, advanced)
- Position management
- Websocket streaming

**Use case**: When connecting to live exchanges for trading

### 7. **ml-models**
- Classification and regression models
- Feature engineering
- Training best practices
- Time-series validation
- Data leakage prevention

**Use case**: When building ML-based strategies

### 8. **autonomous-workflows**
- 6-stage development pipeline
- Continuous monitoring
- Automated retraining
- CI/CD workflows
- Success criteria

**Use case**: When setting up autonomous agent workflows

### 9. **code-quality**
- PEP 8 compliance
- Import organization
- Docstring standards
- Unit testing
- Error handling

**Use case**: When reviewing code quality

### 10. **performance-analysis**
- Return decomposition
- Risk metrics and volatility
- Drawdown analysis
- Risk-adjusted returns
- Trade analysis visualization

**Use case**: When analyzing strategy performance

### 11. **documentation**
- README templates
- Changelog formats
- API documentation
- Performance reports
- Change management

**Use case**: When documenting strategies

## Code Mode Skills

Specialized for code mode (`/mode code`):

### 1. **testing-strategies** (skills-code)
- Unit test templates
- Integration test patterns
- Edge case testing
- Parametrized tests
- Test fixtures

**Use case**: When writing test suites for strategies

### 2. **advanced-debugging** (skills-code)
- Signal tracing instrumentation
- Performance profiling
- Memory usage tracking
- Structured logging
- Error tracking

**Use case**: When debugging complex strategy behavior

## Directory Structure

```
.kilocode/
├── skills/                          # Generic skills (all modes)
│   ├── strategy-development/SKILL.md
│   ├── strategy-backtesting/SKILL.md
│   ├── strategy-optimization/SKILL.md
│   ├── strategy-evaluation/SKILL.md
│   ├── data-management/SKILL.md
│   ├── exchange-integration/SKILL.md
│   ├── ml-models/SKILL.md
│   ├── autonomous-workflows/SKILL.md
│   ├── code-quality/SKILL.md
│   ├── performance-analysis/SKILL.md
│   └── documentation/SKILL.md
│
├── skills-code/                     # Code mode specific skills
│   ├── testing-strategies/SKILL.md
│   └── advanced-debugging/SKILL.md
│
└── SKILLS_README.md                 # This file
```

## Quick Start

### 1. Install Kilo Code CLI

```bash
npm install -g @kilocode/cli
```

### 2. Start Interactive Session

```bash
cd /path/to/BTQuant
kilocode
```

### 3. Use Skills in Tasks

The agent will automatically have access to these skills and apply them to your tasks:

```
> "Generate 3 new moving average crossover strategy variations"
> "Run comprehensive backtest on the new strategies"
> "Optimize parameters using Bayesian search"
> "Evaluate strategies against production criteria"
```

### 4. Switch Modes

```
/mode code        # For writing tests and debugging
/mode architect   # For strategy design
/mode orchestrator # For complex workflows
```

## Example Workflows

### Workflow 1: Strategy Development + Optimization

```
1. Create new strategy using strategy-development skill
2. Run backtest with strategy-backtesting
3. Optimize parameters with strategy-optimization
4. Evaluate with strategy-evaluation
5. Document with documentation skill
```

### Workflow 2: Autonomous Development Pipeline

```
1. Agent generates strategy variants (autonomous-workflows)
2. Validates code quality (code-quality)
3. Runs backtests (strategy-backtesting)
4. Optimizes parameters (strategy-optimization)
5. Performs evaluation (strategy-evaluation)
6. Generates reports (documentation)
```

### Workflow 3: ML Strategy Development

```
1. Design ML model (ml-models)
2. Engineer features (data-management)
3. Train with time-series validation (ml-models)
4. Backtest strategy (strategy-backtesting)
5. Optimize hyperparameters (strategy-optimization)
6. Analyze performance (performance-analysis)
```

## Skills Discovery

List all available skills:

```bash
kilocode /mode architect
# Skills auto-loaded and available
```

View specific skill:

```bash
# Skills are automatically referenced in agent context
# Agents know to apply strategy-development when creating strategies
# Agents know to apply strategy-backtesting when running tests
```

## Configuration

Skills are configured via YAML frontmatter in each SKILL.md:

```yaml
---
name: strategy-development          # Skill identifier
description: ...                    # Skill purpose
---
```

Skills in `skills-code/` are only loaded in code mode.
Skills in `skills/` are available in all modes.

## Best Practices

### For Autonomous Agents

1. **Always validate strategies** before backtesting (code-quality)
2. **Always test out-of-sample** (strategy-evaluation)
3. **Check for overfitting** (strategy-optimization)
4. **Monitor for regime changes** (autonomous-workflows)
5. **Document everything** (documentation)

### For Strategy Development

1. Use **strategy-development** for architecture
2. Implement **code-quality** standards from start
3. Add **testing-strategies** from the beginning
4. Run **strategy-backtesting** frequently
5. Apply **strategy-optimization** iteratively

### For Production Deployment

1. Meet **strategy-evaluation** thresholds
2. Pass **code-quality** review
3. Complete **performance-analysis** review
4. Create comprehensive **documentation**
5. Set up **autonomous-workflows** for monitoring

## Integration with CI/CD

Use with GitHub Actions:

```yaml
- name: Run Kilo Code Strategy Tests
  run: |
    kilocode --auto "Run full backtest and optimization pipeline"
```

## Contributing

To add new skills:

1. Create directory: `.kilocode/skills/skill-name/`
2. Create `SKILL.md` with YAML frontmatter
3. Include practical code examples
4. Document use cases
5. Add to this README

## Support

For Kilo Code CLI documentation, visit: [https://kilo.ai/docs/cli](https://kilo.ai/docs/cli)

For BTQuant documentation, visit: [https://github.com/itsXactlY/BTQuant](https://github.com/itsXactlY/BTQuant)
