# Exit-aware Multi-Task Neural Trading System (V3)

This module adds a complete V3 scaffold for:

- Feature reduction via permutation importance + correlation pruning.
- Exit-aware labeling (MFE/MAE + TP/SL events + timing/volatility targets).
- Multi-task neural architectures (Transformer/CNN-LSTM).
- Unified vectorized backtesting with model-driven exits.
- Walk-forward validation for robust OOS metrics.
- Optional RL-compatible exit-agent interfaces.

## Quickstart

```bash
python neural_trading_system/scripts/run_feature_selection.py
python neural_trading_system/scripts/run_walk_forward.py
```

## Status

This implementation is production-structured and test-backed, with RL PPO training intentionally left as optional extension points.
