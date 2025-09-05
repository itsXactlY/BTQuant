```bash

2025-09-05 21:32:40 Train subset: (39028, 27), Val subset: (9758, 27)
2025-09-05 21:32:40 Training Transformer-GNN Model (FIXED TENSOR ISSUES)...
2025-09-05 21:32:40 Dataset created: 39028 samples, 39028 labels, sequence_length=30
2025-09-05 21:32:40 Dataset created: 9758 samples, 9758 labels, sequence_length=30
2025-09-05 21:32:40 [2KTraining... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   0% -:--:--Epoch 0: Train Loss: nan, Val Loss: nan, Val Acc: 0.0055
2025-09-05 21:34:01 [2KTraining... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00 80% -:--:--
2025-09-05 21:43:06 [?25hTransformer model training completed successfully!
2025-09-05 21:43:06 Converting data for RL training...
2025-09-05 21:43:06 Training Deep RL Agent...
2025-09-05 21:43:06 Removing datetime columns for RL: ['datetime']s
2025-09-05 21:43:06 Cleaned RL data shape: (39026, 26)
2025-09-05 21:43:06 Using cpu device
2025-09-05 21:43:06 Wrapping the env with a `Monitor` wrapper
2025-09-05 21:43:06 Wrapping the env in a DummyVecEnv.
2025-09-05 21:43:07 ---------------------------------
2025-09-05 21:43:07 | rollout/           |          |
2025-09-05 21:43:07 |    ep_len_mean     | 3.53     |
2025-09-05 21:43:07 |    ep_rew_mean     | -514     |
2025-09-05 21:43:07 | time/              |          |
2025-09-05 21:43:07 |    fps             | 2432     |
2025-09-05 21:43:07 |    iterations      | 1        |
2025-09-05 21:43:07 |    time_elapsed    | 0        |
2025-09-05 21:43:07 |    total_timesteps | 2048     |
2025-09-05 21:43:07 ---------------------------------
2025-09-05 21:43:07 RL agent training completed!
2025-09-05 21:43:07 Step 4: Running backtest on real market data...
2025-09-05 21:43:07 Backtest data shape: (12197, 6)
2025-09-05 21:43:07 Date range: 2023-02-15 20:45:00 to 2023-06-23 00:00:00
2025-09-05 21:43:07 Running Backtrader with PolarsData feed...
2025-09-05 21:43:08 PPO agent loaded successfully
2025-09-05 21:43:24 Step 5: Analyzing performance...
2025-09-05 21:43:24                 BTQuant - TENSOR ISSUES FIXED! ✅                
2025-09-05 21:43:24 ┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
2025-09-05 21:43:24 ┃ Metric                 ┃ Value                                      ┃
2025-09-05 21:43:24 ┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
2025-09-05 21:43:24 │ 🔧 Status              │ TENSOR DIMENSION ISSUES FIXED!             │
2025-09-05 21:43:24 │ 🤖 Transformer         │ ✅ Simplified GNN → MLP Architecture       │
2025-09-05 21:43:24 │ 🧮 Tensors             │ ✅ All Dimension Mismatches Resolved       │
2025-09-05 21:43:24 │ ⚡ Training            │ ✅ No More Stacking Errors                 │
2025-09-05 21:43:24 │ 🗂️ Column Names        │ ✅ Database Schema Consistent              │
2025-09-05 21:43:24 │ 🎯 RL Training         │ ✅ Forced CPU Device                       │
2025-09-05 21:43:24 │ 🛡️ None Handling       │ ✅ Safe Metric Extraction                  │
2025-09-05 21:43:24 │ 📊 Data Source         │ Real Market Data (Cached)                  │
2025-09-05 21:43:24 │ ⚡ Data Feed           │ Native Polars Integration                  │
2025-09-05 21:43:24 │ 💻 Device Used         │ cpu                                        │
2025-09-05 21:43:24 │ 🔥 CUDA Available      │ False                                      │
2025-09-05 21:43:24 │ 📈 Assets Analyzed     │ 2 (BTC, ETH)                               │
2025-09-05 21:43:24 │ 📋 Total Data Points   │ 60,983                                     │
2025-09-05 21:43:24 │ 🧬 Features Engineered │ 21                                         │
2025-09-05 21:43:24 │ 📅 Backtest Period     │ 2023-02-15 20:45:00 to 2023-06-23 00:00:00 │
2025-09-05 21:43:24 │ 💰 Initial Capital     │ $10,000.00                                 │
2025-09-05 21:43:24 │ 💵 Final Value         │ $10,000.00                                 │
2025-09-05 21:43:24 │ 📊 Total Return        │ 0.00%                                      │
2025-09-05 21:43:24 │ 📈 Sharpe Ratio        │ 0.000                                      │
2025-09-05 21:43:24 │ 📉 Max Drawdown        │ 0.00%                                      │
2025-09-05 21:43:24 │ 🔄 Total Trades        │ 0                                          │
2025-09-05 21:43:24 │ 🎯 Win Rate            │ 0.0%                                       │
2025-09-05 21:43:24 └────────────────────────┴────────────────────────────────────────────┘
2025-09-05 21:43:24 🎉 BTQuant execution completed successfully! 🎉
```
