# Neural Trading System Handbook

Your stupid-simple guide to running, understanding, and extending the bleeding-edge neural stack that powers `neural_trading_system`. This handbook distills the pipeline into repeatable steps so you can ship changes without reverse‑engineering the codebase every time.

---

## 1. Core Concepts at a Glance
- **Transparency-first data capture** – Backtrader is patched so every intermediate indicator value is recorded bar-by-bar. No black boxes, no missing context.【F:neural_trading_system/neural_pipeline.py†L11-L199】【F:dependencies/backtrader/TransparencyPatch.py†L6-L153】
- **Feature factory on rails** – The `NeuralDataPipeline` handles collection, caching, feature extraction, and return targets with one config dict.【F:neural_trading_system/neural_pipeline.py†L360-L520】
- **Context-aware labels** – Take-profit/stop-loss/hold labels are inferred from regime-aware statistics, not hard thresholds.【F:neural_trading_system/training/context_aware_labels.py†L1-L92】
- **Transformer + regime intelligence** – A market-specific transformer encoder works alongside a VAE regime detector and regime-change heads so you always see *what* the network thinks the market state is.【F:neural_trading_system/models/architecture.py†L1-L200】【F:neural_trading_system/models/architecture.py†L202-L318】
- **End-to-end tooling** – Training, inference, backtesting, and diagnostics scripts are pre-wired. You focus on ideas, not glue code.【F:neural_trading_system/README.md†L120-L205】【F:neural_trading_system/run_neural_backtest.py†L1-L160】

---

## 2. Quickstart Checklist
1. **Collect & cache indicator data**
   ```bash
   python neural_trading_system/neural_pipeline.py
   ```
   *Runs the patched Backtrader collector, exports a date-scoped Parquet, and caches the feature matrix for reuse.*【F:neural_trading_system/neural_pipeline.py†L372-L455】
2. **Train or fine-tune the model**
   ```bash
   python neural_trading_system/train_elite_model.py
   ```
   *Loads cached features, builds context-aware labels, sanitizes inputs, and optimizes the multi-task transformer.*【F:neural_trading_system/training/trainer.py†L1-L152】【F:neural_trading_system/training/context_aware_labels.py†L1-L92】
3. **Inspect what the network learned**
   ```bash
   python neural_trading_system/analyze_model.py
   ```
   *Generates attention maps, feature attributions, regime plots, and calibration diagnostics.*【F:neural_trading_system/README.md†L156-L205】
4. **Backtest the neural strategy**
   ```bash
   python neural_trading_system/run_neural_backtest.py
   ```
   *Streams live indicator batches through the trained model with safe scaling and dashboards for PnL, risk, and signal quality.*【F:neural_trading_system/run_neural_backtest.py†L1-L200】

> **Pro tip:** All scripts auto-discover caches, model checkpoints, and feature extractors. Override paths only when you really need to.

---

## 3. Transparency Patch Deep Dive
- `TransparencyPatch` monkey-patches `bt.Indicator.__setattr__` so every indicator assignment is registered with its owning class and variable name.【F:dependencies/backtrader/TransparencyPatch.py†L23-L78】
- During each `next()` call, the patch pulls OHLCV + every captured indicator value into a batch-friendly dictionary, deduplicates identical tensors, and appends it to an in-memory buffer.【F:dependencies/backtrader/TransparencyPatch.py†L80-L146】
- `export_data()` flushes the buffer into a Polars DataFrame sorted by bar index, then stores Parquet + IPC copies for the feature extractor.【F:dependencies/backtrader/TransparencyPatch.py†L148-L205】

**Result:** The data collector strategy simply calls `capture_patch(self)` and you get 90+ indicator streams without touching Backtrader internals.【F:neural_trading_system/neural_pipeline.py†L200-L356】

---

## 4. Indicator Coverage (Out of the Box)
The `DataCollectionStrategy` instantiates **cycle, regime, volatility, momentum, trend, market cipher, volume, vortex, and misc** families. A few highlights:
- **Cycle & adaptive filters:** CyberCycle, Roofing, Adaptive Laguerre, iDecycler, iFisher, iTrend.【F:neural_trading_system/neural_pipeline.py†L98-L143】【F:neural_trading_system/neural_pipeline.py†L200-L290】
- **Regime & volatility:** Hurst exponent, Damiani Volatmeter, Squeeze Volatility, standardized ATR, Chaikin Volatility.【F:neural_trading_system/neural_pipeline.py†L107-L137】【F:neural_trading_system/neural_pipeline.py†L214-L244】
- **Momentum & oscillators:** RSX, QQE, Relative Momentum Index, WaveTrend, Laguerre RSI, TSI, DV2, Ultimate Oscillator, Awesome/Pretty Good Oscillators.【F:neural_trading_system/neural_pipeline.py†L127-L167】【F:neural_trading_system/neural_pipeline.py†L244-L276】
- **Trend & structure:** KAMA, MESA, Trend Trigger Factor, Schaff Trend, SuperTrend, SSL Channel, Know Sure Thing, Williams Alligator, HMA, ZLEMA.【F:neural_trading_system/neural_pipeline.py†L145-L191】【F:neural_trading_system/neural_pipeline.py†L276-L320】
- **Market cipher & exotic signals:** VuManch Cipher A/B, Waddah Attar Explosion, ASH, Accumulative Swing Index, Fibonacci levels, Heikin Ashi, Parabolic SAR, DPO.【F:neural_trading_system/neural_pipeline.py†L168-L195】【F:neural_trading_system/neural_pipeline.py†L320-L356】
- **Classics for calibration:** RSI (multiple periods), MACD, Bollinger Bands, Stochastics, ADX, CCI, SMA/EMA stacks, ROC, ATR, etc.【F:neural_trading_system/neural_pipeline.py†L322-L356】

Every indicator output is timestamp-aligned so the feature extractor can build temporal windows or per-bar snapshots without manual plumbing.

---

## 5. Feature Extraction & Caching
- `IndicatorFeatureExtractor` builds multi-horizon windows (default `[5, 10, 20, 50, 100, 200]`) and writes `features_*.pkl` caches under `neural_data/features/` for reproducibility.【F:neural_trading_system/neural_pipeline.py†L363-L520】
- Re-runs automatically reuse the freshest cache unless `force_recollect` or `force_cache_file` is provided, saving hours on long histories.【F:neural_trading_system/neural_pipeline.py†L392-L520】
- Sanitization uses Polars to remove NaN/Inf, clip extreme magnitudes, and stamp a SHA256 integrity hash (`*.sanitymeta`) so you know if a cache changed.【F:neural_trading_system/training/trainer.py†L17-L152】

**Workflow Tip:** After tweaking indicators, delete the old cache or bump `force_recollect=True` to regenerate a consistent feature space.

---

## 6. Label Generation & Targets
The training dataset includes multi-task heads:
- **Entries/Exits (logits):** Dynamic class weights per batch handle imbalanced signals.【F:neural_trading_system/training/trainer.py†L154-L210】
- **TP/SL/Hold labels:** Derived from rolling returns + volatility regimes; emphasizes context instead of fixed pip targets.【F:neural_trading_system/training/context_aware_labels.py†L1-L92】
- **Regression targets:** Expected return (SmoothL1) and volatility/position size (MSE) guide risk-aware sizing.【F:neural_trading_system/training/trainer.py†L189-L212】
- **Regime embeddings:** The VAE latent space feeds the regime-change detector so the transformer understands market structure shifts.【F:neural_trading_system/models/architecture.py†L202-L318】

---

## 7. Model Architecture Overview
- **Input projection:** Dense layer brings the high-dimensional feature vector into the transformer model width.【F:neural_trading_system/models/architecture.py†L1-L80】
- **Temporal encoder:** Custom multi-head self-attention blocks with layer scaling and GELU feed-forward stages model inter-indicator relationships.【F:neural_trading_system/models/architecture.py†L20-L120】
- **Market regime module:** A VAE compresses the feature context, while the `RegimeChangeDetector` issues volatility, volume, trend-break, and liquidity shift scores plus a stability index.【F:neural_trading_system/models/architecture.py†L202-L318】
- **Outputs:** Heads produce logits for entries/exits/tp/sl/hold, regression forecasts for returns & volatility, and risk-adjusted position sizing.【F:neural_trading_system/training/trainer.py†L189-L212】

Everything is built with PyTorch so you can drop in new layers or swap attention variants with minimal refactoring.

---

## 8. Training Loop Essentials
- Data is loaded from caches → sanitized → converted to tensors on-the-fly for GPU/AMP training.【F:neural_trading_system/training/trainer.py†L17-L212】
- Loss is a weighted sum of classification, regression, and KL divergence terms (`kl_weight=1e-3`) to balance reconstruction and regime exploration.【F:neural_trading_system/training/trainer.py†L189-L214】
- Positive class weights are recomputed each batch to stabilize rare-event learning (e.g., stop-loss triggers).【F:neural_trading_system/training/trainer.py†L162-L174】
- Integrity metadata ensures you never silently train on corrupted caches—hash mismatches trigger a clean rebuild automatically.【F:neural_trading_system/training/trainer.py†L17-L120】

---

## 9. Inference & Backtesting
- `run_neural_backtest.py` autodetects checkpoints, feature extractors, and configs (`model_config.json`) and seeds RNGs for deterministic replay.【F:neural_trading_system/run_neural_backtest.py†L1-L130】
- `PerfectNeuralStrategy2` streams live indicators from the TransparencyPatch batch, reuses the production feature extractor, and rescales inputs to the exact model dimensionality (pads/truncates safely).【F:neural_trading_system/run_neural_backtest.py†L130-L220】
- The dashboard summarizes equity curve, hit-rate, risk buckets, and raw logits so you can validate calibration before deployment.【F:neural_trading_system/run_neural_backtest.py†L1-L200】

To deploy live, reuse the same strategy class inside your brokerage connector; the only requirement is access to the TransparencyPatch batch feed and the feature extractor object.

---

## 10. Diagnostics & Explainability
- `analyze_model.py` renders attention heatmaps, feature attributions, regime scatterplots, and calibration charts to show **why** a trade was suggested.【F:neural_trading_system/README.md†L156-L205】
- `comprehensive_diagnostics.py`/`analyze_exit_awareness.py` (see `analysis/`) let you drill into exit timing, hold-duration bias, and label distribution.
- The VAE latent embeddings can be plotted directly; color by realized returns to spot which regimes deserve more samples.

**Always run diagnostics after major indicator or label tweaks** to confirm the transformer is still focusing on meaningful bars.

---

## 11. Extending the System
1. **Add a new indicator**
   - Register it inside `DataCollectionStrategy` and let the TransparencyPatch capture its values automatically.【F:neural_trading_system/neural_pipeline.py†L98-L356】
2. **Customize features**
   - Extend `IndicatorFeatureExtractor` or adjust `lookback_windows` in your config to compute bespoke temporal summaries.【F:neural_trading_system/neural_pipeline.py†L363-L520】
3. **Create new targets**
   - Add heads/losses in `trainer.py` and corresponding labels in `context_aware_labels.py`. The multi-task pattern already supports additional regressions or logits.【F:neural_trading_system/training/trainer.py†L154-L214】
4. **Experiment with architectures**
   - Modify `models/architecture.py` to slot in alternative attention blocks, recurrence, or graph layers while keeping the regime modules intact.【F:neural_trading_system/models/architecture.py†L1-L318】

---

## 12. Troubleshooting Cheatsheet
| Symptom | Quick Fix |
| --- | --- |
| `hash mismatch (file modified)` during training | Delete the stale cache in `neural_data/features/` or rerun collection so the sanitization step can regenerate clean data.【F:neural_trading_system/training/trainer.py†L49-L120】 |
| Model rejects feature vector (`ValueError: scaler mismatch`) | Run `python neural_trading_system/neural_pipeline.py` to rebuild features so the scaler and model dimensions align.【F:neural_trading_system/neural_pipeline.py†L372-L520】【F:neural_trading_system/run_neural_backtest.py†L180-L220】 |
| Attention maps look random after changes | Re-check that new indicators are normalized and not flooding the feature space with NaNs/Inf; the sanitization logs will highlight problematic columns.【F:neural_trading_system/training/trainer.py†L73-L152】 |
| Backtest diverges from training metrics | Ensure `PerfectNeuralStrategy2` is using the same scaler + feature extractor as training (auto-detected, but override paths if you moved files).【F:neural_trading_system/run_neural_backtest.py†L60-L220】 |

---

## 13. TL;DR Flowchart
```
Data (Polars) → TransparencyPatch (Backtrader) → Indicator cache (Parquet/IPCs)
      ↓
Feature extractor + sanitization → Multi-task transformer + regime VAE → Checkpoint
      ↓
Diagnostics (attention, regimes, exits) ↔ Backtesting strategy ↔ Deployment adapter
```

Armed with this handbook you can iterate confidently—capture clean data, tweak features, observe the model's reasoning, and ship improvements without losing transparency.
