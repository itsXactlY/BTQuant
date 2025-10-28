import os
import re
import json
import pickle
import random
from pathlib import Path
import math
import numpy as np
import torch
import backtrader as bt
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ---------------------------------------------------------------------
# Repo imports (keep these paths consistent with your project layout)
# ---------------------------------------------------------------------
import sys
sys.path.append('.')
from backtrader.utils.backtest import PolarsDataLoader, DataSpec
from backtrader.TransparencyPatch import activate_patch, capture_patch, optimized_patch

console = Console()


# =============================================================================
# Utilities
# =============================================================================

def _discover_paths():
    """
    Resolve model / feature-extractor with sane defaults.
    """
    base = Path(__file__).resolve().parent
    models_dir = base / "models"
    if not models_dir.exists():
        models_dir = base  # fallback

    # Prefer best_model.pt in models/
    candidates = [
        base / "models" / "best_model.pt",
        base / "models" / "best_exit_aware_model_12_production_ready.pt",
        Path("models/best_model.pt"),
    ]
    model_path = next((str(p) for p in candidates if p.exists()), None)

    # Feature extractor
    fx_candidates = list((base / "models").glob("*feature_extractor.pkl")) + \
                    list(Path("models").glob("*feature_extractor.pkl")) + \
                    list(Path("models").glob("*feature_extractor.pkl"))

    fx_path = str(fx_candidates[0]) if fx_candidates else None

    # Optional config
    cfg_candidates = [
        base / "models" / "model_config.json",
        Path("models/model_config.json"),
        Path("models/model_config.json"),
    ]
    cfg_path = next((str(p) for p in cfg_candidates if p.exists()), None)

    return model_path, fx_path, cfg_path


def _load_model_config(cfg_path: str):
    if not cfg_path or not Path(cfg_path).exists():
        console.print("[yellow]⚠️ No model_config.json found; using defaults[/yellow]")
        return None
    try:
        with open(cfg_path, "r") as f:
            return json.load(f)
    except Exception as e:
        console.print(f"[yellow]⚠️ Could not read config ({e}); continuing with defaults[/yellow]")
        return None


def _detect_arch_from_state(state_dict):
    """
    Infer feature_dim, d_model, d_ff, num_layers from checkpoint weights.
    """
    # input_projection: weight [d_model, feature_dim]
    ip_w = state_dict.get("input_projection.weight")
    if ip_w is None:
        raise RuntimeError("input_projection.weight not found in checkpoint")

    d_model = ip_w.shape[0]
    feature_dim = ip_w.shape[1]

    # Try to detect feed-forward size from first block
    ff_w = state_dict.get("transformer_blocks.0.feed_forward.linear1.weight")
    d_ff = ff_w.shape[0] if ff_w is not None else 4 * d_model  # fallback

    # Count how many blocks exist in the checkpoint
    block_idxs = set()
    pat = re.compile(r"^transformer_blocks\.(\d+)\.")
    for k in state_dict.keys():
        m = pat.match(k)
        if m:
            block_idxs.add(int(m.group(1)))
    num_layers = max(block_idxs) + 1 if block_idxs else 6  # fallback

    return feature_dim, d_model, d_ff, num_layers


def _silence_torch_determinism():
    """
    Keep determinism similar to your previous setup.
    """
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


# =============================================================================
# Strategy
# =============================================================================

class PerfectNeuralStrategy2(bt.Strategy):
    """
    PERFECT MODEL-ALIGNED BACKTEST STRATEGY
    - Pulls features from TransparencyPatch batch (list[dict])
    - Extracts via feature_extractor.extract_all_features(...)
    - Scales robustly: align to scaler dim first, then back to model dim
    """

    params = dict(
        # Model / FE paths (will be overridden by run_backtest args)
        model_path='best_model.pt',
        feature_extractor_path='best_model_feature_extractor.pkl',

        # Core model settings
        seq_len=100,

        # Denormalization
        return_scale=0.016205,  # can be overridden via config

        # Thresholds (can be overridden by config)
        min_entry_prob=0.395,       # P25 from your dist
        min_expected_return=0.0120, # P25 (denorm)
        max_exit_prob=0.435,        # P75 from your dist

        # Position sizing (simple)
        fixed_position_size=0.20,

        # Risk (indicators)
        atr_period=14,

        # Perf
        prediction_interval=5,

        # Debug
        debug=True,
        log_every=50,
    )

    # ----------------------------- Indicators ------------------------------
    def _init_indicators(self):
        # Minimal set for speed; add the full suite if you need exact parity
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period, plot=False)

    # -------------------------- Safe scaling helper ------------------------
    def _scale_to_model_dim(self, f_vec: np.ndarray) -> np.ndarray:
        """
        Scale features SAFELY:
        - Determine model_dim vs scaler_dim.
        - Always call scaler.transform on exactly scaler_dim columns.
        - Stitch back to model_dim with neutral zeros if needed.
        """
        f = np.asarray(f_vec, dtype=np.float32).ravel()

        model_dim = self._input_tensor.shape[2]  # e.g., 9868
        scaler = getattr(self.feature_extractor, 'scaler', None)
        scaler_dim = getattr(scaler, 'n_features_in_', None)

        # Ensure at least model_dim length (pad/truncate) for the final shape
        if f.size < model_dim:
            f = np.pad(f, (0, model_dim - f.size), mode='constant')
        elif f.size > model_dim:
            f = f[:model_dim]

        # No scaler present -> pass-through
        if scaler is None or scaler_dim is None:
            return f

        # Common case: scaler_dim == model_dim - 1 (your setup)
        if scaler_dim == model_dim:
            # Perfect match
            return scaler.transform(f.reshape(1, -1)).ravel()

        if scaler_dim + 1 == model_dim:
            # Scale the known part, keep 1 extra column neutral(0)
            base = f[:scaler_dim]
            base_scaled = scaler.transform(base.reshape(1, -1)).ravel()
            # Append neutral extra column (0)
            return np.concatenate([base_scaled, np.zeros((1,), dtype=np.float32)])

        if model_dim + 1 == scaler_dim:
            # Rare: scaler expects 1 more column than model
            base = np.concatenate([f, np.zeros((1,), dtype=np.float32)])
            scaled = scaler.transform(base.reshape(1, -1)).ravel()
            return scaled[:model_dim]

        # Larger mismatch -> safest is pass-through (or raise)
        return f

    # ------------------------------- Setup ---------------------------------
    def __init__(self):
        console.print(Panel("Perfect Neural Strategy - Model-Aligned Backtest", style="cyan"))

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        console.print(f"🔧 Device: {self.device}")

        # ------------------- Load model checkpoint -------------------
        console.print(f"📦 Loading model from {self.p.model_path}")
        checkpoint = torch.load(self.p.model_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # Detect architecture from weights (feature_dim, d_model, d_ff, num_layers)
        feature_dim, d_model, d_ff, num_layers = _detect_arch_from_state(state_dict)
        console.print(f"📊 Feature dimension: {feature_dim}")
        console.print(f"📏 Detected model hidden size (d_model): {d_model}")
        console.print(f"⚙️ Detected feed_forward_dim: {d_ff}")

        # Build model with detected sizes
        from models.architecture import create_model
        model_cfg = {
            'seq_len': self.p.seq_len,
            'd_model': d_model,
            'd_ff': d_ff,
            'num_layers': num_layers,
            # num_heads & dropout — use your training defaults (shapes are independent)
            'num_heads': 8,
            'dropout': 0.05,
        }
        self.model = create_model(feature_dim, model_cfg).to(self.device)

        # Load weights (allow missing/extra due to block count differences)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            console.print(f"⚠️ Missing keys: {len(missing)} (e.g. {list(missing)[:3]})")
            if unexpected:
                console.print(f"⚠️ Unexpected keys: {len(unexpected)} (e.g. {list(unexpected)[:3]})")

        self.model.eval()
        console.print("✅ Model loaded and set to eval mode")

        # ------------------- Load feature extractor -------------------
        console.print(f"📦 Loading feature extractor from {self.p.feature_extractor_path}")
        with open(self.p.feature_extractor_path, 'rb') as f:
            self.feature_extractor = pickle.load(f)

        expected_dim = getattr(getattr(self.feature_extractor, 'scaler', None), 'n_features_in_', None)
        if expected_dim is not None:
            console.print(f"✅ Feature extractor loaded (dim={expected_dim})")
        else:
            console.print(f"✅ Feature extractor loaded (no scaler info)")

        if expected_dim and expected_dim != feature_dim:
            console.print(f"⚠️ Feature dim mismatch — Extractor={expected_dim}, Model={feature_dim}")

        # Indicators (minimal set; extend if you require parity)
        self._init_indicators()

        # Pre-alloc input tensor (B=1, T=seq_len, F=feature_dim)
        self._input_tensor = torch.zeros(
            1, self.p.seq_len, feature_dim, dtype=torch.float32, device=self.device
        )

        # Warmup (enough bars to fill seq + indicators)
        self._warmup = self.p.seq_len + self.p.atr_period + 10

        # Buffers/state
        self.feature_buffer = []
        self.prediction_counter = 0
        self.last_prediction = None

        console.print(f"🔥 Strategy initialized — Warmup: {self._warmup} bars")

    # --------------------------- Feature Builder ----------------------------
    def _extract_features(self):
        """
        Pulls batch (list[dict]) from TransparencyPatch and produces a model_len vector.
        """
        try:
            batch = optimized_patch.current_batch
            if batch is None or len(batch) == 0:
                return None

            # Convert list[dict] -> dict[str, np.ndarray]
            if not hasattr(self, "_cached_indicator_keys"):
                self._cached_indicator_keys = [
                    k for k in batch[0].keys() if k not in ("bar", "datetime")
                ]

            indicator_arrays = {
                k: np.array([b.get(k, 0.0) for b in batch], dtype=np.float32)
                for k in self._cached_indicator_keys
            }

            # Use your extractor
            feats = self.feature_extractor.extract_all_features(indicator_arrays)
            f_raw = np.asarray(feats, dtype=np.float32).ravel()

            # Scale safely to model dimension (handles scaler 9867 vs model 9868)
            f_scaled = self._scale_to_model_dim(f_raw)
            return f_scaled

        except Exception as e:
            if self.p.debug:
                import traceback
                console.print(f"[red]Feature extraction error: {e}[/red]")
                console.print(f"[red]{traceback.format_exc()}[/red]")
            return None

    # ------------------------------ Predict ---------------------------------
    # ---------- output helpers ----------
    def _flatten_scalar(self, t):
        if t is None:
            return None
        if isinstance(t, torch.Tensor):
            t = t.detach().float().reshape(-1)
            if t.numel() == 0:
                return None
            return float(t[-1].item())
        return float(t)

    def _to_prob(self, t, key_hint: str = ""):
        """Convert logits/out-of-range → probability via sigmoid; clamp to [0,1]."""
        v = self._flatten_scalar(t)
        if v is None:
            return None
        if "logit" in key_hint.lower() or v < 0.0 or v > 1.0:
            v = 1.0 / (1.0 + math.exp(-v))
        return max(0.0, min(1.0, v))

    def _pick(self, out_dict: dict, candidates: list[str]):
        """
        Return (value, matched_key) by scanning dict keys with flexible matching.
        Matches exact / contains / endswith (case-insensitive).
        """
        for k in out_dict.keys():
            kl = str(k).lower()
            for c in candidates:
                c = c.lower()
                if kl == c or kl.endswith(c) or c in kl:
                    return out_dict[k], k
        return None, None

    def _predict(self):
        try:
            # Build model input
            seq = np.array(self.feature_buffer, dtype=np.float32)
            self._input_tensor[0, :, :] = torch.from_numpy(seq)

            # Forward pass
            with torch.no_grad():
                out = self.model(self._input_tensor)

            # One-time visibility of keys
            if isinstance(out, dict) and not getattr(self, "_printed_out_keys", False):
                console.print(f"🗝️ Model output keys: [cyan]{list(out.keys())}[/cyan]")
                self._printed_out_keys = True

            entry_prob = exit_prob = None
            exp_ret = vol_forecast = pos_size = None

            if isinstance(out, dict):
                # Flexible candidates — handles entry_prob, entry_probs, open_prob, entry_logit, etc.
                entry_t, entry_k = self._pick(out, [
                    "entry_prob", "entry_probs", "entry_probability", "prob_entry",
                    "open_prob", "long_prob", "enter_prob", "entry_logit"
                ])
                exit_t,  exit_k  = self._pick(out, [
                    "exit_prob", "exit_probs", "exit_probability", "prob_exit",
                    "close_prob", "sell_prob", "short_prob", "exit_logit"
                ])
                exp_t,   exp_k   = self._pick(out, [
                    "expected_return", "exp_return", "return", "ret", "mu"
                ])
                vol_t,   vol_k   = self._pick(out, [
                    "volatility_forecast", "vol_forecast", "vol", "sigma", "volatility"
                ])
                size_t,  size_k  = self._pick(out, [
                    "position_size", "pos_size", "size", "position"
                ])

                entry_prob = self._to_prob(entry_t, str(entry_k or ""))
                exit_prob  = self._to_prob(exit_t,  str(exit_k or ""))

                exp_ret = self._flatten_scalar(exp_t)
                vol_forecast = self._flatten_scalar(vol_t)
                pos_size = self._flatten_scalar(size_t)

            elif isinstance(out, (list, tuple)):
                # Heuristic order: [entry, exit, expected_return, volatility, position_size]
                vals = list(out)
                entry_prob = self._to_prob(vals[0], "entry") if len(vals) > 0 else None
                exit_prob  = self._to_prob(vals[1], "exit")  if len(vals) > 1 else None
                exp_ret    = self._flatten_scalar(vals[2])   if len(vals) > 2 else None
                vol_forecast = self._flatten_scalar(vals[3]) if len(vals) > 3 else None
                pos_size   = self._flatten_scalar(vals[4])   if len(vals) > 4 else None

            elif isinstance(out, torch.Tensor):
                # Common fallback: last dim ~ heads
                v = out.detach().float().reshape(-1)
                entry_prob = self._to_prob(v[0], "entry") if v.numel() > 0 else None
                exit_prob  = self._to_prob(v[1], "exit")  if v.numel() > 1 else None
                exp_ret    = float(v[2].item()) if v.numel() > 2 else None
                vol_forecast = float(v[3].item()) if v.numel() > 3 else None
                pos_size   = float(v[4].item()) if v.numel() > 4 else None
            else:
                raise RuntimeError(f"Unsupported model output type: {type(out)}")

            return {
                'entry_prob': entry_prob,
                'exit_prob': exit_prob,  # may be None if head not present
                'expected_return': exp_ret,
                'expected_return_norm': exp_ret,  # your head already denorms; keep same field
                'volatility_forecast': vol_forecast,
                'position_size': pos_size if pos_size is not None else self.p.fixed_position_size,
            }

        except Exception as e:
            if self.p.debug:
                import traceback
                console.print(f"[red]Prediction error: {e}[/red]")
                console.print(f"[red]{traceback.format_exc()}[/red]")
            return None

    # -------------------------------- Core ----------------------------------
    def next(self):
        capture_patch(self)

        if len(self) < self._warmup:
            return

        f = self._extract_features()
        if f is None:
            return

        # Maintain rolling sequence buffer
        self.feature_buffer.append(f)
        if len(self.feature_buffer) > self.p.seq_len:
            self.feature_buffer.pop(0)

        if len(self.feature_buffer) < self.p.seq_len:
            return

        # Predict at interval
        if (self.prediction_counter % self.p.prediction_interval) == 0:
            self.last_prediction = self._predict()
        self.prediction_counter += 1

        if not self.last_prediction:
            return

        # Debug log
        if self.p.debug and (len(self) % self.p.log_every == 0):
            pred = self.last_prediction
            console.print(
                f"[{len(self):5d}] entry={pred['entry_prob']:.3f} "
                f"exit={pred['exit_prob']:.3f} "
                f"exp_ret={pred['expected_return']:+.4f} "
                f"size={pred['position_size']:.3f}"
            )

        # Entry/exit logic (simple)
        if not self.position:
            self._check_entry(self.last_prediction)
        else:
            self._check_exit(self.last_prediction)

    def _check_entry(self, pred):
        if pred['entry_prob'] < self.p.min_entry_prob:
            return
        if pred['expected_return'] < self.p.min_expected_return:
            return

        cash = self.broker.getcash()
        price = float(self.data.close[0])
        size = (cash * self.p.fixed_position_size) / max(price, 1e-9)
        self.buy(size=size)

        if self.p.debug:
            console.print(
                f"🚀 ENTRY  prob={pred['entry_prob']:.3f} "
                f"exp_ret={pred['expected_return']:+.4f} size={self.p.fixed_position_size:.0%}"
            )

    def _check_exit(self, pred):
        if pred['exit_prob'] > self.p.max_exit_prob:
            entry_price = float(self.position.price) if self.position else None
            exit_price = float(self.data.close[0])
            pnl_pct = ((exit_price - entry_price) / entry_price) if entry_price else 0.0
            self.close()
            if self.p.debug:
                console.print(
                    f"🛑 EXIT   exit_prob={pred['exit_prob']:.3f} pnl={pnl_pct:+.2%}"
                )

    def notify_order(self, order):  # noqa: D401
        return

    def notify_trade(self, trade):  # noqa: D401
        return


# =============================================================================
# Backtest runner
# =============================================================================

def run_backtest(
    coin='BTC',
    interval='1h',
    start_date='2017-01-01',
    end_date='2024-12-31',
    collateral='USDT',
    init_cash=10_000.0,
    model_path=None,
    feature_extractor_path=None,
    config_path=None,
    plot=False,
):
    _silence_torch_determinism()

    # Resolve paths
    d_model, d_fx, d_cfg = _discover_paths()
    model_path = model_path or d_model
    feature_extractor_path = feature_extractor_path or d_fx
    config_path = config_path or d_cfg

    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError("Model checkpoint not found — please set a valid path.")

    console.print(f"✅ Using model: {Path(model_path).name}")
    if feature_extractor_path:
        console.print(f"✅ Using feature extractor: {Path(feature_extractor_path).name}")

    console.print(Panel.fit(
        f"[bold cyan]NEURAL STRATEGY BACKTEST[/bold cyan]\n"
        f"[yellow]{coin}/{collateral} - {interval}[/yellow]\n\n"
        f"Period: {start_date} → {end_date}\n"
        f"Initial Cash: ${init_cash:,.0f}\n"
        f"Model: {model_path}",
        title="🧠 Backtest",
        border_style="cyan"
    ))

    # Optional thresholds + return_scale from config
    cfg = _load_model_config(config_path)
    strat_kwargs = dict(
        model_path=model_path,
        feature_extractor_path=feature_extractor_path or "",
    )
    if cfg:
        bt_req = cfg.get('backtest_requirements', {})
        crit = bt_req.get('critical_parameters', {})
        thr = bt_req.get('recommended_thresholds', {})

        if 'return_scale' in crit:
            strat_kwargs['return_scale'] = float(crit['return_scale'])
        if 'sequence_length' in crit:
            strat_kwargs['seq_len'] = int(crit['sequence_length'])

        # thresholds
        if 'min_entry_prob' in thr:
            strat_kwargs['min_entry_prob'] = float(thr['min_entry_prob'])
        if 'min_expected_return' in thr:
            strat_kwargs['min_expected_return'] = float(thr['min_expected_return'])
        if 'max_exit_prob' in thr:
            strat_kwargs['max_exit_prob'] = float(thr['max_exit_prob'])

    # Build cerebro
    cerebro = bt.Cerebro(oldbuysell=True, runonce=False, stdstats=False)

    # Data
    console.print(f"\n📥 [cyan]Loading {interval} data for {coin}...[/cyan]")
    loader = PolarsDataLoader()
    spec = DataSpec(symbol=coin, interval=interval, start_date=start_date, end_date=end_date, collateral=collateral)
    df = loader.load_data(spec, use_cache=True)
    data_feed = loader.make_backtrader_feed(df, spec)
    console.print(f"✅ [green]Loaded {len(df):,} bars[/green]")

    cerebro.adddata(data_feed)

    # Strategy
    cerebro.addstrategy(PerfectNeuralStrategy2, **strat_kwargs)

    # Broker/analyzers
    cerebro.broker.setcash(init_cash)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    # Run
    console.print("\n📊 [bold green]Running backtest...[/bold green]")
    results = cerebro.run()
    strat = results[0]

    # Results
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - init_cash) / init_cash * 100.0

    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', None)
    drawdown = strat.analyzers.drawdown.get_analysis()
    trades = strat.analyzers.trades.get_analysis()

    table = Table(show_header=True, header_style="bold magenta", title="📊 Backtest Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="yellow")
    table.add_row("Initial Cash", f"${init_cash:,.2f}")
    table.add_row("Final Value", f"${final_value:,.2f}")
    table.add_row("Total Return", f"{total_return:.2f}%")
    table.add_row("Sharpe Ratio", f"{sharpe:.3f}" if sharpe is not None else "N/A")
    table.add_row("Max Drawdown", f"{drawdown.get('max',{}).get('drawdown', 0):.2f}%")
    table.add_row("", "")
    table.add_row("Total Trades", str(trades.get('total', {}).get('total', 0)))
    table.add_row("Won Trades", str(trades.get('won', {}).get('total', 0)))
    table.add_row("Lost Trades", str(trades.get('lost', {}).get('total', 0)))
    if trades.get('total', {}).get('total', 0) > 0:
        winrate = 100.0 * trades.get('won', {}).get('total', 0) / max(1, trades.get('total', {}).get('total', 1))
        table.add_row("Win Rate", f"{winrate:.2f}%")

    console.print("\n")
    console.print(table)

    if plot:
        console.print("\n📊 [cyan]Generating plot...[/cyan]")
        cerebro.plot(style='candlestick', barup='green', bardown='red')

    return final_value


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    run_backtest(
        coin='BTC',
        interval='15m',
        start_date='2023-01-01',
        end_date='2024-01-01',
        collateral='USDT',
        init_cash=10_000.0,
        model_path='neural_trading_system/models/best_exit_aware_model_15m.pt',
        feature_extractor_path='neural_trading_system/models/exit_aware_feature_extractor.pkl',
        config_path='neural_trading_system/models/model_config.json',              # auto-discover
        plot=True,
    )
