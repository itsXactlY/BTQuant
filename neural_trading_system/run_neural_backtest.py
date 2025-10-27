#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEURAL STRATEGY BACKTEST — Fixed for 2D flattened input models
"""

import os
import re
import json
import pickle
import random
from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
import backtrader as bt
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ---------------------------------------------------------------------
# Repo imports
# ---------------------------------------------------------------------
import sys
sys.path.append('.')
from backtrader.utils.backtest import PolarsDataLoader, DataSpec
from backtrader.TransparencyPatch import activate_patch, capture_patch, optimized_patch

console = Console()


# =============================================================================
# Fixed 2D Model Wrapper (same as analysis.py)
# =============================================================================

class Fixed2DModelWrapper(nn.Module):
    """
    Wrapper that makes a 3D-expecting model work with 2D flattened input.
    Your model was trained with flattened sequences passed directly to input_projection.
    """
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model
        self.d_model = original_model.d_model
        self.seq_len = original_model.seq_len
        
    def forward(self, x, position_context=None):
        """
        x: [batch, 25100] flattened input
        """
        device = x.device
        B = x.shape[0] if x.dim() > 1 else 1
        
        # Ensure 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, features]
        
        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            # This is the case we have - flattened input
            
            # Safety checks
            if torch.isnan(x).any() or torch.isinf(x).any():
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            x = x.clamp(-100, 100)
            
            # Project flattened input to d_model
            x_proj = self.model.input_projection(x)  # [B, d_model]
            
            # For transformer, we need sequence dimension
            # Create a pseudo-sequence of length 1
            x_seq = x_proj.unsqueeze(1)  # [B, 1, d_model]
            
            # Pass through transformer blocks
            attn_weights_list = []
            for block in self.model.transformer_blocks:
                x_seq, attn_w = block(x_seq, mask=None)
                attn_weights_list.append(attn_w)
            
            # Get representation (squeeze out sequence dim)
            current_repr = x_seq.squeeze(1)  # [B, d_model]
            historical_repr = current_repr  # Same since we have no real sequence
            
            # VAE processing
            recon, mu, logvar, current_z = self.model.regime_vae(current_repr)
            _, _, _, historical_z = self.model.regime_vae(historical_repr)
            
            # Regime change
            regime_change_info = self.model.regime_change_detector(
                current_repr, current_z, historical_z
            )
            
            # Predictions
            cr = torch.cat([current_repr, current_z], dim=1)
            entry_logits = self.model.entry_head(cr)
            entry_prob = torch.sigmoid(entry_logits)
            expected_return = self.model.return_head(cr)
            volatility_forecast = self.model.volatility_head(cr)
            position_size = self.model.position_size_head(cr)
            
            # Exit management
            if position_context is not None:
                unrealized_pnl = position_context.get(
                    'unrealized_pnl', torch.zeros(B, 1, device=device)
                )
                time_in_pos = position_context.get(
                    'time_in_position', torch.zeros(B, 1, device=device)
                )
                
                tp = self.model.profit_taking_module(
                    current_repr, current_z, unrealized_pnl, time_in_pos, expected_return
                )
                sl = self.model.stop_loss_module(
                    current_repr, current_z, unrealized_pnl, time_in_pos, expected_return
                )
                hold = self.model.let_winner_run_module(
                    current_repr, current_z, unrealized_pnl, time_in_pos, expected_return
                )
                
                exit_signals = {
                    'profit_taking': tp,
                    'stop_loss': sl,
                    'let_winner_run': hold
                }
                
                # Unified exit
                pos_mask = (unrealized_pnl > 0).float()
                neg_mask = 1.0 - pos_mask
                unified = (
                    pos_mask * tp['take_profit_prob'] * (1.0 - hold['hold_score']) +
                    neg_mask * sl['stop_loss_prob']
                )
            else:
                exit_signals = {}
                unified = torch.zeros(B, 1, device=device)
            
            return {
                'entry_prob': entry_prob,
                'entry_logits': entry_logits,
                'expected_return': expected_return,
                'volatility_forecast': volatility_forecast,
                'position_size': position_size,
                'regime_mu': mu,
                'regime_logvar': logvar,
                'regime_z': current_z,
                'vae_recon': recon,
                'attention_weights': attn_weights_list,
                'sequence_repr': current_repr,
                'regime_change': regime_change_info,
                'exit_signals': exit_signals,
                'unified_exit_prob': torch.clamp(unified, 0.0, 1.0),
            }
        else:
            # 3D input - reshape to 2D and call recursively
            B, T, D = x.shape
            x_flat = x.reshape(B, T * D)
            return self.forward(x_flat, position_context)


# =============================================================================
# Utilities
# =============================================================================

def _discover_paths():
    """Resolve model / feature-extractor with sane defaults."""
    base = Path(__file__).resolve().parent
    models_dir = base / "models"
    if not models_dir.exists():
        models_dir = base

    # Look for model
    candidates = [
        base / "models" / "best_exit_aware_model_12_production_ready.pt",
        base / "models" / "best_model.pt",
        Path("models/best_exit_aware_model_12_production_ready.pt"),
        Path("models/best_model.pt"),
    ]
    model_path = next((str(p) for p in candidates if p.exists()), None)

    # Feature extractor
    fx_candidates = list((base / "models").glob("*feature_extractor.pkl")) + \
                    list(Path("models").glob("*feature_extractor.pkl"))
    fx_path = str(fx_candidates[0]) if fx_candidates else None

    # Config
    cfg_candidates = [
        base / "models" / "model_config.json",
        Path("models/model_config.json"),
    ]
    cfg_path = next((str(p) for p in cfg_candidates if p.exists()), None)

    return model_path, fx_path, cfg_path


def _silence_torch_determinism():
    """Keep determinism."""
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

class NeuralStrategy2D(bt.Strategy):
    """
    Neural Strategy for 2D flattened input models
    """

    params = dict(
        # Model paths
        model_path='best_model.pt',
        feature_extractor_path='best_model_feature_extractor.pkl',

        # Core settings
        seq_len=100,

        # Trading thresholds - based on your analysis results
        min_entry_prob=0.120,       # Was 0.131
        min_expected_return=-0.002,  # Was -0.0005
        max_exit_prob=0.20,         # Was 0.085


        # Position sizing
        fixed_position_size=0.20,

        # Risk
        atr_period=14,

        # Performance
        prediction_interval=1,  # Predict every bar since model is fast

        # Position tracking
        bars_in_position=0,
        entry_price=None,

        # Debug
        debug=True,
        log_every=100,
    )

    def __init__(self):
        console.print(Panel("Neural Strategy 2D - Backtest", style="cyan"))

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        console.print(f"🔧 Device: {self.device}")

        # Load checkpoint
        console.print(f"📦 Loading model from {self.p.model_path}")
        checkpoint = torch.load(self.p.model_path, map_location=self.device)
        
        # Check actual input dimension from checkpoint
        input_weight_shape = checkpoint['model_state_dict']['input_projection.weight'].shape
        actual_input_dim = input_weight_shape[1]  # Should be 25100
        console.print(f"✅ Model input dimension: {actual_input_dim}")

        # Import and create base model
        from models.architecture import NeuralTradingModel
        
        base_model = NeuralTradingModel(
            feature_dim=actual_input_dim,  # 25100 - the flattened dimension
            d_model=256,
            num_heads=8,
            num_layers=6,
            d_ff=1024,
            dropout=0.0,
            latent_dim=16,
            seq_len=100
        )
        
        # Load weights (allow some missing keys due to architecture variations)
        base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        base_model.eval()
        
        # Wrap with our 2D handler
        self.model = Fixed2DModelWrapper(base_model).to(self.device)
        console.print("✅ Model loaded and wrapped for 2D input")

        # Load feature extractor
        console.print(f"📦 Loading feature extractor from {self.p.feature_extractor_path}")
        with open(self.p.feature_extractor_path, 'rb') as f:
            self.feature_extractor = pickle.load(f)
        
        # Get expected dimension
        self.scaler = getattr(self.feature_extractor, 'scaler', None)
        self.scaler_dim = getattr(self.scaler, 'n_features_in_', None) if self.scaler else None
        self.model_dim = actual_input_dim
        
        console.print(f"✅ Feature extractor loaded")
        if self.scaler_dim:
            console.print(f"   Scaler dimension: {self.scaler_dim}")
        console.print(f"   Model dimension: {self.model_dim}")

        # Initialize indicators
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period, plot=False)
        
        # Feature buffer for sequence
        self.feature_buffer = []
        
        # Warmup period
        self._warmup = self.p.seq_len + self.p.atr_period + 10
        
        # State tracking
        self.prediction_counter = 0
        self.last_prediction = None
        self.bars_in_position = 0
        self.entry_price = None
        
        console.print(f"🔥 Strategy initialized — Warmup: {self._warmup} bars")

    def _extract_features(self):
        """Extract features from current batch"""
        try:
            batch = optimized_patch.current_batch
            if batch is None or len(batch) == 0:
                return None

            # Get indicator keys
            if not hasattr(self, "_cached_indicator_keys"):
                self._cached_indicator_keys = [
                    k for k in batch[0].keys() if k not in ("bar", "datetime")
                ]

            # Convert to arrays
            indicator_arrays = {
                k: np.array([b.get(k, 0.0) for b in batch], dtype=np.float32)
                for k in self._cached_indicator_keys
            }

            # Extract features
            feats = self.feature_extractor.extract_all_features(indicator_arrays)
            f_raw = np.asarray(feats, dtype=np.float32).ravel()

            # Scale if we have a scaler
            if self.scaler is not None and self.scaler_dim is not None:
                # Handle dimension mismatch carefully
                if f_raw.size >= self.scaler_dim:
                    # Take first scaler_dim features
                    f_to_scale = f_raw[:self.scaler_dim].reshape(1, -1)
                    f_scaled = self.scaler.transform(f_to_scale).ravel()
                    
                    # If model needs more dims, pad with zeros
                    if self.model_dim > self.scaler_dim:
                        f_scaled = np.pad(f_scaled, (0, self.model_dim - self.scaler_dim), mode='constant')
                    
                    return f_scaled
                else:
                    # Pad to scaler dim, scale, then adjust to model dim
                    f_padded = np.pad(f_raw, (0, self.scaler_dim - f_raw.size), mode='constant')
                    f_scaled = self.scaler.transform(f_padded.reshape(1, -1)).ravel()
                    
                    if self.model_dim != self.scaler_dim:
                        if self.model_dim > self.scaler_dim:
                            f_scaled = np.pad(f_scaled, (0, self.model_dim - self.scaler_dim), mode='constant')
                        else:
                            f_scaled = f_scaled[:self.model_dim]
                    
                    return f_scaled
            
            # No scaler - just ensure correct dimension
            if f_raw.size < self.model_dim:
                f_raw = np.pad(f_raw, (0, self.model_dim - f_raw.size), mode='constant')
            elif f_raw.size > self.model_dim:
                f_raw = f_raw[:self.model_dim]
            
            return f_raw

        except Exception as e:
            if self.p.debug:
                console.print(f"[red]Feature extraction error: {e}[/red]")
            return None

    def _predict(self):
        """Make prediction with flattened features"""
        try:
            # Stack features into a single flattened vector
            # Since model expects [batch, 25100], we concatenate all features
            seq = np.array(self.feature_buffer, dtype=np.float32)  # [seq_len, feature_dim]
            
            # Flatten the sequence: [seq_len * feature_dim]
            # This mimics how the model was trained
            features_flat = seq.flatten()
            
            # Ensure correct dimension
            if features_flat.size < self.model_dim:
                features_flat = np.pad(features_flat, (0, self.model_dim - features_flat.size), mode='constant')
            elif features_flat.size > self.model_dim:
                features_flat = features_flat[:self.model_dim]
            
            # Convert to tensor [1, model_dim]
            x = torch.from_numpy(features_flat).float().unsqueeze(0).to(self.device)
            
            # Position context if we have a position
            position_context = None
            if self.position and self.entry_price is not None:
                current_price = float(self.data.close[0])
                unrealized_pnl = (current_price - self.entry_price) / self.entry_price
                
                position_context = {
                    'unrealized_pnl': torch.tensor([[unrealized_pnl]], dtype=torch.float32, device=self.device),
                    'time_in_position': torch.tensor([[float(self.bars_in_position)]], dtype=torch.float32, device=self.device)
                }
            
            # Forward pass
            with torch.no_grad():
                out = self.model(x, position_context=position_context)
            
            # Extract predictions
            entry_prob = float(out['entry_prob'].cpu().item()) if 'entry_prob' in out else None
            exit_prob = float(out['unified_exit_prob'].cpu().item()) if 'unified_exit_prob' in out else None
            exp_ret = float(out['expected_return'].cpu().item()) if 'expected_return' in out else None
            vol = float(out['volatility_forecast'].cpu().item()) if 'volatility_forecast' in out else None
            pos_size = float(out['position_size'].cpu().item()) if 'position_size' in out else self.p.fixed_position_size
            
            return {
                'entry_prob': entry_prob,
                'exit_prob': exit_prob,
                'expected_return': exp_ret,
                'volatility_forecast': vol,
                'position_size': pos_size,
            }

        except Exception as e:
            if self.p.debug:
                import traceback
                console.print(f"[red]Prediction error: {e}[/red]")
                console.print(traceback.format_exc())
            return None

    def next(self):
        """Main strategy logic"""
        capture_patch(self)

        # Wait for warmup
        if len(self) < self._warmup:
            return

        # Extract features for current bar
        f = self._extract_features()
        if f is None:
            return

        # For the buffer, we need individual feature vectors, not flattened sequences
        # So we'll store just the current features
        # The model will get the full flattened sequence when we predict
        
        # Maintain a buffer of individual feature vectors
        # Each element is the features for one bar
        feature_dim_per_bar = self.model_dim // self.p.seq_len  # e.g., 25100 / 100 = 251
        
        # Reshape current features to per-bar dimension
        if f.size >= feature_dim_per_bar:
            f_bar = f[:feature_dim_per_bar]
        else:
            f_bar = np.pad(f, (0, feature_dim_per_bar - f.size), mode='constant')
        
        self.feature_buffer.append(f_bar)
        
        # Keep only seq_len bars
        if len(self.feature_buffer) > self.p.seq_len:
            self.feature_buffer.pop(0)
        
        # Need full sequence to predict
        if len(self.feature_buffer) < self.p.seq_len:
            return

        # Make prediction at interval
        if (self.prediction_counter % self.p.prediction_interval) == 0:
            self.last_prediction = self._predict()
        self.prediction_counter += 1

        if not self.last_prediction:
            return

        # Update position tracking
        if self.position:
            self.bars_in_position += 1

        # Debug logging
        if self.p.debug and (len(self) % self.p.log_every == 0):
            pred = self.last_prediction
            console.print(
                f"[{len(self):5d}] "
                f"entry={pred['entry_prob']:.4f} "
                f"exit={pred['exit_prob']:.4f} "
                f"exp_ret={pred['expected_return']:+.6f} "
                f"pos={'LONG' if self.position else 'NONE'}"
            )

        # Trading logic
        if not self.position:
            self._check_entry(self.last_prediction)
        else:
            self._check_exit(self.last_prediction)

    def _check_entry(self, pred):
        """Check entry conditions"""
        if pred['entry_prob'] is None or pred['expected_return'] is None:
            return
        
        # Entry conditions
        if pred['entry_prob'] < self.p.min_entry_prob:
            return
        if pred['expected_return'] < self.p.min_expected_return:
            return

        # Calculate position size
        cash = self.broker.getcash()
        price = float(self.data.close[0])
        size = (cash * self.p.fixed_position_size) / max(price, 1e-9)
        
        # Enter position
        self.buy(size=size)
        self.entry_price = price
        self.bars_in_position = 0

        if self.p.debug:
            console.print(
                f"🚀 [green]ENTRY[/green] @ {price:.2f} | "
                f"prob={pred['entry_prob']:.4f} "
                f"exp_ret={pred['expected_return']:+.6f}"
            )

    def _check_exit(self, pred):
        """Check exit conditions"""
        if pred['exit_prob'] is None:
            return
        
        # Exit condition
        if pred['exit_prob'] > self.p.max_exit_prob:
            exit_price = float(self.data.close[0])
            
            if self.entry_price:
                pnl_pct = ((exit_price - self.entry_price) / self.entry_price) * 100
            else:
                pnl_pct = 0.0
            
            self.close()
            
            if self.p.debug:
                color = "green" if pnl_pct > 0 else "red"
                console.print(
                    f"🛑 [{color}]EXIT[/{color}] @ {exit_price:.2f} | "
                    f"exit_prob={pred['exit_prob']:.4f} "
                    f"PnL={pnl_pct:+.2f}% "
                    f"bars={self.bars_in_position}"
                )
            
            # Reset position tracking
            self.entry_price = None
            self.bars_in_position = 0

    def notify_order(self, order):
        """Handle order notifications"""
        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_price = order.executed.price
                self.bars_in_position = 0

    def notify_trade(self, trade):
        """Handle trade notifications"""
        if trade.isclosed:
            self.entry_price = None
            self.bars_in_position = 0


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
    plot=False,
):
    """Run neural strategy backtest"""
    _silence_torch_determinism()

    # Discover paths
    d_model, d_fx, _ = _discover_paths()
    model_path = model_path or d_model
    feature_extractor_path = feature_extractor_path or d_fx

    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError("Model checkpoint not found")

    console.print(Panel.fit(
        f"[bold cyan]NEURAL STRATEGY BACKTEST (2D)[/bold cyan]\n"
        f"[yellow]{coin}/{collateral} - {interval}[/yellow]\n\n"
        f"Period: {start_date} → {end_date}\n"
        f"Initial Cash: ${init_cash:,.0f}\n"
        f"Model: {Path(model_path).name}",
        title="🧠 Backtest",
        border_style="cyan"
    ))

    # Setup Cerebro
    cerebro = bt.Cerebro(oldbuysell=True, runonce=False)

    # Load data
    console.print(f"\n📥 Loading {interval} data for {coin}...")
    loader = PolarsDataLoader()
    spec = DataSpec(
        symbol=coin,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        collateral=collateral
    )
    df = loader.load_data(spec, use_cache=True)
    data_feed = loader.make_backtrader_feed(df, spec)
    console.print(f"✅ Loaded {len(df):,} bars")

    cerebro.adddata(data_feed)

    # Add strategy
    cerebro.addstrategy(
        NeuralStrategy2D,
        model_path=model_path,
        feature_extractor_path=feature_extractor_path,
    )

    # Broker settings
    cerebro.broker.setcash(init_cash)
    cerebro.broker.setcommission(commission=0.001)

    # Analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    # Run backtest
    console.print("\n📊 [bold green]Running backtest...[/bold green]")
    results = cerebro.run()
    strat = results[0]

    # Get results
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - init_cash) / init_cash * 100.0

    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', None)
    drawdown = strat.analyzers.drawdown.get_analysis()
    trades = strat.analyzers.trades.get_analysis()

    # Display results
    table = Table(show_header=True, header_style="bold magenta", title="📊 Backtest Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="yellow")
    
    table.add_row("Initial Cash", f"${init_cash:,.2f}")
    table.add_row("Final Value", f"${final_value:,.2f}")
    table.add_row("Total Return", f"{total_return:.2f}%")
    table.add_row("Sharpe Ratio", f"{sharpe:.3f}" if sharpe else "N/A")
    table.add_row("Max Drawdown", f"{drawdown.get('max',{}).get('drawdown', 0):.2f}%")
    table.add_row("", "")
    
    total_trades = trades.get('total', {}).get('total', 0)
    won_trades = trades.get('won', {}).get('total', 0)
    lost_trades = trades.get('lost', {}).get('total', 0)
    
    table.add_row("Total Trades", str(total_trades))
    table.add_row("Won Trades", str(won_trades))
    table.add_row("Lost Trades", str(lost_trades))
    
    if total_trades > 0:
        winrate = 100.0 * won_trades / total_trades
        table.add_row("Win Rate", f"{winrate:.2f}%")
        
        avg_win = trades.get('won', {}).get('pnl', {}).get('average', 0)
        avg_loss = abs(trades.get('lost', {}).get('pnl', {}).get('average', 0))
        if avg_loss > 0:
            profit_factor = avg_win / avg_loss
            table.add_row("Profit Factor", f"{profit_factor:.2f}")

    console.print("\n")
    console.print(table)

    if plot:
        console.print("\n📊 Generating plot...")
        cerebro.plot(style='candlestick', barup='green', bardown='red')

    return final_value


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    result = run_backtest(
        coin='BTC',
        interval='1h',
        start_date='2021-01-01',  # Shorter period for faster testing
        end_date='2024-12-31',
        collateral='USDT',
        init_cash=10_000.0,
        model_path=None,  # auto-discover
        feature_extractor_path=None,  # auto-discover
        plot=False,
    )
    
    console.print(f"\n[bold green]✅ Backtest complete![/bold green]")
    console.print(f"Final portfolio value: ${result:,.2f}")