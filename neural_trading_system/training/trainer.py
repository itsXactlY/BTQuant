import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import polars as pl
import time
from pathlib import Path
from rich.console import Console
import hashlib
import json
import torch

console = Console()

# =============================================================================
# POLARS-BASED SANITIZATION & CACHE OVERWRITE (HASH-ONLY VALIDATION)
# =============================================================================

def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file for integrity verification."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def check_cache_integrity(cache_path: str) -> tuple[bool, str]:
    """
    Check cache integrity by comparing stored hash with current file hash.
    Returns (is_clean, reason).
    """
    meta_path = Path(str(cache_path) + ".sanitymeta")
    if not meta_path.exists():
        return False, "no metadata found"

    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)

        saved_hash = meta.get("hash")
        if not saved_hash:
            return False, "missing hash in metadata"

        current_hash = compute_file_hash(cache_path)
        if current_hash != saved_hash:
            return False, "hash mismatch (file modified)"

        return True, "hash unchanged"
    except Exception as e:
        return False, f"integrity check failed: {e}"


def sanitize_and_overwrite_cache(df, config, pipeline, find_latest_cache, console: Console):
    """
    Polars-accelerated sanitization with hash-only validation.
    Detects NaN/Inf/extreme values, clips, and safely overwrites caches.
    """
    start_time = time.time()

    latest_cache = find_latest_cache()
    if not latest_cache:
        console.print("[red]❌ No cache file found for sanitization.[/red]")
        return df, None

    cache_path = Path(latest_cache)
    meta_path = Path(str(cache_path) + ".sanitymeta")

    # ──────────────────────────────────────────────────────────────
    # 🧩 Step 1 — Hash integrity check
    # ──────────────────────────────────────────────────────────────
    clean, reason = check_cache_integrity(cache_path)
    if clean:
        console.print(f"[green]✅ Cache verified clean — skipping redundant sanitization[/green] ({reason})")
        df = pl.read_ipc(cache_path)
        features_np = df.to_numpy()
        returns_np = df['returns'].to_numpy() if 'returns' in df.columns else None
        return features_np, returns_np

    console.print(f"[yellow]⚙️  Cache needs sanitization ({reason})[/yellow]")

    # ──────────────────────────────────────────────────────────────
    # 🧩 Step 2 — Load and sanitize
    # ──────────────────────────────────────────────────────────────
    console.rule("[bold blue]🧹 POLARS SANITIZATION STARTED[/bold blue]")
    console.print(f"[cyan]📦 Loading feature cache: {cache_path.name}[/cyan]")

    try:
        df = pl.read_ipc(cache_path)
    except Exception as e:
        console.print(f"[yellow]⚠️ Polars IPC read failed ({e}) — trying pandas pickle fallback...[/yellow]")
        import pandas as pd
        try:
            df_pd = pd.read_pickle(cache_path)
            df = pl.from_pandas(df_pd)
            console.print(f"[green]✅ Loaded via pandas pickle fallback.[/green]")
        except Exception as e2:
            console.print(f"[red]❌ Failed to load cache: {e2}[/red]")
            return pl.DataFrame(), None

    console.print(f"[cyan]🧪 Initial Feature Shape: {df.shape}[/cyan]")
    console.print("[cyan]🔍 Checking for NaN / Inf / extreme values...[/cyan]")

    nan_counts = df.null_count()
    initial_nan = int(nan_counts.to_series().sum())

    np_arr = df.to_numpy()
    initial_inf = int(np.isinf(np_arr).sum())

    max_val = float(np.nanmax(np.abs(np_arr)))
    clip_threshold = 1e8 if max_val > 1e8 else 1e6
    clipped = 0

    if max_val > clip_threshold:
        console.print(f"[yellow]⚠️ Extreme values detected (max={max_val:.2e}). Clipping...[/yellow]")
        np.clip(np_arr, -clip_threshold, clip_threshold, out=np_arr)
        clipped = int(np.sum(np.abs(np_arr) >= clip_threshold))

    np_arr = np.nan_to_num(np_arr, nan=0.0, posinf=0.0, neginf=0.0)
    df_clean = pl.DataFrame(np_arr, schema=df.columns)

    output_path_pkl = cache_path
    output_path_parquet = cache_path.with_suffix(".parquet")

    df_clean.write_ipc(output_path_pkl)
    df_clean.write_parquet(output_path_parquet)

    duration = time.time() - start_time
    total_cleaned = initial_nan + initial_inf
    percent_cleaned = (total_cleaned / np_arr.size) * 100

    console.print(f"[green]✅ Sanitization complete in {duration:.2f}s[/green]")
    console.print(f"   • Cleaned values: {total_cleaned:,} ({percent_cleaned:.4f}%)")
    console.print(f"   • Clipped values: {clipped:,}")
    console.print(f"   • Output cache: {output_path_pkl.name}")
    console.print(f"   • Parquet: {output_path_parquet.name}")

    # ──────────────────────────────────────────────────────────────
    # 🧩 Step 3 — Write hash metadata only
    # ──────────────────────────────────────────────────────────────
    try:
        meta_data = {
            "hash": compute_file_hash(output_path_pkl),
            "clean": True,
        }
        with open(meta_path, "w") as f:
            json.dump(meta_data, f, indent=4)
    except Exception as e:
        console.print(f"[red]⚠️ Failed to write sanitymeta: {e}[/red]")

    console.rule("[green]Sanitization Done[/green]")
    console.print(f"[green]✅ Cache sanitized successfully ({config.get('mode', 'system')} mode) in {duration:.2f}s[/green]")

    features_df = df_clean.drop("returns") if "returns" in df_clean.columns else df_clean
    features_np = features_df.to_numpy()
    returns_np = df_clean["returns"].to_numpy() if "returns" in df_clean.columns else None
    return features_np, returns_np

# =============================================================================
# TRADING DATASET
# =============================================================================

def _pos_weight_of(labels: torch.Tensor) -> torch.Tensor:
    p = labels.mean().clamp(1e-6, 1-1e-6)
    return ((1.0 - p) / p).detach()

def multitask_loss(out, y, loss_weights, kl_weight=1e-3):
    """
    out: dict from model forward
    y: dict of targets (all as float tensors)
    """
    bce = nn.BCEWithLogitsLoss
    huber = nn.SmoothL1Loss(beta=0.01)
    mse   = nn.MSELoss()

    # dynamic pos_weights per batch (robust for class imbalance)
    w_entry = _pos_weight_of(y['entry'])
    w_exit  = _pos_weight_of(y['exit'])
    w_tp    = _pos_weight_of(y['tp'])
    w_sl    = _pos_weight_of(y['sl'])
    w_hold  = _pos_weight_of(y['hold'])

    loss_entry = bce(pos_weight=w_entry)(out['entry_logit'], y['entry'])
    loss_exit  = bce(pos_weight=w_exit )(out['exit_logit'],  y['exit'])
    loss_tp    = bce(pos_weight=w_tp   )(out['tp_logit'],    y['tp'])
    loss_sl    = bce(pos_weight=w_sl   )(out['sl_logit'],    y['sl'])
    loss_hold  = bce(pos_weight=w_hold )(out['hold_logit'],  y['hold'])

    loss_ret   = huber(out['expected_return'],    y['ret'])
    loss_vol   = mse  (out['volatility_forecast'],y['vol'])
    loss_pos   = mse  (out['position_size'],      y['pos'])

    loss = (loss_weights['entry'] * loss_entry
          + loss_weights['exit']  * loss_exit
          + loss_weights['tp']    * loss_tp
          + loss_weights['sl']    * loss_sl
          + loss_weights['hold']  * loss_hold
          + loss_weights['ret']   * loss_ret
          + loss_weights['vol']   * loss_vol
          + loss_weights['pos']   * loss_pos
          + kl_weight * out['kl_loss'])

    return loss, {
        "entry": loss_entry.item(), "exit": loss_exit.item(),
        "tp": loss_tp.item(), "sl": loss_sl.item(), "hold": loss_hold.item(),
        "ret": loss_ret.item(), "vol": loss_vol.item(), "pos": loss_pos.item()
    }

class TradingDataset(Dataset):
    """Enhanced dataset with exit management labels"""
    def __init__(self, features: np.ndarray, returns: np.ndarray, prices: np.ndarray, 
                 seq_len: int = 100, prediction_horizon: int = 1):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.returns = torch.as_tensor(returns, dtype=torch.float32)
        self.prices = torch.as_tensor(prices, dtype=torch.float32)
        self.seq_len = seq_len
        self.prediction_horizon = prediction_horizon

        # Safety checks
        if torch.isnan(self.features).any() or torch.isinf(self.features).any():
            self.features = torch.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)
        
        features_max = self.features.abs().max()
        if features_max > 100:
            self.features = torch.clamp(self.features, min=-100, max=100)

        self.valid_length = max(0, len(self.features) - seq_len - prediction_horizon + 1)

    def __len__(self):
        return self.valid_length

    def __getitem__(self, idx):
        start = idx
        end = idx + self.seq_len
        feature_seq = self.features[start:end]

        label_idx = end + self.prediction_horizon - 1
        future_return = self.returns[label_idx]
        current_price = self.prices[end - 1]
        future_price = self.prices[label_idx]

        # 🆕 Compute exit labels by simulating different holding periods
        max_lookforward = min(50, len(self.returns) - label_idx)  # Look up to 50 bars ahead
        
        if max_lookforward > 5:
            future_prices = self.prices[label_idx:label_idx + max_lookforward]
            future_rets = (future_prices - current_price) / current_price
            
            # Find optimal exit points
            peak_return = future_rets.max()
            peak_idx = future_rets.argmax()
            trough_return = future_rets.min()
            trough_idx = future_rets.argmin()
            
            # 💰 Take-profit label: Did price reach good profit then reverse?
            # If peak > threshold and current position would be near peak
            take_profit_target = 0.0
            if peak_return > 0.02:  # 2% profit threshold
                # Was this bar near the peak? (within 20% of time to peak)
                bars_to_peak = peak_idx
                if bars_to_peak <= 5:  # Peak is soon
                    take_profit_target = 1.0
                elif bars_to_peak <= 10:
                    take_profit_target = 0.7
            
            # 🛑 Stop-loss label: Did price fall significantly?
            stop_loss_target = 0.0
            if trough_return < -0.015:  # -1.5% loss threshold
                bars_to_trough = trough_idx
                if bars_to_trough <= 3:  # Loss is imminent
                    stop_loss_target = 1.0
                elif bars_to_trough <= 7:
                    stop_loss_target = 0.7
            
            # 🚀 Let-winner-run label: Is trend continuing strongly?
            let_run_target = 0.0
            if future_return > 0.01:  # Currently profitable
                # Check if returns continue to grow
                if max_lookforward >= 20:
                    near_returns = future_rets[:10].mean()
                    far_returns = future_rets[10:20].mean()
                    if far_returns > near_returns and far_returns > 0.02:
                        let_run_target = 1.0  # Strong continuation
            
            # 🚨 Regime change label: Volatility spike or pattern break
            regime_change_target = 0.0
            if max_lookforward >= 10:
                recent_vol = future_rets[:5].std()
                future_vol = future_rets[5:10].std()
                if future_vol > recent_vol * 2:  # Volatility doubled
                    regime_change_target = 1.0
                    
        else:
            # Not enough data - neutral labels
            take_profit_target = 0.0
            stop_loss_target = 0.0
            let_run_target = 0.0
            regime_change_target = 0.0

        # Volatility calculation
        return_window = self.returns[idx + self.seq_len: idx + self.seq_len + self.prediction_horizon]
        if return_window.numel() == 0:
            actual_volatility = torch.tensor(0.0, dtype=torch.float32)
        elif return_window.numel() == 1:
            actual_volatility = torch.abs(return_window[0])
        else:
            actual_volatility = torch.std(return_window, unbiased=False)

        # Entry labels (original soft labels)
        returns_std = self.returns.std()
        entry_threshold = max(0.01, 1.5 * returns_std)
        entry_confidence = torch.sigmoid(
            (future_return - entry_threshold) / (0.5 * returns_std)
        )
        entry_label = torch.clamp(entry_confidence, 0.0, 1.0)

        return {
            'features': feature_seq,
            'future_return': future_return,
            'actual_volatility': actual_volatility,
            'entry_label': entry_label,
            
            # 🆕 NEW: Exit management labels
            'take_profit_label': torch.tensor(take_profit_target, dtype=torch.float32),
            'stop_loss_label': torch.tensor(stop_loss_target, dtype=torch.float32),
            'let_winner_run_label': torch.tensor(let_run_target, dtype=torch.float32),
            'regime_change_label': torch.tensor(regime_change_target, dtype=torch.float32),
            
            # Position context (simulated for training)
            'unrealized_pnl': future_return,  # Simulated P&L
            'time_in_position': torch.tensor(float(self.prediction_horizon), dtype=torch.float32),
        }

class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with explicit supervision for exit heads (TP/SL/LR/RC).
    Uses focal for entry; BCE on exit probabilities; return/vol/vae as before.
    Includes diversity + confidence penalties. Exit loss weight is ramped.
    """
    def __init__(self, num_tasks=5):
        super().__init__()
        # you can keep num_tasks for compatibility; we don't use log_vars for exits now
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.entry_temp = nn.Parameter(torch.ones(1))
        
        self.exit_boost = 1.0
        self.exit_base  = 3.0   # your previous baseline boost for exit
        self.exit_max   = 8.0   # cap for the ramp
        
        # base task weights
        self.task_weights = {
            "entry": 1.0,
            "return": 1.5,
            "volatility": 2.0,
            "vae": 1.0,
            # a single scalar that scales the sum of (tp/sl/lr/rc) exit losses
            "exit_bundle": 3.0,   # base boost for exits
        }
        # ramp params (set at runtime)
        self._exit_max_boost = 6.0
        self._exit_ramp_epochs = 10
        self._current_exit_boost = self.task_weights["exit_bundle"]

    def _focal_loss(self,
                    logits: torch.Tensor,
                    targets: torch.Tensor,
                    alpha: torch.Tensor,
                    gamma: float = 2.0) -> torch.Tensor:
        """Focal BCE on logits (all tensors already on the right device/dtype)."""
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt  = torch.exp(-bce).clamp(1e-6, 1.0)
        loss = alpha * (1 - pt) ** gamma * bce
        return loss.mean()

    def _bce_on_probs(self,
                    probs: torch.Tensor,
                    targets: torch.Tensor,
                    alpha: torch.Tensor) -> torch.Tensor:
        """Stable BCE directly on probabilities (no logits)."""
        eps = 1e-6
        p = probs.clamp(eps, 1.0 - eps)
        t = targets
        loss = -(alpha * t * torch.log(p) + (1 - alpha) * (1 - t) * torch.log(1 - p))
        return loss.mean()

    def _batch_alpha(self, y: torch.Tensor) -> torch.Tensor:
        """Dynamic α = 1 - pos_rate (clipped), returned on the same device as y."""
        p = y.mean().clamp(1e-6, 1 - 1e-6)
        return (1.0 - p).clamp(0.05, 0.95).detach()

    def set_exit_weight_ramp(self, epoch: int, ramp_epochs: int = 10, max_boost: float = 8.0):
        """Call once per epoch to ramp exit losses from base to max_boost."""
        if ramp_epochs <= 0:
            self.exit_boost = max_boost
        else:
            t = min(max(epoch, 0), ramp_epochs) / float(ramp_epochs)
            self.exit_boost = float(self.exit_base + (max_boost - self.exit_base) * t)

        self._exit_ramp_epochs = ramp_epochs
        self._exit_max_boost = max_boost
        if ramp_epochs <= 0:
            self._current_exit_boost = self.task_weights["exit_bundle"]
            return
        f = min(1.0, epoch / float(ramp_epochs))
        self._current_exit_boost = self.task_weights["exit_bundle"] * (1.0 + f * (max_boost / max(self.task_weights["exit_bundle"], 1e-6) - 1.0))

    def forward(self, predictions: dict, targets: dict):
        # keep all loss math on the same device + fp32
        device = predictions['entry_logits'].device
        def _f32(x): return x.float().to(device)

        # ─── ENTRY (focal) ─────────────────────────────────────────
        temp = torch.clamp(self.entry_temp.to(device), min=0.1, max=10.0)
        entry_logits = _f32(predictions['entry_logits']).view(-1) / temp
        entry_true   = _f32(targets['entry_label']).view(-1)
        alpha_entry  = self._batch_alpha(entry_true)
        entry_loss   = self._focal_loss(entry_logits, entry_true, alpha=alpha_entry, gamma=2.0)

        # ─── EXIT HEADS (BCE on probabilities) ────────────────────
        exit_signals = predictions.get('exit_signals', {})
        tp_prob = sl_prob = lr_prob = rc_prob = None

        if isinstance(exit_signals, dict):
            tp = exit_signals.get('profit_taking', None)
            sl = exit_signals.get('stop_loss', None)
            lr = exit_signals.get('let_winner_run', None)
            rc = predictions.get('regime_change', None) or exit_signals.get('regime_change', None)

            tp_prob = _f32(tp.get('take_profit_prob')) if (tp and tp.get('take_profit_prob') is not None) else None
            sl_prob = _f32(sl.get('stop_loss_prob'))   if (sl and sl.get('stop_loss_prob')   is not None) else None
            lr_prob = _f32(lr.get('hold_score'))       if (lr and lr.get('hold_score')       is not None) else None
            if isinstance(rc, dict) and rc.get('regime_change_score') is not None:
                rc_prob = _f32(rc.get('regime_change_score'))

        tp_loss = torch.tensor(0.0, device=device)
        sl_loss = torch.tensor(0.0, device=device)
        lr_loss = torch.tensor(0.0, device=device)
        rc_loss = torch.tensor(0.0, device=device)
        exit_terms = []
        exit_probs_for_spread = []

        if tp_prob is not None and 'take_profit_label' in targets:
            y = _f32(targets['take_profit_label']).view(-1)
            a = self._batch_alpha(y)
            tp_loss = self._bce_on_probs(tp_prob.view(-1), y, alpha=a)
            exit_terms.append(tp_loss)
            exit_probs_for_spread.append(tp_prob.view(-1))

        if sl_prob is not None and 'stop_loss_label' in targets:
            y = _f32(targets['stop_loss_label']).view(-1)
            a = self._batch_alpha(y)
            sl_loss = self._bce_on_probs(sl_prob.view(-1), y, alpha=a)
            exit_terms.append(sl_loss)
            exit_probs_for_spread.append(sl_prob.view(-1))

        if lr_prob is not None and 'let_winner_run_label' in targets:
            y = _f32(targets['let_winner_run_label']).view(-1)
            a = self._batch_alpha(y)
            lr_loss = self._bce_on_probs(lr_prob.view(-1), y, alpha=a)
            exit_terms.append(lr_loss)
            exit_probs_for_spread.append(lr_prob.view(-1))

        if rc_prob is not None and 'regime_change_label' in targets:
            y = _f32(targets['regime_change_label']).view(-1)
            a = self._batch_alpha(y)
            rc_loss = self._bce_on_probs(rc_prob.view(-1), y, alpha=a)
            exit_terms.append(rc_loss)
            exit_probs_for_spread.append(rc_prob.view(-1))

        exit_bundle_loss = (torch.stack(exit_terms).mean()
                            if len(exit_terms) else torch.tensor(0.0, device=device))

        # ─── RETURN / VOL / VAE ───────────────────────────────────
        ret_pred = _f32(predictions['expected_return']).view(-1)
        ret_true = _f32(targets['future_return']).view(-1)
        return_loss = F.smooth_l1_loss(ret_pred, ret_true)

        vol_pred = _f32(predictions['volatility_forecast']).view(-1)
        vol_true = _f32(targets['actual_volatility']).view(-1)
        volatility_loss = F.mse_loss(vol_pred, vol_true)

        vae_recon = predictions.get('vae_recon', None)
        seq_repr  = predictions.get('sequence_repr', None)
        regime_mu = predictions.get('regime_mu', None)
        regime_logvar = predictions.get('regime_logvar', None)

        if vae_recon is not None and seq_repr is not None:
            vae_recon_loss = F.mse_loss(_f32(vae_recon), _f32(seq_repr))
        else:
            vae_recon_loss = torch.tensor(0.0, device=device)

        if (regime_mu is not None) and (regime_logvar is not None):
            regime_mu      = _f32(regime_mu)
            regime_logvar  = _f32(regime_logvar)
            regime_logvar_clamped = torch.clamp(regime_logvar, min=-10, max=10)
            kl_loss = -0.5 * torch.sum(
                1 + regime_logvar_clamped - regime_mu.pow(2) - regime_logvar_clamped.exp(),
                dim=1
            ).mean()
        else:
            kl_loss = torch.tensor(0.0, device=device)

        vae_loss = vae_recon_loss + 1e-5 * kl_loss

        # ─── DIVERSITY + CONFIDENCE (safe std) ─────────────────────
        entry_probs = predictions.get('entry_prob', torch.sigmoid(predictions['entry_logits'])).to(device).float().view(-1)
        if exit_probs_for_spread:
            exit_probs_cat = torch.cat(exit_probs_for_spread, dim=0)
        else:
            exit_probs_cat = entry_probs  # fallback to avoid degenerate std

        entry_std = (entry_probs.float().std(correction=0)
                    if entry_probs.numel() >= 2 else torch.tensor(0.0, device=device))
        exit_std  = (exit_probs_cat.float().std(correction=0)
                    if exit_probs_cat.numel() >= 2 else torch.tensor(0.0, device=device))

        diversity_penalty  = 1.0 / (entry_std + 1e-3) + 1.0 / (exit_std + 1e-3)
        confidence_penalty = torch.mean(torch.exp(-10 * (entry_probs - 0.5).pow(2)))

        # ─── UNCERTAINTY-WEIGHTED SUM + EXIT RAMP ──────────────────
        precision_entry  = torch.exp(-self.log_vars[0])
        precision_return = torch.exp(-self.log_vars[2])
        precision_vol    = torch.exp(-self.log_vars[3])
        precision_vae    = torch.exp(-self.log_vars[4])

        exit_boost = getattr(self, "_current_exit_boost", 1.0)

        total_loss = (
            self.task_weights["entry"]      * (precision_entry  * entry_loss      + self.log_vars[0]) +
            exit_boost                      * (exit_bundle_loss) +
            self.task_weights["return"]     * (precision_return * return_loss     + self.log_vars[2]) +
            self.task_weights["volatility"] * (precision_vol    * volatility_loss + self.log_vars[3]) +
            self.task_weights["vae"]        * (precision_vae    * vae_loss        + self.log_vars[4]) +
            0.01 * diversity_penalty +
            0.05 * confidence_penalty
        )

        return {
            'total_loss': total_loss,
            'entry_loss': float(entry_loss.detach()),
            'return_loss': float(return_loss.detach()),
            'volatility_loss': float(volatility_loss.detach()),
            'vae_loss': float(vae_loss.detach()),
            'take_profit_loss': float(tp_loss.detach()),
            'stop_loss_loss': float(sl_loss.detach()),
            'let_run_loss': float(lr_loss.detach()),
            'regime_change_loss': float(rc_loss.detach()),
            'exit_bundle_loss': float(exit_bundle_loss.detach()),
            'diversity_penalty': float(diversity_penalty.detach()),
            'confidence_penalty': float(confidence_penalty.detach()),
            'uncertainties': self.log_vars.detach().cpu().numpy(),
        }

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance"""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    probs = torch.sigmoid(logits)
    pt = torch.where(targets == 1, probs, 1 - probs)
    focal_weight = (1 - pt) ** gamma
    loss = alpha * focal_weight * bce
    return loss.mean()


class NeuralTrainer:
    """Enhanced trainer with exit management tracking"""
    def __init__(self, model, train_loader, val_loader, config: dict, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('lr', 0.0003),
            weight_decay=config.get('weight_decay', 1e-4),
            betas=(0.9, 0.999),
            eps=1e-8
        )

        from torch.optim.lr_scheduler import SequentialLR, ConstantLR, CosineAnnealingLR

        base_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('cosine_Tmax', 60),
            eta_min=config.get('min_lr', 1e-6)
        )
        warmup = ConstantLR(self.optimizer, factor=1.0, total_iters=15)  # 15-epoch flat start
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, base_scheduler], milestones=[15])

        self.criterion = MultiTaskLoss(num_tasks=9)
        self.scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.gradient_accumulation_steps = config.get('grad_accum_steps', 2)

        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project="neural-trading-exit-management",
                    config=config,
                    name=config.get('run_name', 'exit_aware_model')
                )
                self.wandb = wandb
            except ImportError:
                print("⚠️ wandb not installed, disabling logging")
                self.use_wandb = False
                self.wandb = None
        else:
            self.wandb = None

    def train_epoch(self, epoch, on_batch=None, batch_update_every: int = 10):
        """Train for one epoch with AMP autocast correctly handled."""
        self.model.train()
        total_loss = 0.0
        loss_components = {
            'entry': 0, 'return': 0, 'volatility': 0, 'vae': 0,
            'regime_change': 0, 'take_profit': 0, 'stop_loss': 0,
            'let_run': 0, 'confidence_penalty': 0
        }

        all_take_profit_probs, all_stop_loss_probs, all_let_run_scores = [], [], []
        self.optimizer.zero_grad()
        n_batches = len(self.train_loader)

        autocast_enabled = (self.device == "cuda" and self.scaler is not None)

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            features = batch['features'].to(self.device)
            future_return = batch['future_return'].to(self.device)
            actual_volatility = batch['actual_volatility'].to(self.device)
            entry_label = batch['entry_label'].to(self.device)
            take_profit_label = batch['take_profit_label'].to(self.device)
            stop_loss_label = batch['stop_loss_label'].to(self.device)
            let_run_label = batch['let_winner_run_label'].to(self.device)
            regime_change_label = batch['regime_change_label'].to(self.device)
            unrealized_pnl = batch['unrealized_pnl'].to(self.device).unsqueeze(1)
            time_in_position = batch['time_in_position'].to(self.device).unsqueeze(1)

            if torch.isnan(features).any() or torch.isinf(features).any():
                self.optimizer.zero_grad()
                continue

            # ───────────────────────────────────────────────
            # forward pass under autocast
            # ───────────────────────────────────────────────
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
                position_context = {
                    'unrealized_pnl': unrealized_pnl,
                    'time_in_position': time_in_position
                }
                predictions = self.model(features, position_context=position_context)
                targets = {
                    'features': features,
                    'future_return': future_return,
                    'actual_volatility': actual_volatility,
                    'entry_label': entry_label,
                    'take_profit_label': take_profit_label,
                    'stop_loss_label': stop_loss_label,
                    'let_winner_run_label': let_run_label,
                    'regime_change_label': regime_change_label,
                }

                loss_dict = self.criterion(predictions, targets)
                # ensure loss tensor stays in same dtype as model output
                loss = loss_dict['total_loss'].to(self.device) / self.gradient_accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):
                self.optimizer.zero_grad()
                continue

            # ───────────────────────────────────────────────
            # backward pass outside autocast
            # ───────────────────────────────────────────────
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            # aggregate metrics
            total_loss += loss_dict['total_loss'].detach()
            for k in loss_components.keys():
                if f'{k}_loss' in loss_dict:
                    loss_components[k] += loss_dict[f'{k}_loss']

            # exit-signal diagnostics
            with torch.no_grad():
                exit_signals = predictions.get('exit_signals', {})
                if 'profit_taking' in exit_signals:
                    tp = exit_signals['profit_taking'].get('take_profit_prob')
                    if tp is not None:
                        all_take_profit_probs.extend(tp.detach().cpu().numpy().flatten())
                if 'let_winner_run' in exit_signals:
                    lr = exit_signals['let_winner_run'].get('hold_score')
                    if lr is not None:
                        all_let_run_scores.extend(lr.detach().cpu().numpy().flatten())
                if 'stop_loss' in exit_signals:
                    sl = exit_signals['stop_loss'].get('stop_loss_prob')
                    if sl is not None:
                        all_stop_loss_probs.extend(sl.detach().cpu().numpy().flatten())

            # rich callback
            if on_batch and ((batch_idx + 1) % batch_update_every == 0):
                on_batch(batch_idx + 1, n_batches, float(loss.item()), self.optimizer.param_groups[0]['lr'])

        # epoch aggregates
        avg_loss = total_loss / max(1, n_batches)
        avg_components = {k: v / max(1, n_batches) for k, v in loss_components.items()}

        if all_take_profit_probs or all_stop_loss_probs or all_let_run_scores:
            print(f"\n📊 Exit Signal Distribution (Epoch {epoch}):")
            if all_take_profit_probs:
                tp = np.array(all_take_profit_probs)
                print(f"   Take-Profit  Mean={tp.mean():.3f}, Std={tp.std():.3f}, >0.7={100*(tp>0.7).mean():.1f}%")
            if all_stop_loss_probs:
                sl = np.array(all_stop_loss_probs)
                print(f"   Stop-Loss    Mean={sl.mean():.3f}, Std={sl.std():.3f}, >0.7={100*(sl>0.7).mean():.1f}%")
            if all_let_run_scores:
                lr = np.array(all_let_run_scores)
                print(f"   Let-Run      Mean={lr.mean():.3f}, Std={lr.std():.3f}, >0.7={100*(lr>0.7).mean():.1f}%")

        return avg_loss, avg_components

    def train(self, num_epochs):
        """Full training loop with non-flickering Rich dashboard + per-batch Train & Val progress (tensor-safe)."""
        from rich.live import Live
        from rich.table import Table
        from rich.progress import (
            Progress, BarColumn, TextColumn, TimeElapsedColumn,
            TimeRemainingColumn, MofNCompleteColumn
        )
        from rich.console import Group
        import numpy as np
        import psutil
        import torch

        console = Console()

        # fallback
        if not self.config.get('rich_dashboard', True):
            console.print("[yellow]⚠️ Rich dashboard disabled — using tqdm output[/yellow]")
            return self._legacy_train(num_epochs)

        # --- helper: safe float ---
        def _sf(x):
            import torch
            if isinstance(x, torch.Tensor):
                x = x.detach()
                if x.numel() == 1:
                    x = x.item()
                else:
                    x = x.float().mean().item()
            try:
                return float(x)
            except Exception:
                return 0.0

        # --- epoch progress (outer) ---
        epoch_progress = Progress(
            TextColumn("[cyan]Epoch[/cyan] {task.fields[epoch]:03d}"),
            BarColumn(complete_style="bright_magenta"),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•", TextColumn("[green]Train:[/green] {task.fields[loss]:.4f}"),
            "•", TextColumn("[yellow]Val:[/yellow] {task.fields[val]:.4f}"),
            "•", TextColumn("[blue]LR:[/blue] {task.fields[lr]:.6f}"),
            TimeElapsedColumn(),
            expand=True,
        )
        epoch_task = epoch_progress.add_task(
            "Epochs",
            total=num_epochs,
            epoch=0,
            loss=0.0,
            val=0.0,
            lr=_sf(self.optimizer.param_groups[0]['lr'])
        )

        # --- training batch progress ---
        train_progress = Progress(
            TextColumn("[white]Train[/white]"),
            BarColumn(complete_style="magenta"),
            MofNCompleteColumn(),
            "•", TextColumn("loss {task.fields[bloss]:.4f}"),
            "•", TextColumn("lr {task.fields[lr]:.6f}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=True,
        )
        train_task = train_progress.add_task(
            "Batches",
            total=max(1, len(self.train_loader)),
            bloss=0.0,
            lr=_sf(self.optimizer.param_groups[0]['lr'])
        )

        # --- validation batch progress ---
        val_progress = Progress(
            TextColumn("[white]Validation[/white]"),
            BarColumn(complete_style="bright_blue"),
            MofNCompleteColumn(),
            "•", TextColumn("loss {task.fields[bloss]:.4f}"),
            "•", TextColumn("lr {task.fields[lr]:.6f}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=True,
        )
        val_task = val_progress.add_task(
            "Batches",
            total=max(1, len(self.val_loader)),
            bloss=0.0,
            lr=_sf(self.optimizer.param_groups[0]['lr'])
        )

        # --- metrics dashboard ---
        dashboard = Table(expand=True, show_header=True, header_style="bold white")
        dashboard.add_column("Metric", justify="left", style="bold cyan")
        dashboard.add_column("Value", justify="right")
        self.loss_history = []

        def update_dashboard(epoch, train_loss, val_loss, entry_acc, exit_acc, lr, grad_norm):
            epoch     = int(epoch)
            train_loss = _sf(train_loss)
            val_loss   = _sf(val_loss)
            entry_acc  = _sf(entry_acc)
            exit_acc   = _sf(exit_acc)
            lr         = _sf(lr)
            grad_norm  = None if grad_norm is None else _sf(grad_norm)

            table = Table(expand=True, show_header=True, header_style="bold white")
            table.add_column("Metric", justify="left", style="bold cyan")
            table.add_column("Value", justify="right")

            table.add_row("Epoch", f"{epoch}")
            table.add_row("Train Loss", f"{train_loss:.5f}")
            table.add_row("Val Loss",   f"{val_loss:.5f}")
            table.add_row("Entry Acc",  f"{entry_acc:.4f}")
            table.add_row("Exit Acc",   f"{exit_acc:.4f}")
            table.add_row("Grad Norm",  f"{grad_norm:.4f}" if grad_norm is not None else "—")
            table.add_row("LR",         f"{lr:.6f}")
            table.add_row(
                "GPU Mem (MB)" if torch.cuda.is_available() else "CPU Mem (%)",
                f"{(torch.cuda.memory_allocated(self.device) / 1e6) if torch.cuda.is_available() else psutil.virtual_memory().percent:.1f}"
            )

            if len(self.loss_history) > 2:
                chars = "▁▂▃▄▅▆▇█"
                vals = [ _sf(v) for v in self.loss_history ]
                lo, hi = min(vals), max(vals)
                if hi > lo:
                    scaled = np.interp(vals, (lo, hi), (1, len(chars)))
                    spark = "".join(chars[int(x) - 1] for x in scaled.astype(int))
                    table.add_row("Loss Trend", spark)

            group.renderables[-1] = table

        # --- combine all into one live layout ---
        group = Group(epoch_progress, train_progress, val_progress, dashboard)

        with Live(group, refresh_per_second=6, console=console, transient=False):
            # seed dashboard
            update_dashboard(
                epoch=0,
                train_loss=0.0,
                val_loss=0.0,
                entry_acc=0.0,
                exit_acc=0.0,
                lr=_sf(self.optimizer.param_groups[0]['lr']),
                grad_norm=None,
            )
            for epoch in range(num_epochs):
                # reset train bar
                train_progress.reset(
                    train_task,
                    total=max(1, len(self.train_loader)),
                    completed=0,
                    bloss=0.0,
                    lr=_sf(self.optimizer.param_groups[0]['lr'])
                )

                def on_train_batch(bi, total, loss, lr):
                    shown_loss = 0.0 if (loss is None or (isinstance(loss, float) and np.isnan(loss))) else _sf(loss)
                    train_progress.update(train_task, completed=bi, bloss=shown_loss, lr=_sf(lr))
                
                self.criterion.set_exit_weight_ramp(
                    epoch=epoch,
                    ramp_epochs=self.config.get('exit_ramp_epochs', 10),
                    max_boost=self.config.get('exit_max_boost', 8.0),
                )
                
                train_loss, _ = self.train_epoch(epoch, on_batch=on_train_batch, batch_update_every=10)

                # reset val bar
                val_progress.reset(
                    val_task,
                    total=max(1, len(self.val_loader)),
                    completed=0,
                    bloss=0.0,
                    lr=_sf(self.optimizer.param_groups[0]['lr'])
                )

                def on_val_batch(bi, total, loss, lr):
                    shown_loss = 0.0 if (loss is None or (isinstance(loss, float) and np.isnan(loss))) else _sf(loss)
                    val_progress.update(val_task, completed=bi, bloss=shown_loss, lr=_sf(lr))

                val_loss, _, entry_acc, exit_acc = self.validate(on_batch=on_val_batch, batch_update_every=10)

                # step scheduler
                self.scheduler.step()

                lr_now = _sf(self.optimizer.param_groups[0]['lr'])
                grad_norm = getattr(self, 'last_grad_norm', None)
                self.loss_history.append(_sf(val_loss))

                # update epoch prog + metrics
                epoch_progress.update(
                    epoch_task,
                    advance=1,
                    epoch=int(epoch),
                    loss=_sf(train_loss),
                    val=_sf(val_loss),
                    lr=lr_now
                )
                update_dashboard(epoch, train_loss, val_loss, entry_acc, exit_acc, lr_now, grad_norm)

                # logging & checkpoints
                if self.use_wandb and self.wandb is not None:
                    self.wandb.log({
                        'epoch': epoch,
                        'train_loss': _sf(train_loss),
                        'val_loss': _sf(val_loss),
                        'entry_accuracy': _sf(entry_acc),
                        'exit_accuracy': _sf(exit_acc),
                        'learning_rate': lr_now
                    })

                if _sf(val_loss) < _sf(self.best_val_loss):
                    self.best_val_loss = _sf(val_loss)
                    self.patience_counter = 0
                    best_path = self.config.get('best_model_path', 'models/best_model.pt')
                    self.save_checkpoint(best_path, epoch, _sf(val_loss))
                    console.print(f"[green]💾 New best model saved![/green] Val loss: {_sf(val_loss):.4f}")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.get('patience', 25):
                        console.print(f"[red]⏹️ Early stopping triggered at epoch {epoch + 1}[/red]")
                        break

        console.print("\n[bold green]✅ Training completed successfully![/bold green]")

    def validate(self, on_batch=None, batch_update_every: int = 10):
        """Validate model on the current val_loader with Rich dashboard support (returns pure floats)."""
        import numpy as np
        import torch

        self.model.eval()
        total_loss = 0.0
        loss_components = {
            'entry': 0, 'return': 0, 'volatility': 0, 'vae': 0,
            'regime_change': 0, 'take_profit': 0, 'stop_loss': 0,
            'let_run': 0, 'confidence_penalty': 0
        }

        entry_correct, total_samples = 0, 0
        all_entry_probs, all_entry_labels = [], []

        n_batches = len(self.val_loader)
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Move data to device
                features = batch['features'].to(self.device)
                future_return = batch['future_return'].to(self.device)
                actual_volatility = batch['actual_volatility'].to(self.device)
                entry_label = batch['entry_label'].to(self.device)

                # new exit labels
                take_profit_label = batch['take_profit_label'].to(self.device)
                stop_loss_label   = batch['stop_loss_label'].to(self.device)
                let_run_label     = batch['let_winner_run_label'].to(self.device)
                regime_change_label = batch['regime_change_label'].to(self.device)

                unrealized_pnl   = batch['unrealized_pnl'].to(self.device).unsqueeze(1)
                time_in_position = batch['time_in_position'].to(self.device).unsqueeze(1)

                # Forward
                position_context = {'unrealized_pnl': unrealized_pnl, 'time_in_position': time_in_position}
                predictions = self.model(features, position_context=position_context)

                targets = {
                    'features': features,
                    'future_return': future_return,
                    'actual_volatility': actual_volatility,
                    'entry_label': entry_label,
                    'take_profit_label': take_profit_label,
                    'stop_loss_label': stop_loss_label,
                    'let_winner_run_label': let_run_label,
                    'regime_change_label': regime_change_label,
                }

                loss_dict = self.criterion(predictions, targets)
                total_loss += float(loss_dict['total_loss'].detach().cpu())

                for k in loss_components.keys():
                    key = f'{k}_loss'
                    if key in loss_dict:
                        loss_components[k] += float(loss_dict[key])

                # Entry accuracy (threshold 0.5)
                entry_prob = predictions.get('entry_prob', torch.sigmoid(predictions['entry_logits']))
                entry_pred = (entry_prob > 0.5).float().squeeze()
                entry_true = (entry_label > 0.5).float()
                entry_correct += (entry_pred == entry_true).sum().item()
                total_samples += len(entry_label)

                # Spread (optional)
                all_entry_probs.extend(entry_prob.detach().cpu().numpy().ravel().tolist())
                all_entry_labels.extend(entry_label.detach().cpu().numpy().ravel().tolist())

                # live batch update
                if on_batch and ((batch_idx + 1) % batch_update_every == 0):
                    on_batch(
                        batch_idx + 1,
                        n_batches,
                        float(loss_dict['total_loss']),
                        self.optimizer.param_groups[0]['lr'],
                    )

        avg_loss = total_loss / max(1, n_batches)
        avg_components = {k: v / max(1, n_batches) for k, v in loss_components.items()}
        entry_accuracy = float(entry_correct) / max(1, float(total_samples))
        exit_accuracy = 0.0  # placeholder with multi-exit heads

        # Save spread for dashboard
        if all_entry_probs:
            ap = np.array(all_entry_probs)
            al = np.array(all_entry_labels) if all_entry_labels else np.array([0.0])
            self._last_val_spread = {
                "std": float(ap.std()) if ap.size else 0.0,
                "hi": float((ap > 0.7).mean()) if ap.size else 0.0,
                "lo": float((ap < 0.3).mean()) if ap.size else 0.0,
                "label_mean": float(al.mean()) if al.size else 0.0,
            }

        return float(avg_loss), avg_components, float(entry_accuracy), float(exit_accuracy)

    def save_checkpoint(self, filename, epoch, val_loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['val_loss']