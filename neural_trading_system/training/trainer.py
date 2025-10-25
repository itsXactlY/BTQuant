import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import polars as pl
import time
from pathlib import Path
from rich.console import Console
import wandb
import hashlib
import json

import psutil
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
class TradingDataset(Dataset):
    """Enhanced dataset with writable safety, soft labels, and adaptive thresholds."""
    def __init__(self, features: np.ndarray, returns: np.ndarray, seq_len: int = 100, prediction_horizon: int = 1):
        # 🧩 Ensure NumPy arrays are writable before Torch conversion
        if not features.flags.writeable:
            features = np.array(features, copy=True)
            console.print("[yellow]⚠️ Copied read-only feature array to writable memory.[/yellow]")
        if not returns.flags.writeable:
            returns = np.array(returns, copy=True)
            console.print("[yellow]⚠️ Copied read-only returns array to writable memory.[/yellow]")

        # 🔢 Convert to tensors
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.returns = torch.as_tensor(returns, dtype=torch.float32)
        self.seq_len = seq_len
        self.prediction_horizon = prediction_horizon

        # 🧹 Handle NaN/Inf values
        if torch.isnan(self.features).any() or torch.isinf(self.features).any():
            console.print("[yellow]⚠️ NaN/Inf detected in features — replacing with zeros.[/yellow]")
            self.features = torch.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)

        # ⚠️ Clamp extreme feature magnitudes
        features_max = self.features.abs().max()
        if features_max > 100:
            console.print(f"[yellow]⚠️ Extreme feature values detected (max={features_max:.2e}). Clamping to [-100, 100].[/yellow]")
            self.features = torch.clamp(self.features, min=-100, max=100)

        # 📏 Dataset structure
        self.valid_length = max(0, len(self.features) - seq_len - prediction_horizon + 1)

        # 📈 Return statistics for adaptive thresholds
        self.returns_std = float(np.std(returns))
        self.returns_mean = float(np.mean(returns))
        self.entry_threshold = max(0.01, 1.5 * self.returns_std)
        self.exit_threshold = max(0.005, 0.75 * self.returns_std)

        # 🧠 Console summary
        console.print(f"[cyan]📊 Dataset Statistics:[/cyan]")
        console.print(f"   • Returns mean: {self.returns_mean:.6f}")
        console.print(f"   • Returns std: {self.returns_std:.6f}")
        console.print(f"   • Entry threshold: {self.entry_threshold:.4f}")
        console.print(f"   • Exit threshold: {self.exit_threshold:.4f}")

    def __len__(self):
        return self.valid_length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.valid_length:
            raise IndexError(f"Index {idx} out of range for dataset of length {self.valid_length}")

        start = idx
        end = idx + self.seq_len
        label_idx = end + self.prediction_horizon - 1

        feature_seq = self.features[start:end]
        future_return = self.returns[label_idx]

        # 🔮 Compute short-term realized volatility
        return_window = self.returns[end:end + self.prediction_horizon]
        if return_window.numel() == 0:
            actual_volatility = torch.tensor(0.0, dtype=torch.float32)
        elif return_window.numel() == 1:
            actual_volatility = torch.abs(return_window[0])
        else:
            actual_volatility = torch.std(return_window, unbiased=False)

        # 🎯 Soft labels for entry/exit prediction
        entry_label = torch.sigmoid(future_return * 100)
        exit_label = torch.sigmoid(-future_return * 100)

        return {
            'features': feature_seq,
            'future_return': future_return,
            'actual_volatility': actual_volatility,
            'entry_label': entry_label,
            'exit_label': exit_label
        }


class MultiTaskLoss(nn.Module):
    """Multi-task loss with temperature scaling, confidence penalty, focal loss, and Huber loss"""
    def __init__(self, num_tasks=5):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.entry_temp = nn.Parameter(torch.ones(1))
        self.exit_temp = nn.Parameter(torch.ones(1))

    def focal_loss(self, logits, labels, alpha=0.75, gamma=2.0):
        """Focal Loss for addressing class imbalance."""
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        pt = torch.exp(-bce)
        focal = alpha * (1 - pt) ** gamma * bce
        return focal.mean()

    def forward(self, predictions: dict, targets: dict):
        """Multi-task loss with focal loss, diversity penalty, and proper device handling."""
        device = predictions['entry_logits'].device

        # 1. ENTRY SIGNAL LOSS
        temp = torch.clamp(self.entry_temp, min=0.1, max=10.0).to(device)
        logits_entry = predictions['entry_logits'].view(-1) / temp
        labels_entry = targets['entry_label'].view(-1)
        entry_loss = self.focal_loss(logits_entry, labels_entry, alpha=0.75, gamma=2.0)

        # 2. EXIT SIGNAL LOSS
        temp_exit = torch.clamp(self.exit_temp, min=0.1, max=10.0).to(device)
        logits_exit = predictions['exit_logits'].view(-1) / temp_exit
        labels_exit = targets['exit_label'].view(-1)
        exit_loss = self.focal_loss(logits_exit, labels_exit, alpha=0.75, gamma=2.0)

        # 3. PREDICTION DIVERSITY PENALTY
        entry_probs = predictions['entry_prob'].view(-1)
        exit_probs = predictions['exit_prob'].view(-1)

        entry_std = entry_probs.std()
        exit_std = exit_probs.std()

        diversity_penalty = 1.0 / (entry_std + 1e-3) + 1.0 / (exit_std + 1e-3)

        confidence_penalty = (
            torch.mean(torch.exp(-10 * (entry_probs - 0.5).pow(2))) +
            torch.mean(torch.exp(-10 * (exit_probs - 0.5).pow(2)))
        )

        # 4. RETURN PREDICTION LOSS
        ret_pred = predictions['expected_return'].view(-1)
        ret_true = targets['future_return'].view(-1)
        return_loss = F.smooth_l1_loss(ret_pred, ret_true)

        # 5. VOLATILITY FORECASTING LOSS
        vol_pred = predictions['volatility_forecast'].view(-1)
        vol_true = targets['actual_volatility'].view(-1)
        volatility_loss = F.mse_loss(vol_pred, vol_true)

        # 6. VAE LOSS
        vae_recon_loss = F.mse_loss(
            predictions['vae_recon'],
            predictions['sequence_repr']
        )

        regime_logvar_clamped = torch.clamp(predictions['regime_logvar'], min=-10, max=10)

        kl_loss = -0.5 * torch.sum(
            1 + regime_logvar_clamped - 
            predictions['regime_mu'].pow(2) - 
            regime_logvar_clamped.exp(),
            dim=1
        ).mean()

        vae_loss = vae_recon_loss + 0.00001 * kl_loss

        # 7. UNCERTAINTY-WEIGHTED COMBINATION
        precision_entry = torch.exp(-self.log_vars[0])
        precision_exit = torch.exp(-self.log_vars[1])
        precision_return = torch.exp(-self.log_vars[2])
        precision_volatility = torch.exp(-self.log_vars[3])
        precision_vae = torch.exp(-self.log_vars[4])

        total_loss = (
            precision_entry * entry_loss + self.log_vars[0] +
            precision_exit * exit_loss + self.log_vars[1] +
            precision_return * return_loss + self.log_vars[2] +
            precision_volatility * volatility_loss + self.log_vars[3] +
            precision_vae * vae_loss + self.log_vars[4] +
            0.01 * diversity_penalty +
            0.05 * confidence_penalty
        )

        return {
            'total_loss': total_loss,
            'entry_loss': entry_loss.item(),
            'exit_loss': exit_loss.item(),
            'return_loss': return_loss.item(),
            'volatility_loss': volatility_loss.item(),
            'vae_loss': vae_loss.item(),
            'diversity_penalty': diversity_penalty.item(),
            'confidence_penalty': confidence_penalty.item(),
            'entry_temp': self.entry_temp.item(),
            'exit_temp': self.exit_temp.item(),
            'entry_std': entry_std.item(),
            'exit_std': exit_std.item(),
            'uncertainties': self.log_vars.detach().cpu().numpy()
        }


class NeuralTrainer:
    """Training loop with proper GradScaler handling and gradient stability"""
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config: dict,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
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

        # ✅ FIXED: Use StepLR instead of CosineAnnealingWarmRestarts
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10,
            gamma=0.5
        )

        self.criterion = MultiTaskLoss(num_tasks=5)
        self.scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None

        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.gradient_accumulation_steps = config.get('grad_accum_steps', 2)

        self.use_wandb = config.get('use_wandb', True)
        if self.use_wandb:
            try:
                wandb.init(
                    project="neural-trading-system",
                    config=config,
                    name=config.get('run_name', 'experiment')
                )
                self.wandb = wandb
            except ImportError:
                print("⚠️ wandb not installed, disabling logging")
                self.use_wandb = False
                self.wandb = None
        else:
            self.wandb = None

    def train_epoch(self, epoch, on_batch=None, batch_update_every: int = 10):
        """Train for one epoch with optional per-batch UI callback."""
        self.model.train()
        total_loss = 0.0
        loss_components = {
            'entry': 0, 'exit': 0, 'return': 0,
            'volatility': 0, 'vae': 0, 'confidence_penalty': 0
        }

        all_entry_probs, all_exit_probs = [], []
        nan_batches_skipped = 0
        self.optimizer.zero_grad()

        n_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            features = batch['features'].to(self.device)
            future_return = batch['future_return'].to(self.device)
            actual_volatility = batch['actual_volatility'].to(self.device)
            entry_label = batch['entry_label'].to(self.device)
            exit_label = batch['exit_label'].to(self.device)

            if torch.isnan(features).any() or torch.isinf(features).any():
                nan_batches_skipped += 1
                self.optimizer.zero_grad()
                # notify UI even if we skip
                if callable(on_batch) and (batch_idx % batch_update_every == 0):
                    on_batch(batch_idx + 1, n_batches, float('nan'),
                            self.optimizer.param_groups[0]['lr'])
                continue

            with (torch.amp.autocast('cuda') if self.scaler else torch.enable_grad()):
                predictions = self.model(features)
                targets = {
                    'features': features,
                    'future_return': future_return,
                    'actual_volatility': actual_volatility,
                    'entry_label': entry_label,
                    'exit_label': exit_label
                }
                loss_dict = self.criterion(predictions, targets)
                loss = loss_dict['total_loss'] / self.gradient_accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches_skipped += 1
                self.optimizer.zero_grad()
                if callable(on_batch) and (batch_idx % batch_update_every == 0):
                    on_batch(batch_idx + 1, n_batches, float('nan'),
                            self.optimizer.param_groups[0]['lr'])
                continue

            # backward
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # grad step (with accumulation)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                try:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                except RuntimeError as e:
                    if "already been called" not in str(e):
                        raise

                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.last_grad_norm = float(grad_norm)

                has_nan_grad = any(
                    (p.grad is not None and torch.isnan(p.grad).any())
                    for p in self.model.parameters()
                )
                if has_nan_grad:
                    nan_batches_skipped += 1
                    self.optimizer.zero_grad()
                else:
                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

            # accounting
            total_loss += loss_dict['total_loss']
            loss_components['entry'] += loss_dict['entry_loss']
            loss_components['exit'] += loss_dict['exit_loss']
            loss_components['return'] += loss_dict['return_loss']
            loss_components['volatility'] += loss_dict['volatility_loss']
            loss_components['vae'] += loss_dict['vae_loss']
            loss_components['confidence_penalty'] += loss_dict['confidence_penalty']

            with torch.no_grad():
                all_entry_probs.extend(predictions['entry_prob'].detach().cpu().numpy().ravel().tolist())
                all_exit_probs.extend(predictions['exit_prob'].detach().cpu().numpy().ravel().tolist())

            # 🔁 push per-batch update to Rich every N batches
            if callable(on_batch) and ((batch_idx + 1) % batch_update_every == 0):
                on_batch(
                    batch_idx + 1,
                    n_batches,
                    float(loss_dict['total_loss']),
                    self.optimizer.param_groups[0]['lr'],
                )

        n_valid_batches = max(n_batches - nan_batches_skipped, 1)
        avg_loss = total_loss / n_valid_batches
        avg_components = {k: v / n_valid_batches for k, v in loss_components.items()}

        # (keep the prediction spread prints if you like)
        return avg_loss, avg_components

    def _legacy_train(self, num_epochs):
        """Classic tqdm-only training loop (fallback mode)."""
        console.print("[cyan]Running legacy tqdm training loop...[/cyan]")

        for epoch in range(num_epochs):
            train_loss, train_components = self.train_epoch(epoch)
            val_loss, val_components, entry_acc, exit_acc = self.validate()
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]['lr']
            console.print(
                f"[bold cyan]Epoch {epoch}[/bold cyan] "
                f"Train Loss: [green]{train_loss:.4f}[/green] | "
                f"Val Loss: [yellow]{val_loss:.4f}[/yellow] | "
                f"Entry Acc: {entry_acc:.4f} | Exit Acc: {exit_acc:.4f} | LR: {lr:.6f}"
            )

            if self.use_wandb and self.wandb is not None:
                self.wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "entry_acc": entry_acc,
                    "exit_acc": exit_acc,
                    "lr": lr,
                })

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                best_path = self.config.get('best_model_path', 'models/best_model.pt')
                self.save_checkpoint(best_path, epoch, val_loss)
                console.print(f"[green]💾 New best model saved![/green] Val loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.get('patience', 25):
                console.print(f"[red]⏹️ Early stopping triggered at epoch {epoch + 1}[/red]")
                break

        console.print("\n[bold green]✅ Training completed successfully (legacy mode)![/bold green]")

    def train(self, num_epochs):
        """Full training loop with non-flickering Rich dashboard + per-batch Train & Val progress."""
        from rich.live import Live
        from rich.table import Table
        from rich.progress import (
            Progress, BarColumn, TextColumn, TimeElapsedColumn,
            TimeRemainingColumn, MofNCompleteColumn
        )
        from rich.console import Group

        if not self.config.get('rich_dashboard', True):
            console.print("[yellow]⚠️ Rich dashboard disabled — using tqdm output[/yellow]")
            return self._legacy_train(num_epochs)

        # ── epoch progress bar (outer) ─────────────────────────────────────────────
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
            "Epochs", total=num_epochs, epoch=0, loss=0.0, val=0.0, lr=self.optimizer.param_groups[0]['lr']
        )

        # ── training batch progress ────────────────────────────────────────────────
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
            "Batches", total=max(1, len(self.train_loader)), bloss=0.0, lr=self.optimizer.param_groups[0]['lr']
        )

        # ── validation batch progress (NEW) ────────────────────────────────────────
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
            "Batches", total=max(1, len(self.val_loader)), bloss=0.0, lr=self.optimizer.param_groups[0]['lr']
        )

        # ── metrics table ──────────────────────────────────────────────────────────
        dashboard = Table(expand=True, show_header=True, header_style="bold white")
        dashboard.add_column("Metric", justify="left", style="bold cyan")
        dashboard.add_column("Value", justify="right")
        self.loss_history = []

        def update_dashboard(epoch, train_loss, val_loss, entry_acc, exit_acc, lr, grad_norm):
            """Safely rebuild the metrics table each refresh (avoids IndexError in Live)."""
            table = Table(expand=True, show_header=True, header_style="bold white")
            table.add_column("Metric", justify="left", style="bold cyan")
            table.add_column("Value", justify="right")

            table.add_row("Epoch", f"{epoch}")
            table.add_row("Train Loss", f"{train_loss:.5f}")
            table.add_row("Val Loss", f"{val_loss:.5f}")
            table.add_row("Entry Acc", f"{entry_acc:.4f}")
            table.add_row("Exit Acc", f"{exit_acc:.4f}")
            table.add_row("Grad Norm", f"{grad_norm:.4f}" if grad_norm else "—")
            table.add_row("LR", f"{lr:.6f}")
            table.add_row(
                "GPU Mem (MB)" if torch.cuda.is_available() else "CPU Mem (%)",
                f"{(torch.cuda.memory_allocated(self.device) / 1e6) if torch.cuda.is_available() else psutil.virtual_memory().percent:.1f}"
            )

            if hasattr(self, "_last_val_spread"):
                s = self._last_val_spread
                table.add_row(
                    "Val Spread (std/hi/lo/ym)",
                    f"{s['std']:.3f} / {100*s['hi']:.1f}% / {100*s['lo']:.1f}% / {s['label_mean']:.3f}"
                )

            if len(self.loss_history) > 2:
                chars = "▁▂▃▄▅▆▇█"
                # safely convert CUDA tensors to CPU floats
                loss_values = [
                    float(x.detach().cpu()) if torch.is_tensor(x) else float(x)
                    for x in self.loss_history
                ]
                # normalize into [1, len(chars)]
                scaled = np.interp(loss_values, (min(loss_values), max(loss_values)), (1, len(chars)))
                spark = "".join(chars[int(x) - 1] for x in scaled.astype(int))
                table.add_row("Loss Trend", spark)

            group.renderables[-1] = table

        # compose a single Live group to avoid flicker
        group = Group(epoch_progress, train_progress, val_progress, dashboard)

        with Live(group, refresh_per_second=6, console=console, transient=False):
            for epoch in range(num_epochs):
                # reset train progress
                train_progress.reset(
                    train_task, total=max(1, len(self.train_loader)), completed=0,
                    bloss=0.0, lr=self.optimizer.param_groups[0]['lr']
                )

                # per-batch train callback
                def on_train_batch(bi, total, loss, lr):
                    shown_loss = 0.0 if (loss is None or (isinstance(loss, float) and np.isnan(loss))) else loss
                    train_progress.update(train_task, completed=bi, bloss=shown_loss, lr=lr)

                # train epoch
                train_loss, _ = self.train_epoch(epoch, on_batch=on_train_batch, batch_update_every=10)

                # reset val progress
                val_progress.reset(
                    val_task, total=max(1, len(self.val_loader)), completed=0,
                    bloss=0.0, lr=self.optimizer.param_groups[0]['lr']
                )

                # per-batch validation callback
                def on_val_batch(bi, total, loss, lr):
                    shown_loss = 0.0 if (loss is None or (isinstance(loss, float) and np.isnan(loss))) else loss
                    val_progress.update(val_task, completed=bi, bloss=shown_loss, lr=lr)

                # validate
                val_loss, _, entry_acc, exit_acc = self.validate(on_batch=on_val_batch, batch_update_every=10)

                # scheduler
                self.scheduler.step()

                lr = self.optimizer.param_groups[0]['lr']
                grad_norm = getattr(self, 'last_grad_norm', None)
                self.loss_history.append(float(val_loss.detach().cpu()) if torch.is_tensor(val_loss) else float(val_loss))

                # update epoch progress & dashboard
                epoch_progress.update(epoch_task, advance=1, epoch=epoch, loss=train_loss, val=val_loss, lr=lr)
                update_dashboard(epoch, train_loss, val_loss, entry_acc, exit_acc, lr, grad_norm)

                # logging & checkpoints
                if self.use_wandb and self.wandb is not None:
                    self.wandb.log({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'entry_accuracy': entry_acc,
                        'exit_accuracy': exit_acc,
                        'learning_rate': lr
                    })

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    best_path = self.config.get('best_model_path', 'models/best_model.pt')
                    self.save_checkpoint(best_path, epoch, val_loss)
                    console.print(f"[green]💾 New best model saved![/green] Val loss: {val_loss:.4f}")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.get('patience', 25):
                        console.print(f"[red]⏹️ Early stopping triggered at epoch {epoch + 1}[/red]")
                        break

        console.print("\n[bold green]✅ Training completed successfully![/bold green]")

    def validate(self, on_batch=None, batch_update_every: int = 10):
        """Validate with optional per-batch UI callback (no tqdm, no console spam)."""
        self.model.eval()
        total_loss = 0.0
        loss_components = {
            'entry': 0, 'exit': 0, 'return': 0,
            'volatility': 0, 'vae': 0, 'confidence_penalty': 0
        }

        entry_correct = 0
        exit_correct = 0
        total_samples = 0

        all_entry_probs = []
        all_entry_labels = []

        n_batches = len(self.val_loader)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                features = batch['features'].to(self.device)
                future_return = batch['future_return'].to(self.device)
                actual_volatility = batch['actual_volatility'].to(self.device)
                entry_label = batch['entry_label'].to(self.device)
                exit_label = batch['exit_label'].to(self.device)

                predictions = self.model(features)
                targets = {
                    'features': features,
                    'future_return': future_return,
                    'actual_volatility': actual_volatility,
                    'entry_label': entry_label,
                    'exit_label': exit_label
                }

                loss_dict = self.criterion(predictions, targets)
                total_loss += loss_dict['total_loss']
                loss_components['entry'] += loss_dict['entry_loss']
                loss_components['exit'] += loss_dict['exit_loss']
                loss_components['return'] += loss_dict['return_loss']
                loss_components['volatility'] += loss_dict['volatility_loss']
                loss_components['vae'] += loss_dict['vae_loss']
                loss_components['confidence_penalty'] += loss_dict['confidence_penalty']

                # accuracy (threshold 0.5 on probs / labels)
                entry_pred = (predictions['entry_prob'] > 0.5).float().squeeze()
                exit_pred = (predictions['exit_prob'] > 0.5).float().squeeze()
                entry_true = (entry_label > 0.5).float()
                exit_true = (exit_label > 0.5).float()
                entry_correct += (entry_pred == entry_true).sum().item()
                exit_correct += (exit_pred == exit_true).sum().item()
                total_samples += len(entry_label)

                # spread (optional use later)
                all_entry_probs.extend(predictions['entry_prob'].cpu().numpy().ravel().tolist())
                all_entry_labels.extend(entry_label.cpu().numpy().ravel().tolist())

                # 🔁 push per-batch update to UI every N batches
                if callable(on_batch) and ((batch_idx + 1) % batch_update_every == 0):
                    on_batch(
                        batch_idx + 1,
                        n_batches,
                        float(loss_dict['total_loss']),
                        self.optimizer.param_groups[0]['lr'],
                    )

        avg_loss = total_loss / max(1, n_batches)
        avg_components = {k: v / max(1, n_batches) for k, v in loss_components.items()}
        entry_accuracy = entry_correct / max(total_samples, 1)
        exit_accuracy = exit_correct / max(total_samples, 1)

        # (Keep spread if you want to display somewhere else in the dashboard)
        self._last_val_spread = {
            "std": float(np.std(np.array(all_entry_probs))) if all_entry_probs else 0.0,
            "hi": float((np.array(all_entry_probs) > 0.7).mean()) if all_entry_probs else 0.0,
            "lo": float((np.array(all_entry_probs) < 0.3).mean()) if all_entry_probs else 0.0,
            "label_mean": float(np.mean(np.array(all_entry_labels))) if all_entry_labels else 0.0,
        }

        return avg_loss, avg_components, entry_accuracy, exit_accuracy

    def save_checkpoint(self, filename, epoch, val_loss):
        """Save model checkpoint."""
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
        """Load model checkpoint."""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['val_loss']
