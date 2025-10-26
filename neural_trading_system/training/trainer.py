import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from rich.console import Console
import torch
import math

# ============================================================
# MultiTaskLoss (single, stable implementation)
# ============================================================

class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with ramped exit weight and numerically stable (logits-based) supervision.
    - Entry: focal on logits
    - Exits: BCEWithLogits on logits (tp/sl/hold)
    - Returns/Vol/VAE: standard losses
    """
    def __init__(self, num_tasks: int = 5):
        super().__init__()
        # log_vars: entry, (unused placeholder), return, vol, vae
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.entry_temp = nn.Parameter(torch.ones(1))

        self.task_weights = {
            "entry": 1.0,
            "return": 1.5,
            "volatility": 2.0,
            "vae": 1.0,
        }
        self._exit_boost = 1.0  # ramped externally per epoch

    # ---------- helpers ----------
    @staticmethod
    def _batch_alpha(y: torch.Tensor, lo: float = 0.25, hi: float = 0.75) -> float:
        """Dynamic α for focal loss based on batch positive rate."""
        with torch.no_grad():
            p = y.float().mean().clamp_(1e-6, 1 - 1e-6).item()
        return float(max(lo, min(hi, 1.0 - p)))

    @staticmethod
    def _focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.5, gamma: float = 2.0) -> torch.Tensor:
        """Numerically stable focal loss on logits (no in-place ops)."""
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt  = torch.exp(-bce).clamp(1e-6, 1.0)
        loss = alpha * (1.0 - pt) ** gamma * bce
        return loss.mean()

    def set_exit_weight_ramp(self, epoch: int, ramp_epochs: int = 10, max_boost: float = 12.0):  # INCREASED from 8.0 for stronger exit learning):
        """Call from trainer each epoch to ramp exit importance."""


    def compute_attention_entropy_loss(self, attention_weights: list, weight: float = 0.01) -> torch.Tensor:
        """Compute attention entropy regularization to prevent collapse."""
        if not attention_weights:
            return torch.tensor(0.0, device=self.log_vars.device)

        total_entropy_loss = 0.0
        count = 0

        for attn in attention_weights:
            if not isinstance(attn, torch.Tensor):
                continue

            # attn shape: [B, num_heads, seq_len, seq_len]
            # Average over heads and batch
            attn_avg = attn.mean(dim=1)  # [B, seq_len, seq_len]

            # Compute entropy for each position
            entropy = -torch.sum(
                attn_avg * torch.log(attn_avg + 1e-8),
                dim=-1
            ).mean()

            # We want to MAXIMIZE entropy (prevent collapse)
            # So we minimize the negative entropy
            total_entropy_loss -= entropy
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=self.log_vars.device)

        return weight * (total_entropy_loss / count)

        r = 1.0 if ramp_epochs <= 0 else min(1.0, float(max(0, epoch)) / float(ramp_epochs))
        self._exit_boost = 1.0 + r * (max_boost - 1.0)

    def forward(self, predictions: dict, targets: dict):
        device = predictions['entry_logits'].device

        # ----- ENTRY (focal on logits) -----
        temp = torch.clamp(self.entry_temp, min=0.1, max=10.0).to(device)
        entry_logits = predictions['entry_logits'].view(-1) / temp
        entry_true   = targets['entry_label'].view(-1)
        alpha_entry  = self._batch_alpha(entry_true)
        entry_loss   = self._focal_loss(entry_logits, entry_true, alpha=alpha_entry, gamma=2.0)

        # ----- EXIT HEADS (prefer logits; fallback to prob->logit) -----
        def prob_to_logit(prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
            prob = prob.clamp(eps, 1 - eps)
            return torch.log(prob) - torch.log1p(-prob)

        exit_signals = predictions.get('exit_signals', {}) or {}

        # take-profit
        tp_logit = None
        tp_pack  = exit_signals.get('profit_taking', {}) or {}
        if 'take_profit_logit' in tp_pack:
            tp_logit = tp_pack['take_profit_logit']
        elif 'take_profit_prob' in tp_pack:
            tp_logit = prob_to_logit(tp_pack['take_profit_prob'])

        tp_loss = torch.tensor(0.0, device=device)
        if tp_logit is not None and 'take_profit_label' in targets:
            y = targets['take_profit_label'].view(-1)
            tp_loss = F.binary_cross_entropy_with_logits(tp_logit.view(-1), y)

        # stop-loss
        sl_logit = None
        sl_pack  = exit_signals.get('stop_loss', {}) or {}
        if 'stop_loss_logit' in sl_pack:
            sl_logit = sl_pack['stop_loss_logit']
        elif 'stop_loss_prob' in sl_pack:
            sl_logit = prob_to_logit(sl_pack['stop_loss_prob'])

        sl_loss = torch.tensor(0.0, device=device)
        if sl_logit is not None and 'stop_loss_label' in targets:
            y = targets['stop_loss_label'].view(-1)
            sl_loss = F.binary_cross_entropy_with_logits(sl_logit.view(-1), y)

        # let-winner-run (hold)
        hold_logit = None
        hold_pack  = exit_signals.get('let_winner_run', {}) or {}
        if 'hold_logit' in hold_pack:
            hold_logit = hold_pack['hold_logit']
        elif 'hold_score' in hold_pack:
            hold_logit = prob_to_logit(hold_pack['hold_score'])

        hold_loss = torch.tensor(0.0, device=device)
        if hold_logit is not None and 'let_winner_run_label' in targets:
            y = targets['let_winner_run_label'].view(-1)
            hold_loss = F.binary_cross_entropy_with_logits(hold_logit.view(-1), y)

        exit_terms = [t for t in (tp_loss, sl_loss, hold_loss)]
        exit_bundle_loss = torch.stack(exit_terms).mean() if len(exit_terms) else torch.tensor(0.0, device=device)

        # ----- RETURN / VOL / VAE -----
        ret_pred = predictions['expected_return'].view(-1)
        ret_true = targets['future_return'].view(-1)
        return_loss = F.smooth_l1_loss(ret_pred, ret_true)

        vol_pred = predictions['volatility_forecast'].view(-1)
        vol_true = targets['actual_volatility'].view(-1)
        volatility_loss = F.mse_loss(vol_pred, vol_true)

        vae_recon  = predictions.get('vae_recon', None)
        seq_repr   = predictions.get('sequence_repr', None)
        regime_mu  = predictions.get('regime_mu', None)
        regime_lv  = predictions.get('regime_logvar', None)

        if vae_recon is not None and seq_repr is not None:
            vae_recon_loss = F.mse_loss(vae_recon, seq_repr)
        else:
            vae_recon_loss = torch.tensor(0.0, device=device)

        if regime_mu is not None and regime_lv is not None:
            regime_lv = torch.clamp(regime_lv, min=-10, max=10)
            kl_loss = -0.5 * torch.sum(1 + regime_lv - regime_mu.pow(2) - regime_lv.exp(), dim=1).mean()
        else:
            kl_loss = torch.tensor(0.0, device=device)

        vae_loss = vae_recon_loss + 1e-5 * kl_loss

        # ----- DIVERSITY + CONFIDENCE (guarded) -----
        entry_probs = predictions.get('entry_prob', torch.sigmoid(predictions['entry_logits'])).view(-1)
        if entry_probs.numel() >= 2:
            entry_std = entry_probs.float().std(correction=0)
            exit_std  = entry_probs.float().std(correction=0)  # reuse if no unified exit prob
        else:
            entry_std = torch.tensor(0.0, device=device)
            exit_std  = torch.tensor(0.0, device=device)

        diversity_penalty  = 1.0 / (entry_std + 1e-3) + 1.0 / (exit_std + 1e-3)
        confidence_penalty = torch.mean(torch.exp(-10 * (entry_probs - 0.5).pow(2)))

        # ----- uncertainty-weighted sum -----
        precision_entry  = torch.exp(-self.log_vars[0])
        precision_return = torch.exp(-self.log_vars[2])
        precision_vol    = torch.exp(-self.log_vars[3])
        precision_vae    = torch.exp(-self.log_vars[4])

        # Compute attention entropy loss (prevent collapse)
        attention_weights = predictions.get('attention_weights', [])
        entropy_reg_loss = self.compute_attention_entropy_loss(
            attention_weights,
            weight=0.15  # INCREASED: Stronger regularization to prevent collapse  # Regularization strength
        )

        # DEBUG: Log entropy loss value (remove after confirming it works)
        if torch.is_tensor(entropy_reg_loss) and entropy_reg_loss.item() != 0.0:
            if not hasattr(self, '_entropy_log_count'):
                self._entropy_log_count = 0
            self._entropy_log_count += 1
            # if self._entropy_log_count % 100 == 0:  # Log every 100 batches
            #     print(f"[DEBUG] Entropy reg loss: {entropy_reg_loss.item():.6f}")

        total_loss = (
            self.task_weights["entry"] * (precision_entry * entry_loss + self.log_vars[0]) +
            self._exit_boost * (exit_bundle_loss) +
            self.task_weights["return"] * (precision_return * return_loss + self.log_vars[2]) +
            self.task_weights["volatility"] * (precision_vol * volatility_loss + self.log_vars[3]) +
            self.task_weights["vae"] * (precision_vae * vae_loss + self.log_vars[4]) +
            0.01 * diversity_penalty +
            0.05 * confidence_penalty +
            entropy_reg_loss  # NEW: Prevent attention collapse
        )

        return {
            'total_loss': total_loss,
            'entry_loss': float(entry_loss.detach()),
            'return_loss': float(return_loss.detach()),
            'volatility_loss': float(volatility_loss.detach()),
            'vae_loss': float(vae_loss.detach()),
            'take_profit_loss': float(tp_loss.detach()),
            'stop_loss_loss': float(sl_loss.detach()),
            'let_run_loss': float(hold_loss.detach()),
            'exit_bundle_loss': float(exit_bundle_loss.detach()),
            'diversity_penalty': float(diversity_penalty.detach()),
            'confidence_penalty': float(confidence_penalty.detach()),
            'entropy_loss': float(entropy_reg_loss.detach())  # NEW: Track entropy regularization: self.log_vars.detach().cpu().numpy(),
        }

# ============================================================
# Trainer
# ============================================================

class NeuralTrainer:
    """Enhanced trainer with exit management tracking + rich dashboard + epoch probes."""
    def __init__(self, model, train_loader, val_loader, config: dict,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('lr', 3e-4),
            weight_decay=config.get('weight_decay', 1e-4),
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Use ReduceLROnPlateau and step with val loss
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min',
            factor=config.get('lr_factor', 0.5))

        # Learning rate warmup scheduler
        self.warmup_epochs = config.get('warmup_epochs', 5)
        self.base_lr = config.get('lr', 7e-5)

        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                # Linear warmup
                return float(epoch + 1) / float(self.warmup_epochs)
            return 1.0

        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
                                                                  lr_lambda=lr_lambda, 
                                                                  last_epoch=-1) # Fallback to default

        self.criterion = MultiTaskLoss(num_tasks=5)
        self.gradient_accumulation_steps = config.get('grad_accum_steps', 2)

        self.use_amp = (device == 'cuda') and config.get('amp', True)
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

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

        self.best_val_loss = float('inf')
        self.patience_counter = 0

    # ---------- small helpers ----------
    def _sf(self, x):
        """safe float for dashboard formatting"""
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

    def _quick_brier(self, probs: torch.Tensor, y: torch.Tensor) -> float:
        with torch.no_grad():
            p = probs.detach().float().view(-1).clamp(0, 1).cpu()
            t = y.detach().float().view(-1).cpu()
            if p.numel() == 0:
                return float('nan')
            return float(((p - t)**2).mean().item())

    def _epoch_probe(self, epoch: int, batch: dict, predictions: dict):
        """Light diagnostics on one minibatch: calibration, quick correlations, attn entropy."""
        import numpy as np
        with torch.no_grad():
            entry_prob = predictions.get('entry_prob', torch.sigmoid(predictions['entry_logits']))
            entry_brier = self._quick_brier(entry_prob, batch['entry_label'].to(self.device))

            er = predictions['expected_return'].view(-1).detach().float().cpu().numpy()
            rt = batch['future_return'].view(-1).detach().float().cpu().numpy()
            vr = predictions['volatility_forecast'].view(-1).detach().float().cpu().numpy()
            vt = batch['actual_volatility'].view(-1).detach().float().cpu().numpy()

            def safe_corr(a, b):
                if len(a) < 2: return float('nan')
                if np.std(a) == 0 or np.std(b) == 0: return float('nan')
                return float(np.corrcoef(a, b)[0, 1])

            corr_ret = safe_corr(er, rt)
            corr_vol = safe_corr(vr, vt)

            attns = predictions.get('attention_weights', [])
            entropies = []
            for A in attns:
                if not isinstance(A, torch.Tensor): continue
                P = A.float().clamp(1e-6, 1).view(-1, A.shape[-1])  # (B*heads*T, T)
                H = (-P * P.log()).sum(dim=1).mean().item()
                entropies.append(H)
            attn_txt = " / ".join(f"{h:.2f}" for h in entropies) if entropies else "—"

            exit_signals = predictions.get('exit_signals', {})
            def mean_or_dash(x):
                if isinstance(x, torch.Tensor): return f"{x.detach().float().mean().item():.3f}"
                return "—"

            tp = exit_signals.get('profit_taking') or {}
            sl = exit_signals.get('stop_loss') or {}
            lr = exit_signals.get('let_winner_run') or {}

            print(f"\n🔎 Epoch {epoch} probe:")
            print(f"   Entry  Brier: {entry_brier:.4f} | Ret Corr: {corr_ret:.3f} | Vol Corr: {corr_vol:.3f}")
            print(f"   Attn entropy per layer: {attn_txt}")
            print(f"   TP μ: {mean_or_dash(tp.get('take_profit_prob'))} | SL μ: {mean_or_dash(sl.get('stop_loss_prob'))} | HOLD μ: {mean_or_dash(lr.get('hold_score'))}")

    # ---------- core loops ----------
    def train_epoch(self, epoch, on_batch=None, batch_update_every: int = 10):
        """Train for one epoch (AMP-aware, FP32-sanitized exits live in the model)."""
        self.model.train()
        total_loss = 0.0
        loss_components = {}

        self.optimizer.zero_grad()
        n_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
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

            position_context = {
                'unrealized_pnl': unrealized_pnl,
                'time_in_position': time_in_position
            }

            with torch.amp.autocast(device_type='cuda', enabled=False):
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

                out = self.criterion(predictions, targets)
                loss = out['total_loss'] / self.gradient_accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):
                self.optimizer.zero_grad()
                continue

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                # Track gradient norm BEFORE clipping
                    total_grad_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_grad_norm += param_norm.item() ** 2
                    total_grad_norm = total_grad_norm ** 0.5
                    self.last_grad_norm = total_grad_norm  # Store for dashboard

                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += self._sf(out['total_loss'])
            for k, v in out.items():
                if k == 'total_loss': continue
                loss_components[k] = loss_components.get(k, 0.0) + self._sf(v)

            if on_batch and ((batch_idx + 1) % batch_update_every == 0):
                on_batch(
                    batch_idx + 1,
                    n_batches,
                    float(self._sf(loss)),
                    self.optimizer.param_groups[0]['lr']
                )

        avg_loss = total_loss / max(1, n_batches)
        avg_components = {k: v / max(1, n_batches) for k, v in loss_components.items()}
        return avg_loss, avg_components

    def validate(self, on_batch=None, batch_update_every: int = 10):
        """Validate model on val_loader with guarded metrics."""
        self.model.eval()
        total_loss = 0.0
        loss_components = {}
        entry_correct, total_samples = 0, 0

        n_batches = len(self.val_loader)
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
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

                out = self.criterion(predictions, targets)
                total_loss += self._sf(out['total_loss'])
                for k, v in out.items():
                    if k == 'total_loss': continue
                    loss_components[k] = loss_components.get(k, 0.0) + self._sf(v)

                entry_pred = (predictions['entry_prob'] > 0.5).float().squeeze()
                entry_true = (entry_label > 0.5).float()
                entry_correct += (entry_pred == entry_true).sum().item()
                total_samples += len(entry_label)

                if on_batch and ((batch_idx + 1) % batch_update_every == 0):
                    on_batch(
                        batch_idx + 1,
                        n_batches,
                        float(self._sf(out['total_loss'])),
                        self.optimizer.param_groups[0]['lr'],
                    )

        avg_loss = total_loss / max(1, n_batches)
        avg_components = {k: v / max(1, n_batches) for k, v in loss_components.items()}
        entry_accuracy = entry_correct / max(1, total_samples)
        # Calculate actual exit accuracy
        exit_correct = 0
        exit_total = 0

        # Check if we have exit predictions and labels
        if 'unified_exit_prob' in predictions:
            unified_exit = predictions['unified_exit_prob']

            # We need to create a unified exit label from individual exit labels
            # For samples in profit: use take_profit_label
            # For samples in loss: use stop_loss_label
            if 'unrealized_pnl' in batch:
                pnl = batch['unrealized_pnl'].to(self.device).unsqueeze(-1)
                in_profit = (pnl > 0).float()
                in_loss = (pnl <= 0).float()

                # Create unified exit label
                unified_label = (
                    in_profit * batch['take_profit_label'].to(self.device).float() +
                    in_loss * batch['stop_loss_label'].to(self.device).float()
                )

                # Calculate accuracy
                exit_pred = (unified_exit > 0.5).float().squeeze()
                exit_true = (unified_label > 0.5).float().squeeze()
                exit_correct = (exit_pred == exit_true).sum().item()
                exit_total = len(exit_true)

        exit_accuracy = exit_correct / max(1, exit_total)
        return avg_loss, avg_components, entry_accuracy, exit_accuracy

    def save_checkpoint(self, filename, epoch, val_loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': int(epoch),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': float(self._sf(val_loss)),
            'config': self.config
        }
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['val_loss']

    # ---------- main loop with Rich ----------
    def train(self, num_epochs):
        """Full training loop with non-flickering Rich dashboard + per-batch Train & Val progress (tensor-safe)."""
        from rich.live import Live
        from rich.table import Table
        from rich.progress import (
            Progress, BarColumn, TextColumn, TimeElapsedColumn,
            TimeRemainingColumn, MofNCompleteColumn
        )
        from rich.console import Group
        import psutil

        console = Console()

        if not self.config.get('rich_dashboard', True):
            console.print("[yellow]⚠️ Rich dashboard disabled — using simple loop[/yellow]")
            return self._legacy_train(num_epochs)

        # epoch progress
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
            lr=self._sf(self.optimizer.param_groups[0]['lr'])
        )

        # training batch progress
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
            lr=self._sf(self.optimizer.param_groups[0]['lr'])
        )

        # validation batch progress
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
            lr=self._sf(self.optimizer.param_groups[0]['lr'])
        )

        # metrics dashboard
        dashboard = Table(expand=True, show_header=True, header_style="bold white")
        dashboard.add_column("Metric", justify="left", style="bold cyan")
        dashboard.add_column("Value", justify="right")
        self.loss_history = []

        def update_dashboard(epoch, train_loss, val_loss, entry_acc, exit_acc, lr, grad_norm):
            table = Table(expand=True, show_header=True, header_style="bold white")
            table.add_column("Metric", justify="left", style="bold cyan")
            table.add_column("Value", justify="right")

            table.add_row("Epoch", f"{int(epoch)}")
            table.add_row("Train Loss", f"{self._sf(train_loss):.5f}")
            table.add_row("Val Loss",   f"{self._sf(val_loss):.5f}")
            table.add_row("Entry Acc",  f"{self._sf(entry_acc):.4f}")
            table.add_row("Exit Acc",   f"{self._sf(exit_acc):.4f}")
            table.add_row("Grad Norm",  f"{self._sf(grad_norm):.4f}" if grad_norm is not None else "—")
            table.add_row("LR",         f"{self._sf(lr):.6f}")
            table.add_row(
                "GPU Mem (MB)" if torch.cuda.is_available() else "CPU Mem (%)",
                f"{(torch.cuda.memory_allocated(self.device) / 1e6) if torch.cuda.is_available() else psutil.virtual_memory().percent:.1f}"
            )

            if len(self.loss_history) > 2:
                chars = "▁▂▃▄▅▆▇█"
                vals = [ self._sf(v) for v in self.loss_history ]
                lo, hi = min(vals), max(vals)
                if hi > lo:
                    scaled = np.interp(vals, (lo, hi), (1, len(chars)))
                    spark = "".join(chars[int(x) - 1] for x in scaled.astype(int))
                    table.add_row("Loss Trend", spark)

            group.renderables[-1] = table

        group = Group(epoch_progress, train_progress, val_progress, dashboard)

        with Live(group, refresh_per_second=6, console=console, transient=False):
            for epoch in range(num_epochs):
                # ramp exit weight this epoch
                self.criterion.set_exit_weight_ramp(
                    epoch=epoch,
                    ramp_epochs=self.config.get('exit_ramp_epochs', 10),
                    max_boost=self.config.get('exit_max_boost', 8.0),
                )

                # reset train bar
                train_progress.reset(
                    train_task,
                    total=max(1, len(self.train_loader)),
                    completed=0,
                    bloss=0.0,
                    lr=self._sf(self.optimizer.param_groups[0]['lr'])
                )

                def on_train_batch(bi, total, loss, lr):
                    shown_loss = 0.0 if (loss is None or (isinstance(loss, float) and math.isnan(loss))) else self._sf(loss)
                    train_progress.update(train_task, completed=bi, bloss=shown_loss, lr=self._sf(lr))

                train_loss, _ = self.train_epoch(epoch, on_batch=on_train_batch, batch_update_every=10)

                # reset val bar
                val_progress.reset(
                    val_task,
                    total=max(1, len(self.val_loader)),
                    completed=0,
                    bloss=0.0,
                    lr=self._sf(self.optimizer.param_groups[0]['lr'])
                )

                def on_val_batch(bi, total, loss, lr):
                    shown_loss = 0.0 if (loss is None or (isinstance(loss, float) and math.isnan(loss))) else self._sf(loss)
                    val_progress.update(val_task, completed=bi, bloss=shown_loss, lr=self._sf(lr))

                val_loss, _, entry_acc, exit_acc = self.validate(on_batch=on_val_batch, batch_update_every=10)

                # small batch probe
                try:
                    self.model.eval()
                    first_val = next(iter(self.val_loader))
                    features = first_val['features'].to(self.device)
                    position_context = {
                        'unrealized_pnl': first_val['unrealized_pnl'].to(self.device).unsqueeze(1),
                        'time_in_position': first_val['time_in_position'].to(self.device).unsqueeze(1),
                    }
                    with torch.no_grad():
                        preds = self.model(features, position_context=position_context)
                    self._epoch_probe(epoch, first_val, preds)
                except StopIteration:
                    pass

                # step ReduceLROnPlateau with the metric
                val_loss_float = float(self._sf(val_loss))
                # Apply warmup scheduler for first N epochs, then use ReduceLROnPlateau
                if epoch < self.warmup_epochs:
                    self.warmup_scheduler.step()
                else:
                    self.scheduler.step(val_loss_float)

                lr_now = self._sf(self.optimizer.param_groups[0]['lr'])
                grad_norm = getattr(self, 'last_grad_norm', None)
                self.loss_history.append(self._sf(val_loss))

                epoch_progress.update(
                    epoch_task,
                    advance=1,
                    epoch=int(epoch),
                    loss=self._sf(train_loss),
                    val=self._sf(val_loss),
                    lr=lr_now
                )
                update_dashboard(epoch, train_loss, val_loss, entry_acc, exit_acc, lr_now, grad_norm)

                # logging & checkpoints
                if self.use_wandb and self.wandb is not None:
                    self.wandb.log({
                        'epoch': epoch,
                        'train_loss': self._sf(train_loss),
                        'val_loss': self._sf(val_loss),
                        'entry_accuracy': self._sf(entry_acc),
                        'exit_accuracy': self._sf(exit_acc),
                        'learning_rate': lr_now
                    })

                if self._sf(val_loss) < self._sf(self.best_val_loss):
                    self.best_val_loss = self._sf(val_loss)
                    self.patience_counter = 0
                    best_path = self.config.get('best_model_path', 'models/best_model.pt')
                    self.save_checkpoint(best_path, epoch, self._sf(val_loss))
                    console.print(f"[green]💾 New best model saved![/green] Val loss: {self._sf(val_loss):.4f}")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.get('patience', 25):
                        console.print(f"[red]⏹️ Early stopping triggered at epoch {epoch + 1}[/red]")
                        break

        console.print("\n[bold green]✅ Training completed successfully![/bold green]")

    # Very small fallback if rich_dashboard=False
    def _legacy_train(self, num_epochs):
        for epoch in range(num_epochs):
            self.criterion.set_exit_weight_ramp(
                epoch=epoch,
                ramp_epochs=self.config.get('exit_ramp_epochs', 10),
                max_boost=self.config.get('exit_max_boost', 8.0),
            )
            train_loss, _ = self.train_epoch(epoch)
            val_loss, _, _, _ = self.validate()
            self.scheduler.step(float(self._sf(val_loss)))
            if self._sf(val_loss) < self._sf(self.best_val_loss):
                self.best_val_loss = self._sf(val_loss)
                best_path = self.config.get('best_model_path', 'models/best_model.pt')
                self.save_checkpoint(best_path, epoch, self._sf(val_loss))

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
