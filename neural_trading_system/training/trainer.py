from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from rich.console import Console
import torch
import math

class MultiTaskLoss(nn.Module):
\
\
\
\
\

    def __init__(self, num_tasks: int = 5):
        super().__init__()

        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.entry_temp = nn.Parameter(torch.ones(1))
        self.task_weights = {
            'entry': 1.0,
            'return': 1.5,
            'volatility': 2.0,
            'vae': 1.0,
        }
        self.alpha = 1.5
        self._exit_boost = 1.0

    def compute_grad_norm(self, loss, model_params):

        grads = torch.autograd.grad(loss, model_params, retain_graph=True, create_graph=True)
        grad_norm = torch.norm(torch.stack([torch.norm(g) for g in grads]))
        return grad_norm

    @staticmethod
    def _batch_alpha(y: torch.Tensor, lo: float = 0.25, hi: float = 0.75) -> float:

        with torch.no_grad():
            p = y.float().mean().clamp_(1e-6, 1 - 1e-6).item()
        return float(max(lo, min(hi, 1.0 - p)))

    @staticmethod
    def _focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.5, gamma: float = 2.0) -> torch.Tensor:

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt  = torch.exp(-bce).clamp(1e-6, 1.0)
        loss = alpha * (1.0 - pt) ** gamma * bce
        return loss.mean()

    def compute_attention_entropy_loss(self, attention_weights: list, weight: float = 10.0):
        if not attention_weights:
            return torch.tensor(0.0, device=self.log_vars.device)

        total_penalty = 0.0
        for layer_idx, attn in enumerate(attention_weights):
            if not isinstance(attn, torch.Tensor):
                continue

            attn_avg = attn.mean(dim=1)
            entropy = -torch.sum(attn_avg * torch.log(attn_avg + 1e-8), dim=-1).mean()

            target_entropy = torch.log(torch.tensor(attn.shape[-1], dtype=torch.float32))

            min_acceptable = 0.7 * target_entropy

            if entropy < min_acceptable:

                layer_scale = 3.0 if layer_idx == 0 else 1.0
                penalty = layer_scale * ((min_acceptable - entropy) / target_entropy) ** 2
                total_penalty += penalty

        return weight * total_penalty

    def set_exit_weight_ramp(self, epoch: int, ramp_epochs: int = 20, max_boost: float = 4.0):

        if ramp_epochs == 0:
            self.exit_boost = max_boost
        else:
            progress = min(1.0, float(epoch) / float(ramp_epochs))

            self.exit_boost = 1.0 + (max_boost - 1.0) * (np.log1p(progress * 9) / np.log1p(9))

    def forward(self, predictions: dict, targets: dict):

        device = predictions['entry_logits'].device

        entry_logits = predictions['entry_logits'].view(-1)
        entry_true = targets['entry_label'].view(-1).float()
        alpha_entry = self._batch_alpha(entry_true)
        entry_loss = self._focal_loss(entry_logits, entry_true, alpha=alpha_entry, gamma=2.0)

        ret_pred = predictions['expected_return'].view(-1)
        ret_true = targets['future_return'].view(-1)
        return_loss = F.smooth_l1_loss(ret_pred, ret_true)

        vol_pred = predictions['volatility_forecast'].view(-1)
        vol_true = targets['actual_volatility'].view(-1)
        volatility_loss = F.mse_loss(vol_pred, vol_true)

        vae_recon_loss = torch.tensor(0.0, device=device)
        kl_loss = torch.tensor(0.0, device=device)
        if 'vae_recon' in predictions and 'sequence_repr' in predictions:
            vae_recon_loss = F.mse_loss(predictions['vae_recon'], predictions['sequence_repr'])
        if 'regime_mu' in predictions and 'regime_logvar' in predictions:
            mu = predictions['regime_mu']
            lv = torch.clamp(predictions['regime_logvar'], min=-10, max=10)
            kl_loss = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1).mean()
        vae_loss = vae_recon_loss + 1e-5 * kl_loss

        exit_logits_tp = predictions.get('take_profit_logits', torch.zeros(len(entry_true), device=device)).view(-1)
        exit_logits_sl = predictions.get('stop_loss_logits', torch.zeros(len(entry_true), device=device)).view(-1)
        exit_true_tp = targets['take_profit_label'].view(-1).float()
        exit_true_sl = targets['stop_loss_label'].view(-1).float()

        exit_loss_tp = F.binary_cross_entropy_with_logits(exit_logits_tp, exit_true_tp)
        exit_loss_sl = F.binary_cross_entropy_with_logits(exit_logits_sl, exit_true_sl)
        exit_bundle_loss = exit_loss_tp + exit_loss_sl

        attention_weights = predictions.get('attention_weights', [])
        entropy_reg_loss = self.compute_attention_entropy_loss(attention_weights, weight=5.0)

        clamped_logvars = torch.clamp(self.log_vars, min=-3.0, max=3.0)

        prec_entry = torch.exp(-clamped_logvars[0]).clamp(max=10.0)
        prec_return = torch.exp(-clamped_logvars[2]).clamp(max=10.0)
        prec_vol = torch.exp(-clamped_logvars[3]).clamp(max=10.0)
        prec_vae = torch.exp(-clamped_logvars[4]).clamp(max=10.0)

        w_entry = float(self.task_weights['entry'])
        w_return = float(self.task_weights['return'])
        w_vol = float(self.task_weights['volatility'])
        w_vae = float(self.task_weights['vae'])

        term1 = w_entry * (prec_entry * entry_loss + clamped_logvars[0])
        term2 = self._exit_boost * exit_bundle_loss
        term3 = w_return * (prec_return * return_loss + clamped_logvars[2])
        term4 = w_vol * (prec_vol * volatility_loss + clamped_logvars[3])
        term5 = w_vae * (prec_vae * vae_loss + clamped_logvars[4])
        term6 = entropy_reg_loss

        total_loss = term1 + term2 + term3 + term4 + term5 + term6

        if total_loss.item() < 0:
            print(f"⚠️ NEGATIVE LOSS DETECTED! Clamping to 0.1")
            total_loss = torch.clamp(total_loss, min=0.1)

        return total_loss, {
            'entry_loss': entry_loss,
            'return_loss': return_loss,
            'volatility_loss': volatility_loss,
            'vae_loss': vae_loss,
            'exit_bundle_loss': exit_bundle_loss,
            'entropy_reg_loss': entropy_reg_loss,
        }

class NeuralTrainer:

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

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min',
            factor=config.get('lr_factor', 0.5))

        self.warmup_epochs = config.get('warmup_epochs', 5)
        self.base_lr = config.get('lr', 7e-5)

        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:

                return float(epoch + 1) / float(self.warmup_epochs)
            return 1.0

        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                                  lr_lambda=lr_lambda,
                                                                  last_epoch=-1)

        self.criterion = MultiTaskLoss(num_tasks=5)
        self.gradient_accumulation_steps = config.get('grad_accum_steps', 2)

        self.use_amp = (device == 'cuda') and config.get('amp', True)
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        self.use_wandb = config.get('use_wandb', False)
        self.global_batch_step = 0
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project="neural-trading-exit-management",
                    config=config,
                    name=config.get('run_name', 'exit_aware_model')
                )
                watch_freq = config.get('wandb_watch_freq', 100)
                wandb.watch(self.model, log='all', log_freq=watch_freq)
                wandb.config.update({'wandb_watch_freq': watch_freq}, allow_val_change=True)
                wandb.define_metric('epoch')
                for metric_name in ('train_loss', 'val_loss', 'entry_accuracy', 'exit_accuracy', 'learning_rate'):
                    wandb.define_metric(metric_name, step_metric='epoch')
                self.wandb = wandb
            except ImportError:
                print("⚠️ wandb not installed, disabling logging")
                self.use_wandb = False
                self.wandb = None
        else:
            self.wandb = None

        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.loss_history = []
        self.last_epoch_train_components = {}
        self.last_epoch_val_components = {}

    def _sf(self, x):

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

    def _log_to_wandb(self, *, step: int, epoch: int, train_loss, train_components, val_loss,
                      val_components, entry_acc, exit_acc, lr_now, probe_metrics=None):

        if not (self.use_wandb and self.wandb is not None):
            return

        import psutil

        payload = {
            'epoch': epoch,
            'train/loss': self._sf(train_loss),
            'val/loss': self._sf(val_loss),
            'val/entry_accuracy': self._sf(entry_acc),
            'val/exit_accuracy': self._sf(exit_acc),
            'optimizer/lr': self._sf(lr_now),
        }

        for name, value in (train_components or {}).items():
            payload[f'train/{name}'] = self._sf(value)
        for name, value in (val_components or {}).items():
            payload[f'val/{name}'] = self._sf(value)

        grad_norm = getattr(self, 'last_grad_norm', None)
        if grad_norm is not None:
            payload['diagnostics/grad_norm'] = self._sf(grad_norm)

        exit_boost = getattr(self.criterion, '_exit_boost', None)
        if exit_boost is not None:
            payload['diagnostics/exit_boost'] = float(exit_boost)

        payload['diagnostics/scheduler_lr'] = self._sf(lr_now)

        if torch.cuda.is_available():
            payload['diagnostics/gpu_memory_mb'] = float(torch.cuda.memory_allocated(self.device) / 1e6)
        else:
            payload['diagnostics/cpu_memory_percent'] = float(psutil.virtual_memory().percent)

        if probe_metrics:
            payload.update(probe_metrics)

        loss_values = [self._sf(v) for v in getattr(self, 'loss_history', [])]
        sparkline = None
        if len(loss_values) > 2:
            lo, hi = min(loss_values), max(loss_values)
            if hi > lo:
                chars = "▁▂▃▄▅▆▇█"
                scaled = np.interp(loss_values, (lo, hi), (1, len(chars)))
                sparkline = "".join(chars[int(x) - 1] for x in scaled.astype(int))

        if sparkline:
            payload['dashboard/loss_sparkline'] = sparkline

        if loss_values:
            history_table = self.wandb.Table(columns=['epoch', 'val_loss'])
            start_idx = max(0, len(loss_values) - 50)
            for idx in range(start_idx, len(loss_values)):
                history_table.add_data(idx, loss_values[idx])
            payload['dashboard/loss_history'] = history_table

        self.wandb.log(payload, step=step)

    def _epoch_probe(self, epoch: int, batch: dict, predictions: dict):

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
            layer0_heads = []
            for layer_idx, A in enumerate(attns):
                if not isinstance(A, torch.Tensor):
                    continue

                per_head_entropy = []
                for head in range(A.shape[1]):
                    head_attn = A[:, head, :, :].float().clamp(1e-6, 1)
                    H = (-head_attn * head_attn.log()).sum(dim=-1).mean().item()
                    per_head_entropy.append(H)

                if layer_idx == 0:
                    layer0_heads = per_head_entropy

                print(f"   Layer {layer_idx} per-head entropy: {per_head_entropy}")

                P = A.float().clamp(1e-6, 1).view(-1, A.shape[-1])
                H_layer = (-P * P.log()).sum(dim=1).mean().item()
                entropies.append(H_layer)

            attn_txt = " / ".join(f"{h:.2f}" for h in entropies) if entropies else "—"

            exit_signals = predictions.get('exit_signals', {})

            dead_heads = sum(1 for h in layer0_heads if h < 0.1)
            avg_entropy = (sum(layer0_heads) / len(layer0_heads)) if layer0_heads else 0.0

            if dead_heads > 0:
                print(f"   🔴 ALERT: {dead_heads}/8 heads collapsed (< 0.1 entropy)")
            if layer0_heads and avg_entropy < 0.5:
                print(f"   🟡 WARNING: Layer 0 avg entropy = {avg_entropy:.2f}")

            if dead_heads >= 4 and epoch > 2:
                print(f"   ☠️ STOPPING: {dead_heads} heads collapsed - model is dead")
                raise KeyboardInterrupt("Layer 0 collapse detected")

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

            def tensor_mean(x):
                if isinstance(x, torch.Tensor):
                    return float(x.detach().float().mean().item())
                return None

            probe_metrics = {
                'probe/entry_brier': float(entry_brier),
            }

            if not math.isnan(corr_ret):
                probe_metrics['probe/return_corr'] = float(corr_ret)
            if not math.isnan(corr_vol):
                probe_metrics['probe/vol_corr'] = float(corr_vol)
            if entropies:
                probe_metrics['probe/attn_entropy_mean'] = float(np.nanmean(entropies))
                probe_metrics['probe/attn_entropy_min'] = float(np.nanmin(entropies))
            if layer0_heads:
                probe_metrics['probe/layer0_dead_heads'] = float(dead_heads)
                probe_metrics['probe/layer0_entropy_mean'] = float(avg_entropy)

            tp_prob = tensor_mean(tp.get('take_profit_prob'))
            sl_prob = tensor_mean(sl.get('stop_loss_prob'))
            hold_score = tensor_mean(lr.get('hold_score'))

            if tp_prob is not None:
                probe_metrics['probe/take_profit_prob_mean'] = tp_prob
            if sl_prob is not None:
                probe_metrics['probe/stop_loss_prob_mean'] = sl_prob
            if hold_score is not None:
                probe_metrics['probe/hold_score_mean'] = hold_score

            return probe_metrics

    def validate(self, on_batch=None, batch_update_every: int = 10):

        self.model.eval()
        total_loss = 0.0
        loss_components = {}

        entry_correct = 0
        entry_total = 0
        exit_correct = 0
        exit_total = 0

        n_batches = len(self.val_loader)
        debug_mode = (n_batches > 0)
        batch_counter = 0

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
                unrealized_pnl = batch['unrealized_pnl'].to(self.device)
                time_in_position = batch['time_in_position'].to(self.device).unsqueeze(1)

                if unrealized_pnl.dim() == 2:
                    unrealized_pnl = unrealized_pnl.squeeze(-1)
                if entry_label.dim() == 2:
                    entry_label = entry_label.squeeze(-1)
                if take_profit_label.dim() == 2:
                    take_profit_label = take_profit_label.squeeze(-1)
                if stop_loss_label.dim() == 2:
                    stop_loss_label = stop_loss_label.squeeze(-1)

                position_context = {
                    'unrealized_pnl': unrealized_pnl.unsqueeze(-1),
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

                total_loss_tensor, loss_components_dict = self.criterion(predictions, targets)

                total_loss += self._sf(total_loss_tensor)

                for k, v in loss_components_dict.items():
                    loss_components[k] = loss_components.get(k, 0.0) + self._sf(v)

                entry_pred = (predictions['entry_prob'] > 0.5).float().squeeze()
                entry_true = (entry_label > 0.5).float()

                entry_pred = entry_pred.view(-1)
                entry_true = entry_true.view(-1)

                batch_entry_correct = (entry_pred == entry_true).sum().item()
                batch_entry_total = len(entry_label)

                entry_correct += batch_entry_correct
                entry_total += batch_entry_total

                if 'unified_exit_prob' in predictions:
                    unified_exit = predictions['unified_exit_prob']

                    unified_exit = unified_exit.squeeze(-1)

                    in_profit = (unrealized_pnl > 0).float()
                    in_loss = (unrealized_pnl <= 0).float()

                    unified_label = (
                        in_profit * take_profit_label.float() +
                        in_loss * stop_loss_label.float()
                    )

                    exit_pred = (unified_exit > 0.5).float()
                    exit_true = (unified_label > 0.5).float()

                    exit_pred = exit_pred.view(-1)
                    exit_true = exit_true.view(-1)

                    batch_exit_correct = (exit_pred == exit_true).sum().item()
                    batch_exit_total = len(exit_true)

                    exit_correct += batch_exit_correct
                    exit_total += batch_exit_total

                    if on_batch and ((batch_idx + 1) % batch_update_every == 0):
                        on_batch(
                            batch_idx + 1,
                            n_batches,
                            float(self._sf(total_loss_tensor)),
                            self.optimizer.param_groups[0]['lr'],
                        )

                batch_counter += 1

        avg_loss = total_loss / max(1, n_batches)
        avg_components = {k: v / max(1, n_batches) for k, v in loss_components.items()}

        entry_accuracy = entry_correct / max(1, entry_total)
        exit_accuracy = exit_correct / max(1, exit_total)

        return avg_loss, avg_components, entry_accuracy, exit_accuracy

    def train_epoch(self, epoch, on_batch=None, batch_update_every: int = 10):

        self.model.train()
        total_loss_accum = 0.0
        component_sums = {}
        self.optimizer.zero_grad()

        n_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            batch_idx_global = self.global_batch_step
            self.global_batch_step += 1

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
                if batch_idx == 0 and epoch == 0:
                    exit_keys = [k for k in predictions.keys() if 'exit' in k.lower() or 'logit' in k.lower()]
                    print(f"Exit-related keys in predictions: {exit_keys}")
                    for key in exit_keys:
                        val = predictions[key]
                        if isinstance(val, torch.Tensor):
                            print(f"  {key}: shape={val.shape}, sample values={val[:3].detach().cpu().numpy()}")
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

            total_loss_tensor, batch_components = self.criterion(predictions, targets)
            loss = total_loss_tensor / self.gradient_accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARNING] NaN/Inf loss detected at batch {batch_idx}, skipping")
                self.optimizer.zero_grad()
                continue

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)

                if batch_idx == 0 and epoch == 0:
                    print(f"\n{'='*80}")
                    print(f"CHECKPOINT: GRADIENT FLOW (EPOCH 0, BATCH 0)")
                    print(f"{'='*80}")

                    total_grad_norm = 0.0
                    zero_grad_count = 0

                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.data.norm(2).item()
                            total_grad_norm += grad_norm ** 2
                            if grad_norm == 0.0:
                                zero_grad_count += 1
                        else:
                            zero_grad_count += 1

                    total_grad_norm = total_grad_norm ** 0.5

                    print(f"  Total gradient norm (before clip): {total_grad_norm:.6f}")
                    print(f"  Gradient norm is exploding (>10): {'YES ⚠️  BUG' if total_grad_norm > 10.0 else 'NO ✓'}")
                    print(f"  Gradient norm is dead (<1e-6): {'YES ⚠️  BUG' if total_grad_norm < 1e-6 else 'NO ✓'}")
                    print(f"  Layers with zero gradient: {zero_grad_count}")
                    print(f"{'='*80}\n")

                total_grad_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2

                total_grad_norm = total_grad_norm ** 0.5
                self.last_grad_norm = total_grad_norm

                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                if hasattr(self, 'grad_history'):
                    self.grad_history.append(grad_norm)
                    if len(self.grad_history) > 100:
                        avg_norm = sum(self.grad_history[-100:]) / 100
                        if avg_norm < 0.5:
                            self.current_clip_norm = min(2.0, self.current_clip_norm * 1.01)
                        elif avg_norm > 0.9:
                            self.current_clip_norm = max(0.5, self.current_clip_norm * 0.99)
                else:
                    self.grad_history = []
                    self.current_clip_norm = 1.0

                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            total_loss_accum += self._sf(total_loss_tensor)

            for name, value in batch_components.items():
                component_sums[name] = component_sums.get(name, 0.0) + self._sf(value)

            if on_batch and ((batch_idx + 1) % batch_update_every == 0):
                on_batch(
                    batch_idx + 1,
                    n_batches,
                    float(self._sf(loss)),
                    self.optimizer.param_groups[0]['lr']
                )

            if self.use_wandb and self.wandb is not None:
                wandb_interval = self.config.get('wandb_log_interval', 0)
                if wandb_interval and wandb_interval > 0 and ((batch_idx_global + 1) % wandb_interval == 0):
                    batch_payload = {
                        'batch/train/loss': self._sf(total_loss_tensor),
                        'batch/train/lr': self._sf(self.optimizer.param_groups[0]['lr']),
                    }

                    for name, value in batch_components.items():
                        batch_payload[f'batch/train/{name}'] = self._sf(value)

                    grad_norm = getattr(self, 'last_grad_norm', None)
                    if grad_norm is not None:
                        batch_payload['batch/grad_norm'] = self._sf(grad_norm)

                    exit_boost = getattr(self.criterion, '_exit_boost', None)
                    if exit_boost is not None:
                        batch_payload['batch/exit_boost'] = float(exit_boost)

                    self.wandb.log(batch_payload, step=batch_idx_global)

        avg_loss = total_loss_accum / max(1, n_batches)
        avg_components = {k: v / max(1, n_batches) for k, v in component_sums.items()}

        return avg_loss, avg_components

    def set_exit_weight_ramp(self, epoch: int, ramp_epochs: int = 10, max_boost: float = 12.0):

        self.criterion.set_exit_weight_ramp(
            epoch=epoch,
            ramp_epochs=ramp_epochs,
            max_boost=max_boost,
        )

        exit_boost = getattr(self.criterion, "_exit_boost", None)
        if exit_boost is not None:
            print(f"[DEBUG] Epoch {epoch}: exit boost = {exit_boost:.3f}")

    def save_checkpoint(self, epoch: int, path: Path, config: dict):

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': config,
            'feature_names': getattr(config, 'selected_features', None),
            'scaler_state': None,
            'metadata': {
                'timestamp': str(datetime.now()),
                'n_features': config['n_features'],
                'seq_len': config['seq_len'],
                'total_windows': len(self.splits['train'])
            }
        }
        torch.save(checkpoint, path)
        Console.print(f"💾 Saved checkpoint: {path} (epoch {epoch})")

    def load_checkpoint(self, path: Path):

        checkpoint = torch.load(path, map_location=self.device)

        if 'config' not in checkpoint:
            raise ValueError(f"Checkpoint missing config. Old format?")

        loaded_config = checkpoint['config']

        if loaded_config['n_features'] != self.model.n_features:
            raise ValueError(f"Feature mismatch: checkpoint={loaded_config['n_features']}, "
                            f"model={self.model.n_features}")

        if loaded_config['seq_len'] != self.model.seq_len:
            raise ValueError(f"Seq length mismatch: checkpoint={loaded_config['seq_len']}, "
                            f"model={self.model.seq_len}")

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        console.print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
        console.print(f"📋 Config: {loaded_config['n_features']} features, {loaded_config['seq_len']} seq_len")

        return checkpoint['config']

    def train(self, num_epochs):

        from rich.live import Live
        from rich.table import Table
        from rich.progress import (
            Progress, BarColumn, TextColumn, TimeElapsedColumn,
            TimeRemainingColumn, MofNCompleteColumn
        )
        from rich.console import Group
        import psutil

        console = Console()
        wandb_active = self.use_wandb and self.wandb is not None

        try:
            if not self.config.get('rich_dashboard', True):
                console.print("[yellow]⚠️ Rich dashboard disabled — using simple loop[/yellow]")
                return self._legacy_train(num_epochs)

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

                    self.set_exit_weight_ramp(
                        epoch=epoch,
                        ramp_epochs=self.config.get('exit_ramp_epochs', 10),
                        max_boost=self.config.get('exit_max_boost', 8.0),
                    )

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

                    val_loss_float = float(self._sf(val_loss))

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
        finally:
            if wandb_active:
                try:
                    self.wandb.finish()
                except Exception:
                    pass

    def _legacy_train(self, num_epochs):
        if not hasattr(self, 'loss_history'):
            self.loss_history = []

        for epoch in range(num_epochs):
            self.set_exit_weight_ramp(
                epoch=epoch,
                ramp_epochs=self.config.get('exit_ramp_epochs', 10),
                max_boost=self.config.get('exit_max_boost', 8.0),
            )
            train_loss, train_components = self.train_epoch(epoch)
            val_loss, val_components, entry_acc, exit_acc = self.validate()

            self.last_epoch_train_components = dict(train_components or {})
            self.last_epoch_val_components = dict(val_components or {})

            probe_metrics = {}
            try:
                first_val = next(iter(self.val_loader))
                features = first_val['features'].to(self.device)
                position_context = {
                    'unrealized_pnl': first_val['unrealized_pnl'].to(self.device).unsqueeze(1),
                    'time_in_position': first_val['time_in_position'].to(self.device).unsqueeze(1),
                }
                with torch.no_grad():
                    preds = self.model(features, position_context=position_context)
                probe_metrics = self._epoch_probe(epoch, first_val, preds) or {}
            except StopIteration:
                probe_metrics = {}

            val_loss_float = float(self._sf(val_loss))
            if epoch < self.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                self.scheduler.step(val_loss_float)

            lr_now = self._sf(self.optimizer.param_groups[0]['lr'])
            self.loss_history.append(self._sf(val_loss))

            self._log_to_wandb(
                step=epoch,
                epoch=epoch,
                train_loss=train_loss,
                train_components=train_components,
                val_loss=val_loss,
                val_components=val_components,
                entry_acc=entry_acc,
                exit_acc=exit_acc,
                lr_now=lr_now,
                probe_metrics=probe_metrics,
            )

            if self._sf(val_loss) < self._sf(self.best_val_loss):
                self.best_val_loss = self._sf(val_loss)
                best_path = self.config.get('best_model_path', 'models/best_model.pt')
                self.save_checkpoint(best_path, epoch, self._sf(val_loss))

class TradingDataset(Dataset):

    def __init__(self, features: np.ndarray, returns: np.ndarray, prices: np.ndarray,
                 seq_len: int = 100, prediction_horizon: int = 1):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.returns = torch.as_tensor(returns, dtype=torch.float32)
        self.prices = torch.as_tensor(prices, dtype=torch.float32)
        self.seq_len = seq_len
        self.prediction_horizon = prediction_horizon

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

        max_lookforward = min(50, len(self.returns) - label_idx)

        if max_lookforward > 5:
            future_prices = self.prices[label_idx:label_idx + max_lookforward]
            future_rets = (future_prices - current_price) / current_price

            peak_return = future_rets.max()
            peak_idx = future_rets.argmax()
            trough_return = future_rets.min()
            trough_idx = future_rets.argmin()

            take_profit_target = 0.0
            if peak_return > 0.02:

                bars_to_peak = peak_idx
                if bars_to_peak <= 5:
                    take_profit_target = 1.0
                elif bars_to_peak <= 10:
                    take_profit_target = 0.7

            stop_loss_target = 0.0
            if trough_return < -0.015:
                bars_to_trough = trough_idx
                if bars_to_trough <= 3:
                    stop_loss_target = 1.0
                elif bars_to_trough <= 7:
                    stop_loss_target = 0.7

            let_run_target = 0.0
            if future_return > 0.01:

                if max_lookforward >= 20:
                    near_returns = future_rets[:10].mean()
                    far_returns = future_rets[10:20].mean()
                    if far_returns > near_returns and far_returns > 0.02:
                        let_run_target = 1.0

            regime_change_target = 0.0
            if max_lookforward >= 10:
                recent_vol = future_rets[:5].std()
                future_vol = future_rets[5:10].std()
                if future_vol > recent_vol * 2:
                    regime_change_target = 1.0

        else:

            take_profit_target = 0.0
            stop_loss_target = 0.0
            let_run_target = 0.0
            regime_change_target = 0.0

        return_window = self.returns[idx + self.seq_len: idx + self.seq_len + self.prediction_horizon]
        if return_window.numel() == 0:
            actual_volatility = torch.tensor(0.0, dtype=torch.float32)
        elif return_window.numel() == 1:
            actual_volatility = torch.abs(return_window[0])
        else:
            actual_volatility = torch.std(return_window, unbiased=False)

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

            'take_profit_label': torch.tensor(take_profit_target, dtype=torch.float32),
            'stop_loss_label': torch.tensor(stop_loss_target, dtype=torch.float32),
            'let_winner_run_label': torch.tensor(let_run_target, dtype=torch.float32),
            'regime_change_label': torch.tensor(regime_change_target, dtype=torch.float32),

            'unrealized_pnl': future_return,
            'time_in_position': torch.tensor(float(self.prediction_horizon), dtype=torch.float32),
        }
