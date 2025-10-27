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

    def compute_attention_entropy_loss(self, attention_weights: list, weight: float = 0.5) -> torch.Tensor:
        """Prevent attention collapse, especially in early layers."""
        if not attention_weights:
            return torch.tensor(0.0, device=self.log_vars.device)
        
        total_penalty = 0.0
        
        for layer_idx, attn in enumerate(attention_weights):
            if not isinstance(attn, torch.Tensor):
                continue
            
            # attn: [B, num_heads, seq_len, seq_len]
            attn_avg = attn.mean(dim=1)  # [B, seq_len, seq_len]
            
            # Compute entropy per query position
            entropy = -torch.sum(
                attn_avg * torch.log(attn_avg + 1e-8),
                dim=-1  # Over key positions
            ).mean()  # Average over batch and queries
            
            # Target entropy (uniform distribution over 100 tokens = log(100) ≈ 4.6)
            target_entropy = torch.log(torch.tensor(attn.shape[-1], dtype=torch.float32))
            
            # Penalty grows quadratically as entropy drops below 50% of target
            min_acceptable = 0.5 * target_entropy  # e.g., 2.3 for seq_len=100
            
            if entropy < min_acceptable:
                # Strong penalty for collapsed attention
                penalty = ((min_acceptable - entropy) / target_entropy) ** 2
                
                # Layer 0 gets EXTRA penalty (it's collapsing first)
                if layer_idx == 0:
                    penalty *= 10.0  # Triple penalty for first layer
                
                total_penalty += penalty
        
        return weight * total_penalty

    def set_exit_weight_ramp(self, epoch: int, ramp_epochs: int = 10, max_boost: float = 12.0):
        """Ramp exit weight linearly from 1.0 to max_boost over ramp_epochs."""
        if ramp_epochs <= 0:
            r = 1.0
        else:
            r = min(1.0, float(max(0, epoch)) / float(ramp_epochs))
        
        self._exit_boost = 1.0 + r * (max_boost - 1.0)

    def forward(self, predictions: dict, targets: dict):
        """Compute multi-task loss with debug output."""
        device = predictions['entry_logits'].device
        
        # ======== ENTRY ========
        entry_logits = predictions['entry_logits'].view(-1)
        entry_true = targets['entry_label'].view(-1).float()
        
        alpha_entry = self._batch_alpha(entry_true)
        entry_loss = self._focal_loss(entry_logits, entry_true, alpha=alpha_entry, gamma=2.0)
        
        # ======== RETURN ========
        ret_pred = predictions['expected_return'].view(-1)
        ret_true = targets['future_return'].view(-1)
        return_loss = F.smooth_l1_loss(ret_pred, ret_true)
        
        # ======== VOLATILITY ========
        vol_pred = predictions['volatility_forecast'].view(-1)
        vol_true = targets['actual_volatility'].view(-1)
        volatility_loss = F.mse_loss(vol_pred, vol_true)
        
        # ======== VAE ========
        vae_recon_loss = torch.tensor(0.0, device=device)
        kl_loss = torch.tensor(0.0, device=device)
        
        if 'vae_recon' in predictions and 'sequence_repr' in predictions:
            vae_recon_loss = F.mse_loss(predictions['vae_recon'], predictions['sequence_repr'])
        
        if 'regime_mu' in predictions and 'regime_logvar' in predictions:
            mu = predictions['regime_mu']
            lv = torch.clamp(predictions['regime_logvar'], min=-10, max=10)
            kl_loss = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1).mean()
        
        vae_loss = vae_recon_loss + 1e-5 * kl_loss
        
        # ======== EXIT (TP/SL) ========
        exit_logits_tp = predictions.get('take_profit_logits', torch.zeros(len(entry_true), device=device)).view(-1)
        exit_logits_sl = predictions.get('stop_loss_logits', torch.zeros(len(entry_true), device=device)).view(-1)
        exit_true_tp = targets['take_profit_label'].view(-1).float()
        exit_true_sl = targets['stop_loss_label'].view(-1).float()
        
        exit_loss_tp = F.binary_cross_entropy_with_logits(exit_logits_tp, exit_true_tp)
        exit_loss_sl = F.binary_cross_entropy_with_logits(exit_logits_sl, exit_true_sl)
        exit_bundle_loss = exit_loss_tp + exit_loss_sl
        
        # ======== REGULARIZATION ========
        # attention_weights = predictions.get('attention_weights', [])
        # entropy_reg_loss = self.compute_attention_entropy_loss(attention_weights, weight=1.0) # From 0.1
        
        # TODO :: PLAN B
        # entropy_reg_loss = torch.tensor(0.0, device=device)
        attention_weights = predictions.get('attention_weights', [])
        entropy_reg_loss = self.compute_attention_entropy_loss(
            attention_weights, 
            weight=5.0  # Increase from 0.1 to 0.5 for stronger regularization
        )
        
        # diversity_penalty = torch.tensor(0.0, device=device)
        # confidence_penalty = torch.tensor(0.0, device=device)
        
        # ======== DEBUG OUTPUT ========
        # print(f"\n{'='*80}")
        # print(f"LOSS COMPONENT BREAKDOWN")
        # print(f"{'='*80}")
        # print(f"entry_loss:         {entry_loss.item():.6f}")
        # print(f"return_loss:        {return_loss.item():.6f}")
        # print(f"volatility_loss:    {volatility_loss.item():.6f}")
        # print(f"vae_loss:           {vae_loss.item():.6f}")
        # print(f"exit_bundle_loss:   {exit_bundle_loss.item():.6f}")
        # print(f"entropy_reg_loss:   {entropy_reg_loss.item():.6f}")
        # print(f"diversity_penalty:  {diversity_penalty.item():.6f}")
        # print(f"confidence_penalty: {confidence_penalty.item():.6f}")
        # print(f"\nlog_vars: {self.log_vars.detach()}")
        # print(f"_exit_boost: {self._exit_boost}")
        
        # ======== COMPUTE TOTAL LOSS (FIXED FORMULA) ========
        # CLAMPED log_vars to prevent extreme negative values
        log_vars_clamped = torch.clamp(self.log_vars, min=-10, max=10)
        
        # Precisions (inverse variance weighting)
        precision_entry = torch.exp(-log_vars_clamped[0])
        precision_return = torch.exp(-log_vars_clamped[2])
        precision_vol = torch.exp(-log_vars_clamped[3])
        precision_vae = torch.exp(-log_vars_clamped[4])
        
        # Build loss with uncertainty weighting
        term1 = self.task_weights["entry"] * (precision_entry * entry_loss + log_vars_clamped[0])
        term2 = self._exit_boost * exit_bundle_loss
        term3 = self.task_weights["return"] * (precision_return * return_loss + log_vars_clamped[2])
        term4 = self.task_weights["volatility"] * (precision_vol * volatility_loss + log_vars_clamped[3])
        term5 = self.task_weights["vae"] * (precision_vae * vae_loss + log_vars_clamped[4])
        term6 = entropy_reg_loss
        
        # print(f"\nTERM BREAKDOWN:")
        # print(f"term1 (entry):       {term1.item():.6f}")
        # print(f"term2 (exit):        {term2.item():.6f}")
        # print(f"term3 (return):      {term3.item():.6f}")
        # print(f"term4 (volatility):  {term4.item():.6f}")
        # print(f"term5 (vae):         {term5.item():.6f}")
        # print(f"term6 (entropy):     {term6.item():.6f}")
        
        total_loss = term1 + term2 + term3 + term4 + term5 + term6
        
        # print(f"\nFINAL: total_loss = {total_loss.item():.6f}")
        # print(f"⚠️  IS_NEGATIVE: {total_loss.item() < 0}")
        # print(f"⚠️  IS_NaN: {torch.isnan(total_loss)}")
        # print(f"⚠️  IS_Inf: {torch.isinf(total_loss)}")
        # print(f"{'='*80}\n")
        
        # Clamp total loss to prevent negative values from propagating
        if total_loss.item() < 0:
            print(f"⚠️  NEGATIVE LOSS DETECTED! Clamping to 0.1")
            total_loss = torch.clamp(total_loss, min=0.1)
        
        return {
            'total_loss': total_loss,
            'entry_loss': entry_loss,
            'return_loss': return_loss,
            'volatility_loss': volatility_loss,
            'vae_loss': vae_loss,
            'exit_bundle_loss': exit_bundle_loss,
            'entropy_reg_loss': entropy_reg_loss,
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
            for layer_idx, A in enumerate(attns):
                if not isinstance(A, torch.Tensor): continue
                
                # Check per-head entropy
                per_head_entropy = []
                for head in range(A.shape[1]):
                    head_attn = A[:, head, :, :].float().clamp(1e-6, 1)
                    H = (-head_attn * head_attn.log()).sum(dim=-1).mean().item()
                    per_head_entropy.append(H)
                
                print(f"   Layer {layer_idx} per-head entropy: {per_head_entropy}")
            
            for A in attns:
                if not isinstance(A, torch.Tensor): continue
                P = A.float().clamp(1e-6, 1).view(-1, A.shape[-1])  # (B*heads*T, T)
                H = (-P * P.log()).sum(dim=1).mean().item()
                entropies.append(H)
            attn_txt = " / ".join(f"{h:.2f}" for h in entropies) if entropies else "—"

            exit_signals = predictions.get('exit_signals', {})

            layer0_heads = per_head_entropy  # From your existing code
            dead_heads = sum(1 for h in layer0_heads if h < 0.1)
            avg_entropy = sum(layer0_heads) / len(layer0_heads)

            if dead_heads > 0:
                print(f"   🔴 ALERT: {dead_heads}/8 heads collapsed (< 0.1 entropy)")
            if avg_entropy < 0.5:
                print(f"   🟡 WARNING: Layer 0 avg entropy = {avg_entropy:.2f}")
                
            # AUTO-STOP if too many dead heads
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

    # ---------- core loops ----------

    def validate(self, on_batch=None, batch_update_every: int = 10):
        """Validate model on val_loader with FIXED accuracy calculations."""
        self.model.eval()
        total_loss = 0.0
        loss_components = {}
        
        # FIX: Initialize accumulators BEFORE the loop
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
                
                # FIX: Ensure shapes are consistent [B] not [B, 1]
                if unrealized_pnl.dim() == 2:
                    unrealized_pnl = unrealized_pnl.squeeze(-1)  # [B, 1] → [B]
                if entry_label.dim() == 2:
                    entry_label = entry_label.squeeze(-1)
                if take_profit_label.dim() == 2:
                    take_profit_label = take_profit_label.squeeze(-1)
                if stop_loss_label.dim() == 2:
                    stop_loss_label = stop_loss_label.squeeze(-1)
                
                batch_size = len(unrealized_pnl)
                
                position_context = {
                    'unrealized_pnl': unrealized_pnl.unsqueeze(-1),  # [B] → [B, 1] for model
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
                    if k == 'total_loss':
                        continue
                    loss_components[k] = loss_components.get(k, 0.0) + self._sf(v)
                
                # ====== CHECKPOINT: ENTRY ACCURACY (FIXED) ======
                entry_pred = (predictions['entry_prob'] > 0.5).float().squeeze()
                entry_true = (entry_label > 0.5).float()
                
                # Ensure same shape for comparison
                entry_pred = entry_pred.view(-1)  # [B]
                entry_true = entry_true.view(-1)  # [B]
                
                batch_entry_correct = (entry_pred == entry_true).sum().item()
                batch_entry_total = len(entry_label)
                
                entry_correct += batch_entry_correct
                entry_total += batch_entry_total
                
                # if debug_mode and batch_idx == 0:
                #     print(f"\n{'='*80}")
                #     print(f"CHECKPOINT: ENTRY ACCURACY (BATCH 0)")
                #     print(f"{'='*80}")
                #     print(f"  entry_pred shape: {entry_pred.shape}")
                #     print(f"  entry_true shape: {entry_true.shape}")
                #     print(f"  Batch entry_correct: {batch_entry_correct} / {batch_entry_total}")
                #     print(f"  Running total: {entry_correct} / {entry_total}")
                #     print(f"  Batch accuracy: {batch_entry_correct / max(1, batch_entry_total):.4f}")
                
                # ====== CHECKPOINT: EXIT ACCURACY (FIXED) ======
                if 'unified_exit_prob' in predictions:
                    unified_exit = predictions['unified_exit_prob']  # [B, 1]
                    
                    # FIX: Ensure all tensors are 1D [B] before comparison
                    unified_exit = unified_exit.squeeze(-1)  # [B, 1] → [B]
                    
                    # Create unified exit label - SAME DIMENSION AS unified_exit
                    in_profit = (unrealized_pnl > 0).float()  # [B]
                    in_loss = (unrealized_pnl <= 0).float()   # [B]
                    
                    # FIXED: Element-wise multiplication stays [B]
                    unified_label = (
                        in_profit * take_profit_label.float() +
                        in_loss * stop_loss_label.float()
                    )  # [B]
                    
                    # Calculate accuracy with matching shapes
                    exit_pred = (unified_exit > 0.5).float()  # [B]
                    exit_true = (unified_label > 0.5).float()  # [B]
                    
                    # Ensure both 1D
                    exit_pred = exit_pred.view(-1)  # [B]
                    exit_true = exit_true.view(-1)  # [B]
                    
                    batch_exit_correct = (exit_pred == exit_true).sum().item()
                    batch_exit_total = len(exit_true)
                    
                    exit_correct += batch_exit_correct
                    exit_total += batch_exit_total
                    
                    # if debug_mode and batch_idx == 0:
                    #     print(f"\n{'='*80}")
                    #     print(f"CHECKPOINT: EXIT ACCURACY (BATCH 0)")
                    #     print(f"{'='*80}")
                    #     print(f"  unified_exit shape AFTER squeeze: {unified_exit.shape}")
                    #     print(f"  unified_label shape: {unified_label.shape}")
                    #     print(f"  exit_pred shape: {exit_pred.shape}")
                    #     print(f"  exit_true shape: {exit_true.shape}")
                    #     print(f"  Batch exit_correct: {batch_exit_correct} / {batch_exit_total}")
                    #     print(f"  Running total: {exit_correct} / {exit_total}")
                    #     print(f"  Batch accuracy: {batch_exit_correct / max(1, batch_exit_total):.4f}")
                    #     print(f"  in_profit count: {in_profit.sum().item()}")
                    #     print(f"  in_loss count: {in_loss.sum().item()}")
                
                if on_batch and ((batch_idx + 1) % batch_update_every == 0):
                    on_batch(
                        batch_idx + 1,
                        n_batches,
                        float(self._sf(out['total_loss'])),
                        self.optimizer.param_groups[0]['lr'],
                    )
                
                batch_counter += 1
        
        # ====== FINAL CALCULATIONS ======
        avg_loss = total_loss / max(1, n_batches)
        avg_components = {k: v / max(1, n_batches) for k, v in loss_components.items()}
        
        entry_accuracy = entry_correct / max(1, entry_total)
        exit_accuracy = exit_correct / max(1, exit_total)
        
        # print(f"\n{'='*80}")
        # print(f"CHECKPOINT: FINAL VALIDATION METRICS")
        # print(f"{'='*80}")
        # print(f"  Total batches processed: {batch_counter}")
        # print(f"  Entry accuracy: {entry_correct} correct / {entry_total} total = {entry_accuracy:.4f}")
        # print(f"  Exit accuracy: {exit_correct} correct / {exit_total} total = {exit_accuracy:.4f}")
        # print(f"  Avg val loss: {avg_loss:.6f}")
        # print(f"  Entry accuracy valid [0,1]: {'YES ✓' if 0.0 <= entry_accuracy <= 1.0 else 'NO ⚠️  BUG'}")
        # print(f"  Exit accuracy valid [0,1]: {'YES ✓' if 0.0 <= exit_accuracy <= 1.0 else 'NO ⚠️  BUG'}")
        # print(f"{'='*80}\n")
        
        return avg_loss, avg_components, entry_accuracy, exit_accuracy

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
            
            out = self.criterion(predictions, targets)
            loss = out['total_loss'] / self.gradient_accumulation_steps
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[WARNING] NaN/Inf loss detected at batch {batch_idx}, skipping")
                self.optimizer.zero_grad()
                continue
            
            # ====== CHECKPOINT: LOSS SANITY (First batch only) ======
            # if batch_idx == 0 and epoch == 0:
            #     print(f"\n{'='*80}")
            #     print(f"CHECKPOINT: LOSS COMPONENTS (EPOCH 0, BATCH 0)")
            #     print(f"{'='*80}")
            #     print(f"  entry_loss: {out.get('entry_loss', 'N/A')}")
            #     print(f"  return_loss: {out.get('return_loss', 'N/A')}")
            #     print(f"  volatility_loss: {out.get('volatility_loss', 'N/A')}")
            #     print(f"  vae_loss: {out.get('vae_loss', 'N/A')}")
            #     print(f"  exit_bundle_loss: {out.get('exit_bundle_loss', 'N/A')}")
            #     print(f"  total_loss: {out['total_loss'].item():.6f}")
            #     print(f"  total_loss is positive: {'YES ✓' if out['total_loss'].item() > 0 else 'NO ⚠️  BUG'}")
            #     print(f"  total_loss is NaN: {'YES ⚠️  BUG' if torch.isnan(out['total_loss']) else 'NO ✓'}")
            #     print(f"  total_loss is Inf: {'YES ⚠️  BUG' if torch.isinf(out['total_loss']) else 'NO ✓'}")
            #     print(f"{'='*80}\n")
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                # ====== CHECKPOINT: GRADIENT FLOW (First batch only) ======
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
                
                # Track gradient norm BEFORE clipping
                total_grad_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                
                total_grad_norm = total_grad_norm ** 0.5
                self.last_grad_norm = total_grad_norm
                
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
                if k == 'total_loss':
                    continue
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

    def set_exit_weight_ramp(self, epoch: int, ramp_epochs: int = 10, max_boost: float = 12.0):
        """Call from trainer each epoch to ramp exit importance."""
        # Linear ramp from 1.0 to max_boost over ramp_epochs
        self.criterion.set_exit_weight_ramp(
            epoch=epoch,
            ramp_epochs=ramp_epochs,
            max_boost=max_boost,
        )

        exit_boost = getattr(self.criterion, "_exit_boost", None)
        if exit_boost is not None:
            print(f"[DEBUG] Epoch {epoch}: exit boost = {exit_boost:.3f}")  # Optional debug

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
                self.set_exit_weight_ramp(
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
            self.set_exit_weight_ramp(
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
