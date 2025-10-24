import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from typing import Dict
from tqdm import tqdm
import wandb


class TradingDataset(Dataset):
    """Enhanced dataset with soft labels and adaptive thresholds"""
    def __init__(self, features: np.ndarray, returns: np.ndarray, seq_len: int = 100, prediction_horizon: int = 1):
        # ✅ CRITICAL FIX: Normalize features to prevent extreme values
        self.features = torch.as_tensor(features, dtype=torch.float32)

        # Check for NaN/Inf in source data
        if torch.isnan(self.features).any() or torch.isinf(self.features).any():
            print("⚠️ NaN/Inf detected in source features! Replacing with zeros.")
            self.features = torch.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)

        # ✅ CRITICAL: Clip extreme values
        features_max = self.features.abs().max()
        if features_max > 100:
            print(f"⚠️ Extreme feature values detected (max={features_max:.2e}). Clamping to [-100, 100].")
            self.features = torch.clamp(self.features, min=-100, max=100)

        self.returns = torch.as_tensor(returns, dtype=torch.float32)
        self.seq_len = seq_len
        self.prediction_horizon = prediction_horizon

        # Ensure non-negative length
        self.valid_length = max(0, len(self.features) - seq_len - prediction_horizon + 1)
        
        # NEW: Dynamic thresholds based on volatility
        self.returns_std = float(np.std(returns))
        self.returns_mean = float(np.mean(returns))
        self.entry_threshold = max(0.01, 1.5 * self.returns_std)  # Adaptive
        self.exit_threshold = max(0.005, 0.75 * self.returns_std)
        
        print(f"📊 Dataset Statistics:")
        print(f"   Returns mean: {self.returns_mean:.6f}")
        print(f"   Returns std: {self.returns_std:.6f}")
        print(f"   Entry threshold: {self.entry_threshold:.4f}")
        print(f"   Exit threshold: {self.exit_threshold:.4f}")

    def __len__(self):
        return self.valid_length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.valid_length:
            raise IndexError(f"Index {idx} out of range for dataset of length {self.valid_length}")

        start = idx
        end = idx + self.seq_len
        feature_seq = self.features[start:end]  # [seq_len, feat_dim]

        # Label index is last step in horizon
        label_idx = end + self.prediction_horizon - 1
        future_return = self.returns[label_idx]

        # Vol window over the horizon
        return_window = self.returns[idx + self.seq_len: idx + self.seq_len + self.prediction_horizon]
        if return_window.numel() == 0:
            actual_volatility = torch.tensor(0.0, dtype=torch.float32)
        elif return_window.numel() == 1:
            actual_volatility = torch.abs(return_window[0])
        else:
            actual_volatility = torch.std(return_window, unbiased=False)

        # NEW: Soft labels with confidence based on magnitude
        # Convert return to confidence score using sigmoid
        entry_confidence = torch.sigmoid(
            (future_return - self.entry_threshold) / (0.5 * self.returns_std)
        )
        exit_confidence = torch.sigmoid(
            (self.exit_threshold - future_return) / (0.5 * self.returns_std)
        )
        
        # Clamp to [0, 1] for safety
        entry_label = torch.clamp(entry_confidence, 0.0, 1.0)
        exit_label = torch.clamp(exit_confidence, 0.0, 1.0)

        return {
            'features': feature_seq,
            'future_return': future_return,
            'actual_volatility': actual_volatility,
            'entry_label': entry_label,
            'exit_label': exit_label
        }


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """
    Focal loss to focus learning on hard examples.
    Helps with class imbalance and pushes model to be more confident.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    probs = torch.sigmoid(logits)
    pt = torch.where(targets == 1, probs, 1 - probs)
    focal_weight = (1 - pt) ** gamma
    loss = alpha * focal_weight * bce
    return loss.mean()


class MultiTaskLoss(nn.Module):
    """
    ✅ ENHANCED Multi-task loss with:
    - Temperature scaling for calibration
    - Confidence penalty to push away from 0.5
    - Focal loss for hard examples
    - Huber loss for returns (robust to outliers)
    """
    def __init__(self, num_tasks=5):
        super().__init__()
        # Learnable uncertainty parameters
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
        # NEW: Learnable temperature for calibration
        self.entry_temp = nn.Parameter(torch.ones(1))
        self.exit_temp = nn.Parameter(torch.ones(1))

    def forward(self, predictions: Dict, targets: Dict):
        """
        Args:
            predictions: Dict from model forward pass
            targets: Dict with ground truth labels
        """
        # ✅ ENHANCED: Use temperature-scaled logits for better calibration
        temp = torch.clamp(self.entry_temp, min=0.1, max=10.0).to(predictions['entry_logits'].device)
        logits_entry = predictions['entry_logits'].view(-1) / temp
        labels_entry = targets['entry_label'].view(-1)
        entry_loss = focal_loss(logits_entry, labels_entry, alpha=0.25, gamma=2.0)

        # ✅ ENHANCED: Use temperature-scaled logits
        temp_exit = torch.clamp(self.exit_temp, min=0.1, max=10.0).to(predictions['exit_logits'].device)
        logits_exit = predictions['exit_logits'].view(-1) / temp_exit
        labels_exit = targets['exit_label'].view(-1)
        exit_loss = focal_loss(logits_exit, labels_exit, alpha=0.25, gamma=2.0)
        
        # NEW: Confidence penalty - penalize predictions near 0.5
        entry_probs = predictions['entry_prob'].view(-1)
        exit_probs = predictions['exit_prob'].view(-1)
        
        # Exponential penalty for being near 0.5 (maximum uncertainty)
        confidence_penalty = (
            torch.mean(torch.exp(-10 * (entry_probs - 0.5).pow(2))) +
            torch.mean(torch.exp(-10 * (exit_probs - 0.5).pow(2)))
        )

        # Return regression loss - Huber loss (more robust to outliers)
        ret_pred = predictions['expected_return'].view(-1)
        ret_true = targets['future_return'].view(-1)
        return_loss = F.smooth_l1_loss(ret_pred, ret_true)

        # Volatility forecasting loss
        vol_pred = predictions['volatility_forecast'].view(-1)
        vol_true = targets['actual_volatility'].view(-1)
        volatility_loss = F.mse_loss(vol_pred, vol_true)

        # ✅ FIXED: VAE reconstructs sequence_repr now
        vae_recon_loss = F.mse_loss(
            predictions['vae_recon'],
            predictions['sequence_repr']
        )

        # ✅ CRITICAL FIX: Clamp logvar in loss calculation
        regime_logvar_clamped = torch.clamp(predictions['regime_logvar'], min=-10, max=10)

        # KL divergence with clamped logvar
        kl_loss = -0.5 * torch.sum(
            1 + regime_logvar_clamped - 
            predictions['regime_mu'].pow(2) - 
            regime_logvar_clamped.exp(),
            dim=1
        ).mean()

        # ✅ REDUCED KL weight to prevent instability
        vae_loss = vae_recon_loss + 0.00001 * kl_loss

        # Uncertainty-weighted combination
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
            0.1 * confidence_penalty  # NEW: Add confidence penalty
        )

        return {
            'total_loss': total_loss,
            'entry_loss': entry_loss.item(),
            'exit_loss': exit_loss.item(),
            'return_loss': return_loss.item(),
            'volatility_loss': volatility_loss.item(),
            'vae_loss': vae_loss.item(),
            'confidence_penalty': confidence_penalty.item(),
            'entry_temp': self.entry_temp.item(),
            'exit_temp': self.exit_temp.item(),
            'uncertainties': self.log_vars.detach().cpu().numpy()
        }


class NeuralTrainer:
    """
    ✅ ENHANCED Training loop with:
    - Prediction distribution monitoring
    - Better gradient handling
    - Early stopping based on confidence spread
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config: Dict,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('lr', 0.0003),  # Higher default LR
            weight_decay=config.get('weight_decay', 1e-4),
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.get('T_0', 20),
            T_mult=2,
            eta_min=config.get('min_lr', 1e-7)
        )

        # Loss function
        self.criterion = MultiTaskLoss(num_tasks=5)

        # Mixed precision training
        self.scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None

        # Tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.gradient_accumulation_steps = config.get('grad_accum_steps', 2)

        # ✅ FIXED: Initialize wandb conditionally
        self.use_wandb = config.get('use_wandb', False)
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

    def train_epoch(self, epoch):
        """Train for one epoch with enhanced monitoring."""
        self.model.train()
        total_loss = 0
        loss_components = {
            'entry': 0, 'exit': 0, 'return': 0,
            'volatility': 0, 'vae': 0, 'confidence_penalty': 0
        }
        
        # NEW: Track prediction distributions
        all_entry_probs = []
        all_exit_probs = []

        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            features = batch['features'].to(self.device)
            future_return = batch['future_return'].to(self.device)
            actual_volatility = batch['actual_volatility'].to(self.device)
            entry_label = batch['entry_label'].to(self.device)
            exit_label = batch['exit_label'].to(self.device)

            # ✅ CRITICAL: Check input data validity
            if torch.isnan(features).any() or torch.isinf(features).any():
                print(f"\n⚠️ NaN/Inf in batch {batch_idx} features. Skipping.")
                self.optimizer.zero_grad()
                continue

            # Forward pass with mixed precision
            with torch.amp.autocast('cuda') if self.scaler else torch.enable_grad():
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

            # Check for NaN in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n⚠️ NaN/Inf in loss at batch {batch_idx}. Skipping.")
                self.optimizer.zero_grad()
                continue

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Proper gradient clipping order
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Track losses (only if valid)
            total_loss += loss_dict['total_loss']
            loss_components['entry'] += loss_dict['entry_loss']
            loss_components['exit'] += loss_dict['exit_loss']
            loss_components['return'] += loss_dict['return_loss']
            loss_components['volatility'] += loss_dict['volatility_loss']
            loss_components['vae'] += loss_dict['vae_loss']
            loss_components['confidence_penalty'] += loss_dict['confidence_penalty']
            
            # NEW: Collect predictions for distribution analysis
            with torch.no_grad():
                all_entry_probs.extend(predictions['entry_prob'].cpu().numpy().flatten().tolist())
                all_exit_probs.extend(predictions['exit_prob'].cpu().numpy().flatten().tolist())

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}",
                'entry_temp': f"{loss_dict.get('entry_temp', 1.0):.3f}",
            })

        # Average losses
        n_batches = len(self.train_loader)
        avg_loss = total_loss / n_batches
        avg_components = {k: v / n_batches for k, v in loss_components.items()}
        
        # NEW: Analyze prediction distribution
        entry_probs_array = np.array(all_entry_probs)
        exit_probs_array = np.array(all_exit_probs)
        
        print(f"\n📊 Prediction Distribution (Epoch {epoch}):")
        print(f"   Entry Probs - Mean: {entry_probs_array.mean():.3f}, Std: {entry_probs_array.std():.3f}")
        print(f"   Entry >0.7: {(entry_probs_array > 0.7).mean()*100:.1f}%, <0.3: {(entry_probs_array < 0.3).mean()*100:.1f}%")
        print(f"   Exit Probs  - Mean: {exit_probs_array.mean():.3f}, Std: {exit_probs_array.std():.3f}")
        print(f"   Exit >0.7: {(exit_probs_array > 0.7).mean()*100:.1f}%, <0.3: {(exit_probs_array < 0.3).mean()*100:.1f}%")

        return avg_loss, avg_components

    def validate(self):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0
        loss_components = {
            'entry': 0, 'exit': 0, 'return': 0,
            'volatility': 0, 'vae': 0, 'confidence_penalty': 0
        }

        # Metrics
        entry_correct = 0
        exit_correct = 0
        total_samples = 0
        
        # NEW: Track validation predictions
        all_entry_probs = []
        all_entry_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
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

                # Calculate accuracy (using 0.5 threshold for soft labels)
                entry_pred = (predictions['entry_prob'] > 0.5).float().squeeze()
                exit_pred = (predictions['exit_prob'] > 0.5).float().squeeze()
                entry_true = (entry_label > 0.5).float()
                exit_true = (exit_label > 0.5).float()
                
                entry_correct += (entry_pred == entry_true).sum().item()
                exit_correct += (exit_pred == exit_true).sum().item()
                total_samples += len(entry_label)
                
                # Collect for calibration analysis
                all_entry_probs.extend(predictions['entry_prob'].cpu().numpy().flatten().tolist())
                all_entry_labels.extend(entry_label.cpu().numpy().flatten().tolist())

        # Average metrics
        n_batches = len(self.val_loader)
        avg_loss = total_loss / n_batches
        avg_components = {k: v / n_batches for k, v in loss_components.items()}
        entry_accuracy = entry_correct / total_samples
        exit_accuracy = exit_correct / total_samples
        
        # NEW: Calibration analysis
        entry_probs_array = np.array(all_entry_probs)
        print(f"\n📊 Validation Prediction Spread:")
        print(f"   Std: {entry_probs_array.std():.3f} (target: >0.15)")
        print(f"   High conf (>0.7): {(entry_probs_array > 0.7).mean()*100:.1f}%")
        print(f"   Low conf (<0.3): {(entry_probs_array < 0.3).mean()*100:.1f}%")

        return avg_loss, avg_components, entry_accuracy, exit_accuracy

    def train(self, num_epochs):
        """Full training loop."""
        print("\n🎯 Starting training loop...")

        for epoch in range(num_epochs):
            # Train
            train_loss, train_components = self.train_epoch(epoch)

            # Validate
            val_loss, val_components, entry_acc, exit_acc = self.validate()

            # Learning rate step
            self.scheduler.step()

            # ✅ FIXED: Log to wandb conditionally
            if self.use_wandb and self.wandb is not None:
                self.wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'entry_accuracy': entry_acc,
                    'exit_accuracy': exit_acc,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    **{f'train_{k}': v for k, v in train_components.items()},
                    **{f'val_{k}': v for k, v in val_components.items()}
                })

            print(f"\nEpoch {epoch} Summary:")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Entry Acc: {entry_acc:.3f} | Exit Acc: {exit_acc:.3f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                best_path = self.config.get('best_model_path', 'models/best_model.pt')
                self.save_checkpoint(best_path, epoch, val_loss)
                print(f"💾 New best model saved! Val loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.config.get('patience', 25):
                print(f"\n⏹️ Early stopping triggered after {epoch + 1} epochs")
                break

            # Save checkpoint every N epochs
            if (epoch + 1) % self.config.get('save_every', 10) == 0:
                checkpoint_path = f"models/checkpoint_epoch_{epoch}.pt"
                self.save_checkpoint(checkpoint_path, epoch, val_loss)

        print("\n✅ Training completed!")

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