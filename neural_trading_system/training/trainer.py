import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from typing import Dict
from tqdm import tqdm
import wandb  # For experiment tracking

class TradingDataset(Dataset):
    def __init__(self, features: np.ndarray, returns: np.ndarray, seq_len: int = 100, prediction_horizon: int = 1):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.returns = torch.as_tensor(returns, dtype=torch.float32)
        self.seq_len = seq_len
        self.prediction_horizon = prediction_horizon
        # Ensure non-negative length
        self.valid_length = max(0, len(self.features) - seq_len - prediction_horizon + 1)

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

        entry_label = torch.tensor(1.0 if future_return > 0.01 else 0.0, dtype=torch.float32)
        exit_label = torch.tensor(1.0 if future_return < -0.005 else 0.0, dtype=torch.float32)

        return {
            'features': feature_seq,
            'future_return': future_return,
            'actual_volatility': actual_volatility,
            'entry_label': entry_label,
            'exit_label': exit_label
        }



class MultiTaskLoss(nn.Module):
    """
    Custom multi-task loss with uncertainty weighting.
    
    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses"
    https://arxiv.org/abs/1705.07115
    """
    
    def __init__(self, num_tasks=5):
        super().__init__()
        # Learnable uncertainty parameters
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, predictions: Dict, targets: Dict):
        """
        Args:
            predictions: Dict from model forward pass
            targets: Dict with ground truth labels
        """
        # Entry classification loss
        logits_entry = predictions['entry_prob'].view(-1)         # logits shape [B]
        labels_entry = targets['entry_label'].view(-1)            # [B]
        entry_loss = F.binary_cross_entropy_with_logits(logits_entry, labels_entry)

        logits_exit = predictions['exit_prob'].view(-1)           # [B]
        labels_exit = targets['exit_label'].view(-1)              # [B]
        exit_loss = F.binary_cross_entropy_with_logits(logits_exit, labels_exit)

        ret_pred = predictions['expected_return'].view(-1)        # [B]
        ret_true = targets['future_return'].view(-1)              # [B]
        return_loss = F.mse_loss(ret_pred, ret_true)

        vol_pred = predictions['volatility_forecast'].view(-1)    # [B]
        vol_true = targets['actual_volatility'].view(-1)          # [B]
        volatility_loss = F.mse_loss(vol_pred, vol_true)

        if predictions['entry_prob'].ndim == 0:
            predictions['entry_prob'] = predictions['entry_prob'].unsqueeze(0)

        # Return regression loss
        return_loss = F.mse_loss(
            predictions['expected_return'].squeeze(),
            targets['future_return']
        )
        
        # Volatility forecasting loss
        volatility_loss = F.mse_loss(
            predictions['volatility_forecast'].squeeze(),
            targets['actual_volatility']
        )
        
        # VAE loss (reconstruction + KL divergence)
        vae_recon_loss = F.mse_loss(
            predictions['vae_recon'],
            targets['features'][:, -1, :]  # Reconstruct last feature vector
        )
        
        kl_loss = -0.5 * torch.sum(
            1 + predictions['regime_logvar'] - 
            predictions['regime_mu'].pow(2) - 
            predictions['regime_logvar'].exp()
        )
        
        vae_loss = vae_recon_loss + 0.001 * kl_loss  # Weight KL term
        
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
            precision_vae * vae_loss + self.log_vars[4]
        )

        return {
            'total_loss': total_loss,
            'entry_loss': entry_loss.item(),
            'exit_loss': exit_loss.item(),
            'return_loss': return_loss.item(),
            'volatility_loss': volatility_loss.item(),
            'vae_loss': vae_loss.item(),
            'uncertainties': self.log_vars.detach().cpu().numpy()
        }


class NeuralTrainer:
    """
    Training loop with advanced techniques:
    - Mixed precision training
    - Gradient clipping
    - Learning rate scheduling
    - Early stopping
    - Gradient accumulation
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
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5),
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler - Cosine annealing with warm restarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.get('T_0', 10),
            T_mult=2,
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # Loss function
        self.criterion = MultiTaskLoss(num_tasks=5)
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
        
        # Tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.gradient_accumulation_steps = config.get('grad_accum_steps', 4)
        
        # Initialize Weights & Biases
        if config.get('use_wandb', True):
            wandb.init(
                project="neural-trading-system",
                config=config,
                name=config.get('run_name', 'experiment')
            )
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        loss_components = {
            'entry': 0, 'exit': 0, 'return': 0, 
            'volatility': 0, 'vae': 0
        }
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            features = batch['features'].to(self.device)
            future_return = batch['future_return'].to(self.device)
            actual_volatility = batch['actual_volatility'].to(self.device)
            entry_label = batch['entry_label'].to(self.device)
            exit_label = batch['exit_label'].to(self.device)
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast() if self.scaler else torch.enable_grad():
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
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Track losses
            total_loss += loss_dict['total_loss']
            loss_components['entry'] += loss_dict['entry_loss']
            loss_components['exit'] += loss_dict['exit_loss']
            loss_components['return'] += loss_dict['return_loss']
            loss_components['volatility'] += loss_dict['volatility_loss']
            loss_components['vae'] += loss_dict['vae_loss']
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Average losses
        n_batches = len(self.train_loader)
        avg_loss = total_loss / n_batches
        avg_components = {k: v / n_batches for k, v in loss_components.items()}
        
        return avg_loss, avg_components
    
    def validate(self):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0
        loss_components = {
            'entry': 0, 'exit': 0, 'return': 0,
            'volatility': 0, 'vae': 0
        }
        
        # Metrics
        entry_correct = 0
        exit_correct = 0
        total_samples = 0
        
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
                
                # Calculate accuracy
                entry_pred = (predictions['entry_prob'] > 0.5).float().squeeze()
                exit_pred = (predictions['exit_prob'] > 0.5).float().squeeze()

                entry_correct += (entry_pred == entry_label).sum().item()
                exit_correct += (exit_pred == exit_label).sum().item()
                total_samples += len(entry_label)
        
        # Average metrics
        n_batches = len(self.val_loader)
        avg_loss = total_loss / n_batches
        avg_components = {k: v / n_batches for k, v in loss_components.items()}
        
        entry_accuracy = entry_correct / total_samples
        exit_accuracy = exit_correct / total_samples
        
        assert predictions['entry_prob'].shape[0] == targets['entry_label'].shape[0], "Batch size mismatch for entry head"
        assert predictions['exit_prob'].shape[0] == targets['exit_label'].shape[0], "Batch size mismatch for exit head"

        return avg_loss, avg_components, entry_accuracy, exit_accuracy
    
    def train(self, num_epochs):
        """Full training loop."""
        for epoch in range(num_epochs):
            # Train
            train_loss, train_components = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_components, entry_acc, exit_acc = self.validate()
            
            # Learning rate step
            self.scheduler.step()
            
            # Log to wandb
            # if self.config.get('use_wandb', True):
            #     wandb.log({
            #         'epoch': epoch,
            #         'train_loss': train_loss,
            #         'val_loss': val_loss,
            #         'entry_accuracy': entry_acc,
            #         'exit_accuracy': exit_acc,
            #         'learning_rate': self.optimizer.param_groups[0]['lr'],
            #         **{f'train_{k}': v for k, v in train_components.items()},
            #         **{f'val_{k}': v for k, v in val_components.items()}
            #     })
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Entry Acc: {entry_acc:.3f} | Exit Acc: {exit_acc:.3f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                # 1) Save best model checkpoint
                best_path = self.config.get('best_model_path', 'models/best_model.pt')
                self.save_checkpoint(best_path, epoch, val_loss)

                # 2) Save the matched feature extractor
                # Expect the pipeline passed a reference via config
                feature_extractor = self.config.get('feature_extractor_ref', None)
                if feature_extractor is not None:
                    import pickle, os
                    fe_path = os.path.splitext(best_path)[0] + '_feature_extractor.pkl'
                    with open(fe_path, 'wb') as f:
                        pickle.dump(feature_extractor, f)

                # 3) Optional: sync to W&B
                if self.config.get('use_wandb', True):
                    try:
                        import wandb
                        wandb.save(best_path)
                        if feature_extractor is not None:
                            wandb.save(fe_path)
                    except Exception:
                        pass

                print(f"💾 New best model saved! Val loss: {val_loss:.4f}")

            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.get('patience', 15):
                print(f"\n⏹️ Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Save checkpoint every N epochs
            if (epoch + 1) % self.config.get('save_every', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt', epoch, val_loss)
        
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