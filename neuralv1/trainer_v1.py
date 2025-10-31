
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from rich.console import Console

console = Console()

class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks: int = 11):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.l1 = nn.SmoothL1Loss()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks // 2))

    def forward(self, preds: dict, targets: torch.Tensor, epoch: int = 0):

        bce_loss = self.bce(preds['entry'], targets[:, 0])
        bce_loss += self.bce(preds['tp'], targets[:, 1])
        bce_loss += self.bce(preds['sl'], targets[:, 2])

        if epoch > 10:
            bce_loss *= 0.5

        reg_loss = self.mse(preds['others'][:, :4], targets[:, 3:7])
        reg_loss += self.l1(preds['others'][:, 4:], targets[:, 7:])

        total = bce_loss + reg_loss
        return total, {'bce': bce_loss.item(), 'reg': reg_loss.item()}

class V1FixedTrainer:
    def __init__(self, model, splits: dict, config: dict):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.config = config
        self.splits = splits
        self.loss_fn = MultiTaskLoss()
        self.optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config['epochs'])
        self.scaler = torch.amp.GradScaler('cuda') if 'cuda' in str(self.device) else None
        self.best_val = float('inf')
        self.patience = 0

    def train(self) -> str:
        model_dir = self.config['model_dir']
        best_path = model_dir / 'v1_fixed_best.pt'

        num_windows = min(5, len(self.splits['train']))
        console.print(f"🎯 Training on {num_windows} windows")

        for window_idx in range(num_windows):
            self.patience = 0
            self.best_val = float('inf')
            train_f, train_l = self.splits['train'][window_idx]

            console.print(f"\n📊 Window {window_idx+1}/{num_windows}: Train shape {train_f.shape}, Labels {train_l.shape}")

            train_ds = TensorDataset(
                torch.tensor(train_f, dtype=torch.float32).to(self.device),
                torch.tensor(train_l, dtype=torch.float32).to(self.device)
            )
            train_loader = DataLoader(train_ds, batch_size=self.config['batch_size'], shuffle=False)

            for epoch in range(self.config['epochs']):
                self.model.train()
                tot_loss = 0
                batch_count = 0

                for feat, label in train_loader:
                    context = torch.zeros((feat.size(0), 2), device=self.device)
                    self.optimizer.zero_grad()

                    if self.scaler:
                        with torch.amp.autocast('cuda'):
                            preds = self.model(feat, context)
                            loss, comp = self.loss_fn(preds, label, epoch)

                        self.scaler.scale(loss).backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        preds = self.model(feat, context)
                        loss, comp = self.loss_fn(preds, label, epoch)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()

                    tot_loss += loss.item()
                    batch_count += 1

                val_loss = self.validate(self.splits['val'][window_idx])
                self.scheduler.step()

                avg_loss = tot_loss / max(batch_count, 1)
                wandb.log({
                    'window': window_idx,
                    'epoch': epoch,
                    'train_loss': avg_loss,
                    'val_loss': val_loss,
                    'lr': self.scheduler.get_last_lr()[0]
                })

                if epoch % 10 == 0:
                    console.print(f"  Epoch {epoch}/{self.config['epochs']}: Train={avg_loss:.4f}, Val={val_loss:.4f}")

                if val_loss < self.best_val:
                    self.best_val = val_loss
                    torch.save(self.model.state_dict(), best_path)
                    self.patience = 0
                    console.print(f"  💾 Saved best model: val_loss={val_loss:.4f}")
                else:
                    self.patience += 1
                    if self.patience > 45:
                        console.print(f"  ⏹️ Early stopping at epoch {epoch}")
                        break

            console.print(f"✅ Window {window_idx+1}/{num_windows} complete; Best val loss: {self.best_val:.4f}")

        console.print(f"🎉 Training complete! Best model saved to {best_path}")
        return str(best_path)

    def validate(self, val_split: tuple) -> float:
        val_f, val_l = val_split
        val_ds = TensorDataset(
            torch.tensor(val_f, dtype=torch.float32).to(self.device),
            torch.tensor(val_l, dtype=torch.float32).to(self.device)
        )
        val_loader = DataLoader(val_ds, batch_size=self.config['batch_size'])

        self.model.eval()
        tot_loss = 0
        batch_count = 0

        with torch.no_grad():
            for feat, label in val_loader:
                context = torch.zeros((feat.size(0), 2), device=self.device)
                preds = self.model(feat, context)
                loss, _ = self.loss_fn(preds, label, epoch=999)
                tot_loss += loss.item()
                batch_count += 1

        return tot_loss / max(batch_count, 1)
