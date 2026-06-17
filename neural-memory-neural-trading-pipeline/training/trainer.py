from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .loss import MultiTaskLoss


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    early_stopping_patience: int = 15
    gradient_clip: float = 1.0
    batch_size: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    def __init__(self, model: torch.nn.Module, config: TrainConfig | None = None) -> None:
        self.model = model
        self.cfg = config or TrainConfig()
        self.loss_fn = MultiTaskLoss()

    def _step(self, batch, optimizer=None):
        x, y = batch
        x = x.to(self.cfg.device)
        y = {k: v.to(self.cfg.device) for k, v in y.items()}
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip)
            optimizer.step()
        return float(loss.detach().cpu())

    def train(self, train_ds, val_ds):
        self.model.to(self.cfg.device)
        self.loss_fn.to(self.cfg.device)
        tr_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True)
        va_loader = DataLoader(val_ds, batch_size=self.cfg.batch_size, shuffle=False)

        optimizer = AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

        best_val = float("inf")
        patience = 0
        history = []
        for epoch in range(self.cfg.epochs):
            self.model.train()
            train_loss = sum(self._step(b, optimizer) for b in tr_loader) / max(len(tr_loader), 1)

            self.model.eval()
            with torch.no_grad():
                val_loss = sum(self._step(b, None) for b in va_loader) / max(len(va_loader), 1)
            scheduler.step(val_loss)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= self.cfg.early_stopping_patience:
                    break

        self.model.load_state_dict(best_state)
        return history
