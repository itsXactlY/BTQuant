from __future__ import annotations
import math
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 10000, scale: float = 1.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = scale

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        x = x + self.scale * self.pe[:, :T]
        return self.dropout(x)

import torch.nn.utils.spectral_norm as spectral_norm

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, layer_idx=0, num_layers=6):
        super().__init__()

        attn_dropout = 0.2 if layer_idx == 0 else 0.15
        self.mha = nn.MultiheadAttention(
            d_model, num_heads, dropout=attn_dropout, batch_first=True
        )
        self.mha.in_proj_weight = nn.Parameter(
            spectral_norm(nn.Linear(d_model, 3*d_model, bias=False)).weight
        )

        self.sigma_scale = nn.Parameter(torch.ones(1))

        self.ff = nn.Sequential(
            spectral_norm(nn.Linear(d_model, d_ff)),
            nn.GELU(),
            spectral_norm(nn.Linear(d_ff, d_model))
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.15)
        self.dropout2 = nn.Dropout(0.15)

    def forward(self, x, mask=None):
        attn_out, attn_weights = self.mha(
            self.norm1(x), self.norm1(x), self.norm1(x),
            attn_mask=mask, need_weights=True, average_attn_weights=False
        )
        x = x + self.sigma_scale * self.dropout1(attn_out)

        ff_out = self.ff(self.norm2(x))
        x = x + self.sigma_scale * self.dropout2(ff_out)
        return x, attn_weights

class MarketRegimeVAE(nn.Module):
    def __init__(self, d_model: int, latent_dim: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, d_model),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
\
\
\

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

class RegimeChangeDetector(nn.Module):
    def __init__(self, d_model: int, latent_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model + 2 * latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.GELU(),
        )
        self.stability_head = nn.Linear(64, 1)

    def forward(
        self,
        current_repr: torch.Tensor,
        current_z: torch.Tensor,
        historical_z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        delta_z = current_z - historical_z
        x = torch.cat([current_repr, current_z, delta_z], dim=1)
        h = self.mlp(x)
        stability = torch.sigmoid(self.stability_head(h))
        regime_change_score = 1.0 - stability
        return {
            "stability": stability,
            "regime_change_score": regime_change_score,
        }

from torch.nn.utils import spectral_norm

class LetWinnerRunModule(nn.Module):
    def __init__(self, d_model, latent_dim):
        super().__init__()

        self.hold_analyzer = nn.Sequential(
            spectral_norm(nn.Linear(d_model + latent_dim + 3, 256)),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
            spectral_norm(nn.Linear(256, 128)),
            nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            spectral_norm(nn.Linear(128, 64)),
            nn.GELU(),
        )

        self.let_run_head = nn.Sequential(
            spectral_norm(nn.Linear(64, 32)),
            nn.GELU(),
            spectral_norm(nn.Linear(32, 1))
        )

    def forward(self, current_repr, regime_z, unrealized_pnl, time_in_position, expected_return):
        position_context = torch.cat([unrealized_pnl, time_in_position, expected_return], dim=1)
        x = torch.cat([current_repr, regime_z, position_context], dim=1)

        features = self.hold_analyzer(x)

        feat32 = torch.nan_to_num(features.float(), 0.0, 0.0, 0.0).clamp(-1e4, 1e4)

        hold_logit = self.let_run_head(feat32)
        hold_prob = torch.sigmoid(hold_logit)

        return {"hold_logit": hold_logit, "hold_score": hold_prob}

class ProfitTakingModule(nn.Module):
    def __init__(self, d_model, latent_dim):
        super().__init__()

        self.profit_analyzer = nn.Sequential(
            spectral_norm(nn.Linear(d_model + latent_dim + 3, 256)),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
            spectral_norm(nn.Linear(256, 128)),
            nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            spectral_norm(nn.Linear(128, 64)),
            nn.GELU(),
        )

        self.take_profit_head = nn.Sequential(
            spectral_norm(nn.Linear(64, 32)),
            nn.GELU(),
            spectral_norm(nn.Linear(32, 1))
        )

    def forward(self, current_repr, regime_z, unrealized_pnl, time_in_position, expected_return):
        position_context = torch.cat([unrealized_pnl, time_in_position, expected_return], dim=1)
        x = torch.cat([current_repr, regime_z, position_context], dim=1)
        features = self.profit_analyzer(x)

        feat32 = torch.nan_to_num(features.float(), 0.0, 0.0, 0.0).clamp(-1e4, 1e4)

        tp_logit = self.take_profit_head(feat32)
        tp_prob = torch.sigmoid(tp_logit)

        return {"take_profit_logits": tp_logit, "take_profit_prob": tp_prob}

class StopLossModule(nn.Module):
    def __init__(self, d_model, latent_dim):
        super().__init__()

        self.loss_analyzer = nn.Sequential(
            spectral_norm(nn.Linear(d_model + latent_dim + 3, 256)),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
            spectral_norm(nn.Linear(256, 128)),
            nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            spectral_norm(nn.Linear(128, 64)),
            nn.GELU(),
        )

        self.stop_loss_head = nn.Sequential(
            spectral_norm(nn.Linear(64, 32)),
            nn.GELU(),
            spectral_norm(nn.Linear(32, 1))
        )

    def forward(self, current_repr, regime_z, unrealized_pnl, time_in_position, expected_return):
        position_context = torch.cat([unrealized_pnl, time_in_position, expected_return], dim=1)
        x = torch.cat([current_repr, regime_z, position_context], dim=1)
        features = self.loss_analyzer(x)

        feat32 = torch.nan_to_num(features.float(), 0.0, 0.0, 0.0).clamp(-1e4, 1e4)

        sl_logit = self.stop_loss_head(feat32)
        sl_prob = torch.sigmoid(sl_logit)

        return {"stop_loss_logits": sl_logit, "stop_loss_prob": sl_prob}

class NeuralTradingModel(nn.Module):
\
\

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 512,
        num_heads: int = 16,
        num_layers: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.15,
        latent_dim: int = 16,
        seq_len: int = 100,
        positional_encoding_scale: float = 1.0,
        input_projection_gain: float = 1.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        self.input_projection = nn.Linear(feature_dim, d_model)
        nn.init.xavier_uniform_(self.input_projection.weight, gain=input_projection_gain)
        if self.input_projection.bias is not None:
            nn.init.zeros_(self.input_projection.bias)
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len, scale=positional_encoding_scale)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout,
                            layer_idx=i, num_layers=num_layers)
            for i in range(num_layers)
        ])

        self.regime_vae = MarketRegimeVAE(d_model, latent_dim)

        self.regime_change_detector = RegimeChangeDetector(d_model, latent_dim)
        self.profit_taking_module   = ProfitTakingModule(d_model, latent_dim)
        self.stop_loss_module       = StopLossModule(d_model, latent_dim)
        self.let_winner_run_module  = LetWinnerRunModule(d_model, latent_dim)

        def head(output_activation: Optional[nn.Module] = None, hidden_dim: int = 256):
            layers: List[nn.Module] = [
                nn.Linear(d_model + latent_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, 1),
            ]
            if output_activation is not None:
                layers.append(output_activation)
            return nn.Sequential(*layers)

        self.entry_head          = head(hidden_dim=256)
        self.return_head         = head(nn.Tanh(), hidden_dim=256)
        self.volatility_head     = head(nn.Softplus(), hidden_dim=128)
        self.position_size_head  = head(nn.Sigmoid(), hidden_dim=128)

        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if hasattr(self, "profit_taking_module"):
            nn.init.constant_(self.profit_taking_module.take_profit_head[-1].bias, -1.5)
        if hasattr(self, "stop_loss_module"):
            nn.init.constant_(self.stop_loss_module.stop_loss_head[-1].bias, -1.2)
        if hasattr(self, "let_winner_run_module"):
            nn.init.constant_(self.let_winner_run_module.let_run_head[-1].bias, -0.5)

    def forward(
        self,
        x: torch.Tensor,
        position_context: Optional[Dict[str, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        device = x.device
        B, T, _ = x.shape

        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = x.clamp(-100, 100)

        if self.training and hasattr(self, '_log_input_stats'):
            print(f"Input projection output: mean={x.mean():.4f}, std={x.std():.4f}, "
                f"min={x.min():.4f}, max={x.max():.4f}")

        x = self.input_projection(x)
        x = self.pos_encoding(x)

        attn_weights_list = []
        for block in self.transformer_blocks:
            x, attn_w = block(x, mask=mask)
            attn_weights_list.append(attn_w)

        current_repr   = x[:, -1, :]
        lookback       = min(20, T)
        historical_repr = x[:, -lookback:, :].mean(dim=1)

        recon, mu, logvar, current_z = self.regime_vae(current_repr)
        _, _, _, historical_z        = self.regime_vae(historical_repr)

        regime_change_info = self.regime_change_detector(current_repr, current_z, historical_z)

        cr = torch.cat([current_repr, current_z], dim=1)
        entry_logits        = self.entry_head(cr)
        entry_prob          = torch.sigmoid(entry_logits)
        expected_return     = self.return_head(cr)
        volatility_forecast = self.volatility_head(cr)
        position_size       = self.position_size_head(cr)

        take_profit_logits = None
        stop_loss_logits = None
        if position_context is not None:

            unrealized_pnl = position_context.get('unrealized_pnl',  torch.zeros(B, 1, device=device))
            time_in_pos    = position_context.get('time_in_position', torch.zeros(B, 1, device=device))
            exp_ret_ctx    = expected_return

            tp   = self.profit_taking_module(current_repr, current_z, unrealized_pnl, time_in_pos, exp_ret_ctx)
            sl   = self.stop_loss_module(current_repr, current_z, unrealized_pnl, time_in_pos, exp_ret_ctx)
            hold = self.let_winner_run_module(current_repr, current_z, unrealized_pnl, time_in_pos, exp_ret_ctx)

            exit_signals = {'profit_taking': tp, 'stop_loss': sl, 'let_winner_run': hold}
            take_profit_logits = tp.get('take_profit_logits')
            stop_loss_logits = sl.get('stop_loss_logits')

            pos_mask = (unrealized_pnl > 0).float()
            neg_mask = 1.0 - pos_mask

            tp_now   = tp['take_profit_prob']
            sl_now   = sl['stop_loss_prob']
            hold_now = hold['hold_score']

            unified = pos_mask * (tp_now * (1.0 - hold_now)) + neg_mask * sl_now
        else:
            exit_signals = {'profit_taking': None, 'stop_loss': None, 'let_winner_run': None}
            unified = torch.zeros(B, 1, device=device)

        return {
            'entry_logits': entry_logits,
            'entry_prob': entry_prob,
            'expected_return': expected_return,
            'volatility_forecast': volatility_forecast,
            'position_size': position_size,
            'regime_mu': mu, 'regime_logvar': logvar, 'regime_z': current_z,
            'vae_recon': recon,
            'attention_weights': attn_weights_list,
            'sequence_repr': current_repr,
            'regime_change': regime_change_info,
            'exit_signals': exit_signals,
            'unified_exit_prob': torch.clamp(unified, 0.0, 1.0),
            'take_profit_logits': take_profit_logits,
            'stop_loss_logits': stop_loss_logits,
        }

    @staticmethod
    def _compute_unified_exit(
        exit_signals: Dict[str, Any],
        regime_change_info: Dict[str, torch.Tensor],
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
\
\
\

        if isinstance(exit_signals, dict):
            if "should_exit_profit" in exit_signals and exit_signals["should_exit_profit"] is not None:
                return exit_signals["should_exit_profit"]
            if "should_exit_loss" in exit_signals and exit_signals["should_exit_loss"] is not None:
                return exit_signals["should_exit_loss"]

        return torch.zeros(batch, 1, device=device)

def create_model(feature_dim, config=None):

    if config is None:
        config = {}

    model_params = {
        'd_model': config.get('d_model', 512),
        'num_heads': config.get('num_heads', 16),
        'num_layers': config.get('num_layers', 8),
        'd_ff': config.get('d_ff', 2048),
        'dropout': config.get('dropout', 0.15),
        'latent_dim': config.get('latent_dim', 16),
        'seq_len': config.get('seq_len', 100),
        'positional_encoding_scale': config.get('positional_encoding_scale', 1.0),
        'input_projection_gain': config.get('input_projection_gain', 1.0),
    }

    model = NeuralTradingModel(
        feature_dim=feature_dim,
        **model_params
    )

    return model
