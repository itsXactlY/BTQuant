import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadSelfAttention(nn.Module):
    """Custom multi-head attention for indicator relationships."""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(context)

        # Residual connection + layer norm
        return self.layer_norm(x + self.dropout(output)), attn_weights

class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return self.layer_norm(residual + x)

class TransformerBlock(nn.Module):
    """Single transformer encoder block."""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        x, attn_weights = self.attention(x, mask)
        x = self.feed_forward(x)
        return x, attn_weights

class MarketRegimeVAE(nn.Module):
    """
    ✅ FIXED: Variational Autoencoder with stability safeguards.
    """
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # ✅ CRITICAL: Check input for NaN/Inf FIRST
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"⚠️ NaN/Inf in VAE input! Setting to zero.")
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        mu, logvar = self.encode(x)

        # ✅ CRITICAL FIX: Clamp logvar to prevent explosion
        logvar = torch.clamp(logvar, min=-10, max=10)

        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        # ✅ CRITICAL FIX: Check for NaN in output
        if torch.isnan(recon).any():
            recon = torch.nan_to_num(recon, nan=0.0, posinf=0.0, neginf=0.0)

        return recon, mu, logvar, z

class NeuralTradingModel(nn.Module):
    """
    ✅ FIXED: Main neural trading architecture with improved initialization.
    """
    def __init__(
        self,
        feature_dim,
        d_model=256,
        num_heads=8,
        num_layers=6,
        d_ff=1024,
        dropout=0.1,
        latent_dim=8,
        seq_len=100
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        # Input projection
        self.input_projection = nn.Linear(feature_dim, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len)

        # Transformer encoder stack
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Market regime VAE
        self.regime_vae = MarketRegimeVAE(d_model, latent_dim)

        # Multi-task heads
        def head(output_activation=None):
            layers = [
                nn.Linear(d_model + latent_dim, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            ]
            if output_activation is not None:
                layers.append(output_activation)
            return nn.Sequential(*layers)

        # Heads
        self.entry_head = head()
        self.exit_head = head()
        self.return_head = head(nn.Tanh())
        self.volatility_head = head(nn.Softplus())
        self.position_size_head = head(nn.Sigmoid())

        self._init_weights()

    def _init_weights(self):
        """✅ FIXED: Improved initialization for stability."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Standard Xavier for most layers
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # ✅ CRITICAL: Check and clip input features
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"⚠️ NaN/Inf in model input! Replacing with zeros.")
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Clamp extreme values to prevent overflow
        x = torch.clamp(x, min=-100, max=100)

        # Project input to model dimension
        x = self.input_projection(x)
        x = self.pos_encoding(x)

        # Transformer stack
        attn_weights_list = []
        for block in self.transformer_blocks:
            x, attn_weights = block(x, mask)
            attn_weights_list.append(attn_weights)

        # Last token representation
        sequence_repr = x[:, -1, :]  # [batch_size, d_model]

        # VAE for regime detection
        recon, mu, logvar, regime_z = self.regime_vae(sequence_repr)

        # Combine sequence representation with regime embedding
        combined_repr = torch.cat([sequence_repr, regime_z], dim=1)

        # Task heads
        entry_logits = self.entry_head(combined_repr)
        exit_logits = self.exit_head(combined_repr)

        # Derived activations
        entry_prob = torch.sigmoid(entry_logits)
        exit_prob = torch.sigmoid(exit_logits)
        expected_return = self.return_head(combined_repr)
        volatility_forecast = self.volatility_head(combined_repr)
        position_size = self.position_size_head(combined_repr)

        return {
            'entry_logits': entry_logits,
            'exit_logits': exit_logits,
            'entry_prob': entry_prob,
            'exit_prob': exit_prob,
            'expected_return': expected_return,
            'volatility_forecast': volatility_forecast,
            'position_size': position_size,
            'regime_mu': mu,
            'regime_logvar': logvar,
            'regime_z': regime_z,
            'vae_recon': recon,
            'attention_weights': attn_weights_list,
            'sequence_repr': sequence_repr
        }

def create_model(feature_dim, config=None):
    """Factory function to create model."""
    if config is None:
        config = {}

    model_params = {
        'd_model': config.get('d_model', 256),
        'num_heads': config.get('num_heads', 8),
        'num_layers': config.get('num_layers', 6),
        'd_ff': config.get('d_ff', 1024),
        'dropout': config.get('dropout', 0.1),
        'latent_dim': config.get('latent_dim', 8),
        'seq_len': config.get('seq_len', 100)
    }

    model = NeuralTradingModel(
        feature_dim=feature_dim,
        **model_params
    )

    return model
