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
    Variational Autoencoder for unsupervised market regime discovery.
    Learns a latent space where similar market conditions cluster together.
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
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


class NeuralTradingModel(nn.Module):
    """
    Main neural trading architecture.
    
    Input: Sequence of feature vectors [batch, seq_len, feature_dim]
    Output: Multi-task predictions
        - entry_prob: [batch, 1] - probability of successful entry
        - exit_prob: [batch, 1] - probability should exit now
        - expected_return: [batch, 1] - expected return if entered
        - volatility_forecast: [batch, 1] - predicted volatility
        - position_size: [batch, 1] - optimal position size (0-1)
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
        self.regime_vae = MarketRegimeVAE(feature_dim, latent_dim)
        
        # Multi-task heads
        self.entry_head = nn.Sequential(
            nn.Linear(d_model + latent_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.exit_head = nn.Sequential(
            nn.Linear(d_model + latent_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.return_head = nn.Sequential(
            nn.Linear(d_model + latent_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh()  # Returns can be positive or negative
        )
        
        self.volatility_head = nn.Sequential(
            nn.Linear(d_model + latent_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Volatility is always positive
        )
        
        self.position_size_head = nn.Sequential(
            nn.Linear(d_model + latent_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Position size 0-1
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, feature_dim]
            mask: [batch, seq_len] optional attention mask
        
        Returns:
            Dict with predictions and attention weights
        """
        batch_size, seq_len, _ = x.size()
        
        # Project input to model dimension
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through transformer blocks
        attn_weights_list = []
        for block in self.transformer_blocks:
            x, attn_weights = block(x, mask)
            attn_weights_list.append(attn_weights)
        
        # Extract representation from last position (most recent data)
        sequence_repr = x[:, -1, :]  # [batch, d_model]
        
        # Get market regime embedding from VAE
        current_features = x[:, -1, :]  # Use transformed features
        # Project back to feature space for VAE
        features_for_vae = self.input_projection.weight.t() @ sequence_repr.t()
        features_for_vae = features_for_vae.t()[:, :self.feature_dim]
        
        recon, mu, logvar, regime_z = self.regime_vae(features_for_vae)
        
        # Concatenate sequence representation with regime embedding
        combined_repr = torch.cat([sequence_repr, regime_z], dim=1)
        
        # Multi-task predictions
        entry_prob = self.entry_head(combined_repr)
        exit_prob = self.exit_head(combined_repr)
        expected_return = self.return_head(combined_repr)
        volatility_forecast = self.volatility_head(combined_repr)
        position_size = self.position_size_head(combined_repr)
        
        return {
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
    """Factory function to create model with default or custom config."""
    if config is None:
        config = {
            'd_model': 256,
            'num_heads': 8,
            'num_layers': 6,
            'd_ff': 1024,
            'dropout': 0.1,
            'latent_dim': 8,
            'seq_len': 100
        }
    
    model = NeuralTradingModel(
        feature_dim=feature_dim,
        **config
    )
    
    return model