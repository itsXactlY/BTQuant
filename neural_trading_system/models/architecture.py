# architecture_v2.py
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

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(context)

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
    """Enhanced transformer encoder block with layer scaling."""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.layer_scale_1 = nn.Parameter(torch.ones(d_model) * 0.1)
        self.layer_scale_2 = nn.Parameter(torch.ones(d_model) * 0.1)

    def forward(self, x, mask=None):
        attn_out, attn_weights = self.attention(x, mask)
        x = x + self.layer_scale_1 * (attn_out - x)
        
        ff_out = self.feed_forward(x)
        x = x + self.layer_scale_2 * (ff_out - x)
        
        return x, attn_weights

class MarketRegimeVAE(nn.Module):
    """Variational Autoencoder for regime detection with stability."""
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()

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
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        mu, logvar = self.encode(x)
        logvar = torch.clamp(logvar, min=-10, max=10)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)

        if torch.isnan(recon).any():
            recon = torch.nan_to_num(recon, nan=0.0, posinf=0.0, neginf=0.0)

        return recon, mu, logvar, z


class RegimeChangeDetector(nn.Module):
    """
    🚨 NEW: Detects when market regime is shifting.
    Looks at volatility changes, volume anomalies, and pattern breaks.
    """
    def __init__(self, d_model, latent_dim):
        super().__init__()
        
        # Temporal comparison network
        self.regime_comparator = nn.Sequential(
            nn.Linear(d_model + latent_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        
        # Multi-scale regime change scores
        self.volatility_change_head = nn.Linear(64, 1)  # Sudden vol shift
        self.volume_anomaly_head = nn.Linear(64, 1)     # Unusual volume
        self.trend_break_head = nn.Linear(64, 1)        # Trend reversal
        self.liquidity_shift_head = nn.Linear(64, 1)    # Market depth change
        
        # Overall regime stability score
        self.stability_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 0 = unstable, 1 = stable
        )
        
    def forward(self, current_repr, current_regime, historical_regime_mean):
        """
        Args:
            current_repr: [B, d_model] - current market state
            current_regime: [B, latent_dim] - current regime embedding
            historical_regime_mean: [B, latent_dim] - avg regime from window
        """
        # Concatenate for comparison
        x = torch.cat([current_repr, current_regime, historical_regime_mean], dim=1)
        
        features = self.regime_comparator(x)
        
        # Individual change signals
        vol_change = torch.sigmoid(self.volatility_change_head(features))
        volume_anomaly = torch.sigmoid(self.volume_anomaly_head(features))
        trend_break = torch.sigmoid(self.trend_break_head(features))
        liquidity_shift = torch.sigmoid(self.liquidity_shift_head(features))
        
        # Overall stability (inverse of change)
        stability = self.stability_head(features)
        
        # Regime change score: weighted combination
        regime_change_score = (
            0.3 * vol_change +
            0.25 * volume_anomaly +
            0.25 * trend_break +
            0.2 * liquidity_shift
        )
        
        return {
            'regime_change_score': regime_change_score,
            'stability': stability,
            'vol_change': vol_change,
            'volume_anomaly': volume_anomaly,
            'trend_break': trend_break,
            'liquidity_shift': liquidity_shift
        }


class ProfitTakingModule(nn.Module):
    """
    💰 NEW: Learns optimal profit-taking timing.
    Considers: unrealized profit, momentum fade, resistance levels, regime stability.
    """
    def __init__(self, d_model, latent_dim):
        super().__init__()
        
        # Takes current state + position context
        self.profit_analyzer = nn.Sequential(
            nn.Linear(d_model + latent_dim + 3, 256),  # +3 for position context
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        
        # Multiple profit-taking signals
        self.momentum_fade_head = nn.Linear(64, 1)      # Momentum weakening
        self.resistance_near_head = nn.Linear(64, 1)    # Near resistance
        self.profit_ratio_optimal_head = nn.Linear(64, 1)  # Good risk/reward hit
        self.extension_risk_head = nn.Linear(64, 1)     # Overextended
        
        # Final take-profit probability
        self.take_profit_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Probability to take profit NOW
        )
        
    def forward(self, current_repr, regime_z, unrealized_pnl, time_in_position, expected_return):
        """
        Args:
            current_repr: [B, d_model]
            regime_z: [B, latent_dim]
            unrealized_pnl: [B, 1] - current P&L ratio (e.g., 0.05 = 5%)
            time_in_position: [B, 1] - normalized time holding
            expected_return: [B, 1] - model's return forecast
        """
        # Position context
        position_context = torch.cat([unrealized_pnl, time_in_position, expected_return], dim=1)
        
        x = torch.cat([current_repr, regime_z, position_context], dim=1)
        features = self.profit_analyzer(x)
        
        # Individual signals
        momentum_fade = torch.sigmoid(self.momentum_fade_head(features))
        resistance_near = torch.sigmoid(self.resistance_near_head(features))
        profit_optimal = torch.sigmoid(self.profit_ratio_optimal_head(features))
        extension_risk = torch.sigmoid(self.extension_risk_head(features))
        
        # Overall take-profit decision
        take_profit_prob = self.take_profit_head(features)
        
        return {
            'take_profit_prob': take_profit_prob,
            'momentum_fade': momentum_fade,
            'resistance_near': resistance_near,
            'profit_optimal': profit_optimal,
            'extension_risk': extension_risk
        }


class StopLossModule(nn.Module):
    """
    🛑 NEW: Learns when to cut losses intelligently.
    Not just "price down X%" - considers regime breaks, failed patterns, acceleration.
    """
    def __init__(self, d_model, latent_dim):
        super().__init__()
        
        self.loss_analyzer = nn.Sequential(
            nn.Linear(d_model + latent_dim + 3, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        
        # Loss-cutting signals
        self.pattern_failure_head = nn.Linear(64, 1)    # Setup invalidated
        self.acceleration_down_head = nn.Linear(64, 1)  # Loss accelerating
        self.support_break_head = nn.Linear(64, 1)      # Key level broken
        self.regime_hostile_head = nn.Linear(64, 1)     # Regime turned bad
        
        # Final stop-loss probability
        self.stop_loss_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Probability to cut NOW
        )
        
    def forward(self, current_repr, regime_z, unrealized_pnl, regime_change_score, time_in_position):
        """
        Args:
            current_repr: [B, d_model]
            regime_z: [B, latent_dim]
            unrealized_pnl: [B, 1] - current P&L ratio (negative)
            regime_change_score: [B, 1] - from RegimeChangeDetector
            time_in_position: [B, 1]
        """
        position_context = torch.cat([unrealized_pnl, regime_change_score, time_in_position], dim=1)
        
        x = torch.cat([current_repr, regime_z, position_context], dim=1)
        features = self.loss_analyzer(x)
        
        # Individual signals
        pattern_failure = torch.sigmoid(self.pattern_failure_head(features))
        acceleration_down = torch.sigmoid(self.acceleration_down_head(features))
        support_break = torch.sigmoid(self.support_break_head(features))
        regime_hostile = torch.sigmoid(self.regime_hostile_head(features))
        
        # Overall stop-loss decision
        stop_loss_prob = self.stop_loss_head(features)
        
        return {
            'stop_loss_prob': stop_loss_prob,
            'pattern_failure': pattern_failure,
            'acceleration_down': acceleration_down,
            'support_break': support_break,
            'regime_hostile': regime_hostile
        }


class LetWinnerRunModule(nn.Module):
    """
    🚀 NEW: Decides when to HOLD and let a winner run.
    Looks for: trend continuation, momentum acceleration, breakout confirmation.
    """
    def __init__(self, d_model, latent_dim):
        super().__init__()
        
        self.continuation_analyzer = nn.Sequential(
            nn.Linear(d_model + latent_dim + 3, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        
        # Continuation signals
        self.trend_strength_head = nn.Linear(64, 1)     # Strong trend intact
        self.momentum_accel_head = nn.Linear(64, 1)     # Momentum building
        self.breakout_confirm_head = nn.Linear(64, 1)   # Breakout extending
        self.regime_favorable_head = nn.Linear(64, 1)   # Regime supports move
        
        # Hold recommendation
        self.hold_score_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # High = strong HOLD signal
        )
        
    def forward(self, current_repr, regime_z, unrealized_pnl, momentum_indicator, regime_stability):
        """
        Args:
            current_repr: [B, d_model]
            regime_z: [B, latent_dim]
            unrealized_pnl: [B, 1] - current profit
            momentum_indicator: [B, 1] - from model's momentum features
            regime_stability: [B, 1] - from RegimeChangeDetector
        """
        position_context = torch.cat([unrealized_pnl, momentum_indicator, regime_stability], dim=1)
        
        x = torch.cat([current_repr, regime_z, position_context], dim=1)
        features = self.continuation_analyzer(x)
        
        # Individual signals
        trend_strength = torch.sigmoid(self.trend_strength_head(features))
        momentum_accel = torch.sigmoid(self.momentum_accel_head(features))
        breakout_confirm = torch.sigmoid(self.breakout_confirm_head(features))
        regime_favorable = torch.sigmoid(self.regime_favorable_head(features))
        
        # Overall hold recommendation
        hold_score = self.hold_score_head(features)
        
        return {
            'hold_score': hold_score,
            'trend_strength': trend_strength,
            'momentum_accel': momentum_accel,
            'breakout_confirm': breakout_confirm,
            'regime_favorable': regime_favorable
        }


class NeuralTradingModel(nn.Module):
    """
    🎯 ENHANCED: Self-aware trading model with intelligent exit management.
    No fixed TP/SL - learns optimal timing from data.
    """
    def __init__(
        self,
        feature_dim,
        d_model=512,
        num_heads=16,
        num_layers=8,
        d_ff=2048,
        dropout=0.15,
        latent_dim=16,
        seq_len=100
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        # Input projection
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len)

        # Transformer encoder stack
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Market regime VAE
        self.regime_vae = MarketRegimeVAE(d_model, latent_dim)

        # 🆕 NEW MODULES FOR INTELLIGENT EXIT MANAGEMENT
        self.regime_change_detector = RegimeChangeDetector(d_model, latent_dim)
        self.profit_taking_module = ProfitTakingModule(d_model, latent_dim)
        self.stop_loss_module = StopLossModule(d_model, latent_dim)
        self.let_winner_run_module = LetWinnerRunModule(d_model, latent_dim)

        # Original heads (entry, return, volatility, position sizing)
        def head(output_activation=None, hidden_dim=256):
            layers = [
                nn.Linear(d_model + latent_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, 1)
            ]
            if output_activation is not None:
                layers.append(output_activation)
            return nn.Sequential(*layers)

        self.entry_head = head(hidden_dim=256)
        self.return_head = head(nn.Tanh(), hidden_dim=256)
        self.volatility_head = head(nn.Softplus(), hidden_dim=128)
        self.position_size_head = head(nn.Sigmoid(), hidden_dim=128)

        self._init_weights()

    def _init_weights(self):
        """Improved initialization for stability."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, position_context=None, mask=None):
        """
        Args:
            x: [B, seq_len, feature_dim]
            position_context: Dict with:
                - 'unrealized_pnl': [B, 1] - current P&L if in position
                - 'time_in_position': [B, 1] - bars since entry
                - 'entry_price': [B, 1] - entry price
                (None if not in position)
        """
        batch_size, seq_len, _ = x.size()

        # Safety checks
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, min=-100, max=100)

        # Project and encode
        x = self.input_projection(x)
        x = self.pos_encoding(x)

        # Transformer stack
        attn_weights_list = []
        for block in self.transformer_blocks:
            x, attn_weights = block(x, mask)
            attn_weights_list.append(attn_weights)

        # Current state representation
        current_repr = x[:, -1, :]  # [B, d_model]
        
        # Historical regime (mean of last 20 bars)
        lookback = min(20, seq_len)
        historical_repr = x[:, -lookback:, :].mean(dim=1)  # [B, d_model]

        # VAE for regime detection
        recon, mu, logvar, current_regime_z = self.regime_vae(current_repr)
        _, _, _, historical_regime_z = self.regime_vae(historical_repr)

        # 🚨 Regime change detection
        regime_change_info = self.regime_change_detector(
            current_repr, 
            current_regime_z, 
            historical_regime_z
        )

        # Original predictions
        combined_repr = torch.cat([current_repr, current_regime_z], dim=1)
        entry_logits = self.entry_head(combined_repr)
        entry_prob = torch.sigmoid(entry_logits)
        expected_return = self.return_head(combined_repr)
        volatility_forecast = self.volatility_head(combined_repr)
        position_size = self.position_size_head(combined_repr)

        # 🆕 EXIT MANAGEMENT PREDICTIONS
        exit_signals = {}
        
        if position_context is not None:
            # In position - compute exit signals
            unrealized_pnl = position_context.get('unrealized_pnl', torch.zeros(batch_size, 1, device=x.device))
            time_in_pos = position_context.get('time_in_position', torch.zeros(batch_size, 1, device=x.device))
            
            # Momentum proxy from expected return
            momentum_indicator = expected_return
            
            if unrealized_pnl.mean() > 0:
                # In profit - check take-profit signals
                profit_signals = self.profit_taking_module(
                    current_repr,
                    current_regime_z,
                    unrealized_pnl,
                    time_in_pos,
                    expected_return
                )
                
                # Check if should let winner run
                winner_signals = self.let_winner_run_module(
                    current_repr,
                    current_regime_z,
                    unrealized_pnl,
                    momentum_indicator,
                    regime_change_info['stability']
                )
                
                exit_signals.update({
                    'profit_taking': profit_signals,
                    'let_winner_run': winner_signals,
                    # Final exit decision balances both
                    'should_exit_profit': profit_signals['take_profit_prob'] * (1 - winner_signals['hold_score'])
                })
            else:
                # In loss - check stop-loss signals
                loss_signals = self.stop_loss_module(
                    current_repr,
                    current_regime_z,
                    unrealized_pnl,
                    regime_change_info['regime_change_score'],
                    time_in_pos
                )
                
                exit_signals.update({
                    'stop_loss': loss_signals,
                    'should_exit_loss': loss_signals['stop_loss_prob']
                })
        else:
            # Not in position - no exit signals needed
            exit_signals = {
                'profit_taking': None,
                'stop_loss': None,
                'let_winner_run': None
            }

        return {
            # Original outputs
            'entry_logits': entry_logits,
            'entry_prob': entry_prob,
            'expected_return': expected_return,
            'volatility_forecast': volatility_forecast,
            'position_size': position_size,
            'regime_mu': mu,
            'regime_logvar': logvar,
            'regime_z': current_regime_z,
            'vae_recon': recon,
            'attention_weights': attn_weights_list,
            'sequence_repr': current_repr,
            
            # 🆕 NEW: Intelligent exit management
            'regime_change': regime_change_info,
            'exit_signals': exit_signals,
            
            # 🆕 Unified exit probability (combines all exit logic)
            'unified_exit_prob': self._compute_unified_exit(exit_signals, regime_change_info)
        }
    
    def _compute_unified_exit(self, exit_signals, regime_change_info):
        """
        Combines all exit signals into one unified exit probability.
        Higher value = stronger exit signal.
        """
        # Check if we have exit signals (in position)
        if 'should_exit_profit' in exit_signals:
            # In profit: balance take-profit vs let-run
            return exit_signals['should_exit_profit']
        elif 'should_exit_loss' in exit_signals:
            # In loss: stop-loss signal
            return exit_signals['should_exit_loss']
        else:
            # Not in position - return zeros with correct shape
            # Get batch size from regime_change_info
            batch_size = regime_change_info['stability'].size(0)
            device = regime_change_info['stability'].device
            return torch.zeros(batch_size, 1, device=device)


def create_model(feature_dim, config=None):
    """Factory function to create model."""
    if config is None:
        config = {}

    model_params = {
        'd_model': config.get('d_model', 512),
        'num_heads': config.get('num_heads', 16),
        'num_layers': config.get('num_layers', 8),
        'd_ff': config.get('d_ff', 2048),
        'dropout': config.get('dropout', 0.15),
        'latent_dim': config.get('latent_dim', 16),
        'seq_len': config.get('seq_len', 100)
    }

    model = NeuralTradingModel(
        feature_dim=feature_dim,
        **model_params
    )

    return model