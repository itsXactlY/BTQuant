#!/usr/bin/env python3
"""
FULL MODEL ANALYSIS - Fixed for 2D flattened input models
"""

import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.progress import track
from rich.table import Table
from rich.panel import Panel

console = Console()

# PATHS
MODEL_PATH = "models/best_exit_aware_model_12_production_ready.pt"
FEATURE_CACHE = "neural_data/features/features_36db232ee667373f_seq100_hor5.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# FIXED MODEL WRAPPER FOR 2D INPUT
# ============================================

class Fixed2DModelWrapper(nn.Module):
    """
    Wrapper that makes a 3D-expecting model work with 2D flattened input.
    Your model was trained with flattened sequences passed directly to input_projection.
    """
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model
        self.d_model = original_model.d_model
        self.seq_len = original_model.seq_len
        
    def forward(self, x, position_context=None):
        """
        x: [batch, 25100] flattened input
        """
        device = x.device
        B = x.shape[0]
        
        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            # This is the case we have - flattened input
            # The model was trained this way!
            
            # Safety checks
            if torch.isnan(x).any() or torch.isinf(x).any():
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            x = x.clamp(-100, 100)
            
            # Project flattened input to d_model
            x_proj = self.model.input_projection(x)  # [B, d_model]
            
            # For transformer, we need sequence dimension
            # Create a pseudo-sequence of length 1
            x_seq = x_proj.unsqueeze(1)  # [B, 1, d_model]
            
            # Skip positional encoding (not meaningful for single timestep)
            
            # Pass through transformer blocks
            attn_weights_list = []
            for block in self.model.transformer_blocks:
                x_seq, attn_w = block(x_seq, mask=None)
                attn_weights_list.append(attn_w)
            
            # Get representation (squeeze out sequence dim)
            current_repr = x_seq.squeeze(1)  # [B, d_model]
            historical_repr = current_repr  # Same since we have no real sequence
            
            # VAE processing
            recon, mu, logvar, current_z = self.model.regime_vae(current_repr)
            _, _, _, historical_z = self.model.regime_vae(historical_repr)
            
            # Regime change
            regime_change_info = self.model.regime_change_detector(
                current_repr, current_z, historical_z
            )
            
            # Predictions
            cr = torch.cat([current_repr, current_z], dim=1)
            entry_logits = self.model.entry_head(cr)
            entry_prob = torch.sigmoid(entry_logits)
            expected_return = self.model.return_head(cr)
            volatility_forecast = self.model.volatility_head(cr)
            position_size = self.model.position_size_head(cr)
            
            # Exit management
            if position_context is not None:
                unrealized_pnl = position_context.get(
                    'unrealized_pnl', torch.zeros(B, 1, device=device)
                )
                time_in_pos = position_context.get(
                    'time_in_position', torch.zeros(B, 1, device=device)
                )
                
                tp = self.model.profit_taking_module(
                    current_repr, current_z, unrealized_pnl, time_in_pos, expected_return
                )
                sl = self.model.stop_loss_module(
                    current_repr, current_z, unrealized_pnl, time_in_pos, expected_return
                )
                hold = self.model.let_winner_run_module(
                    current_repr, current_z, unrealized_pnl, time_in_pos, expected_return
                )
                
                exit_signals = {
                    'profit_taking': tp,
                    'stop_loss': sl,
                    'let_winner_run': hold
                }
                
                # Unified exit
                pos_mask = (unrealized_pnl > 0).float()
                neg_mask = 1.0 - pos_mask
                unified = (
                    pos_mask * tp['take_profit_prob'] * (1.0 - hold['hold_score']) +
                    neg_mask * sl['stop_loss_prob']
                )
            else:
                exit_signals = {}
                unified = torch.zeros(B, 1, device=device)
            
            return {
                'entry_logits': entry_logits,
                'entry_prob': entry_prob,
                'expected_return': expected_return,
                'volatility_forecast': volatility_forecast,
                'position_size': position_size,
                'regime_mu': mu,
                'regime_logvar': logvar,
                'regime_z': current_z,
                'vae_recon': recon,
                'attention_weights': attn_weights_list,
                'sequence_repr': current_repr,
                'regime_change': regime_change_info,
                'exit_signals': exit_signals,
                'unified_exit_prob': torch.clamp(unified, 0.0, 1.0),
            }
        else:
            # 3D input - reshape to 2D and call recursively
            B, T, D = x.shape
            x_flat = x.reshape(B, T * D)
            return self.forward(x_flat, position_context)


# ============================================
# LOAD MODEL
# ============================================

console.print(Panel.fit(
    "[bold cyan]NEURAL TRADING MODEL ANALYSIS[/bold cyan]\n"
    "[yellow]Complete diagnostic and performance evaluation[/yellow]",
    border_style="cyan"
))

# Load checkpoint
console.print(f"\n📥 Loading model from: [cyan]{MODEL_PATH}[/cyan]")
checkpoint = torch.load(MODEL_PATH, map_location=device)

# Check input dimension
input_weight_shape = checkpoint['model_state_dict']['input_projection.weight'].shape
actual_input_dim = input_weight_shape[1]  # Should be 25100
console.print(f"✅ Model input dimension: [green]{actual_input_dim}[/green]")

# Import and create model
from models.architecture import NeuralTradingModel

base_model = NeuralTradingModel(
    feature_dim=actual_input_dim,  # 25100 - the flattened dimension
    d_model=256,
    num_heads=8,
    num_layers=6,
    d_ff=1024,
    dropout=0.0,
    latent_dim=16,
    seq_len=100
)

# Load weights
base_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
base_model.eval()

# Wrap with our fixed wrapper
model = Fixed2DModelWrapper(base_model).to(device)
console.print("✅ Model loaded and wrapped for 2D input")

# ============================================
# LOAD DATA
# ============================================

console.print(f"\n📥 Loading features from: [cyan]{FEATURE_CACHE}[/cyan]")
with open(FEATURE_CACHE, 'rb') as f:
    cache = pickle.load(f)

features = cache['features']  # [N, 25100]
prices = cache['prices']
console.print(f"✅ Loaded {len(features):,} samples")
console.print(f"   Shape: {features.shape}")

# Calculate returns
returns = np.diff(prices) / prices[:-1]
returns = np.append(returns, 0)
labels = (returns > 0).astype(int)

# ============================================
# INFERENCE
# ============================================

console.print("\n" + "="*80)
console.print("[bold cyan]RUNNING INFERENCE[/bold cyan]")
console.print("="*80)

BATCH_SIZE = 512
all_outputs = {
    'entry_prob': [],
    'exit_prob': [],
    'expected_return': [],
    'volatility': [],
    'position_size': [],
    'tp_prob': [],
    'sl_prob': [],
    'hold_score': []
}

with torch.no_grad():
    for i in track(range(0, len(features), BATCH_SIZE), description="Processing batches"):
        batch = features[i:i+BATCH_SIZE]
        X = torch.from_numpy(batch).float().to(device)
        
        # Simulate position context
        position_context = {
            'unrealized_pnl': torch.randn(len(batch), 1, device=device) * 0.02,
            'time_in_position': torch.ones(len(batch), 1, device=device) * 5
        }
        
        outputs = model(X, position_context=position_context)
        
        all_outputs['entry_prob'].append(outputs['entry_prob'].cpu().numpy())
        all_outputs['exit_prob'].append(outputs['unified_exit_prob'].cpu().numpy())
        all_outputs['expected_return'].append(outputs['expected_return'].cpu().numpy())
        all_outputs['volatility'].append(outputs['volatility_forecast'].cpu().numpy())
        all_outputs['position_size'].append(outputs['position_size'].cpu().numpy())
        
        # Exit signals
        if 'exit_signals' in outputs and outputs['exit_signals']:
            if 'profit_taking' in outputs['exit_signals']:
                all_outputs['tp_prob'].append(
                    outputs['exit_signals']['profit_taking']['take_profit_prob'].cpu().numpy()
                )
            if 'stop_loss' in outputs['exit_signals']:
                all_outputs['sl_prob'].append(
                    outputs['exit_signals']['stop_loss']['stop_loss_prob'].cpu().numpy()
                )
            if 'let_winner_run' in outputs['exit_signals']:
                all_outputs['hold_score'].append(
                    outputs['exit_signals']['let_winner_run']['hold_score'].cpu().numpy()
                )

# Concatenate all predictions
for key in all_outputs:
    if all_outputs[key]:
        all_outputs[key] = np.concatenate(all_outputs[key]).flatten()
    else:
        all_outputs[key] = np.zeros(len(features))

console.print(f"\n✅ Generated predictions for {len(all_outputs['entry_prob']):,} samples")

# ============================================
# ANALYSIS
# ============================================

console.print("\n" + "="*80)
console.print("[bold cyan]STATISTICAL ANALYSIS[/bold cyan]")
console.print("="*80)

# Create analysis table
table = Table(show_header=True, header_style="bold magenta")
table.add_column("Metric", style="cyan", width=30)
table.add_column("Value", justify="right", style="yellow")

# Entry statistics
entry_probs = all_outputs['entry_prob']
table.add_row("Entry Prob Mean", f"{entry_probs.mean():.6f}")
table.add_row("Entry Prob Std", f"{entry_probs.std():.6f}")
table.add_row("Entry Prob Min/Max", f"{entry_probs.min():.6f} / {entry_probs.max():.6f}")

# Correlations
def safe_corr(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return np.nan
    return np.corrcoef(a[mask], b[mask])[0, 1]

corr_entry = safe_corr(entry_probs, returns)
corr_return = safe_corr(all_outputs['expected_return'], returns)
corr_vol = safe_corr(all_outputs['volatility'], np.abs(returns))

table.add_row("", "")  # Separator
table.add_row("Entry↔Return Correlation", f"{corr_entry:+.6f}" if np.isfinite(corr_entry) else "N/A")
table.add_row("ExpReturn↔Return Correlation", f"{corr_return:+.6f}" if np.isfinite(corr_return) else "N/A")
table.add_row("VolForecast↔|Return| Correlation", f"{corr_vol:+.6f}" if np.isfinite(corr_vol) else "N/A")

# Win rates by confidence
table.add_row("", "")  # Separator
table.add_row("[bold]WIN RATES BY CONFIDENCE[/bold]", "")
for percentile in [90, 75, 50, 25, 10]:
    threshold = np.percentile(entry_probs, percentile)
    mask = entry_probs >= threshold
    if mask.sum() > 0:
        win_rate = labels[mask].mean() * 100
        count = mask.sum()
        table.add_row(f"Top {100-percentile}% (≥{threshold:.4f})", 
                     f"{win_rate:.1f}% ({count:,} trades)")

console.print(table)

# Performance metrics
top10 = entry_probs >= np.percentile(entry_probs, 90)
bot10 = entry_probs <= np.percentile(entry_probs, 10)

if top10.sum() > 0 and bot10.sum() > 0:
    top10_wr = labels[top10].mean()
    bot10_wr = labels[bot10].mean()
    top10_ret = returns[top10].mean()
    bot10_ret = returns[bot10].mean()
    
    edge_wr = top10_wr - bot10_wr
    edge_ret = top10_ret - bot10_ret
    
    console.print("\n" + "="*80)
    console.print("[bold cyan]PERFORMANCE VERDICT[/bold cyan]")
    console.print("="*80)
    
    verdict_table = Table(show_header=False)
    verdict_table.add_column("", style="cyan", width=25)
    verdict_table.add_column("Top 10%", justify="right", style="green")
    verdict_table.add_column("Bottom 10%", justify="right", style="red")
    verdict_table.add_column("Edge", justify="right", style="yellow")
    
    verdict_table.add_row("Sample Count", 
                          f"{top10.sum():,}", 
                          f"{bot10.sum():,}", 
                          "")
    verdict_table.add_row("Win Rate", 
                          f"{top10_wr*100:.1f}%", 
                          f"{bot10_wr*100:.1f}%", 
                          f"{edge_wr*100:+.1f}%")
    verdict_table.add_row("Avg Return", 
                          f"{top10_ret*100:+.4f}%", 
                          f"{bot10_ret*100:+.4f}%", 
                          f"{edge_ret*100:+.4f}%")
    
    console.print(verdict_table)
    
    # Final assessment
    if edge_wr > 0.05 and edge_ret > 0.001:
        console.print("\n[bold green]✅ STRONG EDGE - Model shows clear predictive power[/bold green]")
    elif edge_wr > 0.02 or edge_ret > 0.0005:
        console.print("\n[bold yellow]⚠️ MODERATE EDGE - Some predictive ability[/bold yellow]")
    else:
        console.print("\n[bold red]❌ WEAK EDGE - Limited predictive power[/bold red]")

# ============================================
# VISUALIZATION
# ============================================

console.print("\n" + "="*80)
console.print("[bold cyan]GENERATING VISUALIZATIONS[/bold cyan]")
console.print("="*80)

fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.25)

# Style settings
plt.style.use('seaborn-v0_8-darkgrid')

# 1. Entry probability distribution
ax = fig.add_subplot(gs[0, 0])
ax.hist(entry_probs, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(entry_probs.mean(), color='red', linestyle='--', linewidth=2, 
          label=f'Mean: {entry_probs.mean():.3f}')
ax.set_xlabel('Entry Probability')
ax.set_ylabel('Count')
ax.set_title('Entry Signal Distribution')
ax.legend()

# 2. Expected return distribution
ax = fig.add_subplot(gs[0, 1])
exp_ret_pct = all_outputs['expected_return'] * 100
ax.hist(exp_ret_pct, bins=50, color='green', alpha=0.7, edgecolor='black')
ax.axvline(0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Expected Return (%)')
ax.set_ylabel('Count')
ax.set_title(f'Expected Returns (μ={exp_ret_pct.mean():.3f}%)')

# 3. Volatility forecast
ax = fig.add_subplot(gs[0, 2])
vol_pct = all_outputs['volatility'] * 100
ax.hist(vol_pct, bins=50, color='orange', alpha=0.7, edgecolor='black')
ax.set_xlabel('Volatility Forecast (%)')
ax.set_ylabel('Count')
ax.set_title('Volatility Predictions')

# 4. Entry prob vs actual return (scatter)
ax = fig.add_subplot(gs[1, 0])
sample_idx = np.random.choice(len(entry_probs), min(2000, len(entry_probs)), replace=False)
ax.scatter(entry_probs[sample_idx], returns[sample_idx]*100, 
          alpha=0.3, s=5, c='blue')
ax.set_xlabel('Entry Probability')
ax.set_ylabel('Actual Return (%)')
ax.set_title(f'Entry Signal vs Return (r={corr_entry:.3f})')
ax.axhline(0, color='gray', linestyle='-', alpha=0.3)

# 5. Win rate by confidence bins
ax = fig.add_subplot(gs[1, 1])
n_bins = 10
prob_bins = np.linspace(0, 1, n_bins+1)
win_rates = []
bin_centers = []
bin_counts = []

for i in range(n_bins):
    mask = (entry_probs >= prob_bins[i]) & (entry_probs < prob_bins[i+1])
    if mask.sum() > 0:
        wr = labels[mask].mean() * 100
        win_rates.append(wr)
        bin_centers.append((prob_bins[i] + prob_bins[i+1]) / 2)
        bin_counts.append(mask.sum())

bars = ax.bar(bin_centers, win_rates, width=0.08, color='darkgreen', alpha=0.7)
ax.axhline(50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
ax.set_xlabel('Entry Probability Bin')
ax.set_ylabel('Win Rate (%)')
ax.set_title('Win Rate by Confidence Level')
ax.legend()

# Add count labels on bars
for bar, count in zip(bars, bin_counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
           f'n={count}', ha='center', va='bottom', fontsize=8)

# 6. Cumulative returns by signal strength
ax = fig.add_subplot(gs[1, 2])
sorted_idx = np.argsort(-entry_probs)
cumulative_returns = np.cumsum(returns[sorted_idx])
percentiles = np.arange(len(cumulative_returns)) / len(cumulative_returns) * 100
ax.plot(percentiles, cumulative_returns * 100, linewidth=2)
ax.set_xlabel('Percentile (by entry signal strength)')
ax.set_ylabel('Cumulative Return (%)')
ax.set_title('Cumulative Returns (sorted by signal)')
ax.grid(True, alpha=0.3)

# 7. Exit signals distribution
ax = fig.add_subplot(gs[2, 0])
exit_data = [
    all_outputs['tp_prob'],
    all_outputs['sl_prob'],
    all_outputs['hold_score']
]
labels_exit = ['Take Profit', 'Stop Loss', 'Hold']
bp = ax.boxplot(exit_data, labels=labels_exit, patch_artist=True)
colors = ['green', 'red', 'blue']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
ax.set_ylabel('Probability')
ax.set_title('Exit Signal Distributions')

# 8. Position sizing distribution
ax = fig.add_subplot(gs[2, 1])
ax.hist(all_outputs['position_size'], bins=50, color='purple', alpha=0.7, edgecolor='black')
ax.set_xlabel('Position Size (0-1)')
ax.set_ylabel('Count')
ax.set_title(f'Position Sizing (μ={all_outputs["position_size"].mean():.3f})')

# 9. Signal correlation heatmap
ax = fig.add_subplot(gs[2, 2])
corr_matrix = np.corrcoef([
    entry_probs,
    all_outputs['exit_prob'],
    all_outputs['expected_return'],
    all_outputs['volatility'],
    all_outputs['position_size']
])
im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks(range(5))
ax.set_yticks(range(5))
labels_corr = ['Entry', 'Exit', 'ExpRet', 'Vol', 'Size']
ax.set_xticklabels(labels_corr, rotation=45)
ax.set_yticklabels(labels_corr)
ax.set_title('Signal Correlations')
plt.colorbar(im, ax=ax)

# Add correlation values
for i in range(5):
    for j in range(5):
        text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=8)

# 10. Time series of signals (sample)
ax = fig.add_subplot(gs[3, :])
window = min(500, len(entry_probs))
idx_range = range(window)
ax.plot(idx_range, entry_probs[:window], label='Entry', alpha=0.7, linewidth=1)
ax.plot(idx_range, all_outputs['exit_prob'][:window], label='Exit', alpha=0.7, linewidth=1)
ax.plot(idx_range, all_outputs['position_size'][:window], label='Size', alpha=0.7, linewidth=1)
ax.set_xlabel('Time')
ax.set_ylabel('Signal Value')
ax.set_title(f'Signal Time Series (first {window} samples)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('NEURAL TRADING MODEL - COMPREHENSIVE ANALYSIS', fontsize=16, fontweight='bold', y=1.02)
plt.savefig('model_analysis_complete.png', dpi=150, bbox_inches='tight')
console.print("✅ Saved: [green]model_analysis_complete.png[/green]")

# ============================================
# EXPORT RESULTS
# ============================================

console.print("\n" + "="*80)
console.print("[bold cyan]EXPORTING RESULTS[/bold cyan]")
console.print("="*80)

# Create comprehensive DataFrame
df = pd.DataFrame({
    'entry_prob': entry_probs,
    'exit_prob': all_outputs['exit_prob'],
    'expected_return': all_outputs['expected_return'],
    'expected_return_pct': all_outputs['expected_return'] * 100,
    'volatility_forecast': all_outputs['volatility'],
    'volatility_forecast_pct': all_outputs['volatility'] * 100,
    'position_size': all_outputs['position_size'],
    'take_profit_prob': all_outputs['tp_prob'],
    'stop_loss_prob': all_outputs['sl_prob'],
    'hold_score': all_outputs['hold_score'],
    'actual_return': returns,
    'actual_return_pct': returns * 100,
    'win_label': labels,
    'price': prices
})

# Add derived metrics
df['signal_strength'] = df['entry_prob'] * df['position_size']
df['risk_adjusted_signal'] = df['expected_return'] / (df['volatility_forecast'] + 1e-6)

# Save to CSV
df.to_csv('model_predictions_full.csv', index=False)
console.print(f"✅ Saved: [green]model_predictions_full.csv[/green] ({len(df):,} rows)")

# Save summary statistics
summary = df.describe()
summary.to_csv('model_summary_stats.csv')
console.print("✅ Saved: [green]model_summary_stats.csv[/green]")

console.print("\n" + "="*80)
console.print("[bold green]✨ ANALYSIS COMPLETE![/bold green]")
console.print("="*80)