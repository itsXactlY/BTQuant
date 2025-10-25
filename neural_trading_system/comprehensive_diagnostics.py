#!/usr/bin/env python3
"""
comprehensive_diagnostics.py

Complete diagnostic suite for neural trading models.
Checks data quality, model health, training stability, and outputs detailed reports.
"""

import sys
import glob
import pickle
import numpy as np
import torch
import polars as pl
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

console = Console()

# ============================================================================
# DATA QUALITY DIAGNOSTICS
# ============================================================================

class DataQualityChecker:
    """Comprehensive data quality validation"""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.stats = {}
    
    def check_parquet_data(self, parquet_path: str) -> Dict:
        """Check raw indicator data quality"""
        console.print("\n" + "="*80)
        console.print(Panel.fit("[bold cyan]DATA QUALITY CHECK: Raw Indicators[/bold cyan]", border_style="cyan"))
        
        if not Path(parquet_path).exists():
            self.issues.append(f"File not found: {parquet_path}")
            console.print(f"[red]❌ File not found: {parquet_path}[/red]")
            return {}
        
        df = pl.read_parquet(parquet_path)
        # Ensure datetime column is parsed correctly
        if df["datetime"].dtype == pl.Utf8:
            df = df.with_columns(
                pl.when(pl.col("datetime").str.contains(r"\.\d+"))
                .then(pl.col("datetime").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False))
                .otherwise(pl.col("datetime").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False))
                .alias("datetime")
            )

        # Drop rows with null datetime
        df = df.filter(pl.col("datetime").is_not_null())

        dates = df["datetime"].to_list()

        # Skip gaps if we have fewer than 2 timestamps
        if len(dates) > 1:
            for i in range(1, len(dates)):
                if dates[i] is None or dates[i-1] is None:
                    continue
                try:
                    delta = (dates[i] - dates[i-1]).total_seconds()
                    if delta > 7200:  # >2h gap (for 1h interval)
                        print(f"⚠️  Gap detected between {dates[i-1]} and {dates[i]} (Δ={delta/3600:.2f}h)")
                except Exception as e:
                    print(f"⚠️  Skipped bad timestamp pair: {dates[i-1]}, {dates[i]} ({e})")
        else:
            print("⚠️  Not enough valid timestamps to check continuity.")
        df.write_parquet(parquet_path)
        console.print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Basic stats
        self.stats['raw_rows'] = len(df)
        self.stats['raw_cols'] = len(df.columns)
        
        # Check for critical columns
        critical_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        missing_critical = [c for c in critical_cols if c not in df.columns]
        if missing_critical:
            self.issues.append(f"Missing critical columns: {missing_critical}")
            console.print(f"[red]❌ Missing: {missing_critical}[/red]")
        
        # Check for NaN/Inf in each column
        console.print("\n🔍 [yellow]Checking for NaN/Inf values...[/yellow]")
        nan_summary = []
        inf_summary = []
        
        for col in df.columns:
            if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                nan_count = df[col].null_count()
                values = df[col].to_numpy()
                inf_count = np.isinf(values).sum() if values.dtype.kind == 'f' else 0
                
                if nan_count > 0:
                    nan_summary.append((col, nan_count, nan_count/len(df)*100))
                if inf_count > 0:
                    inf_summary.append((col, inf_count, inf_count/len(df)*100))
        
        if nan_summary:
            console.print(f"\n⚠️  [yellow]Found NaN in {len(nan_summary)} columns:[/yellow]")
            table = Table(show_header=True, header_style="bold yellow")
            table.add_column("Column", style="cyan")
            table.add_column("NaN Count", justify="right")
            table.add_column("Percentage", justify="right")
            
            for col, count, pct in sorted(nan_summary, key=lambda x: -x[1])[:20]:
                table.add_row(col, f"{count:,}", f"{pct:.2f}%")
            console.print(table)
            self.warnings.append(f"NaN values in {len(nan_summary)} columns")
        
        if inf_summary:
            console.print(f"\n⚠️  [yellow]Found Inf in {len(inf_summary)} columns:[/yellow]")
            table = Table(show_header=True, header_style="bold yellow")
            table.add_column("Column", style="cyan")
            table.add_column("Inf Count", justify="right")
            table.add_column("Percentage", justify="right")
            
            for col, count, pct in sorted(inf_summary, key=lambda x: -x[1])[:20]:
                table.add_row(col, f"{count:,}", f"{pct:.2f}%")
            console.print(table)
            self.warnings.append(f"Inf values in {len(inf_summary)} columns")
        
        # Check for extreme values
        console.print("\n🔍 [yellow]Checking for extreme values...[/yellow]")
        extreme_cols = []
        
        for col in df.columns:
            if df[col].dtype in [pl.Float32, pl.Float64]:
                values = df[col].to_numpy()
                values = values[~np.isnan(values)]
                if len(values) > 0:
                    std = np.std(values)
                    mean = np.mean(values)
                    max_val = np.max(np.abs(values))
                    
                    # Flag if values exceed 10 standard deviations or very large absolute values
                    if max_val > abs(mean) + 10 * std or max_val > 1e6:
                        extreme_cols.append((col, mean, std, max_val))
        
        if extreme_cols:
            console.print(f"⚠️  [yellow]Found extreme values in {len(extreme_cols)} columns:[/yellow]")
            table = Table(show_header=True, header_style="bold yellow")
            table.add_column("Column", style="cyan")
            table.add_column("Mean", justify="right")
            table.add_column("Std", justify="right")
            table.add_column("Max Abs", justify="right")
            
            for col, mean, std, max_val in extreme_cols[:10]:
                table.add_row(col, f"{mean:.2e}", f"{std:.2e}", f"{max_val:.2e}")
            console.print(table)
            self.warnings.append(f"Extreme values in {len(extreme_cols)} columns")
        
        # Check temporal continuity
        if 'datetime' in df.columns:
            console.print("\n🔍 [yellow]Checking temporal continuity...[/yellow]")
            dates = df['datetime'].to_list()
            gaps = []
            for i in range(1, len(dates)):
                # This is a simple check; adjust based on your expected interval
                if (dates[i] - dates[i-1]).total_seconds() > 7200:  # >2 hours for 1h data
                    gaps.append((i, dates[i-1], dates[i]))
            
            if gaps:
                console.print(f"⚠️  [yellow]Found {len(gaps)} temporal gaps[/yellow]")
                self.warnings.append(f"Temporal gaps: {len(gaps)}")
        
        self.stats['nan_columns'] = len(nan_summary)
        self.stats['inf_columns'] = len(inf_summary)
        self.stats['extreme_columns'] = len(extreme_cols)
        
        return self.stats
    
    def check_feature_cache(self, cache_path: str) -> Dict:
        """Check processed feature cache quality"""
        console.print("\n" + "="*80)
        console.print(Panel.fit("[bold cyan]DATA QUALITY CHECK: Feature Cache[/bold cyan]", border_style="cyan"))
        
        if not Path(cache_path).exists():
            self.issues.append(f"Cache not found: {cache_path}")
            console.print(f"[red]❌ Cache not found: {cache_path}[/red]")
            return {}
        
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        
        features = cached_data['features']
        returns = cached_data['returns']
        
        console.print(f"✅ Loaded feature cache")
        console.print(f"   Features shape: {features.shape}")
        console.print(f"   Returns shape: {returns.shape}")
        
        # Detailed feature statistics
        console.print("\n📊 [cyan]Feature Statistics:[/cyan]")
        
        nan_count = np.isnan(features).sum()
        inf_count = np.isinf(features).sum()
        zero_count = (features == 0).sum()
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="yellow")
        table.add_column("Percentage", justify="right", style="green")
        
        total_values = features.size
        table.add_row("Total Values", f"{total_values:,}", "100.00%")
        table.add_row("NaN Values", f"{nan_count:,}", f"{nan_count/total_values*100:.4f}%")
        table.add_row("Inf Values", f"{inf_count:,}", f"{inf_count/total_values*100:.4f}%")
        table.add_row("Zero Values", f"{zero_count:,}", f"{zero_count/total_values*100:.4f}%")
        
        console.print(table)
        
        if nan_count > 0:
            self.issues.append(f"Features contain {nan_count:,} NaN values")
            console.print(f"[red]❌ Features contain NaN: {nan_count:,}[/red]")
        
        if inf_count > 0:
            self.issues.append(f"Features contain {inf_count:,} Inf values")
            console.print(f"[red]❌ Features contain Inf: {inf_count:,}[/red]")
        
        # Distribution analysis
        console.print("\n📊 [cyan]Feature Distribution:[/cyan]")
        
        valid_features = features[~np.isnan(features) & ~np.isinf(features)]
        if len(valid_features) > 0:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Statistic", style="cyan")
            table.add_column("Value", justify="right", style="yellow")
            
            table.add_row("Min", f"{np.min(valid_features):.6f}")
            table.add_row("Max", f"{np.max(valid_features):.6f}")
            table.add_row("Mean", f"{np.mean(valid_features):.6f}")
            table.add_row("Median", f"{np.median(valid_features):.6f}")
            table.add_row("Std", f"{np.std(valid_features):.6f}")
            table.add_row("Q1 (25%)", f"{np.percentile(valid_features, 25):.6f}")
            table.add_row("Q3 (75%)", f"{np.percentile(valid_features, 75):.6f}")
            
            console.print(table)
        
        # Check returns
        console.print("\n📊 [cyan]Returns Statistics:[/cyan]")
        
        nan_returns = np.isnan(returns).sum()
        inf_returns = np.isinf(returns).sum()
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="yellow")
        
        table.add_row("Total", f"{len(returns):,}")
        table.add_row("NaN", f"{nan_returns:,}")
        table.add_row("Inf", f"{inf_returns:,}")
        
        if nan_returns == 0 and inf_returns == 0:
            valid_returns = returns
            table.add_row("Min", f"{np.min(valid_returns):.6f}")
            table.add_row("Max", f"{np.max(valid_returns):.6f}")
            table.add_row("Mean", f"{np.mean(valid_returns):.6f}")
            table.add_row("Std", f"{np.std(valid_returns):.6f}")
        
        console.print(table)
        
        if nan_returns > 0:
            self.issues.append(f"Returns contain {nan_returns:,} NaN values")
        if inf_returns > 0:
            self.issues.append(f"Returns contain {inf_returns:,} Inf values")
        
        self.stats['feature_shape'] = features.shape
        self.stats['feature_nan'] = nan_count
        self.stats['feature_inf'] = inf_count
        self.stats['returns_nan'] = nan_returns
        self.stats['returns_inf'] = inf_returns
        
        return self.stats
    
    def generate_feature_report(self, cache_path: str, output_dir: str = 'diagnostics'):
        """Generate detailed feature analysis plots"""
        console.print("\n📊 [cyan]Generating feature analysis plots...[/cyan]")
        
        Path(output_dir).mkdir(exist_ok=True)
        
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        
        features = cached_data['features']
        returns = cached_data['returns']
        
        # 1. Feature distribution heatmap
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Sample features for visualization (too many to show all)
        sample_features = features[:1000, :100]  # First 1000 samples, first 100 features
        
        # Heatmap of feature values
        im = axes[0, 0].imshow(sample_features.T, aspect='auto', cmap='RdBu_r', 
                               vmin=-3, vmax=3, interpolation='nearest')
        axes[0, 0].set_title('Feature Values Heatmap (first 100 features)')
        axes[0, 0].set_xlabel('Sample Index')
        axes[0, 0].set_ylabel('Feature Index')
        plt.colorbar(im, ax=axes[0, 0])
        
        # Histogram of all feature values
        valid_features = features[~np.isnan(features) & ~np.isinf(features)]
        axes[0, 1].hist(valid_features.ravel(), bins=100, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Feature Value Distribution')
        axes[0, 1].set_xlabel('Feature Value')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_yscale('log')
        
        # Returns distribution
        valid_returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        axes[1, 0].hist(valid_returns, bins=100, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Returns Distribution')
        axes[1, 0].set_xlabel('Forward Return')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
        
        # Feature variance across samples
        feature_vars = np.var(features, axis=0)
        # 🧠 Safe histogram plotting
        finite_vals = feature_vars[np.isfinite(feature_vars)]
        if len(finite_vals) > 0:
            finite_vals = np.clip(finite_vals, -1e6, 1e6)
            axes[1, 1].hist(finite_vals, bins=100, alpha=0.7, edgecolor='black')
        else:
            axes[1, 1].text(0.5, 0.5, "No finite feature vars", ha='center', va='center', color='red')
        axes[1, 1].set_title('Feature Variance Distribution')
        axes[1, 1].set_xlabel('Variance')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plot_path = Path(output_dir) / 'feature_analysis.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        console.print(f"   ✅ Saved: {plot_path}")
        plt.close()
        
        # 2. Correlation analysis (sample)
        console.print("   Computing feature correlations...")
        sample_size = min(5000, len(features))
        sample_indices = np.random.choice(len(features), sample_size, replace=False)
        sample_features = features[sample_indices, :50]  # First 50 features
        
        fig, ax = plt.subplots(figsize=(12, 10))
        corr = np.corrcoef(sample_features.T)
        im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_title(f'Feature Correlation Matrix (first 50 features, n={sample_size})')
        plt.colorbar(im, ax=ax)
        
        plot_path = Path(output_dir) / 'feature_correlation.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        console.print(f"   ✅ Saved: {plot_path}")
        plt.close()


# ============================================================================
# MODEL HEALTH DIAGNOSTICS
# ============================================================================

class ModelHealthChecker:
    """Comprehensive model health validation"""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.stats = {}
    
    def check_model_weights(self, model_path: str) -> Dict:
        """Check model checkpoint for NaN/Inf in weights"""
        console.print("\n" + "="*80)
        console.print(Panel.fit("[bold cyan]MODEL HEALTH CHECK: Weights[/bold cyan]", border_style="cyan"))
        
        if not Path(model_path).exists():
            self.issues.append(f"Model not found: {model_path}")
            console.print(f"[red]❌ Model not found: {model_path}[/red]")
            return {}
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        console.print("✅ Checkpoint loaded")
        console.print(f"   Keys: {list(checkpoint.keys())}")
        
        if 'model_state_dict' not in checkpoint:
            self.issues.append("No model_state_dict in checkpoint")
            console.print("[red]❌ No model_state_dict found[/red]")
            return {}
        
        state_dict = checkpoint['model_state_dict']
        
        # Analyze each parameter
        console.print(f"\n🔍 [yellow]Analyzing {len(state_dict)} parameters...[/yellow]")
        
        nan_params = []
        inf_params = []
        zero_params = []
        large_params = []
        
        total_params = 0
        total_elements = 0
        
        for name, tensor in track(state_dict.items(), description="Checking weights"):
            total_params += 1
            total_elements += tensor.numel()
            
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()
            all_zero = (tensor == 0).all().item()
            max_abs = torch.abs(tensor).max().item()
            
            if has_nan:
                nan_params.append(name)
            if has_inf:
                inf_params.append(name)
            if all_zero:
                zero_params.append(name)
            if max_abs > 1e6:
                large_params.append((name, max_abs))
        
        # Summary table
        console.print("\n📊 [cyan]Weight Health Summary:[/cyan]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="yellow")
        table.add_column("Details", style="white")
        
        # NaN check
        if nan_params:
            table.add_row("NaN Weights", "[red]❌ FAIL[/red]", f"{len(nan_params)} parameters")
            self.issues.append(f"NaN in {len(nan_params)} parameters")
        else:
            table.add_row("NaN Weights", "[green]✅ PASS[/green]", "None found")
        
        # Inf check
        if inf_params:
            table.add_row("Inf Weights", "[red]❌ FAIL[/red]", f"{len(inf_params)} parameters")
            self.issues.append(f"Inf in {len(inf_params)} parameters")
        else:
            table.add_row("Inf Weights", "[green]✅ PASS[/green]", "None found")
        
        # Zero check
        if zero_params:
            table.add_row("All-Zero Weights", "[yellow]⚠️  WARN[/yellow]", f"{len(zero_params)} parameters")
            self.warnings.append(f"All-zero in {len(zero_params)} parameters")
        else:
            table.add_row("All-Zero Weights", "[green]✅ PASS[/green]", "None found")
        
        # Large values
        if large_params:
            table.add_row("Large Values", "[yellow]⚠️  WARN[/yellow]", f"{len(large_params)} parameters >1e6")
            self.warnings.append(f"Large values in {len(large_params)} parameters")
        else:
            table.add_row("Large Values", "[green]✅ PASS[/green]", "All reasonable")
        
        console.print(table)
        
        # Detailed lists
        if nan_params:
            console.print(f"\n[red]❌ Parameters with NaN:[/red]")
            for name in nan_params[:10]:
                console.print(f"   - {name}")
            if len(nan_params) > 10:
                console.print(f"   ... and {len(nan_params)-10} more")
        
        if inf_params:
            console.print(f"\n[red]❌ Parameters with Inf:[/red]")
            for name in inf_params[:10]:
                console.print(f"   - {name}")
            if len(inf_params) > 10:
                console.print(f"   ... and {len(inf_params)-10} more")
        
        if large_params:
            console.print(f"\n[yellow]⚠️  Parameters with large values:[/yellow]")
            for name, val in sorted(large_params, key=lambda x: -x[1])[:10]:
                console.print(f"   - {name}: {val:.2e}")
        
        # Parameter statistics
        console.print("\n📊 [cyan]Parameter Statistics:[/cyan]")
        
        all_params = torch.cat([p.flatten() for p in state_dict.values()])
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Statistic", style="cyan")
        table.add_column("Value", justify="right", style="yellow")
        
        table.add_row("Total Parameters", f"{total_params:,}")
        table.add_row("Total Elements", f"{total_elements:,}")
        table.add_row("Min", f"{all_params.min().item():.6e}")
        table.add_row("Max", f"{all_params.max().item():.6e}")
        table.add_row("Mean", f"{all_params.mean().item():.6e}")
        table.add_row("Std", f"{all_params.std().item():.6e}")
        table.add_row("Median", f"{all_params.median().item():.6e}")
        
        console.print(table)
        
        self.stats['total_params'] = total_params
        self.stats['total_elements'] = total_elements
        self.stats['nan_params'] = len(nan_params)
        self.stats['inf_params'] = len(inf_params)
        self.stats['zero_params'] = len(zero_params)
        
        return self.stats
    
    def check_model_outputs(self, model_path: str, feature_extractor_path: str, 
                           cache_path: str, num_samples: int = 100) -> Dict:
        """Test model on sample data and check outputs"""
        console.print("\n" + "="*80)
        console.print(Panel.fit("[bold cyan]MODEL HEALTH CHECK: Outputs[/bold cyan]", border_style="cyan"))
        
        # Load model
        sys.path.append('neural_trading_system')
        from models.architecture import NeuralTradingModel
        
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint.get('config', {})
        
        model = NeuralTradingModel(
            feature_dim=config.get('feature_dim', 500),
            d_model=config.get('d_model', 256),
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 6),
            d_ff=config.get('d_ff', 1024),
            dropout=0.0,
            latent_dim=config.get('latent_dim', 8),
            seq_len=config.get('seq_len', 100),
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load feature extractor
        with open(feature_extractor_path, 'rb') as f:
            feature_extractor = pickle.load(f)
        
        # Load test data
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        
        features = cached_data['features']
        seq_len = config.get('seq_len', 100)
        
        console.print(f"✅ Loaded model and data")
        console.print(f"   Testing on {num_samples} samples")
        
        # Test forward pass
        sample_indices = np.random.choice(len(features) - seq_len, num_samples, replace=False)
        
        nan_outputs = 0
        inf_outputs = 0
        valid_outputs = 0
        
        entry_probs = []
        expected_returns = []
        regime_embeddings_list = []
        
        console.print(f"\n🔍 [yellow]Running forward passes...[/yellow]")
        
        with torch.no_grad():
            for idx in track(sample_indices, description="Testing"):
                seq = features[idx:idx+seq_len]
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0)
                
                try:
                    outputs = model(seq_tensor)
                    
                    entry_prob = outputs['entry_prob'].item()
                    exp_return = outputs['expected_return'].item()
                    regime_emb = outputs['regime_embedding'].cpu().numpy()
                    
                    if np.isnan(entry_prob) or np.isnan(exp_return):
                        nan_outputs += 1
                    elif np.isinf(entry_prob) or np.isinf(exp_return):
                        inf_outputs += 1
                    else:
                        valid_outputs += 1
                        entry_probs.append(entry_prob)
                        expected_returns.append(exp_return)
                        regime_embeddings_list.append(regime_emb)
                
                except Exception as e:
                    console.print(f"[red]Error on sample {idx}: {e}[/red]")
                    nan_outputs += 1
        
        # Summary
        console.print("\n📊 [cyan]Output Health Summary:[/cyan]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Check", style="cyan")
        table.add_column("Count", justify="right", style="yellow")
        table.add_column("Percentage", justify="right", style="green")
        
        table.add_row("Valid Outputs", str(valid_outputs), f"{valid_outputs/num_samples*100:.1f}%")
        table.add_row("NaN Outputs", str(nan_outputs), f"{nan_outputs/num_samples*100:.1f}%")
        table.add_row("Inf Outputs", str(inf_outputs), f"{inf_outputs/num_samples*100:.1f}%")
        
        console.print(table)
        
        if nan_outputs > 0:
            self.issues.append(f"Model outputs NaN on {nan_outputs}/{num_samples} samples")
            console.print(f"[red]❌ Model produces NaN outputs![/red]")
        
        if inf_outputs > 0:
            self.issues.append(f"Model outputs Inf on {inf_outputs}/{num_samples} samples")
            console.print(f"[red]❌ Model produces Inf outputs![/red]")
        
        # Output statistics (if we have valid outputs)
        if valid_outputs > 0:
            console.print("\n📊 [cyan]Valid Output Statistics:[/cyan]")
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Output", style="cyan")
            table.add_column("Min", justify="right", style="yellow")
            table.add_column("Max", justify="right", style="yellow")
            table.add_column("Mean", justify="right", style="yellow")
            table.add_column("Std", justify="right", style="yellow")
            
            entry_probs = np.array(entry_probs)
            expected_returns = np.array(expected_returns)
            
            table.add_row(
                "Entry Prob",
                f"{entry_probs.min():.4f}",
                f"{entry_probs.max():.4f}",
                f"{entry_probs.mean():.4f}",
                f"{entry_probs.std():.4f}"
            )
            table.add_row(
                "Expected Return",
                f"{expected_returns.min():.6f}",
                f"{expected_returns.max():.6f}",
                f"{expected_returns.mean():.6f}",
                f"{expected_returns.std():.6f}"
            )
            
            console.print(table)
        
        self.stats['valid_outputs'] = valid_outputs
        self.stats['nan_outputs'] = nan_outputs
        self.stats['inf_outputs'] = inf_outputs
        
        return self.stats
    
    def visualize_weight_distributions(self, model_path: str, output_dir: str = 'diagnostics'):
        """Generate weight distribution plots"""
        console.print("\n📊 [cyan]Generating weight distribution plots...[/cyan]")
        
        Path(output_dir).mkdir(exist_ok=True)
        
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        
        # Collect all weights by layer type
        layer_weights = {
            'embedding': [],
            'attention': [],
            'feedforward': [],
            'regime_encoder': [],
            'other': []
        }
        
        for name, tensor in state_dict.items():
            weights = tensor.flatten().cpu().numpy()
            
            if 'embedding' in name.lower():
                layer_weights['embedding'].extend(weights)
            elif 'attention' in name.lower() or 'attn' in name.lower():
                layer_weights['attention'].extend(weights)
            elif 'feedforward' in name.lower() or 'mlp' in name.lower() or 'fc' in name.lower():
                layer_weights['feedforward'].extend(weights)
            elif 'regime' in name.lower():
                layer_weights['regime_encoder'].extend(weights)
            else:
                layer_weights['other'].extend(weights)
        
        # Create subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        axes = axes.flatten()
        
        plot_count = 0
        for idx, (layer_type, weights) in enumerate(layer_weights.items()):
            if len(weights) > 0:
                weights = np.array(weights)
                
                # Filter out NaN and Inf values for plotting
                valid_weights = weights[~np.isnan(weights) & ~np.isinf(weights)]
                
                if len(valid_weights) > 0:
                    # Plot histogram
                    axes[idx].hist(valid_weights, bins=100, alpha=0.7, edgecolor='black')
                    axes[idx].set_title(f'{layer_type.capitalize()} Weights')
                    axes[idx].set_xlabel('Weight Value')
                    axes[idx].set_ylabel('Count')
                    axes[idx].set_yscale('log')
                    axes[idx].axvline(0, color='red', linestyle='--', linewidth=1)
                    axes[idx].grid(True, alpha=0.3)
                    
                    # Add statistics
                    nan_pct = np.isnan(weights).sum() / len(weights) * 100
                    inf_pct = np.isinf(weights).sum() / len(weights) * 100
                    stats_text = f'Valid: {len(valid_weights):,}/{len(weights):,}\n'
                    stats_text += f'Mean: {np.mean(valid_weights):.4f}\n'
                    stats_text += f'Std: {np.std(valid_weights):.4f}'
                    if nan_pct > 0 or inf_pct > 0:
                        stats_text += f'\n⚠️ NaN: {nan_pct:.1f}%'
                        if inf_pct > 0:
                            stats_text += f'\n⚠️ Inf: {inf_pct:.1f}%'
                    
                    axes[idx].text(0.02, 0.98, stats_text, transform=axes[idx].transAxes,
                                verticalalignment='top', fontsize=9,
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
                    plot_count += 1
                else:
                    # All values are NaN/Inf
                    axes[idx].text(0.5, 0.5, f'{layer_type.capitalize()}\n❌ All NaN/Inf\n({len(weights):,} values)',
                                ha='center', va='center', fontsize=12, color='red',
                                transform=axes[idx].transAxes,
                                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
                    axes[idx].set_title(f'{layer_type.capitalize()} Weights')
                    axes[idx].set_xlim([0, 1])
                    axes[idx].set_ylim([0, 1])
                    axes[idx].axis('off')
        
        # Remove empty subplots
        for idx in range(len(layer_weights), 6):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plot_path = Path(output_dir) / 'weight_distributions.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        console.print(f"   ✅ Saved: {plot_path}")
        plt.close()
        
        # Create a summary plot showing NaN distribution across layers
        fig, ax = plt.subplots(figsize=(12, 6))
        
        layer_names = []
        nan_percentages = []
        total_counts = []
        
        for layer_type, weights in layer_weights.items():
            if len(weights) > 0:
                weights = np.array(weights)
                nan_pct = np.isnan(weights).sum() / len(weights) * 100
                layer_names.append(layer_type.capitalize())
                nan_percentages.append(nan_pct)
                total_counts.append(len(weights))
        
        x = np.arange(len(layer_names))
        bars = ax.bar(x, nan_percentages, color=['red' if p > 50 else 'orange' if p > 10 else 'green' for p in nan_percentages])
        ax.set_xlabel('Layer Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('NaN Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_title('NaN Distribution Across Layer Types', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
        ax.axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='10% threshold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        # Add value labels on bars
        for i, (bar, pct, count) in enumerate(zip(bars, nan_percentages, total_counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%\n({count:,} params)',
                ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plot_path = Path(output_dir) / 'nan_distribution.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        console.print(f"   ✅ Saved: {plot_path}")
        plt.close()


# ============================================================================
# TRAINING STABILITY DIAGNOSTICS
# ============================================================================

class TrainingStabilityChecker:
    """Analyze training logs for stability issues"""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.stats = {}
    
    def analyze_training_log(self, log_path: str) -> Dict:
        """Parse and analyze training log file"""
        console.print("\n" + "="*80)
        console.print(Panel.fit("[bold cyan]TRAINING STABILITY CHECK[/bold cyan]", border_style="cyan"))
        
        if not Path(log_path).exists():
            self.issues.append(f"Log not found: {log_path}")
            console.print(f"[red]❌ Log not found: {log_path}[/red]")
            return {}
        
        # Parse log file
        epochs = []
        losses = []
        gradient_norms = []
        learning_rates = []
        
        console.print(f"📖 [yellow]Parsing log file...[/yellow]")
        
        with open(log_path, 'r') as f:
            for line in f:
                # Extract metrics (adjust based on your log format)
                if 'Epoch' in line and 'Loss' in line:
                    # Example: "Epoch 10/100 | Loss: 0.1234 | Grad Norm: 1.23 | LR: 0.001"
                    try:
                        parts = line.split('|')
                        epoch_part = [p for p in parts if 'Epoch' in p][0]
                        loss_part = [p for p in parts if 'Loss' in p][0]
                        
                        epoch = int(epoch_part.split('/')[0].split()[-1])
                        loss = float(loss_part.split(':')[-1].strip())
                        
                        epochs.append(epoch)
                        losses.append(loss)
                        
                        # Optional: extract gradient norm and LR if present
                        if 'Grad Norm' in line:
                            grad_part = [p for p in parts if 'Grad' in p][0]
                            grad_norm = float(grad_part.split(':')[-1].strip())
                            gradient_norms.append(grad_norm)
                        
                        if 'LR' in line:
                            lr_part = [p for p in parts if 'LR' in p][0]
                            lr = float(lr_part.split(':')[-1].strip())
                            learning_rates.append(lr)
                    except Exception as e:
                        continue
        
        console.print(f"✅ Parsed {len(epochs)} training epochs")
        
        if len(losses) == 0:
            self.issues.append("No loss values found in log")
            console.print("[red]❌ No loss values found[/red]")
            return {}
        
        # Analyze loss trajectory
        console.print("\n📊 [cyan]Loss Trajectory Analysis:[/cyan]")
        
        losses = np.array(losses)
        
        # Check for NaN/Inf
        nan_count = np.isnan(losses).sum()
        inf_count = np.isinf(losses).sum()
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="yellow")
        table.add_column("Details", style="white")
        
        if nan_count > 0:
            table.add_row("NaN Losses", "[red]❌ FAIL[/red]", f"{nan_count} epochs")
            self.issues.append(f"NaN loss in {nan_count} epochs")
        else:
            table.add_row("NaN Losses", "[green]✅ PASS[/green]", "None")
        
        if inf_count > 0:
            table.add_row("Inf Losses", "[red]❌ FAIL[/red]", f"{inf_count} epochs")
            self.issues.append(f"Inf loss in {inf_count} epochs")
        else:
            table.add_row("Inf Losses", "[green]✅ PASS[/green]", "None")
        
        # Check for exploding loss
        valid_losses = losses[~np.isnan(losses) & ~np.isinf(losses)]
        if len(valid_losses) > 10:
            initial_loss = np.mean(valid_losses[:5])
            final_loss = np.mean(valid_losses[-5:])
            
            if final_loss > initial_loss * 2:
                table.add_row("Loss Trend", "[red]❌ FAIL[/red]", f"Loss increased {final_loss/initial_loss:.2f}x")
                self.issues.append("Loss increased during training")
            elif final_loss < initial_loss * 0.5:
                table.add_row("Loss Trend", "[green]✅ PASS[/green]", f"Loss decreased {initial_loss/final_loss:.2f}x")
            else:
                table.add_row("Loss Trend", "[yellow]⚠️  WARN[/yellow]", "Limited improvement")
                self.warnings.append("Limited loss improvement")
        
        # Check for loss spikes
        if len(valid_losses) > 5:
            rolling_mean = np.convolve(valid_losses, np.ones(5)/5, mode='valid')
            rolling_std = np.array([np.std(valid_losses[max(0,i-5):i]) for i in range(5, len(valid_losses))])
            
            spikes = []
            for i in range(len(rolling_mean)):
                if valid_losses[i+5] > rolling_mean[i] + 3*rolling_std[i]:
                    spikes.append(i+5)
            
            if len(spikes) > 0:
                table.add_row("Loss Spikes", "[yellow]⚠️  WARN[/yellow]", f"{len(spikes)} spikes detected")
                self.warnings.append(f"{len(spikes)} loss spikes")
            else:
                table.add_row("Loss Spikes", "[green]✅ PASS[/green]", "None detected")
        
        console.print(table)
        
        # Gradient norm analysis
        if len(gradient_norms) > 0:
            console.print("\n📊 [cyan]Gradient Norm Analysis:[/cyan]")
            
            gradient_norms = np.array(gradient_norms)
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Statistic", style="cyan")
            table.add_column("Value", justify="right", style="yellow")
            
            table.add_row("Min", f"{np.min(gradient_norms):.6f}")
            table.add_row("Max", f"{np.max(gradient_norms):.6f}")
            table.add_row("Mean", f"{np.mean(gradient_norms):.6f}")
            table.add_row("Std", f"{np.std(gradient_norms):.6f}")
            
            console.print(table)
            
            # Check for gradient explosion/vanishing
            if np.max(gradient_norms) > 100:
                self.warnings.append(f"Large gradient norms detected (max: {np.max(gradient_norms):.2f})")
                console.print(f"[yellow]⚠️  Large gradient norms detected[/yellow]")
            
            if np.mean(gradient_norms) < 0.001:
                self.warnings.append(f"Very small gradient norms (mean: {np.mean(gradient_norms):.6f})")
                console.print(f"[yellow]⚠️  Very small gradient norms[/yellow]")
        
        self.stats['total_epochs'] = len(epochs)
        self.stats['final_loss'] = valid_losses[-1] if len(valid_losses) > 0 else np.nan
        self.stats['min_loss'] = np.min(valid_losses) if len(valid_losses) > 0 else np.nan
        self.stats['nan_losses'] = nan_count
        self.stats['inf_losses'] = inf_count
        
        return self.stats
    
    def visualize_training_metrics(self, log_path: str, output_dir: str = 'diagnostics'):
        """Generate training metrics plots"""
        console.print("\n📊 [cyan]Generating training metrics plots...[/cyan]")
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Parse log (same as above)
        epochs = []
        losses = []
        gradient_norms = []
        learning_rates = []
        
        with open(log_path, 'r') as f:
            for line in f:
                if 'Epoch' in line and 'Loss' in line:
                    try:
                        parts = line.split('|')
                        epoch_part = [p for p in parts if 'Epoch' in p][0]
                        loss_part = [p for p in parts if 'Loss' in p][0]
                        
                        epoch = int(epoch_part.split('/')[0].split()[-1])
                        loss = float(loss_part.split(':')[-1].strip())
                        
                        epochs.append(epoch)
                        losses.append(loss)
                        
                        if 'Grad Norm' in line:
                            grad_part = [p for p in parts if 'Grad' in p][0]
                            grad_norm = float(grad_part.split(':')[-1].strip())
                            gradient_norms.append(grad_norm)
                        
                        if 'LR' in line:
                            lr_part = [p for p in parts if 'LR' in p][0]
                            lr = float(lr_part.split(':')[-1].strip())
                            learning_rates.append(lr)
                    except:
                        continue
        
        # Create plots
        n_plots = 1 + (len(gradient_norms) > 0) + (len(learning_rates) > 0)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 5*n_plots))
        
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Loss plot
        axes[plot_idx].plot(epochs, losses, linewidth=2)
        axes[plot_idx].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Loss')
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].set_yscale('log')
        plot_idx += 1
        
        # Gradient norm plot
        if len(gradient_norms) > 0:
            axes[plot_idx].plot(epochs[:len(gradient_norms)], gradient_norms, linewidth=2, color='orange')
            axes[plot_idx].set_title('Gradient Norm', fontsize=14, fontweight='bold')
            axes[plot_idx].set_xlabel('Epoch')
            axes[plot_idx].set_ylabel('Gradient Norm')
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1
        
        # Learning rate plot
        if len(learning_rates) > 0:
            axes[plot_idx].plot(epochs[:len(learning_rates)], learning_rates, linewidth=2, color='green')
            axes[plot_idx].set_title('Learning Rate', fontsize=14, fontweight='bold')
            axes[plot_idx].set_xlabel('Epoch')
            axes[plot_idx].set_ylabel('Learning Rate')
            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].set_yscale('log')
        
        plt.tight_layout()
        plot_path = Path(output_dir) / 'training_metrics.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        console.print(f"   ✅ Saved: {plot_path}")
        plt.close()


# ============================================================================
# MAIN DIAGNOSTIC RUNNER
# ============================================================================

def run_full_diagnostics(
    parquet_path: str = None,
    cache_path: str = None,
    model_path: str = None,
    feature_extractor_path: str = None,
    log_path: str = None,
    output_dir: str = 'diagnostics'
):
    """Run complete diagnostic suite"""
    
    console.print("\n" + "="*80)
    console.print(Panel.fit(
        "[bold magenta]COMPREHENSIVE NEURAL TRADING MODEL DIAGNOSTICS[/bold magenta]",
        subtitle=f"[italic]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/italic]",
        border_style="magenta"
    ))
    console.print("="*80)
    
    all_issues = []
    all_warnings = []
    all_stats = {}
    
    # Data Quality Checks
    if parquet_path or cache_path:
        data_checker = DataQualityChecker()
        
        if parquet_path:
            data_checker.check_parquet_data(parquet_path)
        
        if cache_path:
            data_checker.check_feature_cache(cache_path)
            data_checker.generate_feature_report(cache_path, output_dir)
        
        all_issues.extend(data_checker.issues)
        all_warnings.extend(data_checker.warnings)
        all_stats.update(data_checker.stats)
    
    # Model Health Checks
    if model_path:
        model_checker = ModelHealthChecker()
        
        model_checker.check_model_weights(model_path)
        
        if feature_extractor_path and cache_path:
            model_checker.check_model_outputs(model_path, feature_extractor_path, cache_path)
        
        model_checker.visualize_weight_distributions(model_path, output_dir)
        
        all_issues.extend(model_checker.issues)
        all_warnings.extend(model_checker.warnings)
        all_stats.update(model_checker.stats)
    
    # Training Stability Checks
    if log_path:
        training_checker = TrainingStabilityChecker()
        
        training_checker.analyze_training_log(log_path)
        training_checker.visualize_training_metrics(log_path, output_dir)
        
        all_issues.extend(training_checker.issues)
        all_warnings.extend(training_checker.warnings)
        all_stats.update(training_checker.stats)
    
    # Final Summary Report
    console.print("\n" + "="*80)
    console.print(Panel.fit("[bold magenta]FINAL DIAGNOSTIC SUMMARY[/bold magenta]", border_style="magenta"))
    
    summary_table = Table(show_header=True, header_style="bold cyan", title="Health Status")
    summary_table.add_column("Category", style="cyan", width=20)
    summary_table.add_column("Status", justify="center", width=15)
    summary_table.add_column("Count", justify="right", width=10)
    
    if len(all_issues) == 0:
        summary_table.add_row("Critical Issues", "[green]✅ PASS[/green]", "0")
    else:
        summary_table.add_row("Critical Issues", "[red]❌ FAIL[/red]", str(len(all_issues)))
    
    if len(all_warnings) == 0:
        summary_table.add_row("Warnings", "[green]✅ PASS[/green]", "0")
    else:
        summary_table.add_row("Warnings", "[yellow]⚠️  WARN[/yellow]", str(len(all_warnings)))
    
    console.print(summary_table)
    
    # Issues list
    if all_issues:
        console.print("\n[bold red]CRITICAL ISSUES:[/bold red]")
        for i, issue in enumerate(all_issues, 1):
            console.print(f"  {i}. {issue}")
    
    # Warnings list
    if all_warnings:
        console.print("\n[bold yellow]WARNINGS:[/bold yellow]")
        for i, warning in enumerate(all_warnings, 1):
            console.print(f"  {i}. {warning}")
    
    # Statistics
    if all_stats:
        console.print("\n[bold cyan]OVERALL STATISTICS:[/bold cyan]")
        stats_table = Table(show_header=True, header_style="bold magenta")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", justify="right", style="yellow")
        
        for key, value in all_stats.items():
            if isinstance(value, (int, np.integer)):
                stats_table.add_row(key, f"{value:,}")
            elif isinstance(value, (float, np.floating)):
                stats_table.add_row(key, f"{value:.6f}")
            elif isinstance(value, tuple):
                stats_table.add_row(key, str(value))
        
        console.print(stats_table)
    
    # Final verdict
    console.print("\n" + "="*80)
    if len(all_issues) == 0:
        console.print(Panel.fit(
            "[bold green]🎉 ALL CHECKS PASSED! Model appears healthy.[/bold green]",
            border_style="green"
        ))
    else:
        console.print(Panel.fit(
            f"[bold red]⚠️  FOUND {len(all_issues)} CRITICAL ISSUES! Review immediately.[/bold red]",
            border_style="red"
        ))
    
    console.print(f"\n📁 Diagnostic plots saved to: {output_dir}/")
    console.print("="*80 + "\n")
    
    return {
        'issues': all_issues,
        'warnings': all_warnings,
        'stats': all_stats
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Neural Trading Model Diagnostics')
    parser.add_argument('--parquet', type=str, help='Path to raw indicator parquet file')
    parser.add_argument('--cache', type=str, help='Path to feature cache pickle file')
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    parser.add_argument('--feature-extractor', type=str, help='Path to feature extractor pickle')
    parser.add_argument('--log', type=str, help='Path to training log file')
    parser.add_argument('--output', type=str, default='diagnostics', help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Auto-detect files if not specified
    if not any([args.parquet, args.cache, args.model, args.log]):
        console.print("[yellow]No files specified. Attempting auto-detection...[/yellow]")
        
        # Look for common file patterns
        if not args.cache:
            caches = glob.glob('**/feature_cache*.pkl', recursive=True)
            if caches:
                args.cache = caches[0]
                console.print(f"Found cache: {args.cache}")
        
        if not args.model:
            models = glob.glob('**/model_*.pt', recursive=True) + glob.glob('**/checkpoint*.pt', recursive=True)
            if models:
                args.model = models[0]
                console.print(f"Found model: {args.model}")
        
        if not args.feature_extractor:
            extractors = glob.glob('**/feature_extractor*.pkl', recursive=True)
            if extractors:
                args.feature_extractor = extractors[0]
                console.print(f"Found feature extractor: {args.feature_extractor}")
        
        if not args.log:
            logs = glob.glob('**/training*.log', recursive=True) + glob.glob('**/train*.log', recursive=True)
            if logs:
                args.log = logs[0]
                console.print(f"Found log: {args.log}")
    
    # Run diagnostics
    results = run_full_diagnostics(
        parquet_path=args.parquet,
        cache_path=args.cache,
        model_path=args.model,
        feature_extractor_path=args.feature_extractor,
        log_path=args.log,
        output_dir=args.output
    )



'''
python comprehensive_diagnostics.py \
  --parquet neural_data/BTC_1h_2017-01-01_2024-12-31_neural_data.parquet \
  --cache neural_data/features/features_faee3f17ea44d4f8.pkl \
  --output diagnostics_nantrace

'''