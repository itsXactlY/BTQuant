#!/usr/bin/env python3
"""
Analyze trained neural trading model.
Visualize attention patterns, feature importance, and learned regimes.
"""

import glob
import sys
from pathlib import Path
import pickle
import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Ensure package resolution
sys.path.append('neural_trading_system')

# Analysis tools
from analysis.attention_viz import AttentionAnalyzer  # noqa: E402

console = Console()


def load_model_and_data(model_path: str, feature_extractor_path: str):
    """Load trained model checkpoint and feature extractor."""
    console.print(f"\n📥 [cyan]Loading model from: {model_path}[/cyan]")

    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint.get('config', {})

    # Dynamically infer input feature dimension from checkpoint
    try:
        input_dim = checkpoint['model_state_dict']['input_projection.weight'].shape[1]
        console.print(f"[green]✅ Detected input feature dimension from checkpoint:[/green] {input_dim}")
    except Exception:
        input_dim = config.get('feature_dim', 500)
        console.print(f"[yellow]⚠️ Could not infer feature_dim from checkpoint; using config/default:[/yellow] {input_dim}")

    # Lazy import to avoid circular imports
    from models.architecture import NeuralTradingModel  # noqa: E402

    model = NeuralTradingModel(
        feature_dim=input_dim,
        d_model=config.get('d_model', 512),
        num_heads=config.get('num_heads', 16),
        num_layers=config.get('num_layers', 8),
        d_ff=config.get('d_ff', 2048),
        dropout=0.0,
        latent_dim=config.get('latent_dim', 16),
        seq_len=config.get('seq_len', 100),
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    console.print("✅ [green]Model loaded successfully[/green]")

    # Load feature extractor
    console.print(f"\n📥 [cyan]Loading feature extractor from: {feature_extractor_path}[/cyan]")
    with open(feature_extractor_path, 'rb') as f:
        feature_extractor = pickle.load(f)
    console.print("✅ [green]Feature extractor loaded successfully[/green]")

    return model, feature_extractor, config


def load_test_data(data_path: str = 'neural_data/BTC_1h_2017-01-01_2024-12-31_neural_data.parquet'):
    """Load Polars DataFrame for test data."""
    import polars as pl
    console.print(f"\n📥 [cyan]Loading test data from: {data_path}[/cyan]")
    df = pl.read_parquet(data_path)
    console.print(f"✅ [green]Loaded {len(df):,} bars[/green]")
    return df


def print_model_summary(model, config):
    """Pretty-print model architecture and parameter counts."""
    console.print("\n" + "=" * 80)
    console.print(Panel.fit("[bold cyan]MODEL ARCHITECTURE[/bold cyan]", border_style="cyan"))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", justify="right", style="yellow")
    table.add_row("Feature Dimension", str(config.get('feature_dim', 'N/A')))
    table.add_row("Model Dimension", str(config.get('d_model', 512)))
    table.add_row("Attention Heads", str(config.get('num_heads', 16)))
    table.add_row("Transformer Layers", str(config.get('num_layers', 8)))
    table.add_row("Feed-Forward Dim", str(config.get('d_ff', 2048)))
    table.add_row("Latent Dimension", str(config.get('latent_dim', 16)))
    table.add_row("Sequence Length", str(config.get('seq_len', 100)))
    console.print(table)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"\n📊 [bold]Total Parameters:[/bold] {total_params:,}")
    console.print(f"📊 [bold]Trainable Parameters:[/bold] {trainable_params:,}")
    console.print("=" * 80)


class LazySequenceDataset:
    """Memory-efficient sequence generator - loads on demand"""
    def __init__(self, features, returns, seq_len):
        self.features = features
        self.returns = returns
        self.seq_len = seq_len
        self.num_sequences = len(features) - seq_len
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        """Generate sequence on-the-fly"""
        # Handle slicing
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            return self.get_batch(list(indices))
        
        # Handle single index
        seq = self.features[idx:idx + self.seq_len]
        ret = self.returns[idx]
        return seq, ret
    
    def get_batch(self, indices):
        """Get multiple sequences at once"""
        sequences = np.stack([self.features[i:i+self.seq_len] for i in indices])
        returns = self.returns[indices]
        return sequences, returns

def prepare_test_sequences_from_cache(cache_path: str, config: dict):
    """Load pre-computed features and create lazy dataset."""
    import polars as pl
    import pickle
    import os

    console.print(f"\n📥 [cyan]Loading cached features from: {cache_path}[/cyan]")

    # Auto-detect file type
    with open(cache_path, "rb") as f:
        header = f.read(8)

    if header.startswith(b"ARROW1") or header.startswith(b"PAR1"):
        console.print("[cyan]📦 Detected Arrow/Parquet format — loading with Polars[/cyan]")
        df = pl.read_ipc(cache_path)
        if "returns" in df.columns:
            returns = df["returns"].to_numpy()
            features = df.drop("returns").to_numpy()

            # 🩹 Patch for 9867 vs 9868 mismatch
            if features.shape[1] + 1 == 9868:
                console.print("[yellow]⚠️ Adding dummy feature column to match model input dimension.[/yellow]")
                pad = np.zeros((features.shape[0], 1), dtype=features.dtype)
                features = np.hstack([features, pad])
        else:
            returns = np.zeros(len(df))
            features = df.to_numpy()

    elif header.startswith(b"\x80") or header.startswith(b"\x81"):
        console.print("[yellow]📦 Detected Pickle format — loading with pickle[/yellow]")
        with open(cache_path, "rb") as f:
            cached_data = pickle.load(f)
        features = cached_data["features"]
        returns = cached_data["returns"]

    else:
        raise ValueError(f"❌ Unknown cache format: starts with {header!r}")

    console.print(f"✅ [green]Loaded {len(features):,} features[/green]")
    console.print(f"   Feature dimension: {features.shape[1]}")

    seq_len = int(config.get("seq_len", 100))
    if len(features) <= seq_len:
        console.print("[red]Not enough data for sequences[/red]")
        return None, None

    # Create lazy dataset instead of materializing all sequences
    dataset = LazySequenceDataset(features, returns, seq_len)
    console.print(f"✅ [green]Created lazy dataset with {len(dataset):,} sequences[/green]")
    console.print(f"   Memory usage: ~{features.nbytes / 1e9:.2f} GB (features only)")
    console.print(f"   vs {len(dataset) * seq_len * features.shape[1] * 4 / 1e9:.2f} GB if materialized")

    return dataset, returns[:len(dataset)]


def analyze_attention_patterns(analyzer, test_dataset, num_samples=5):
    console.print("\n" + "=" * 80)
    console.print(Panel.fit("[bold cyan]ATTENTION PATTERN ANALYSIS[/bold cyan]", border_style="cyan"))

    n = min(num_samples, len(test_dataset))
    indices = np.random.choice(len(test_dataset), n, replace=False)
    
    for i, idx in enumerate(indices):
        console.print(f"\n🔍 [yellow]Analyzing sample {i + 1}/{n} (index {idx})...[/yellow]")
        sample_seq, _ = test_dataset[idx]  # Load only this one sequence
        
        attention_weights, predictions = analyzer.extract_attention_weights(sample_seq)
        
        console.print(f"   Layers: {len(attention_weights)}")
        p_entry = torch.sigmoid(predictions['entry_prob']).item() if predictions['entry_prob'].ndim == 0 else torch.sigmoid(predictions['entry_prob']).squeeze().item()
        console.print(f"   Entry Probability: {p_entry:.3f}")
        console.print(f"   Expected Return: {predictions['expected_return'].item():.4f}")

        if i == 0:
            console.print("\n📊 [cyan]Generating attention heatmap...[/cyan]")
            analyzer.plot_attention_heatmap(attention_weights[-1], layer_idx=-1, head_idx=0)

            console.print("\n📊 [cyan]Generating attention timeline...[/cyan]")
            analyzer.plot_attention_timeline(attention_weights)


def analyze_feature_importance(analyzer, test_dataset, num_samples=100, feature_dim=None):
    console.print("\n" + "=" * 80)
    console.print(Panel.fit("[bold cyan]FEATURE IMPORTANCE ANALYSIS[/bold cyan]", border_style="cyan"))

    n = min(num_samples, len(test_dataset))
    indices = np.random.choice(len(test_dataset), n, replace=False)
    
    console.print(f"\n🔍 [yellow]Computing feature importance on {n} samples...[/yellow]")
    
    # Get batch of sequences
    test_sequences, _ = test_dataset.get_batch(indices)
    
    importance_vec, feature_names = analyzer.compute_feature_importance(
        test_sequences, num_samples=n, feature_dim=feature_dim, return_names=True
    )

    # Build dict for display
    importance_dict = dict(zip(feature_names, importance_vec.tolist()))
    # Sort for top-N display
    sorted_items = sorted(importance_dict.items(), key=lambda x: -x[1])

    console.print("\n📊 [bold]Top 20 Most Important Features:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("Feature", style="yellow")
    table.add_column("Importance", justify="right", style="green")
    for rank, (feat, imp) in enumerate(sorted_items[:20], 1):
        table.add_row(str(rank), feat, f"{imp:.6f}")
    console.print(table)

    console.print("\n📊 [cyan]Generating feature importance plot...[/cyan]")
    analyzer.plot_feature_importance(importance_dict, top_n=30)


def analyze_regime_space(analyzer, test_dataset, test_returns):
    console.print("\n" + "=" * 80)
    console.print(Panel.fit("[bold cyan]MARKET REGIME SPACE ANALYSIS[/bold cyan]", border_style="cyan"))
    console.print("\n🔍 [yellow]Visualizing learned market regimes...[/yellow]")
    
    # Sample subset for visualization
    n_samples = min(1000, len(test_dataset))
    indices = np.linspace(0, len(test_dataset)-1, n_samples, dtype=int)
    test_sequences, sample_returns = test_dataset.get_batch(indices)
    
    analyzer.visualize_regime_space(test_sequences, sample_returns)


def analyze_decision_boundary(analyzer, test_dataset, test_returns):
    console.print("\n" + "=" * 80)
    console.print(Panel.fit("[bold cyan]DECISION BOUNDARY ANALYSIS[/bold cyan]", border_style="cyan"))
    console.print("\n🔍 [yellow]Analyzing decision boundary...[/yellow]")
    if len(test_dataset) == 0:
        console.print("⚠️  No sequences available for decision boundary analysis")
        return
    
    # Get a batch of up to 1000 sequences
    n_samples = min(1000, len(test_dataset))
    indices = list(range(n_samples))
    test_sequences, sample_returns = test_dataset.get_batch(indices)
    
    analyzer.analyze_decision_boundary(test_sequences, sample_returns)


def main():
    console.print(Panel.fit(
        "[bold cyan]NEURAL TRADING SYSTEM[/bold cyan]\n"
        "[yellow]Model Analysis & Visualization[/yellow]",
        title="🧠 Analysis Tool",
        border_style="cyan"
    ))

    # Configuration
    MODEL_PATH = 'models/best_model.pt'
    FEATURE_EXTRACTOR_PATH = glob.glob('models/*feature_extractor.pkl')[0]
    
    # Find the most recent feature cache
    cache_files = list(Path('neural_data/features').glob('features_*.pkl'))
    if not cache_files:
        console.print("[red]❌ No feature cache found[/red]")
        console.print("[yellow]💡 Run training first to generate cache[/yellow]")
        return
    
    FEATURE_CACHE_PATH = str(sorted(cache_files, key=lambda p: p.stat().st_mtime)[-1])

    # Check files
    if not Path(MODEL_PATH).exists():
        console.print(f"[red]❌ Model not found: {MODEL_PATH}[/red]")
        return
    if not Path(FEATURE_EXTRACTOR_PATH).exists():
        console.print(f"[red]❌ Feature extractor not found: {FEATURE_EXTRACTOR_PATH}[/red]")
        return

    # Load model/extractor
    model, feature_extractor, config = load_model_and_data(MODEL_PATH, FEATURE_EXTRACTOR_PATH)
    print_model_summary(model, config)

    # Load sequences from cache (INSTANT!)
    test_dataset, test_returns = prepare_test_sequences_from_cache(
        FEATURE_CACHE_PATH, 
        config
    )
    
    if test_dataset is None:
        console.print("[red]❌ Failed to load test dataset[/red]")
        return

    # Initialize analyzer
    console.print("\n🔧 [cyan]Initializing analyzer...[/cyan]")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    analyzer = AttentionAnalyzer(model, feature_extractor, device=device)
    model.to(device)
    console.print("✅ [green]Analyzer ready[/green]")

    # Run analyses
    try:
        analyze_attention_patterns(analyzer, test_dataset, num_samples=3)
        analyze_feature_importance(analyzer, test_dataset, num_samples=100, feature_dim=config.get('feature_dim'))
        analyze_regime_space(analyzer, test_dataset, test_returns)
        analyze_decision_boundary(analyzer, test_dataset, test_returns)
    except Exception as e:
        console.print(f"\n[red]❌ Analysis failed: {e}[/red]")
        import traceback
        traceback.print_exc()

    console.print("\n" + "=" * 80)
    console.print(Panel.fit("[bold green]✅ ANALYSIS COMPLETE![/bold green]", border_style="green"))


if __name__ == '__main__':
    main()