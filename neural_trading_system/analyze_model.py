#!/usr/bin/env python3
"""
Analyze trained neural trading model.
Visualize attention patterns, feature importance, and learned regimes.
"""

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

    # Lazy import to avoid circular imports
    from models.architecture import NeuralTradingModel  # noqa: E402

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
    console.print("✅ [green]Model loaded successfully[/green]")

    console.print(f"\n📥 [cyan]Loading feature extractor from: {feature_extractor_path}[/cyan]")
    with open(feature_extractor_path, 'rb') as f:
        feature_extractor = pickle.load(f)
    console.print("✅ [green]Feature extractor loaded successfully[/green]")

    return model, feature_extractor, config


def load_test_data(data_path: str = 'neural_data/BTC_4h_neural_data.parquet'):
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
    table.add_row("Model Dimension", str(config.get('d_model', 256)))
    table.add_row("Attention Heads", str(config.get('num_heads', 8)))
    table.add_row("Transformer Layers", str(config.get('num_layers', 6)))
    table.add_row("Feed-Forward Dim", str(config.get('d_ff', 1024)))
    table.add_row("Latent Dimension", str(config.get('latent_dim', 8)))
    table.add_row("Sequence Length", str(config.get('seq_len', 100)))
    console.print(table)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"\n📊 [bold]Total Parameters:[/bold] {total_params:,}")
    console.print(f"📊 [bold]Trainable Parameters:[/bold] {trainable_params:,}")
    console.print("=" * 80)


def prepare_test_sequences(df, feature_extractor, config):
    """Prepare test sequences (Polars-only) consistent with training."""
    console.print("\n🔧 [cyan]Preparing test sequences (Polars-only)...[/cyan]")
    import polars as pl
    from tqdm import tqdm

    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'close' column")

    seq_len = int(config.get('seq_len', 100))
    horizon = int(config.get('prediction_horizon', 5))

    # Compute forward returns vectorized
    df = df.with_columns([
        ((pl.col('close').shift(-horizon) - pl.col('close')) / pl.col('close'))
        .alias('forward_return')
    ])

    ohlcv_cols = ['bar', 'datetime', 'open', 'high', 'low', 'close', 'volume']
    indicator_cols = [c for c in df.columns if c not in ohlcv_cols]

    # Fill nulls on numeric columns
    from polars import selectors as cs
    df = df.with_columns(cs.numeric().fill_null(0))

    # Build numpy arrays per indicator
    indicator_data_full = {c: df.get_column(c).to_numpy() for c in indicator_cols}

    features_list = []
    expected_dim = None

    total_rows = df.height
    valid_rows = total_rows - horizon  # exclude trailing horizon with null returns

    for i in tqdm(range(seq_len, valid_rows), desc="Extracting features"):
        current_data = {k: v[:i] for k, v in indicator_data_full.items()}
        try:
            feats = feature_extractor.extract_all_features(current_data)
            feats = np.asarray(feats, dtype=np.float32)
            if feats.ndim > 1:
                feats = feats.ravel()
            if expected_dim is None:
                expected_dim = feats.size
            if feats.size != expected_dim:
                if feats.size < expected_dim:
                    feats = np.pad(feats, (0, expected_dim - feats.size))
                else:
                    feats = feats[:expected_dim]
            # Apply same transform as training
            feats = feature_extractor.transform(feats)
            features_list.append(feats)
        except Exception:
            if expected_dim is not None:
                features_list.append(np.zeros(expected_dim, dtype=np.float32))
            else:
                continue

    if not features_list:
        console.print("⚠️  No per-bar features extracted; returning empty sequences")
        return np.empty((0, seq_len, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)

    features = np.vstack(features_list).astype(np.float32)

    if len(features) <= seq_len:
        console.print("⚠️  Not enough features to build at least one sequence")
        return np.empty((0, seq_len, features.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32)

    sequences = np.stack([features[i:i + seq_len] for i in range(len(features) - seq_len)], axis=0)

    # Align returns to sequence starts and clean NaNs
    rets = df.get_column('forward_return')[seq_len:seq_len + len(features)].to_numpy().copy()
    rets = rets[:len(sequences)]
    np.nan_to_num(rets, copy=False)

    console.print(f"✅ [green]Prepared {len(sequences):,} test sequences[/green]")
    return sequences, rets


def analyze_attention_patterns(analyzer, test_sequences, num_samples=5):
    console.print("\n" + "=" * 80)
    console.print(Panel.fit("[bold cyan]ATTENTION PATTERN ANALYSIS[/bold cyan]", border_style="cyan"))

    n = min(num_samples, len(test_sequences))
    for i in range(n):
        console.print(f"\n🔍 [yellow]Analyzing sample {i + 1}/{n}...[/yellow]")
        sample_seq = test_sequences[i]
        attention_weights, predictions = analyzer.extract_attention_weights(sample_seq)

        console.print(f"   Layers: {len(attention_weights)}")
        # If model outputs logits, convert to probability for readability
        p_entry = torch.sigmoid(predictions['entry_prob']).item() if predictions['entry_prob'].ndim == 0 else torch.sigmoid(predictions['entry_prob']).squeeze().item()
        console.print(f"   Entry Probability: {p_entry:.3f}")
        console.print(f"   Expected Return: {predictions['expected_return'].item():.4f}")

        if i == 0:
            console.print("\n📊 [cyan]Generating attention heatmap...[/cyan]")
            analyzer.plot_attention_heatmap(attention_weights[-1], layer_idx=-1, head_idx=0)

            console.print("\n📊 [cyan]Generating attention timeline...[/cyan]")
            analyzer.plot_attention_timeline(attention_weights)


def analyze_feature_importance(analyzer, test_sequences, num_samples=100, feature_dim=None):
    console.print("\n" + "=" * 80)
    console.print(Panel.fit("[bold cyan]FEATURE IMPORTANCE ANALYSIS[/bold cyan]", border_style="cyan"))

    n = min(num_samples, len(test_sequences))
    if n == 0:
        console.print("⚠️  No sequences available for feature importance")
        return

    console.print(f"\n🔍 [yellow]Computing feature importance on {n} samples...[/yellow]")
    importance = analyzer.compute_feature_importance(test_sequences, num_samples=n)

    # Normalize importance output to 1D vector
    importance = np.asarray(importance)
    if importance.ndim == 0:
        # Scalar → broadcast
        dim = int(feature_dim or 1)
        importance = np.full(dim, float(importance), dtype=np.float32)
    elif importance.ndim > 1:
        # Reduce to per-feature vector
        importance = importance.mean(axis=0)

    if feature_dim is not None and importance.shape[0] != int(feature_dim):
        # Pad or truncate to expected feature_dim
        dim = int(feature_dim)
        if importance.shape[0] < dim:
            importance = np.pad(importance, (0, dim - importance.shape[0]))
        else:
            importance = importance[:dim]

    feature_names = [f'feature_{i}' for i in range(int(importance.shape[0]))]
    importance_dict = dict(zip(feature_names, importance.tolist()))

    console.print("\n📊 [bold]Top 20 Most Important Features:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("Feature", style="yellow")
    table.add_column("Importance", justify="right", style="green")

    for rank, (feat, imp) in enumerate(sorted(importance_dict.items(), key=lambda x: -x[1])[:20], 1):
        table.add_row(str(rank), feat, f"{imp:.6f}")

    console.print(table)
    console.print("\n📊 [cyan]Generating feature importance plot...[/cyan]")
    analyzer.plot_feature_importance(importance_dict, top_n=30)


def analyze_regime_space(analyzer, test_sequences, test_returns):
    console.print("\n" + "=" * 80)
    console.print(Panel.fit("[bold cyan]MARKET REGIME SPACE ANALYSIS[/bold cyan]", border_style="cyan"))
    console.print("\n🔍 [yellow]Visualizing learned market regimes...[/yellow]")
    if len(test_sequences) == 0:
        console.print("⚠️  No sequences available for regime visualization")
        return
    analyzer.visualize_regime_space(test_sequences, test_returns)


def analyze_decision_boundary(analyzer, test_sequences, test_returns):
    console.print("\n" + "=" * 80)
    console.print(Panel.fit("[bold cyan]DECISION BOUNDARY ANALYSIS[/bold cyan]", border_style="cyan"))
    console.print("\n🔍 [yellow]Analyzing decision boundary...[/yellow]")
    if len(test_sequences) == 0:
        console.print("⚠️  No sequences available for decision boundary analysis")
        return
    analyzer.analyze_decision_boundary(test_sequences, test_returns)


def main():
    console.print(Panel.fit(
        "[bold cyan]NEURAL TRADING SYSTEM[/bold cyan]\n"
        "[yellow]Model Analysis & Visualization[/yellow]",
        title="🧠 Analysis Tool",
        border_style="cyan"
    ))

    # Configuration
    MODEL_PATH = 'best_model.pt'
    FEATURE_EXTRACTOR_PATH = 'best_model_feature_extractor.pkl'
    DATA_PATH = 'neural_data/BTC_4h_neural_data.parquet'

    # Check files
    if not Path(MODEL_PATH).exists():
        console.print(f"[red]❌ Model not found: {MODEL_PATH}[/red]")
        console.print("[yellow]💡 Run training first: python neural_pipeline.py[/yellow]")
        return
    if not Path(FEATURE_EXTRACTOR_PATH).exists():
        console.print(f"[red]❌ Feature extractor not found: {FEATURE_EXTRACTOR_PATH}[/red]")
        console.print("[yellow]💡 Ensure best_model_feature_extractor.pkl is saved during training[/yellow]")
        return

    # Load model/extractor
    model, feature_extractor, config = load_model_and_data(MODEL_PATH, FEATURE_EXTRACTOR_PATH)
    print_model_summary(model, config)

    # Load test data
    df = load_test_data(DATA_PATH)

    # Prepare test sequences (Polars-only and consistent)
    test_sequences, test_returns = prepare_test_sequences(df, feature_extractor, config)

    # Initialize analyzer
    console.print("\n🔧 [cyan]Initializing analyzer...[/cyan]")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    analyzer = AttentionAnalyzer(model, feature_extractor, device=device)
    model.to(device) # Move to available GPU:0
    console.print("✅ [green]Analyzer ready[/green]")

    # Run analyses
    try:
        analyze_attention_patterns(analyzer, test_sequences, num_samples=3)
        analyze_feature_importance(analyzer, test_sequences, num_samples=100, feature_dim=config.get('feature_dim'))
        analyze_regime_space(analyzer, test_sequences, test_returns)
        analyze_decision_boundary(analyzer, test_sequences, test_returns)
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Analysis interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Analysis failed: {e}[/red]")
        import traceback
        traceback.print_exc()

    console.print("\n" + "=" * 80)
    console.print(Panel.fit("[bold green]✅ ANALYSIS COMPLETE![/bold green]", border_style="green"))


if __name__ == '__main__':
    main()
