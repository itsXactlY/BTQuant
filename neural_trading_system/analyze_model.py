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
    import polars as pl
    from tqdm import tqdm
    console.print("\n🔧 [cyan]Preparing test sequences (Polars-only)...[/cyan]")

    seq_len = int(config.get('seq_len', 100))
    horizon = int(config.get('prediction_horizon', 5))
    feature_dim_expected = int(config.get('feature_dim'))  # enforce training width [web:94]

    # Vectorized forward returns
    df = df.with_columns([
        ((pl.col('close').shift(-horizon) - pl.col('close')) / pl.col('close')).alias('forward_return')
    ])

    ohlcv_cols = ['bar','datetime','open','high','low','close','volume']
    indicator_cols = [c for c in df.columns if c not in ohlcv_cols]

    from polars import selectors as cs
    df = df.with_columns(cs.numeric().fill_null(0))

    arrs = {c: df.get_column(c).to_numpy() for c in indicator_cols}

    features_list = []
    total_rows = df.height
    valid_rows = total_rows - horizon

    for i in tqdm(range(seq_len, valid_rows), desc="Extracting features"):
        current_data = {k: arrs[k][:i] for k in indicator_cols}  # match training boundary [web:94]
        try:
            feats = feature_extractor.extract_all_features(current_data)
            f = np.asarray(feats, dtype=np.float32).ravel()
            # Enforce the training feature width BEFORE transform
            if f.size != feature_dim_expected:
                if f.size < feature_dim_expected:
                    f = np.pad(f, (0, feature_dim_expected - f.size))
                else:
                    f = f[:feature_dim_expected]
            f = feature_extractor.transform(f)  # same scaler as training [web:94]
            features_list.append(f)
        except Exception:
            features_list.append(np.zeros(feature_dim_expected, dtype=np.float32))

    if not features_list:
        return np.empty((0, seq_len, feature_dim_expected), np.float32), np.empty((0,), np.float32)

    features = np.vstack(features_list).astype(np.float32)

    if len(features) <= seq_len:
        return np.empty((0, seq_len, feature_dim_expected), np.float32), np.empty((0,), np.float32)

    sequences = np.stack([features[i:i+seq_len] for i in range(len(features) - seq_len)], axis=0)

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
    MODEL_PATH = 'models/best_model.pt'
    FEATURE_EXTRACTOR_PATH = 'models/neural_BTC_4h_2020-01-01_2020-03-01_feature_extractor.pkl' # TODO :: remove the timeframe etc. or viceversa the loader
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
