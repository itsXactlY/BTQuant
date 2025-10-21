#!/usr/bin/env python3
"""
Analyze trained neural trading model.
Visualize attention patterns, feature importance, and learned regimes.
"""

import torch
import numpy as np
import pickle
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Import analysis tools
import sys
sys.path.append('neural_trading_system')

from analysis.attention_viz import AttentionAnalyzer

console = Console()


def load_model_and_data(model_path: str, feature_extractor_path: str):
    """Load trained model and feature extractor."""
    
    console.print(f"\n📥 [cyan]Loading model from: {model_path}[/cyan]")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint.get('config', {})
    
    # Load model
    from models.architecture import NeuralTradingModel
    
    model = NeuralTradingModel(
        feature_dim=config.get('feature_dim', 500),
        d_model=config.get('d_model', 256),
        num_heads=config.get('num_heads', 8),
        num_layers=config.get('num_layers', 6),
        d_ff=config.get('d_ff', 1024),
        dropout=0.0,
        latent_dim=config.get('latent_dim', 8),
        seq_len=config.get('seq_len', 100)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    console.print("✅ [green]Model loaded successfully[/green]")
    
    # Load feature extractor
    console.print(f"\n📥 [cyan]Loading feature extractor from: {feature_extractor_path}[/cyan]")
    
    with open(feature_extractor_path, 'rb') as f:
        feature_extractor = pickle.load(f)
    
    console.print("✅ [green]Feature extractor loaded successfully[/green]")
    
    return model, feature_extractor, config


def load_test_data(data_path: str = 'neural_data/BTC_4h_neural_data.parquet'):
    """Load test data for analysis."""
    
    import polars as pl
    
    console.print(f"\n📥 [cyan]Loading test data from: {data_path}[/cyan]")
    
    df = pl.read_parquet(data_path)
    
    console.print(f"✅ [green]Loaded {len(df):,} bars[/green]")
    
    return df


def print_model_summary(model, config):
    """Print model architecture summary."""
    
    console.print("\n" + "="*80)
    console.print(Panel.fit(
        "[bold cyan]MODEL ARCHITECTURE[/bold cyan]",
        border_style="cyan"
    ))
    
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
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    console.print(f"\n📊 [bold]Total Parameters:[/bold] {total_params:,}")
    console.print(f"📊 [bold]Trainable Parameters:[/bold] {trainable_params:,}")
    console.print("="*80)


def analyze_attention_patterns(analyzer, test_sequences, num_samples=5):
    """Analyze and visualize attention patterns."""
    
    console.print("\n" + "="*80)
    console.print(Panel.fit(
        "[bold cyan]ATTENTION PATTERN ANALYSIS[/bold cyan]",
        border_style="cyan"
    ))
    
    for i in range(min(num_samples, len(test_sequences))):
        console.print(f"\n🔍 [yellow]Analyzing sample {i+1}/{num_samples}...[/yellow]")
        
        sample_seq = test_sequences[i]
        
        # Extract attention weights
        attention_weights, predictions = analyzer.extract_attention_weights(sample_seq)
        
        console.print(f"   Layers: {len(attention_weights)}")
        console.print(f"   Entry Probability: {predictions['entry_prob'].item():.3f}")
        console.print(f"   Expected Return: {predictions['expected_return'].item():.4f}")
        
        # Plot attention heatmap for last layer
        if i == 0:  # Only plot first sample to avoid too many plots
            console.print("\n📊 [cyan]Generating attention heatmap...[/cyan]")
            analyzer.plot_attention_heatmap(
                attention_weights[-1],  # Last layer
                layer_idx=-1,
                head_idx=0
            )
            
            console.print("\n📊 [cyan]Generating attention timeline...[/cyan]")
            analyzer.plot_attention_timeline(attention_weights)


def analyze_feature_importance(analyzer, test_sequences, num_samples=100):
    """Compute and visualize feature importance."""
    
    console.print("\n" + "="*80)
    console.print(Panel.fit(
        "[bold cyan]FEATURE IMPORTANCE ANALYSIS[/bold cyan]",
        border_style="cyan"
    ))
    
    console.print(f"\n🔍 [yellow]Computing feature importance on {num_samples} samples...[/yellow]")
    
    importance_dict = analyzer.compute_feature_importance(
        test_sequences,
        num_samples=num_samples
    )
    
    # Print top features
    console.print("\n📊 [bold]Top 20 Most Important Features:[/bold]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("Feature", style="yellow")
    table.add_column("Importance", justify="right", style="green")
    
    for rank, (feature, importance) in enumerate(list(importance_dict.items())[:20], 1):
        table.add_row(str(rank), feature, f"{importance:.6f}")
    
    console.print(table)
    
    # Plot
    console.print("\n📊 [cyan]Generating feature importance plot...[/cyan]")
    analyzer.plot_feature_importance(importance_dict, top_n=30)


def analyze_regime_space(analyzer, test_sequences, test_returns):
    """Visualize learned market regime space."""
    
    console.print("\n" + "="*80)
    console.print(Panel.fit(
        "[bold cyan]MARKET REGIME SPACE ANALYSIS[/bold cyan]",
        border_style="cyan"
    ))
    
    console.print("\n🔍 [yellow]Visualizing learned market regimes...[/yellow]")
    
    analyzer.visualize_regime_space(test_sequences, test_returns)


def analyze_decision_boundary(analyzer, test_sequences, test_returns):
    """Analyze model's decision boundary."""
    
    console.print("\n" + "="*80)
    console.print(Panel.fit(
        "[bold cyan]DECISION BOUNDARY ANALYSIS[/bold cyan]",
        border_style="cyan"
    ))
    
    console.print("\n🔍 [yellow]Analyzing decision boundary...[/yellow]")
    
    analyzer.analyze_decision_boundary(test_sequences, test_returns)


def prepare_test_sequences(df, feature_extractor, config):
    """Prepare test sequences from dataframe."""
    
    console.print("\n🔧 [cyan]Preparing test sequences...[/cyan]")
    
    # Convert to pandas
    df_pd = df.to_pandas()
    
    # Build indicator data
    indicator_data_full = {}
    ohlcv_cols = ['bar', 'datetime', 'open', 'high', 'low', 'close', 'volume']
    
    for col in df_pd.columns:
        if col not in ['bar', 'datetime']:
            indicator_data_full[col] = df_pd[col].fillna(0).values
    
    # Extract features
    seq_len = config.get('seq_len', 100)
    features_list = []
    returns = []
    
    close_prices = df_pd['close'].values
    
    from tqdm import tqdm
    
    for i in tqdm(range(seq_len, len(df_pd)), desc="Extracting features"):
        current_data = {key: values[:i+1] for key, values in indicator_data_full.items()}
        
        try:
            features = feature_extractor.extract_all_features(current_data)
            features = feature_extractor.transform(features)
            features_list.append(features)
            
            # Calculate return
            if i + 5 < len(close_prices):
                ret = (close_prices[i + 5] - close_prices[i]) / close_prices[i]
                returns.append(ret)
            else:
                returns.append(0.0)
        except:
            continue
    
    features = np.array(features_list, dtype=np.float32)
    returns = np.array(returns, dtype=np.float32)
    
    # Create sequences
    sequences = []
    for i in range(len(features) - seq_len):
        sequences.append(features[i:i+seq_len])
    
    sequences = np.array(sequences, dtype=np.float32)
    returns = returns[:len(sequences)]
    
    console.print(f"✅ [green]Prepared {len(sequences):,} test sequences[/green]")
    
    return sequences, returns


def main():
    """Main analysis pipeline."""
    
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
    
    # Check if files exist
    if not Path(MODEL_PATH).exists():
        console.print(f"[red]❌ Model not found: {MODEL_PATH}[/red]")
        console.print("[yellow]💡 Run training first: python neural_pipeline.py[/yellow]")
        return
    
    # Load model
    model, feature_extractor, config = load_model_and_data(
        MODEL_PATH,
        FEATURE_EXTRACTOR_PATH
    )
    
    # Print model summary
    print_model_summary(model, config)
    
    # Load test data
    df = load_test_data(DATA_PATH)
    
    # Prepare test sequences
    test_sequences, test_returns = prepare_test_sequences(df, feature_extractor, config)
    
    # Initialize analyzer
    console.print("\n🔧 [cyan]Initializing analyzer...[/cyan]")
    analyzer = AttentionAnalyzer(model, feature_extractor, device='cuda' if torch.cuda.is_available() else 'cpu')
    console.print("✅ [green]Analyzer ready[/green]")
    
    # Run analyses
    try:
        # 1. Attention patterns
        analyze_attention_patterns(analyzer, test_sequences, num_samples=3)
        
        # 2. Feature importance
        analyze_feature_importance(analyzer, test_sequences, num_samples=100)
        
        # 3. Regime space
        analyze_regime_space(analyzer, test_sequences, test_returns)
        
        # 4. Decision boundary
        analyze_decision_boundary(analyzer, test_sequences, test_returns)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Analysis interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Analysis failed: {e}[/red]")
        import traceback
        traceback.print_exc()
    
    console.print("\n" + "="*80)
    console.print(Panel.fit(
        "[bold green]✅ ANALYSIS COMPLETE![/bold green]",
        border_style="green"
    ))


if __name__ == '__main__':
    main()