#!/usr/bin/env python3
"""
Analyze trained neural trading model.
Visualize attention patterns, feature importance, regimes, and prediction diagnostics.
"""

from __future__ import annotations

import glob
import io
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Ensure package resolutio

# Analysis tools
from analysis.attention_viz import AttentionAnalyzer  # noqa: E402

console = Console()
OUT_DIR = Path("analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# File discovery
# =============================================================================

def _latest(path_glob: str) -> Optional[str]:
    files = [Path(p) for p in glob.glob(path_glob)]
    return str(max(files, key=lambda p: p.stat().st_mtime)) if files else None


def discover_model_and_extractor(
    model_path: Optional[str] = None,
    feature_extractor_path: Optional[str] = None
) -> Tuple[str, str]:
    if not model_path:
        # prefer exit-aware naming, else any .pt
        model_path = _latest("models/*exit*aware*.pt") or _latest("models/*.pt")
    if not feature_extractor_path:
        feature_extractor_path = _latest("models/*feature_extractor.pkl")

    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError("❌ Could not find a model checkpoint under models/*.pt")
    if not feature_extractor_path or not Path(feature_extractor_path).exists():
        raise FileNotFoundError("❌ Could not find a feature extractor under models/*feature_extractor.pkl")

    return model_path, feature_extractor_path


def discover_feature_cache(cache_path: Optional[str] = None) -> str:
    if cache_path and Path(cache_path).exists():
        return cache_path
    latest = _latest("neural_data/features/features_*.pkl")
    if not latest:
        raise FileNotFoundError("❌ No feature cache found under neural_data/features/features_*.pkl")
    return latest


# =============================================================================
# Model + extractor
# =============================================================================

def load_model_and_data(model_path: str, feature_extractor_path: str):
    """Load trained model checkpoint and feature extractor."""
    console.print(f"\n📥 [cyan]Loading model from: {model_path}[/cyan]")
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint.get('config', {})

    # Infer input feature dimension from checkpoint
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


# =============================================================================
# Dataset utilities — one cached row == one flattened window [F]
# =============================================================================

class FlatWindowDataset:
    """Each row in `features` is already a flattened seq window (shape = [F])."""
    def __init__(self, features: np.ndarray, returns: np.ndarray, timestamps: Optional[np.ndarray] = None):
        self.features = features  # [N, F]
        self.returns = returns    # [N]
        self.ts = timestamps if (timestamps is not None and len(timestamps) == len(features)) else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.returns[idx]
        t = self.ts[idx] if self.ts is not None else None
        return x, y, t

    def get_batch(self, indices: List[int]):
        X = self.features[indices]
        y = self.returns[indices]
        ts = self.ts[indices] if self.ts is not None else None
        return X, y, ts


def _read_bytes_head(path: str, n: int = 8) -> bytes:
    with open(path, "rb") as f:
        return f.read(n)


def load_cached_features(cache_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load pickled cache produced by your pipeline (dict with features, returns, [timestamps]).
    Fallbacks for Arrow/Parquet IPC if needed.
    """
    header = _read_bytes_head(cache_path, 8)

    # Arrow/IPC or Parquet
    if header.startswith(b"ARROW1") or header.startswith(b"PAR1"):
        import polars as pl
        console.print("[cyan]📦 Detected Arrow/Parquet format — loading with Polars[/cyan]")
        df = pl.read_ipc(cache_path) if header.startswith(b"ARROW1") else pl.read_parquet(cache_path)
        if "returns" in df.columns:
            returns = df["returns"].to_numpy()
            features = df.drop("returns").to_numpy()
        else:
            returns = np.zeros(len(df))
            features = df.to_numpy()
        timestamps = df["datetime"].to_numpy() if "datetime" in df.columns else None
        return features.astype(np.float32), returns.astype(np.float32), timestamps

    # Pickle cache
    console.print("[yellow]📦 Detected Pickle format — loading with pickle[/yellow]")
    with open(cache_path, "rb") as f:
        cached = pickle.load(f)
    features = cached["features"].astype(np.float32)
    returns = cached["returns"].astype(np.float32)
    timestamps = cached.get("timestamps", None)
    if timestamps is not None:
        timestamps = np.asarray(timestamps)

    console.print(f"✅ [green]Loaded {len(features):,} features[/green]")
    console.print(f"   Feature dimension: {features.shape[1]}")
    return features, returns, timestamps


def prepare_dataset_from_cache(cache_path: str, config: dict) -> Tuple[FlatWindowDataset, np.ndarray]:
    features, returns, timestamps = load_cached_features(cache_path)

    # No need to chop for seq_len; each row is a flattened window
    dataset = FlatWindowDataset(features, returns, timestamps)
    console.print(f"✅ [green]Created dataset with {len(dataset):,} samples[/green]")
    console.print(f"   Memory usage: ~{features.nbytes / 1e9:.2f} GB (features only)")
    return dataset, returns


# =============================================================================
# Analyses (all save to OUT_DIR)
# =============================================================================

def analyze_attention_patterns(analyzer: AttentionAnalyzer, ds: FlatWindowDataset, num_samples: int = 5):
    console.print("\n" + "=" * 80)
    console.print(Panel.fit("[bold cyan]ATTENTION PATTERN ANALYSIS[/bold cyan]", border_style="cyan"))

    n = min(num_samples, len(ds))
    indices = np.random.choice(len(ds), n, replace=False)

    for i, idx in enumerate(indices):
        console.print(f"\n🔍 [yellow]Analyzing sample {i + 1}/{n} (index {idx})...[/yellow]")
        x_vec, y, _ = ds[idx]  # one flattened window [F]
        attn_list, preds = analyzer.extract_attention_weights(x_vec)

        p_entry = preds.get("entry_prob", None)
        if p_entry is not None:
            console.print(f"   Entry Probability: {float(p_entry):.3f}")
        exp_ret = preds.get("expected_return", None)
        if exp_ret is not None:
            console.print(f"   Expected Return: {float(exp_ret):+.6f}  (~{float(exp_ret)*100:+.4f}%)")

        if isinstance(attn_list, (list, tuple)) and len(attn_list) > 0:
            console.print(f"   Layers captured: {len(attn_list)}")
            heat_path = OUT_DIR / f"attn_heatmap_{idx}.png"
            tl_path   = OUT_DIR / f"attn_timeline_{idx}.png"
            analyzer.plot_attention_heatmap(attn_list[-1], layer_idx=-1, head_idx=0, save_path=str(heat_path))
            analyzer.plot_attention_timeline(attn_list, save_path=str(tl_path))
            console.print(f"📊 Saved attention heatmap: [green]{heat_path}[/green]")
            console.print(f"📊 Saved attention timeline: [green]{tl_path}[/green]")
        else:
            console.print("[yellow]⚠️ No attention weights available — hooks/return_attn may be off[/yellow]")


def analyze_feature_importance(analyzer: AttentionAnalyzer, ds: FlatWindowDataset, num_samples: int = 100, feature_dim: Optional[int] = None):
    console.print("\n" + "=" * 80)
    console.print(Panel.fit("[bold cyan]FEATURE IMPORTANCE ANALYSIS[/bold cyan]", border_style="cyan"))

    n = min(num_samples, len(ds))
    indices = np.random.choice(len(ds), n, replace=False)

    console.print(f"\n🔍 [yellow]Computing feature importance on {n} samples...[/yellow]")
    X, _, _ = ds.get_batch(indices)  # [B, F]

    importance_vec, feature_names = analyzer.compute_feature_importance(
        X, num_samples=n, feature_dim=feature_dim, return_names=True
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

    fi_path = OUT_DIR / "feature_importance_top30.png"
    analyzer.plot_feature_importance(importance_dict, top_n=30, save_path=str(fi_path))
    console.print(f"📁 Saved: [green]{fi_path}[/green]")


def analyze_regime_space(analyzer: AttentionAnalyzer, ds: FlatWindowDataset, returns_all: np.ndarray):
    console.print("\n" + "=" * 80)
    console.print(Panel.fit("[bold cyan]MARKET REGIME SPACE ANALYSIS[/bold cyan]", border_style="cyan"))
    console.print("\n🔍 [yellow]Visualizing learned market regimes...[/yellow]")

    # Sample subset for visualization
    n_samples = min(1000, len(ds))
    indices = np.linspace(0, len(ds)-1, n_samples, dtype=int)
    X, y, _ = ds.get_batch(list(indices))

    # Use analyzer's helper (now accepts save_path)
    save_path = OUT_DIR / "regime_space.png"
    analyzer.visualize_regime_space(X, y, max_points=n_samples, save_path=str(save_path))
    console.print(f"📁 Saved: [green]{save_path}[/green]")


def analyze_decision_boundary(analyzer: AttentionAnalyzer, ds: FlatWindowDataset, returns_all: np.ndarray):
    console.print("\n" + "=" * 80)
    console.print(Panel.fit("[bold cyan]DECISION BOUNDARY ANALYSIS[/bold cyan]", border_style="cyan"))
    console.print("\n🔍 [yellow]Analyzing decision boundary...[/yellow]")

    # ----- Gather a slice
    n_samples = min(2000, len(ds))
    indices = list(range(n_samples))
    X, y, _ = ds.get_batch(indices)  # X:[B,F], y:[B]

    # ----- Predict probabilities
    probs = analyzer.predict_entry_proba(X)  # np.array [B]
    if probs is None or len(probs) == 0 or np.all(np.isnan(probs)):
        console.print("[red]No predictions produced[/red]")
        return

    # ----- Adaptive thresholds, like before
    valid_mask = ~np.isnan(probs)
    p = probs[valid_mask]
    r = y[valid_mask]
    if p.size == 0:
        console.print("[red]All probabilities are NaN[/red]")
        return

    hi_thr, lo_thr = 0.7, 0.3
    if (p >= hi_thr).sum() == 0:
        hi_thr = float(np.quantile(p, 0.90))
    if (p <= lo_thr).sum() == 0:
        lo_thr = float(np.quantile(p, 0.10))

    hi_idx = p >= hi_thr
    lo_idx = p <= lo_thr

    # ----- Stats helper
    def stats(mask: np.ndarray):
        if mask.sum() == 0:
            return 0, float("nan"), float("nan"), float("nan")
        sub = r[mask]
        wins = (sub > 0).sum()
        winrate = 100.0 * wins / mask.sum()
        mean = float(sub.mean())
        std = float(sub.std(ddof=1)) if mask.sum() > 1 else 0.0
        se = std / np.sqrt(max(1, mask.sum()))
        return int(mask.sum()), winrate, mean, se

    n_hi, wr_hi, mu_hi, se_hi = stats(hi_idx)
    n_lo, wr_lo, mu_lo, se_lo = stats(lo_idx)

    console.print("\n📊 [bold]Decision Boundary Analysis:[/bold]")
    console.print(f"High confidence (≥{hi_thr:.4f}) samples: {n_hi}")
    console.print(f"  - Win rate: {wr_hi:.2f}%")
    console.print(f"  - Avg return: {mu_hi:+.6f}  (~{mu_hi*100:+.4f}%)")
    console.print(f"\nLow confidence (≤{lo_thr:.4f}) samples: {n_lo}")
    console.print(f"  - Win rate: {wr_lo:.2f}%")
    console.print(f"  - Avg return: {mu_lo:+.6f}  (~{mu_lo*100:+.4f}%)")

    # =========================
    # PLOT 1: Return histograms
    # =========================
    if n_hi > 0 or n_lo > 0:
        hi_r = r[hi_idx] if n_hi > 0 else np.array([])
        lo_r = r[lo_idx] if n_lo > 0 else np.array([])

        both = np.concatenate([hi_r, lo_r]) if (n_hi > 0 and n_lo > 0) else (hi_r if n_hi > 0 else lo_r)
        if both.size > 0:
            low_q, high_q = np.quantile(both, [0.01, 0.99])
            bins = np.linspace(low_q, high_q, 50)
        else:
            bins = 40

        plt.figure(figsize=(9, 5))
        if n_hi > 0:
            plt.hist(hi_r * 100.0, bins=bins * 100.0 if isinstance(bins, np.ndarray) else bins,
                     alpha=0.6, label=f"High ≥{hi_thr:.3f}")
        if n_lo > 0:
            plt.hist(lo_r * 100.0, bins=bins * 100.0 if isinstance(bins, np.ndarray) else bins,
                     alpha=0.6, label=f"Low ≤{lo_thr:.3f}")
        plt.title("Future returns (%) — High vs Low confidence")
        plt.xlabel("Future return (%)")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        hpath = OUT_DIR / "decision_boundary_return_hist.png"
        plt.savefig(hpath, bbox_inches="tight"); plt.close()
        console.print(f"📁 Saved: [green]{hpath}[/green]")

    # =========================
    # PLOT 2: Boxplot (percent)
    # =========================
    if n_hi > 0 and n_lo > 0:
        plt.figure(figsize=(7, 5))
        plt.boxplot([lo_r * 100.0, hi_r * 100.0],
                    labels=[f"Low ≤{lo_thr:.3f} (n={n_lo})", f"High ≥{hi_thr:.3f} (n={n_hi})"])
        plt.title("Future return (%) by confidence group")
        plt.ylabel("Future return (%)")
        plt.tight_layout()
        bpath = OUT_DIR / "decision_boundary_boxplot.png"
        plt.savefig(bpath, bbox_inches="tight"); plt.close()
        console.print(f"📁 Saved: [green]{bpath}[/green]")

    # ====================================
    # PLOT 3: Win rate bar chart (percent)
    # ====================================
    groups = []
    wrs = []
    counts = []
    if n_lo > 0:
        groups.append(f"Low ≤{lo_thr:.3f}")
        wrs.append(wr_lo)
        counts.append(n_lo)
    if n_hi > 0:
        groups.append(f"High ≥{hi_thr:.3f}")
        wrs.append(wr_hi)
        counts.append(n_hi)

    if wrs:
        plt.figure(figsize=(7, 5))
        x = np.arange(len(wrs))
        plt.bar(x, wrs)
        for xi, (wr, c) in enumerate(zip(wrs, counts)):
            plt.text(xi, wr + 0.5, f"n={c}", ha="center", va="bottom", fontsize=9)
        plt.xticks(x, groups, rotation=0)
        plt.ylabel("Win rate (%)")
        plt.title("Win rate by confidence group")
        plt.ylim(0, max(100, max(wrs) + 5))
        plt.tight_layout()
        wpath = OUT_DIR / "decision_boundary_winrate.png"
        plt.savefig(wpath, bbox_inches="tight"); plt.close()
        console.print(f"📁 Saved: [green]{wpath}[/green]")

    # ==========================================
    # PLOT 4: Mean return with 95% CI (percent)
    # ==========================================
    mu = []
    ci = []
    g2 = []
    if n_lo > 0:
        mu.append(mu_lo * 100.0)
        ci.append(1.96 * se_lo * 100.0)
        g2.append(f"Low ≤{lo_thr:.3f}")
    if n_hi > 0:
        mu.append(mu_hi * 100.0)
        ci.append(1.96 * se_hi * 100.0)
        g2.append(f"High ≥{hi_thr:.3f}")

    if mu:
        plt.figure(figsize=(7, 5))
        x = np.arange(len(mu))
        plt.bar(x, mu, yerr=ci, capsize=5)
        plt.xticks(x, g2, rotation=0)
        plt.ylabel("Mean future return (%)")
        plt.title("Mean future return (95% CI) by confidence group")
        plt.tight_layout()
        mpath = OUT_DIR / "decision_boundary_mean_return.png"
        plt.savefig(mpath, bbox_inches="tight"); plt.close()
        console.print(f"📁 Saved: [green]{mpath}[/green]")

    # ==========================
    # PLOT 5: Lift curve (TOP-k)
    # ==========================
    order = np.argsort(-p)  # descending by probability
    r_sorted = r[order]
    if r_sorted.size > 10:
        cum_avg = np.cumsum(r_sorted) / np.arange(1, r_sorted.size + 1)
        xpct = np.arange(1, r_sorted.size + 1) / r_sorted.size * 100.0
        baseline = float(r.mean()) * 100.0

        plt.figure(figsize=(8, 5))
        plt.plot(xpct, cum_avg * 100.0, label="Cumulative avg return (sorted by p)")
        plt.axhline(baseline, ls="--", label=f"Baseline avg = {baseline:.4f}%")
        plt.xlabel("Top x% by p(entry)")
        plt.ylabel("Cumulative avg future return (%)")
        plt.title("Lift curve")
        plt.legend()
        plt.tight_layout()
        lpath = OUT_DIR / "decision_boundary_lift_curve.png"
        plt.savefig(lpath, bbox_inches="tight"); plt.close()
        console.print(f"📁 Saved: [green]{lpath}[/green]")

    # ====================
    # CSV summary to disk
    # ====================
    import csv
    csv_path = OUT_DIR / "decision_boundary_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "threshold", "n", "winrate_pct", "mean_return", "mean_return_pct", "stderr", "stderr_pct"])
        w.writerow(["low",  f"<= {lo_thr:.6f}", n_lo, f"{wr_lo:.4f}", f"{mu_lo:+.8f}", f"{mu_lo*100:+.6f}", f"{se_lo:.8f}", f"{se_lo*100:.6f}"])
        w.writerow(["high", f">= {hi_thr:.6f}", n_hi, f"{wr_hi:.4f}", f"{mu_hi:+.8f}", f"{mu_hi*100:+.6f}", f"{se_hi:.8f}", f"{se_hi*100:.6f}"])
    console.print(f"📄 Saved: [green]{csv_path}[/green]")


# ------------------------ NEW: Prediction diagnostics ------------------------

def analyze_prediction_diagnostics(
    analyzer: AttentionAnalyzer,
    ds: FlatWindowDataset,
    returns_all: np.ndarray,
    max_points: int = 5000
):
    console.print("\n" + "=" * 80)
    console.print(Panel.fit("[bold cyan]PREDICTION DIAGNOSTICS[/bold cyan]", border_style="cyan"))

    N = min(max_points, len(ds))
    idx = list(range(N))
    X, y, ts = ds.get_batch(idx)
    probs = analyzer.predict_entry_proba(X)

    # 1) Histogram of entry probabilities
    plt.figure(figsize=(8, 5))
    clean = probs[~np.isnan(probs)]
    plt.hist(clean, bins=40)
    plt.title("Entry Probability — Histogram")
    plt.xlabel("p(entry)")
    plt.ylabel("Count")
    hpath = OUT_DIR / "prob_histogram.png"
    plt.tight_layout()
    plt.savefig(hpath, bbox_inches="tight"); plt.close()
    console.print(f"📁 Saved: [green]{hpath}[/green]")

    # 2) Calibration by decile
    valid_mask = ~np.isnan(probs)
    p = probs[valid_mask]
    r = y[valid_mask]
    if len(p) >= 100:
        qs = np.quantile(p, np.linspace(0, 1, 11))
        mids, avg_ret = [], []
        for i in range(10):
            lo, hi = qs[i], qs[i+1]
            bin_mask = (p >= lo) & (p <= hi) if i == 9 else (p >= lo) & (p < hi)
            if bin_mask.sum() > 0:
                mids.append(float((lo+hi)/2.0))
                avg_ret.append(float(r[bin_mask].mean()))
        plt.figure(figsize=(8, 5))
        plt.plot(mids, np.array(avg_ret)*100.0, marker='o')
        plt.title("Calibration: avg future return by p(entry) decile")
        plt.xlabel("Mid-decile p(entry)")
        plt.ylabel("Avg future return (%)")
        cpath = OUT_DIR / "calibration_by_decile.png"
        plt.tight_layout()
        plt.savefig(cpath, bbox_inches="tight"); plt.close()
        console.print(f"📁 Saved: [green]{cpath}[/green]")

        # Save CSV with bin stats
        import csv
        csv_path = OUT_DIR / "calibration_by_decile.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["mid_p_entry", "avg_future_return"])
            for m, a in zip(mids, avg_ret):
                w.writerow([m, a])
        console.print(f"📄 Saved: [green]{csv_path}[/green]")

    # 3) Scatter: p(entry) vs future return
    plt.figure(figsize=(8, 5))
    plt.scatter(p, r*100.0, s=6, alpha=0.5)
    plt.title("p(entry) vs Future Return")
    plt.xlabel("p(entry)")
    plt.ylabel("Future return (%)")
    spath = OUT_DIR / "prob_vs_return_scatter.png"
    plt.tight_layout()
    plt.savefig(spath, bbox_inches="tight"); plt.close()
    console.print(f"📁 Saved: [green]{spath}[/green]")

    # 4) If timestamps exist, plot a short timeseries of p(entry)
    if ts is not None:
        take = min(2000, len(ts))
        plt.figure(figsize=(10, 4))
        plt.plot(ts[:take], probs[:take])
        plt.title("Entry Probability — Timeseries (first 2k samples)")
        plt.xlabel("time")
        plt.ylabel("p(entry)")
        tpath = OUT_DIR / "prob_timeseries_first2k.png"
        plt.tight_layout()
        plt.savefig(tpath, bbox_inches="tight"); plt.close()
        console.print(f"📁 Saved: [green]{tpath}[/green]")

def analyze_pseudo_temporal_attention(analyzer, ds, num_samples=3, out_dir=Path("analysis")):
    console.print("\n" + "=" * 80)
    console.print(Panel.fit("[bold cyan]PSEUDO-TEMPORAL EXPLANATION (FLAT WINDOWS)[/bold cyan]", border_style="cyan"))

    n = min(num_samples, len(ds))
    indices = np.random.choice(len(ds), n, replace=False)

    for i, idx in enumerate(indices):
        x_vec, _, _ = ds[idx]  # flat [L]
        sal = analyzer.temporal_saliency_from_flat(x_vec)              # [T]
        dlt, _ = analyzer.temporal_occlusion_from_flat(x_vec, block=3) # [T]

        # plot saliency
        plt.figure(figsize=(12, 3))
        plt.plot(sal)
        plt.title(f"Temporal saliency (|grad|, normalized) — sample {idx}")
        plt.xlabel("time step"); plt.ylabel("importance (0..1)")
        p1 = out_dir / f"pseudo_temporal_saliency_{idx}.png"
        plt.tight_layout(); plt.savefig(p1, bbox_inches="tight"); plt.close()
        console.print(f"📁 Saved: [green]{p1}[/green]")

        # plot occlusion delta
        plt.figure(figsize=(12, 3))
        plt.plot(dlt)
        plt.axhline(0.0, ls="--", c="k", lw=0.8)
        plt.title(f"Temporal occlusion Δp(entry) — sample {idx}")
        plt.xlabel("time step"); plt.ylabel("Δp")
        p2 = out_dir / f"pseudo_temporal_occlusion_{idx}.png"
        plt.tight_layout(); plt.savefig(p2, bbox_inches="tight"); plt.close()
        console.print(f"📁 Saved: [green]{p2}[/green]")

# =============================================================================
# Main
# =============================================================================

def main():
    console.print(Panel.fit(
        "[bold cyan]NEURAL TRADING SYSTEM[/bold cyan]\n"
        "[yellow]Model Analysis & Visualization[/yellow]",
        title="🧠 Analysis Tool",
        border_style="cyan"
    ))

    # ----- Discover files
    try:
        MODEL_PATH, FEATURE_EXTRACTOR_PATH = discover_model_and_extractor(
            model_path=None, feature_extractor_path=None
        )
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return

    try:
        FEATURE_CACHE_PATH = discover_feature_cache()
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        console.print("[yellow]💡 Run training first to generate cache[/yellow]")
        return

    # Sanity display
    console.print(f"[cyan]Model:            {MODEL_PATH}[/cyan]")
    console.print(f"[cyan]FeatureExtractor: {FEATURE_EXTRACTOR_PATH}[/cyan]")
    console.print(f"[cyan]Feature cache:    {FEATURE_CACHE_PATH}[/cyan]")

    # Load model/extractor
    model, feature_extractor, config = load_model_and_data(MODEL_PATH, FEATURE_EXTRACTOR_PATH)
    # Preserve true feature_dim into config for printing
    config['feature_dim'] = getattr(getattr(feature_extractor, 'scaler', None), 'n_features_in_', None) or config.get('feature_dim', 'N/A')
    print_model_summary(model, config)

    # Load sequences from cache
    ds, returns_all = prepare_dataset_from_cache(FEATURE_CACHE_PATH, config)

    # Initialize analyzer
    console.print("\n🔧 [cyan]Initializing analyzer...[/cyan]")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    analyzer = AttentionAnalyzer(model, feature_extractor, device=device, auto_scale=False, config=config)
    model.to(device)
    console.print("✅ [green]Analyzer ready[/green]")

    # Run analyses
    try:
        analyze_attention_patterns(analyzer, ds, num_samples=3)
        analyze_pseudo_temporal_attention(analyzer, ds, num_samples=3, out_dir=OUT_DIR)
        analyze_feature_importance(analyzer, ds, num_samples=100, feature_dim=config.get('feature_dim'))
        analyze_regime_space(analyzer, ds, returns_all)
        analyze_decision_boundary(analyzer, ds, returns_all)
        analyze_prediction_diagnostics(analyzer, ds, returns_all, max_points=5000)
    except Exception as e:
        console.print(f"\n[red]❌ Analysis failed: {e}[/red]")
        import traceback
        traceback.print_exc()

    console.print("\n" + "=" * 80)
    console.print(Panel.fit("[bold green]✅ ANALYSIS COMPLETE![/bold green]", border_style="green"))


if __name__ == '__main__':
    main()
