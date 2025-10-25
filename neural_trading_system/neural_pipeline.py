#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elite Neural Trading System — Full Polars Integration
"""

import backtrader as bt
import pandas as pd
import polars as pl
import numpy as np
import torch
import time
import pickle
from pathlib import Path
from typing import Dict
from rich.console import Console
from rich.panel import Panel

from backtrader.TransparencyPatch import (
    activate_patch, capture_patch, export_data, optimized_patch
)
from backtrader.utils.backtest import PolarsDataLoader, DataSpec
from training.trainer import sanitize_and_overwrite_cache

console = Console()


# =====================================================================
# 🔍 Cache Utility
# =====================================================================
def find_latest_cache(cache_dir='neural_data/features'):
    """Find the most recent feature cache file."""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return None
    cache_files = list(cache_path.glob('features_*.pkl'))
    if not cache_files:
        return None
    return str(max(cache_files, key=lambda p: p.stat().st_mtime))


# =====================================================================
# 📊 Data Collection Strategy
# =====================================================================
class DataCollectionStrategy(bt.Strategy):
    """
    Collects data with all 90+ indicators for neural feature extraction.
    """
    params = dict(backtest=True, debug=False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.atr = bt.indicators.ATR(self.data, period=14)

    def next(self):
        capture_patch(self)

    def stop(self):
        if self.p.debug:
            from backtrader.TransparencyPatch import print_patch
            print_patch(auto_export=False)


# =====================================================================
# 🧠 Neural Data Pipeline
# =====================================================================
class NeuralDataPipeline:
    """
    Orchestrates data collection, feature cache handling, sanitization,
    and training data preparation using Polars.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.loader = PolarsDataLoader()
        from data.feature_extractor import IndicatorFeatureExtractor
        self.feature_extractor = IndicatorFeatureExtractor(
            lookback_windows=config.get('lookback_windows', [5, 10, 20, 50, 100, 200])
        )

    # ----------------------------------------------------------
    # SMART CACHE LOADER
    # ----------------------------------------------------------
    @staticmethod
    def smart_cache_loader(path: str):
        """
        Detects file type (Arrow/Parquet/Pickle/JSON) and loads accordingly.
        Returns a Polars DataFrame.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Cache not found: {path}")

        with open(path, "rb") as f:
            header = f.read(8)

        # Arrow IPC or Parquet
        if header.startswith(b"ARROW1") or header.startswith(b"PAR1"):
            console.print(f"[cyan]📦 Detected Arrow/Parquet format — loading with Polars ({path.name})[/cyan]")
            try:
                return pl.read_ipc(path)
            except Exception:
                return pl.read_parquet(path)

        # Pickle
        elif header.startswith(b"\x80") or header.startswith(b"\x81"):
            console.print(f"[yellow]📦 Detected Pickle format — loading with pickle ({path.name})[/yellow]")
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, pd.DataFrame):
                return pl.from_pandas(obj)
            elif isinstance(obj, dict):
                return pl.DataFrame(obj)
            else:
                return obj

        # JSON / dict fallback
        elif header.strip().startswith(b"{"):
            console.print(f"[yellow]📦 Detected JSON/dict-like cache ({path.name})[/yellow]")
            obj = pd.read_pickle(path)
            return pl.from_pandas(obj) if isinstance(obj, pd.DataFrame) else obj

        raise ValueError(f"❌ Unknown cache format: {header!r} ({path})")

    # ----------------------------------------------------------
    # DATA COLLECTION
    # ----------------------------------------------------------
    def collect_data_from_backtrader(
        self,
        coin='BTC',
        interval='4h',
        start_date='2018-01-01',
        end_date='2024-12-31',
        collateral='USDT',
        force_recollect=False
    ) -> pl.DataFrame:
        """
        Runs Backtrader with DataCollectionStrategy and TransparencyPatch.
        """
        export_dir = Path('neural_data')
        export_dir.mkdir(parents=True, exist_ok=True)
        export_stem = f'{coin}_{interval}_{start_date}_{end_date}_neural_data'
        export_parquet = export_dir / f'{export_stem}.parquet'

        if export_parquet.exists() and not force_recollect:
            console.print(f"\n📥 [cyan]Loading cached export: {export_parquet}[/cyan]")
            df_collected = pl.read_parquet(str(export_parquet))
            if df_collected.schema.get("datetime") != pl.Datetime:
                df_collected = df_collected.with_columns(
                    pl.col("datetime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
                )
            start_dt = pl.datetime(int(start_date[:4]), int(start_date[5:7]), int(start_date[8:10]))
            end_dt = pl.datetime(int(end_date[:4]), int(end_date[5:7]), int(end_date[8:10]))
            df_collected = df_collected.filter(
                (pl.col("datetime") >= start_dt) & (pl.col("datetime") <= end_dt)
            )
            console.print(f"✅ [green]Loaded {len(df_collected):,} bars from cache[/green]")
            console.print(f"   Features: {len(df_collected.columns)}")
            return df_collected

        console.print("🔬 [bold cyan]Starting Neural Data Collection[/bold cyan]")
        activate_patch(debug=False)
        cerebro = bt.Cerebro(oldbuysell=True, runonce=False, stdstats=False)

        spec = DataSpec(symbol=coin, interval=interval, start_date=start_date, end_date=end_date, collateral=collateral)
        df = self.loader.load_data(spec, use_cache=True)
        data_feed = self.loader.make_backtrader_feed(df, spec)
        console.print(f"✅ [green]Loaded {len(df):,} bars[/green]")

        cerebro.adddata(data_feed)
        cerebro.addstrategy(DataCollectionStrategy, backtest=True, debug=False)
        cerebro.broker.setcash(10000)
        cerebro.broker.setcommission(commission=0.001)
        cerebro.run()

        df_collected = export_data(filename=export_stem, export_dir=str(export_dir))
        console.print(f"✅ [green]Collection Complete — {len(df_collected):,} bars[/green]")
        return df_collected

    # ----------------------------------------------------------
    # TRAINING DATA PREPARATION
    # ----------------------------------------------------------
    def prepare_training_data(self, df, prediction_horizon: int, force_cache_file: str = None):
        """
        Loads a feature cache (auto-detecting Arrow/Pickle) and prepares features + returns arrays.
        """
        console.rule("[bold blue]🧠 Preparing Training Data[/bold blue]")

        if force_cache_file:
            cache_path = Path(force_cache_file)
            console.print(f"[cyan]📥 Force loading cache:[/cyan] {cache_path}")

            cached = self.smart_cache_loader(cache_path)

            if isinstance(cached, dict):
                df_cached = pl.DataFrame(cached)
            elif isinstance(cached, pd.DataFrame):
                df_cached = pl.from_pandas(cached)
            elif isinstance(cached, pl.DataFrame):
                df_cached = cached
            else:
                raise TypeError(f"Unsupported cache type: {type(cached)}")

            if "returns" in df_cached.columns:
                console.print("[magenta]🧹 Isolating 'returns' column from features[/magenta]")
                returns = df_cached["returns"].to_numpy()
                features = df_cached.drop("returns").to_numpy()
            else:
                return_cols = [c for c in df_cached.columns if "return" in c.lower()]
                if return_cols:
                    main_col = return_cols[0]
                    returns = df_cached[main_col].to_numpy()
                    features = df_cached.drop(main_col).to_numpy()
                else:
                    returns = None
                    features = df_cached.to_numpy()

            console.print(
                f"[green]✅ Loaded![/green] Shape: [bold]{df_cached.shape}[/bold] "
                f"| Features: {features.shape[1]} | Returns: {0 if returns is None else len(returns)}"
            )

            return {
                "features": features,
                "returns": returns,
                "metadata": {
                    "columns": df_cached.columns,
                    "rows": len(df_cached)
                }
            }

        console.print("[yellow]⚠️ No cache provided, using live DataFrame input[/yellow]")
        if isinstance(df, pl.DataFrame):
            if "returns" in df.columns:
                returns = df["returns"].to_numpy()
                features = df.drop("returns").to_numpy()
            else:
                returns = None
                features = df.to_numpy()
        else:
            raise TypeError("Expected Polars DataFrame input")

        return {
            "features": features,
            "returns": returns,
            "metadata": {
                "columns": df.columns,
                "rows": len(df)
            }
        }

    # ----------------------------------------------------------
    # MODEL TRAINING
    # ----------------------------------------------------------
    def train_neural_model(self, training_data: Dict, save_path='best_model.pt'):
        from torch.utils.data import DataLoader
        from training.trainer import NeuralTrainer, TradingDataset
        from models.architecture import create_model
        from neural_pipeline import find_latest_cache

        console.rule("[bold magenta]🧠 NEURAL MODEL TRAINING STARTED")
        features = training_data['features']
        returns = training_data['returns']

        console.print("\n[yellow]🧹 Performing model-level Polars cache sanitization...[/yellow]")
        start_time = time.perf_counter()
        features, returns = sanitize_and_overwrite_cache(features, self.config, self, find_latest_cache, console)
        console.print(f"[green]✅ Model cache sanitized successfully in {time.perf_counter() - start_time:.2f}s[/green]\n")

        feature_dim = features.shape[1]
        console.print(f"[cyan]📏 Feature Dimension:[/cyan] {feature_dim}")

        train_end = int(len(features) * 0.7)
        self.feature_extractor.fit_scaler(features[:train_end])
        features_normalized = []
        expected = getattr(self.feature_extractor.scaler, 'n_features_in_', None)

        for f in features:
            # Safe scaling with automatic ±1 column tolerance
            f_scaled = self.feature_extractor.safe_transform(f.reshape(1, -1)).ravel()
            features_normalized.append(f_scaled)

        features_normalized = np.array(features_normalized)

        val_start = int(len(features) * 0.7)
        test_start = int(len(features) * 0.85)

        train_features = features_normalized[:val_start]
        train_returns = returns[:val_start]
        val_features = features_normalized[val_start:test_start]
        val_returns = returns[val_start:test_start]
        test_features = features_normalized[test_start:]
        test_returns = returns[test_start:]

        train_dataset = TradingDataset(train_features, train_returns, seq_len=self.config['seq_len'])
        val_dataset = TradingDataset(val_features, val_returns, seq_len=self.config['seq_len'])

        train_loader = DataLoader(train_dataset, batch_size=self.config.get('batch_size', 128), shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=self.config.get('batch_size', 128), shuffle=False)

        model = create_model(feature_dim, self.config)
        trainer = NeuralTrainer(model, train_loader, val_loader, self.config, device=self.config.get('device', 'cuda'))

        console.print("\n[green]🎯 Beginning training loop...[/green]")
        trainer.train(self.config.get('num_epochs', 200))
        console.print(f"[bold green]✅ Model training complete![/bold green] Saved → {save_path}")
        return trainer, test_features, test_returns


# =====================================================================
# 🚀 TRAINING ORCHESTRATION
# =====================================================================
def train_neural_system(
    coin='BTC', interval='4h',
    start_date='2018-01-01', end_date='2024-12-31',
    collateral='USDT', config: Dict = None
):
    console.rule("[bold cyan]🚀 ELITE NEURAL TRADING SYSTEM LAUNCH")

    if config is None:
        config = {
            'seq_len': 100,
            'prediction_horizon': 5,
            'lookback_windows': [5, 10, 20, 50, 100, 200],
            'batch_size': 128,
            'num_epochs': 200,
            'lr': 0.0003,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }

    console.print(Panel.fit(
        f"[bold cyan]Neural System Configuration[/bold cyan]\n\n"
        f"Symbol: {coin}/{collateral}\nInterval: {interval}\n"
        f"Period: {start_date} → {end_date}\nDevice: {config['device']}",
        border_style="cyan"
    ))

    pipeline = NeuralDataPipeline(config)
    df = pipeline.collect_data_from_backtrader(coin, interval, start_date, end_date, collateral)
    latest_cache = find_latest_cache()
    console.print(f"[cyan]🔍 Latest cache: {latest_cache}[/cyan]")

    training_data = pipeline.prepare_training_data(df, config['prediction_horizon'], latest_cache)
    console.print("\n[yellow]🧹 Running automatic Polars cache sanitization before training...[/yellow]")
    t0 = time.perf_counter()
    features, returns = sanitize_and_overwrite_cache(df, config, pipeline, find_latest_cache, console)
    console.print(f"[green]✅ Feature cache sanitized in {time.perf_counter() - t0:.2f}s[/green]\n")

    model_path = f'models/elite_neural_{coin}_{interval}_{start_date}_{end_date}.pt'
    trainer, test_features, test_returns = pipeline.train_neural_model(training_data, save_path=model_path)

    console.print(Panel.fit(
        "[bold green]✅ PIPELINE COMPLETE![/bold green]\n\n"
        f"Model saved: {model_path}\n"
        "Next:\n"
        "  python analyze_model.py\n"
        "  python run_neural_backtest.py\n"
        "  python deploy_live.py",
        title="🎉 Success", border_style="green"
    ))
    return pipeline, trainer, training_data


if __name__ == '__main__':
    elite_config = {
        'seq_len': 100,
        'prediction_horizon': 5,
        'lookback_windows': [5, 10, 20, 50, 100, 200],
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 1024,
        'dropout': 0.15,
        'batch_size': 16,
        'num_epochs': 200,
        'lr': 0.0003,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'run_name': 'elite_btc_1h_full_history',
        'rich_dashboard': True,
    }

    train_neural_system(
        coin='BTC', interval='1h',
        start_date='2017-01-01', end_date='2024-12-31',
        collateral='USDT', config=elite_config
    )
