import backtrader as bt
import polars as pl
import numpy as np
import torch
from pathlib import Path
from typing import Dict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time

from backtrader.TransparencyPatch import activate_patch, capture_patch, export_data, optimized_patch
from backtrader.utils.backtest import PolarsDataLoader, DataSpec

# ==============================================================================
# DATA COLLECTION STRATEGY
# ==============================================================================

class DataCollectionStrategy(bt.Strategy):
    """
    Runs ONLY to collect indicator data via TransparencyPatch. No trading.
    """
    params = dict(
        use_cycle_signals=True,
        use_regime_signals=True,
        use_volatility_signals=True,
        use_momentum_signals=True,
        use_trend_signals=True,
        cycle_period=20,
        roofing_hp_period=48,
        roofing_ss_period=10,
        hurst_period=100,
        laguerre_length=20,
        damiani_atr_fast=13,
        damiani_std_fast=20,
        damiani_atr_slow=40,
        damiani_std_slow=100,
        damiani_thresh=1.4,
        squeeze_period=20,
        squeeze_mult=2,
        squeeze_period_kc=20,
        squeeze_mult_kc=1.5,
        satr_atr_period=14,
        satr_std_period=20,
        rsx_period=14,
        qqe_period=6,
        qqe_fast=5,
        qqe_q=3.0,
        rmi_period=20,
        rmi_lookback=5,
        wavetrend_period=10,
        kama_period=30,
        kama_fast=2,
        kama_slow=30,
        mesa_fast=20,
        mesa_slow=50,
        ttf_period=15,
        schaff_cycle=10,
        schaff_fast=23,
        schaff_slow=50,
        backtest=True,
        debug=False,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.atr = bt.indicators.ATR(self.data, period=14)

        from backtrader.indicators.CyberCycle import CyberCycle
        from backtrader.indicators.ElhersDecyclerOscillator import DecyclerOscillator
        from backtrader.indicators.RoofingFilter import RoofingFilter
        from backtrader.indicators.kama import AdaptiveMovingAverage as KAMA
        from backtrader.indicators.hurst import HurstExponent
        from backtrader.indicators.wavetrend import WaveTrend
        from backtrader.indicators.AdaptiveCyberCycle import AdaptiveCyberCycle
        from backtrader.indicators.AdaptiveLaguerreFilter import AdaptiveLaguerreFilter
        from backtrader.indicators.DamianiVolatmeter import DamianiVolatmeter
        from backtrader.indicators.SqueezeVolatility import SqueezeVolatility
        from backtrader.indicators.StandarizedATR import StandarizedATR
        from backtrader.indicators.RSX import RSX
        from backtrader.indicators.qqe import QQE
        from backtrader.indicators.MesaAdaptiveMovingAverage import MAMA
        from backtrader.indicators.TrendTriggerFactor import TrendTriggerFactor
        from backtrader.indicators.rmi import RelativeMomentumIndex
        from backtrader.indicators.SchaffTrendCycle import SchaffTrendCycle

        if self.p.use_cycle_signals:
            self.cyber_cycle = CyberCycle(self.data, period=self.p.cycle_period)
            self.decycler = DecyclerOscillator(self.data)
            self.roofing = RoofingFilter(self.data, hp_period=self.p.roofing_hp_period, ss_period=self.p.roofing_ss_period)

        if self.p.use_regime_signals:
            self.hurst = HurstExponent(self.data, period=self.p.hurst_period)
            self.adaptive_cycle = AdaptiveCyberCycle(self.data)

        if self.p.use_volatility_signals:
            self.laguerre = AdaptiveLaguerreFilter(self.data, length=self.p.laguerre_length)
            self.damiani = DamianiVolatmeter(self.data, atr_fast=self.p.damiani_atr_fast, std_fast=self.p.damiani_std_fast, atr_slow=self.p.damiani_atr_slow, std_slow=self.p.damiani_std_slow, thresh=self.p.damiani_thresh)
            self.squeeze = SqueezeVolatility(self.data, period=self.p.squeeze_period, mult=self.p.squeeze_mult, period_kc=self.p.squeeze_period_kc, mult_kc=self.p.squeeze_mult_kc)
            self.satr = StandarizedATR(self.data, atr_period=self.p.satr_atr_period, std_period=self.p.satr_std_period)

        if self.p.use_momentum_signals:
            self.rsx = RSX(self.data, length=self.p.rsx_period)
            self.qqe = QQE(self.data, period=self.p.qqe_period, fast=self.p.qqe_fast, q=self.p.qqe_q)
            self.rmi = RelativeMomentumIndex(self.data, period=self.p.rmi_period, lookback=self.p.rmi_lookback)
            self.wavetrend = WaveTrend(self.data, period=self.p.wavetrend_period)

        if self.p.use_trend_signals:
            self.kama = KAMA(self.data, period=self.p.kama_period, fast=self.p.kama_fast, slow=self.p.kama_slow)
            self.mesa = MAMA(self.data, fast=self.p.mesa_fast, slow=self.p.mesa_slow)
            self.ttf = TrendTriggerFactor(self.data, period=self.p.ttf_period)
            self.schaff = SchaffTrendCycle(self.data, cycle=self.p.schaff_cycle, fast=self.p.schaff_fast, slow=self.p.schaff_slow)

        if self.p.debug:
            print(f"✅ DataCollectionStrategy initialized with {len(optimized_patch.indicator_registry)} indicators")

    def next(self):
        capture_patch(self)

    def stop(self):
        if self.p.debug:
            from backtrader.TransparencyPatch import print_patch
            print_patch(auto_export=False)


# ==============================================================================
# NEURAL DATA PIPELINE
# ==============================================================================

class NeuralDataPipeline:
    """
    Automated pipeline: PolarsDataLoader → Backtrader → TransparencyPatch → Training
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.loader = PolarsDataLoader()
        
        from data.feature_extractor import IndicatorFeatureExtractor
        self.feature_extractor = IndicatorFeatureExtractor(
            lookback_windows=config.get('lookback_windows', [5, 10, 20, 50, 100])
        )
    
    def collect_data_from_backtrader(
        self,
        coin: str = 'BTC',
        interval: str = '4h',
        start_date: str = '2018-01-01',
        end_date: str = '2024-12-31',
        collateral: str = 'USDT',
        force_recollect: bool = False
    ) -> pl.DataFrame:
        """
        Run backtrader with DataCollectionStrategy. Uses date-scoped caching.
        """
        from rich.console import Console
        console = Console()
        
        export_dir = Path('neural_data')
        export_dir.mkdir(parents=True, exist_ok=True)
        export_stem = f'{coin}_{interval}_{start_date}_{end_date}_neural_data'
        export_parquet = export_dir / f'{export_stem}.parquet'
        
        if export_parquet.exists() and not force_recollect:
            console.print(f"\n📥 [cyan]Loading cached export: {export_parquet}[/cyan]")
            df_collected = pl.read_parquet(str(export_parquet))
            df_collected = df_collected.filter(
                (pl.col('datetime') >= pl.lit(start_date)) & 
                (pl.col('datetime') <= pl.lit(end_date))
            )
            console.print(f"✅ [green]Loaded {len(df_collected):,} bars from cache[/green]")
            console.print(f"   Features: {len(df_collected.columns)}")
            console.print(f"   Indicator features: {len([c for c in df_collected.columns if c not in ['bar', 'datetime', 'open', 'high', 'low', 'close', 'volume']])}")
            return df_collected
        
        console.print("🔬 [bold cyan]Starting Neural Data Collection[/bold cyan]")
        console.print(f"   Symbol: {coin}/{collateral}")
        console.print(f"   Interval: {interval}")
        console.print(f"   Period: {start_date} → {end_date}")
        
        console.print("\n🔧 [yellow]Activating TransparencyPatch...[/yellow]")
        activate_patch(debug=False)
        
        cerebro = bt.Cerebro(oldbuysell=True, runonce=False, stdstats=False)
        
        console.print(f"\n📥 [cyan]Loading data for {coin}...[/cyan]")
        spec = DataSpec(symbol=coin, interval=interval, start_date=start_date, end_date=end_date, collateral=collateral)
        
        df = self.loader.load_data(spec, use_cache=True)
        data_feed = self.loader.make_backtrader_feed(df, spec)
        console.print(f"✅ [green]Loaded {len(df):,} bars[/green]")
        
        cerebro.adddata(data_feed)
        cerebro.addstrategy(DataCollectionStrategy, backtest=True, debug=False)
        cerebro.broker.setcash(10000)
        cerebro.broker.setcommission(commission=0.001)
        
        console.print("\n📊 [bold green]Running Backtrader to collect indicator data...[/bold green]")
        cerebro.run()
        
        console.print("\n💾 [yellow]Exporting collected data...[/yellow]")
        df_collected = export_data(filename=export_stem, export_dir=str(export_dir))
        
        console.print(f"\n✅ [bold green]Collection Complete![/bold green]")
        console.print(f"   Bars: {len(df_collected):,}")
        console.print(f"   Features: {len(df_collected.columns)}")
        console.print(f"   Cached to: {export_parquet}")
        
        del cerebro, data_feed
        import gc; gc.collect()
        
        return df_collected

    def prepare_training_data(self, df: pl.DataFrame, prediction_horizon: int = 5) -> Dict:
        """
        Optimized feature extraction with disk caching. First run extracts and saves; 
        subsequent runs load from cache in seconds.
        """
        from rich.console import Console
        import pickle
        import hashlib
        console = Console()
        
        # Build cache key from data hash and config
        cache_dir = Path('neural_data/features')
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Hash based on: df shape, seq_len, prediction_horizon, lookback_windows
        seq_len = int(self.config.get('seq_len', self.config.get('seqlen', 100)))
        lookback_str = str(sorted(self.config.get('lookback_windows', [5, 10, 20, 50, 100])))
        cache_key_raw = f"{df.shape[0]}_{df.shape[1]}_{seq_len}_{prediction_horizon}_{lookback_str}"
        cache_key = hashlib.md5(cache_key_raw.encode()).hexdigest()[:16]
        cache_file = cache_dir / f'features_{cache_key}.pkl'
        
        # Try to load cached features
        if cache_file.exists():
            console.print(f"[cyan]📥 Loading cached features from {cache_file.name}[/cyan]")
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            console.print(f"[green]✅ Loaded cached features! Shape: {cached['features'].shape}[/green]")
            console.print(f"   Feature dimension: {cached['feature_dim']}")
            return cached
        
        console.print("[cyan]Preparing training data (will cache for next run)...[/cyan]")

        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        # 1) Vectorized forward returns
        df = df.with_columns([
            ((pl.col('close').shift(-prediction_horizon) - pl.col('close')) / pl.col('close')).alias('forward_return')
        ])
        console.print(f"   ✅ Calculated forward returns (horizon={prediction_horizon})")

        # 2) Indicator columns
        ohlcv_cols = ['bar', 'datetime', 'open', 'high', 'low', 'close', 'volume']
        indicator_cols = [c for c in df.columns if c not in ohlcv_cols]
        console.print(f"   ✅ Found {len(indicator_cols)} indicator features")

        # 3) Fill nulls
        from polars import selectors as cs
        df = df.with_columns(cs.numeric().fill_null(0))

        # 4) Build numpy arrays
        indicator_arrays = {c: df.get_column(c).to_numpy() for c in indicator_cols}

        console.print(f"\n   Extracting features (seq_len={seq_len})...")

        total_rows = df.height
        valid_rows = total_rows - prediction_horizon
        start_idx = seq_len
        end_idx = valid_rows
        if end_idx <= start_idx:
            raise ValueError("Not enough rows")

        indices = list(range(start_idx, end_idx))
        N = len(indices)

        # 5) Determine expected dim
        def compute_one(i: int):
            current = {k: indicator_arrays[k][:i] for k in indicator_cols}
            feats = self.feature_extractor.extract_all_features(current)
            return np.asarray(feats, dtype=np.float32).ravel()

        first = compute_one(indices[0])
        expected_dim = int(first.size)
        console.print(f"   Expected feature dimension: {expected_dim}")

        # 6) Feature extraction (single-thread with tqdm)
        t0 = time.perf_counter()
        features_list = []

        console.print(f"   Single-threaded extraction (N={N})")
        for i in tqdm(indices, desc="Extracting", unit="bar"):
            current = {k: indicator_arrays[k][:i] for k in indicator_cols}
            try:
                f = np.asarray(self.feature_extractor.extract_all_features(current), np.float32).ravel()
                if f.size != expected_dim:
                    f = np.pad(f, (0, expected_dim - f.size)) if f.size < expected_dim else f[:expected_dim]
                features_list.append(f)
            except Exception:
                features_list.append(np.zeros(expected_dim, dtype=np.float32))

        dt = time.perf_counter() - t0
        rows_per_s = N / dt if dt > 0 else 0.0
        console.print(f"⏱️ {dt:.2f}s | Rows: {N:,} | {rows_per_s:,.1f} rows/s")

        # 7) Stack
        features = np.vstack(features_list).astype(np.float32)

        # 8) Align returns
        returns = df.get_column('forward_return')[seq_len:seq_len + len(features)].to_numpy().copy()
        np.nan_to_num(returns, copy=False)

        console.print(f"[green]✅ Complete! Shape: {features.shape}, NaN: {np.isnan(features).sum()}, Inf: {np.isinf(features).sum()}[/green]")

        timestamps = df.get_column('datetime')[seq_len:seq_len + len(features)].to_numpy() if 'datetime' in df.columns else None

        result = {
            'features': features,
            'returns': returns[:len(features)],
            'feature_dim': features.shape[1],
            'timestamps': timestamps,
            'indicator_columns': indicator_cols,
        }
        
        # Cache result
        console.print(f"[yellow]💾 Caching features to {cache_file.name}[/yellow]")
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        console.print(f"[green]✅ Cached! Next run will load in ~2 seconds[/green]")
        
        return result

    def train_neural_model(self, training_data: Dict, save_path: str = 'best_model.pt'):
        """
        Train the neural network on collected data.
        """
        from rich.console import Console
        from torch.utils.data import DataLoader
        from training.trainer import NeuralTrainer, TradingDataset
        from models.architecture import create_model
        
        console = Console()
        console.print("\n🧠 [bold magenta]Starting Neural Network Training[/bold magenta]")
        
        features = training_data['features']
        returns = training_data['returns']
        feature_dim = training_data['feature_dim']
        
        self.config['feature_dim'] = feature_dim
        
        train_end = int(len(features) * 0.7)
        console.print(f"\n   Fitting scaler on {train_end:,} training samples...")
        self.feature_extractor.fit_scaler(features[:train_end])
        
        console.print("   Normalizing features...")
        features_normalized = np.array([self.feature_extractor.transform(f) for f in features])
        
        val_start = int(len(features) * 0.7)
        test_start = int(len(features) * 0.85)
        
        train_features = features_normalized[:val_start]
        train_returns = returns[:val_start]
        val_features = features_normalized[val_start:test_start]
        val_returns = returns[val_start:test_start]
        test_features = features_normalized[test_start:]
        test_returns = returns[test_start:]
        
        console.print(f"\n   📊 Data Split:")
        console.print(f"      Train: {len(train_features):>8,} bars ({len(train_features)/len(features)*100:>5.1f}%)")
        console.print(f"      Val:   {len(val_features):>8,} bars ({len(val_features)/len(features)*100:>5.1f}%)")
        console.print(f"      Test:  {len(test_features):>8,} bars ({len(test_features)/len(features)*100:>5.1f}%)")
        
        train_dataset = TradingDataset(train_features, train_returns, seq_len=self.config['seq_len'], prediction_horizon=self.config.get('prediction_horizon', 5))
        val_dataset = TradingDataset(val_features, val_returns, seq_len=self.config['seq_len'], prediction_horizon=self.config.get('prediction_horizon', 5))
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.get('batch_size', 32), shuffle=False, num_workers=4, pin_memory=True if torch.cuda.is_available() else False)
        val_loader = DataLoader(val_dataset, batch_size=self.config.get('batch_size', 32), shuffle=False, num_workers=4, pin_memory=True if torch.cuda.is_available() else False)
        
        console.print("\n🏗️  [cyan]Building neural architecture...[/cyan]")
        model = create_model(feature_dim, self.config)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        console.print(f"   Total parameters:     {total_params:>12,}")
        console.print(f"   Trainable parameters: {trainable_params:>12,}")
        
        trainer = NeuralTrainer(model=model, train_loader=train_loader, val_loader=val_loader, config=self.config, device=self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        console.print("\n🎯 [bold green]Starting training loop...[/bold green]")
        trainer.train(self.config.get('num_epochs', 100))
        console.print(f"[cyan]Train size: {len(train_dataset)}[/cyan]")
        console.print(f"[cyan]Val size:   {len(val_dataset)}[/cyan]")

        import pickle
        feature_extractor_path = save_path.replace('.pt', '_feature_extractor.pkl')
        with open(feature_extractor_path, 'wb') as f:
            pickle.dump(self.feature_extractor, f)
        
        console.print(f"\n✅ [bold green]Training complete![/bold green]")
        console.print(f"   Model saved to: {save_path}")
        console.print(f"   Feature extractor saved to: {feature_extractor_path}")
        
        return trainer, test_features, test_returns


# ==============================================================================
# TRAINING ORCHESTRATION
# ==============================================================================

def train_neural_system(
    coin: str = 'BTC',
    interval: str = '4h',
    start_date: str = '2018-01-01',
    end_date: str = '2024-12-31',
    collateral: str = 'USDT',
    config: Dict = None,
):
    """
    One-command training pipeline.
    """
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    if config is None:
        config = {
            'seq_len': 100,
            'prediction_horizon': 5,
            'lookback_windows': [5, 10, 20, 50, 100],
            'd_model': 256,
            'num_heads': 8,
            'num_layers': 6,
            'd_ff': 1024,
            'dropout': 0.1,
            'latent_dim': 8,
            'batch_size': 32,
            'num_epochs': 100,
            'lr': 1e-4,
            'min_lr': 1e-6,
            'weight_decay': 1e-5,
            'grad_accum_steps': 4,
            'T_0': 10,
            'patience': 15,
            'save_every': 10,
            'use_wandb': True,
            'run_name': f'neural_{coin}_{interval}',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'fe_workers': 4,
            'fe_chunk': 1024,
            'fe_parallel_threshold': 100_000,
        }

    console.print(Panel.fit(
        f"[bold cyan]NEURAL TRADING SYSTEM[/bold cyan]\n"
        f"[yellow]Automated Training Pipeline[/yellow]\n\n"
        f"Symbol: {coin}/{collateral}\n"
        f"Interval: {interval}\n"
        f"Period: {start_date} → {end_date}\n"
        f"Device: {config['device']}",
        title="🧠 Configuration",
        border_style="cyan",
    ))

    pipeline = NeuralDataPipeline(config)
    df = pipeline.collect_data_from_backtrader(coin=coin, interval=interval, start_date=start_date, end_date=end_date, collateral=collateral)

    training_data = pipeline.prepare_training_data(df, prediction_horizon=config['prediction_horizon'])

    model_path = f'models/neural_{coin}_{interval}_{start_date}_{end_date}.pt'
    trainer, test_features, test_returns = pipeline.train_neural_model(training_data, save_path=model_path)

    console.print("\n" + "=" * 80)
    console.print(Panel.fit(
        "[bold green]✅ PIPELINE COMPLETE![/bold green]\n\n"
        f"Model: {model_path}\n"
        f"Feature Extractor: {model_path.replace('.pt', '_feature_extractor.pkl')}\n\n"
        "[yellow]Next steps:[/yellow]\n"
        "  1. Analyze what the network learned:\n"
        "     [cyan]python analyze_model.py[/cyan]\n"
        "  2. Run backtest with neural strategy:\n"
        "     [cyan]python run_neural_backtest.py[/cyan]\n"
        "  3. Deploy to live trading:\n"
        "     [cyan]python deploy_live.py[/cyan]",
        title="🎉 Success",
        border_style="green",
    ))

    return pipeline, trainer, training_data


if __name__ == '__main__':
    '''
    DEBUG_config = {
        'seq_len': 10,
        'prediction_horizon': 1,
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 1,
        'batch_size': 4,
        'num_epochs': 10,
        'lookback_windows': [5, 10],
        'device': 'cuda',
        # parallel controls
        'fe_workers': 4,
        'fe_chunk': 512,
        'fe_parallel_threshold': 100_000,
    }

    train_neural_system(
        coin='BTC',
        interval='4h',
        start_date='2020-01-01',
        end_date='2020-03-01',
        collateral='USDT',
        config=DEBUG_config,
    )
    '''

    full_quant_config = {
        'seq_len': 100,
        'prediction_horizon': 5,
        'lookback_windows': [5, 10, 20, 50, 100],
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 1024,
        'dropout': 0.1,
        'latent_dim': 8,
        'batch_size': 32,
        'num_epochs': 100,
        'lr': 1e-4,
        'min_lr': 1e-6,
        'weight_decay': 1e-5,
        'grad_accum_steps': 4,
        'T_0': 10,
        'patience': 15,
        'save_every': 10,
        'use_wandb': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'fe_workers': 4,
        'fe_chunk': 1024,
        'fe_parallel_threshold': 999999,  # 100_000 Force single-thread 
    }

    train_neural_system(
        coin='BTC',
        interval='4h',
        start_date='2017-01-01',
        end_date='2024-01-01',
        collateral='USDT',
        config=full_quant_config
    )
