
import json
import backtrader as bt
import polars as pl
import numpy as np
import torch
from pathlib import Path
from typing import Dict
import time

from backtrader.TransparencyPatch import activate_patch, capture_patch, export_data, optimized_patch
from backtrader.utils.backtest import PolarsDataLoader, DataSpec

from backtrader.indicators.RoofingFilter import RoofingFilter
from backtrader.indicators.AdaptiveCyberCycle import AdaptiveCyberCycle
from backtrader.indicators.hurst import HurstExponent
from backtrader.indicators.RSX import RSX
from backtrader.indicators.ultimateoscillator import UltimateOscillator
from backtrader.indicators.ChaikinMoneyFlow import ChaikinMoneyFlow
from backtrader.indicators.ChaikinVolatility import ChaikinVolatility
from backtrader.indicators.Klingeroscillator import KlingerOscillator
from backtrader.indicators.SSLChannel import SSLChannel
from backtrader.indicators.VolumeOscillator import VolumeOscillator
from backtrader.indicators.ASH import ASH
from backtrader.indicators.ASI import AccumulativeSwingIndex
from backtrader.indicators.psar import ParabolicSAR

def find_latest_cache(cache_dir='neural_data/features'):

    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return None

    cache_files = list(cache_path.glob('features_*.pkl'))
    if not cache_files:
        return None

    return str(max(cache_files, key=lambda p: p.stat().st_mtime))

class DataCollectionStrategy(bt.Strategy):
\
\
\

    params = dict(

        use_cycle_signals=False,
        cyber_cycle_period=20,
        roofing_hp_period=48,
        roofing_ss_period=10,
        butterworth_period=20,
        supersmooth_period=10,

        use_regime_signals=False,
        hurst_period=100,

        use_volatility_signals=False,
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
        chaikin_vol_period=10,

        use_momentum_signals=False,
        rsx_period=14,
        qqe_period=6,
        qqe_fast=5,
        qqe_q=3.0,
        qqemod_period=14,
        rmi_period=20,
        rmi_lookback=5,
        wavetrend_period=10,
        lrsi_period=14,
        tsi_period_slow=25,
        tsi_period_fast=13,
        dv2_period=2,
        ultimate_osc_period1=7,
        ultimate_osc_period2=14,
        ultimate_osc_period3=28,

        use_trend_signals=False,
        kama_period=30,
        kama_fast=2,
        kama_slow=30,
        mesa_fast=20,
        mesa_slow=50,
        ttf_period=15,
        schaff_cycle=10,
        schaff_fast=23,
        schaff_slow=50,
        supertrend_period=10,
        supertrend_multiplier=3.0,
        ssl_period=10,
        tdfi_period=13,
        kst_roc1=10,
        kst_roc2=15,
        kst_roc3=20,
        kst_roc4=30,
        williams_alligator_jaw=13,
        williams_alligator_teeth=8,
        williams_alligator_lips=5,

        vumanchu_a_period=10,
        vumanchu_b_period=13,
        waddah_fast=20,
        waddah_slow=40,
        waddah_channel=20,
        waddah_mult=2.0,
        waddah_sensitivity=150,

        chaikin_mf_period=20,
        klinger_fast=34,
        klinger_slow=55,
        vol_osc_fast=5,
        vol_osc_slow=10,

        vortex_period=14,

        madr_period=10,
        smaa_period=20,
        hma_period=20,
        zlema_period=20,
        psar_af=0.02,
        psar_max_af=0.2,
        dpo_period=20,

        backtest=True,
        debug=True,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.atr = bt.indicators.ATR(self.data, period=14)

        if self.p.use_cycle_signals:
            self.roofing = RoofingFilter(self.data, hp_period=self.p.roofing_hp_period, ss_period=self.p.roofing_ss_period, plot=False)
            self.adaptive_cycle = AdaptiveCyberCycle(self.data, plot=False)

        if self.p.use_regime_signals:
            self.hurst = HurstExponent(self.data, period=self.p.hurst_period, plot=False)

        if self.p.use_volatility_signals:
            self.chaikin_vol = ChaikinVolatility(self.data, ema_period=self.p.chaikin_vol_period, roc_period=self.p.chaikin_vol_period, plot=False)

        if self.p.use_momentum_signals:
            self.rsx = RSX(self.data, length=self.p.rsx_period, plot=False)
            self.ultimate = UltimateOscillator(self.data, p1=self.p.ultimate_osc_period1, p2=self.p.ultimate_osc_period2, p3=self.p.ultimate_osc_period3, plot=False)

        if self.p.use_trend_signals:
            self.ssl = SSLChannel(self.data, period=self.p.ssl_period, plot=False)

        self.chaikin_mf = ChaikinMoneyFlow(self.data, period=self.p.chaikin_mf_period, plot=False)
        self.klinger = KlingerOscillator(self.data, fast=self.p.klinger_fast, slow=self.p.klinger_slow, plot=False)
        self.vol_osc = VolumeOscillator(self.data, shortlen=self.p.vol_osc_fast, longlen=self.p.vol_osc_slow, plot=False)

        self.ash = ASH(self.data, plot=False)
        self.asi = AccumulativeSwingIndex(self.data, plot=False)
        self.psar = ParabolicSAR(self.data, af=self.p.psar_af, afmax=self.p.psar_max_af, plot=False)
        self.sma200 = bt.indicators.SMA(self.data, period=200)

        if self.p.debug:
            print(f"✅ DataCollectionStrategy initialized with 90+ indicators")

    def next(self):

        capture_patch(self)

    def stop(self):

        if self.p.debug:
            print("✅ Data collection complete")
            from backtrader.TransparencyPatch import print_patch
            print_patch(auto_export=False)

class NeuralDataPipeline:
\
\

    def __init__(self, config: Dict):
        self.config = config
        self.loader = PolarsDataLoader()

        from data.feature_extractor import IndicatorFeatureExtractor
        self.feature_extractor = IndicatorFeatureExtractor(
            lookback_windows=config.get('lookback_windows', [5, 10, 20, 50, 100, 200])
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
\
\

        from rich.console import Console
        console = Console()

        export_dir = Path('neural_data')
        export_dir.mkdir(parents=True, exist_ok=True)
        export_stem = f'{coin}_{interval}_{start_date}_{end_date}_neural_data'
        export_parquet = export_dir / f'{export_stem}.parquet'

        if export_parquet.exists() and not force_recollect:
            console.print(f"\n🔥 [cyan]Loading cached export: {export_parquet}[/cyan]")
            df_collected = pl.read_parquet(str(export_parquet))

            if 'datetime' in df_collected.columns:

                if df_collected['datetime'].dtype != pl.Datetime:
                    df_collected = df_collected.with_columns(
                        pl.col('datetime').str.to_datetime()
                    )

                start_dt = pl.lit(start_date).str.to_datetime()
                end_dt = pl.lit(end_date).str.to_datetime()

                df_collected = df_collected.filter(
                    (pl.col('datetime') >= start_dt) &
                    (pl.col('datetime') <= end_dt)
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

        console.print(f"\n🔥 [cyan]Loading data for {coin}...[/cyan]")
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

    def prepare_training_data(
        self,
        df: pl.DataFrame,
        prediction_horizon: int = 5,
        force_cache_file: str = None
    ) -> Dict:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\

        from rich.console import Console
        import pickle
        import hashlib
        console = Console()

        cache_dir = Path('neural_data/features')
        cache_dir.mkdir(parents=True, exist_ok=True)

        if force_cache_file:
            cache_file = Path(force_cache_file)
            if cache_file.exists():
                console.print(f"[cyan]🔥 Force loading cache: {cache_file}[/cyan]")
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                console.print(f"[green]✅ Loaded! Shape: {cached['features'].shape}[/green]")

                if 'prices' not in cached and 'close' in df.columns:
                    console.print("[yellow]⚠️ Adding prices to cached data...[/yellow]")
                    seq_len = self.config.get('seq_len', 100)
                    prices = df.get_column('close').to_numpy()
                    cached['prices'] = prices[seq_len:seq_len + len(cached['features'])]

                return cached
            else:
                console.print(f"[red]❌ Cache file not found: {cache_file}[/red]")

        existing_caches = list(cache_dir.glob('features_*.pkl'))
        if existing_caches:

            latest_cache = max(existing_caches, key=lambda p: p.stat().st_mtime)
            console.print(f"[yellow]🔍 Found existing cache: {latest_cache.name}[/yellow]")
            console.print(f"[yellow]   Use this cache? (it will skip feature extraction)[/yellow]")

            console.print(f"[cyan]🔥 Auto-loading existing cache: {latest_cache}[/cyan]")
            with open(latest_cache, 'rb') as f:
                cached = pickle.load(f)
            console.print(f"[green]✅ Loaded! Shape: {cached['features'].shape}[/green]")

            if 'prices' not in cached and 'close' in df.columns:
                console.print("[yellow]⚠️ Adding prices to cached data...[/yellow]")
                seq_len = self.config.get('seq_len', 100)
                prices = df.get_column('close').to_numpy()
                cached['prices'] = prices[seq_len:seq_len + len(cached['features'])]

            return cached

        data_hash = hashlib.md5(str(df.shape).encode()).hexdigest()[:16]
        config_str = f"seq{self.config.get('seq_len', 100)}_hor{prediction_horizon}"
        cache_file = cache_dir / f'features_{data_hash}_{config_str}.pkl'

        if cache_file.exists():
            console.print(f"[cyan]🔥 Loading cached features from {cache_file.name}[/cyan]")
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            console.print(f"[green]✅ Loaded cached features! Shape: {cached['features'].shape}[/green]")
            console.print(f"   Feature dimension: {cached['feature_dim']}")

            if 'prices' not in cached and 'close' in df.columns:
                console.print("[yellow]⚠️ Adding prices to cached data...[/yellow]")
                seq_len = self.config.get('seq_len', 100)
                prices = df.get_column('close').to_numpy()
                cached['prices'] = prices[seq_len:seq_len + len(cached['features'])]

            return cached

        console.print("[cyan]Preparing training data (will cache for next run)...[/cyan]")

        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        df = df.with_columns([
            ((pl.col('close').shift(-prediction_horizon) - pl.col('close')) / pl.col('close')).alias('forward_return')
        ])
        console.print(f"   ✅ Calculated forward returns (horizon={prediction_horizon})")

        prices = df.get_column('close').to_numpy()
        console.print(f"   ✅ Extracted {len(prices):,} price points")

        ohlcv_cols = ['bar', 'datetime', 'open', 'high', 'low', 'close', 'volume']
        indicator_cols = [c for c in df.columns if c not in ohlcv_cols + ['forward_return']]
        console.print(f"   ✅ Found {len(indicator_cols)} indicator features")

        from polars import selectors as cs
        df = df.with_columns(cs.numeric().fill_null(0))

        indicator_arrays = {c: df.get_column(c).to_numpy() for c in indicator_cols}
        returns_array = df.get_column('forward_return').to_numpy()

        seq_len = int(self.config.get('seq_len', 100))
        console.print(f"\n   Extracting features (seq_len={seq_len})...")

        total_rows = df.height
        valid_rows = total_rows - prediction_horizon
        start_idx = seq_len
        end_idx = valid_rows
        if end_idx <= start_idx:
            raise ValueError(f"Not enough rows: total={total_rows}, need at least {seq_len + prediction_horizon}")

        indices = list(range(start_idx, end_idx))
        N = len(indices)

        indicator_matrix = df.select(indicator_cols).to_numpy().astype(np.float32)
        console.print(f"   📊 Indicator matrix: {indicator_matrix.shape}")

        from numpy.lib.stride_tricks import sliding_window_view

        console.print(f"   🚀 Vectorized extraction (seq_len={seq_len})...")
        t0 = time.perf_counter()

        n_samples, n_features = indicator_matrix.shape
        sequences = sliding_window_view(
            indicator_matrix,
            window_shape=(seq_len, n_features),
            axis=(0, 1)
        )

        sequences3d = sequences[:, 0, :, :].astype(np.float32)
        features = sequences3d

        dt = time.perf_counter() - t0
        console.print(f"   ✅ Extracted {features.shape[0]} sequences in {dt:.2f}s")
        console.print(f"      Speed: {features.shape[0] / dt:.1f} sequences/s")

        n_sequences = features.shape[0]

        returns = returns_array[seq_len:seq_len + n_sequences].copy()
        aligned_prices = prices[seq_len:seq_len + n_sequences].copy()

        min_len = min(len(features), len(returns), len(aligned_prices))
        features = features[:min_len]
        returns = returns[:min_len]
        aligned_prices = aligned_prices[:min_len]

        np.nan_to_num(returns, copy=False)

        console.print(f"   📏 Aligned: features={len(features)}, returns={len(returns)}")

        console.print("📊 Filtering for high-signal data...")
        signal_threshold = np.percentile(np.abs(returns), 20)
        high_signal_mask = np.abs(returns) >= signal_threshold

        features_filtered = features[high_signal_mask]
        returns_filtered = returns[high_signal_mask]
        prices_filtered = aligned_prices[high_signal_mask]

        console.print(f"   Signal threshold: {signal_threshold:.6f}")
        console.print(f"   Kept {len(features_filtered):,} / {len(features):,} bars ({len(features_filtered)/len(features)*100:.1f}%)")
        console.print(f"   Filtered returns - Mean: {np.mean(returns_filtered):.6f}, Std: {np.std(returns_filtered):.6f}")

        console.print(f"[green]✅ Complete! Shape: {features_filtered.shape}, NaN: {np.isnan(features_filtered).sum()}, Inf: {np.isinf(features_filtered).sum()}[/green]")

        timestamps = df.get_column('datetime')[seq_len:seq_len + len(features)].to_numpy() if 'datetime' in df.columns else None
        if timestamps is not None:
            timestamps = timestamps[high_signal_mask]

        result = {

            'features': features_filtered,
            'returns': returns_filtered,
            'prices': prices_filtered,

            'feature_dim': features_filtered.shape[2],
            'timestamps': timestamps,
            'indicator_columns': indicator_cols,
        }

        console.print(f"[yellow]💾 Caching raw features to {cache_file.name}[/yellow]")
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        console.print(f"[green]✅ Cached! Next run will load in ~2 seconds[/green]")

        return result

    def train_neural_model(self, training_data: Dict, save_path: str = 'best_model.pt'):
\
\

        from rich.console import Console
        from torch.utils.data import DataLoader
        from training.trainer import NeuralTrainer, TradingDataset
        from models.architecture import create_model

        console = Console()
        console.print("\n🧠 [bold magenta]Starting Neural Network Training[/bold magenta]")

        features = training_data['features']
        returns = training_data['returns']
        prices = training_data['prices']
        feature_dim = training_data['feature_dim']

        self.config['feature_dim'] = feature_dim

        train_end = int(len(features) * 0.7)
        console.print(f"\n   Fitting scaler on {train_end:,} training samples...")

        seq_len = features.shape[1]
        train_features_flat = features[:train_end].reshape(-1, feature_dim)
        self.feature_extractor.fit_scaler(train_features_flat)

        console.print("   Normalizing features...")

        features_normalized = np.array([
            self.feature_extractor.safe_transform(f) for f in features
        ])

        val_start = int(len(features) * 0.7)
        test_start = int(len(features) * 0.85)

        train_features = features_normalized[:val_start]
        train_returns = returns[:val_start]
        train_prices = prices[:val_start]

        val_features = features_normalized[val_start:test_start]
        val_returns = returns[val_start:test_start]
        val_prices = prices[val_start:test_start]

        test_features = features_normalized[test_start:]
        test_returns = returns[test_start:]
        test_prices = prices[test_start:]

        console.print(f"\n   📊 Data Split:")
        console.print(f"      Train: {len(train_features):>8,} bars ({len(train_features)/len(features)*100:>5.1f}%)")
        console.print(f"      Val:   {len(val_features):>8,} bars ({len(val_features)/len(features)*100:>5.1f}%)")
        console.print(f"      Test:  {len(test_features):>8,} bars ({len(test_features)/len(features)*100:>5.1f}%)")

        train_dataset = TradingDataset(
            train_features,
            train_returns,
            train_prices,
            seq_len=self.config['seq_len'],
            prediction_horizon=self.config.get('prediction_horizon', 5)
        )
        val_dataset = TradingDataset(
            val_features,
            val_returns,
            val_prices,
            seq_len=self.config['seq_len'],
            prediction_horizon=self.config.get('prediction_horizon', 5)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 128),
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 128),
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )

        console.print("\n🗂️  [cyan]Building neural architecture...[/cyan]")
        model = create_model(feature_dim, self.config)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        console.print(f"   Total parameters:     {total_params:>12,}")
        console.print(f"   Trainable parameters: {trainable_params:>12,}")

        trainer = NeuralTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.config,
            device=self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )

        console.print("\n🎯 [bold green]Starting training loop...[/bold green]")
        trainer.train(self.config.get('num_epochs', 200))
        console.print(f"[cyan]Train size: {len(train_dataset)}[/cyan]")
        console.print(f"[cyan]Val size:   {len(val_dataset)}[/cyan]")

        import pickle
        feature_extractor_path = save_path.replace('.pt', '_feature_extractor.pkl')
        with open(feature_extractor_path, 'wb') as f:
            pickle.dump(self.feature_extractor, f)

        test_data = {
            'features': test_features,
            'returns': test_returns,
            'prices': test_prices
        }
        test_data_path = 'neural_data/test_data.pkl'
        with open(test_data_path, 'wb') as f:
            pickle.dump(test_data, f)
        console.print(f"[green]💾 Saved test data for analysis: {test_data_path}[/green]")

        console.print(f"\n✅ [bold green]Training complete![/bold green]")
        console.print(f"   Model saved to: {save_path}")
        console.print(f"   Feature extractor saved to: {feature_extractor_path}")
        console.print(f"   Test data saved to: {test_data_path}")

        return trainer, test_features, test_returns

def train_neural_system(
    coin: str = 'BTC',
    interval: str = '4h',
    start_date: str = '2000-01-01',
    end_date: str = '2030-12-31',
    collateral: str = 'USDT',
    config: Dict = None,
):
\
\

    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    if config is None:
        config = {

            'seq_len': 100,
            'prediction_horizon': 5,
            'lookback_windows': [5, 10, 20, 50, 100, 200],
            'd_model': 256,
            'num_heads': 8,
            'num_layers': 6,
            'd_ff': 1024,
            'dropout': 0.15,
            'latent_dim': 16,

            'batch_size': 16,
            'num_epochs': 200,
            'lr': 0.0003,
            'min_lr': 1e-7,
            'weight_decay': 1e-4,
            'grad_accum_steps': 2,
            'T_0': 20,
            'patience': 25,
            'save_every': 5,

            'use_wandb': False,
            'run_name': f'exit_aware_neural_{coin}_{interval}',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'best_model_path': 'models/best_exit_aware_model.pt',
        }

    console.print(Panel.fit(
        f"[bold cyan]🚀 STUXNET :: SELF-AWARE NEURAL TRADING SYSTEM[/bold cyan]\n"
        f"[yellow]Intelligent Exit Management - No Fixed TP/SL[/yellow]\n\n"
        f"Symbol: {coin}/{collateral}\n"
        f"Interval: {interval}\n"
        f"Period: {start_date} → {end_date}\n"
        f"Device: {config['device']}\n\n"
        f"Architecture:\n"
        f"  • d_model: {config['d_model']}\n"
        f"  • num_heads: {config['num_heads']}\n"
        f"  • num_layers: {config['num_layers']}\n"
        f"  • latent_dim: {config['latent_dim']}\n\n"
        f"Training:\n"
        f"  • batch_size: {config['batch_size']}\n"
        f"  • epochs: {config['num_epochs']}\n"
        f"  • learning_rate: {config['lr']}\n"
        f"  • patience: {config['patience']}\n\n"
        f"Exit Management:\n"
        f"  • Take-Profit: Momentum fade, resistance, optimal ratios\n"
        f"  • Stop-Loss: Pattern failure, acceleration, regime breaks\n"
        f"  • Let-Winner-Run: Trend strength, continuation signals\n"
        f"  • Regime Change: Vol spikes, volume anomalies, breaks",
        title="🧠 Configuration",
        border_style="cyan",
    ))

    pipeline = NeuralDataPipeline(config)

    df = pipeline.collect_data_from_backtrader(
        coin=coin,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        collateral=collateral
    )

    latest_cache = find_latest_cache()
    if latest_cache:
        console.print(f"[cyan]🔍 Latest cache: {latest_cache}[/cyan]")
    else:
        console.print("[yellow]No feature cache found - will extract features[/yellow]")

    training_data = pipeline.prepare_training_data(
        df,
        prediction_horizon=config['prediction_horizon'],
        force_cache_file=latest_cache
    )

    model_path = f'models/exit_aware_{coin}_{interval}_{start_date}_{end_date}.pt'
    trainer, test_features, test_returns = pipeline.train_neural_model(training_data, save_path=model_path)

    console.print("\n" + "=" * 80)
    console.print(Panel.fit(
        "[bold green]✅ PIPELINE COMPLETE![/bold green]\n\n"
        f"Model: {model_path}\n"
        f"Feature Extractor: {model_path.replace('.pt', '_feature_extractor.pkl')}\n"
        f"Test Data: neural_data/test_data.pkl\n\n"
        "[yellow]Next steps:[/yellow]\n"
        "  1. Analyze exit awareness:\n"
        "     [cyan]python analyze_exit_awareness.py[/cyan]\n"
        "  2. Run backtest with exit-aware strategy:\n"
        "     [cyan]python run_neural_backtest.py[/cyan]\n"
        "  3. Deploy to live trading:\n"
        "     [cyan]python deploy_live.py[/cyan]\n\n"
        "[bold cyan]Key Features:[/bold cyan]\n"
        "  • Takes profit based on momentum fade & resistance\n"
        "  • Cuts losses on pattern failures & regime breaks\n"
        "  • Lets winners run during strong continuation\n"
        "  • Detects regime changes before crashes",
        title="🎉 Success",
        border_style="green",
    ))

    config_path = config['best_model_path'].replace('.pt', '_config.json')
    with open(config_path, 'w') as f:

        json_config = {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool, list, dict))}
        json.dump(json_config, f, indent=2)

    console.print(f"[green]✅ Config saved to {config_path}[/green]")

    return pipeline, trainer, training_data

if __name__ == '__main__':

    exit_aware_config = {

        'feature_dim': 25100,
        'seq_len': 100,
        'prediction_horizon': 5,
        'lookback_windows': [5, 10, 20, 50, 100, 200],
        'd_model': 512,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 2048,
        'dropout': 0.05,
        'latent_dim': 16,
        'positional_encoding_scale': 1.0,
        'input_projection_gain': 1.0,

        'batch_size': 16,
        'num_epochs': 200,
        'lr': 2e-5,
        'min_lr': 1e-7,
        'weight_decay': 1e-4,
        'grad_accum_steps': 2,

        'T_0': 20,
        'patience': 25,
        'save_every': 5,

        'use_wandb': True,
        'run_name': 'exit_aware_15m_full',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'best_model_path': 'models/best_exit_aware_model.pt',
    }

    train_neural_system(
        coin='BTC',
        interval='15m',
        start_date='2017-01-01',
        end_date='2024-01-01',
        collateral='USDT',
        config=exit_aware_config
    )
