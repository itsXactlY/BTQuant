# ==============================================================================
# NEURAL DATA COLLECTION - INTEGRATED WITH YOUR INFRASTRUCTURE
# ==============================================================================

import backtrader as bt
import polars as pl
import numpy as np
import torch
from pathlib import Path
from typing import Dict

from backtrader.TransparencyPatch import activate_patch, capture_patch, export_data, optimized_patch
from backtrader.utils.backtest import PolarsDataLoader, DataSpec

# ===== Module-scope globals for workers =====
W_INDICATORS = None
W_KEYS = None
W_FE = None
W_EXPECTED_DIM = None

def fe_worker_init(indicator_arrays, indicator_cols, fe_params, expected_dim):
    """Initializer runs once per worker process; stash shared state in globals."""
    global W_INDICATORS, W_KEYS, W_FE, W_EXPECTED_DIM
    W_INDICATORS = indicator_arrays
    W_KEYS = indicator_cols
    W_EXPECTED_DIM = int(expected_dim)
    from data.feature_extractor import IndicatorFeatureExtractor
    W_FE = IndicatorFeatureExtractor(**fe_params)

def fe_worker_batch(idx_batch):
    """Top-level worker: process a batch of indices into feature rows."""
    out = []
    for i in idx_batch:
        try:
            current = {k: W_INDICATORS[k][:i] for k in W_KEYS}
            feats = W_FE.extract_all_features(current)
            f = np.asarray(feats, dtype=np.float32).ravel()
            if f.size != W_EXPECTED_DIM:
                if f.size < W_EXPECTED_DIM:
                    f = np.pad(f, (0, W_EXPECTED_DIM - f.size))
                else:
                    f = f[:W_EXPECTED_DIM]
            out.append(f)
        except Exception:
            out.append(np.zeros(W_EXPECTED_DIM, dtype=np.float32))
    return out

class DataCollectionStrategy(bt.Strategy):
    """
    Strategy that runs ONLY to collect indicator data.
    No trading - pure data harvesting with TransparencyPatch.
    """
    
    params = dict(
        # Enable all indicator blocks
        use_cycle_signals=True,
        use_regime_signals=True,
        use_volatility_signals=True,
        use_momentum_signals=True,
        use_trend_signals=True,
        
        # Indicator parameters from your MegaScalpingStrategy
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
        
        # Control flags
        backtest=True,
        debug=False
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize ATR (base indicator)
        self.atr = bt.indicators.ATR(self.data, period=14)
        
        # Import all your indicators
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
        
        # Initialize cycle indicators
        if self.p.use_cycle_signals:
            self.cyber_cycle = CyberCycle(self.data, period=self.p.cycle_period)
            self.decycler = DecyclerOscillator(self.data)
            self.roofing = RoofingFilter(
                self.data,
                hp_period=self.p.roofing_hp_period,
                ss_period=self.p.roofing_ss_period
            )
        
        # Initialize regime indicators
        if self.p.use_regime_signals:
            self.hurst = HurstExponent(self.data, period=self.p.hurst_period)
            self.adaptive_cycle = AdaptiveCyberCycle(self.data)
        
        # Initialize volatility indicators
        if self.p.use_volatility_signals:
            self.laguerre = AdaptiveLaguerreFilter(
                self.data,
                length=self.p.laguerre_length
            )
            self.damiani = DamianiVolatmeter(
                self.data,
                atr_fast=self.p.damiani_atr_fast,
                std_fast=self.p.damiani_std_fast,
                atr_slow=self.p.damiani_atr_slow,
                std_slow=self.p.damiani_std_slow,
                thresh=self.p.damiani_thresh
            )
            self.squeeze = SqueezeVolatility(
                self.data,
                period=self.p.squeeze_period,
                mult=self.p.squeeze_mult,
                period_kc=self.p.squeeze_period_kc,
                mult_kc=self.p.squeeze_mult_kc
            )
            self.satr = StandarizedATR(
                self.data,
                atr_period=self.p.satr_atr_period,
                std_period=self.p.satr_std_period
            )
        
        # Initialize momentum indicators
        if self.p.use_momentum_signals:
            self.rsx = RSX(self.data, length=self.p.rsx_period)
            self.qqe = QQE(
                self.data,
                period=self.p.qqe_period,
                fast=self.p.qqe_fast,
                q=self.p.qqe_q
            )
            self.rmi = RelativeMomentumIndex(
                self.data,
                period=self.p.rmi_period,
                lookback=self.p.rmi_lookback
            )
            self.wavetrend = WaveTrend(
                self.data,
                period=self.p.wavetrend_period
            )
        
        # Initialize trend indicators
        if self.p.use_trend_signals:
            self.kama = KAMA(
                self.data,
                period=self.p.kama_period,
                fast=self.p.kama_fast,
                slow=self.p.kama_slow
            )
            self.mesa = MAMA(
                self.data,
                fast=self.p.mesa_fast,
                slow=self.p.mesa_slow
            )
            self.ttf = TrendTriggerFactor(self.data, period=self.p.ttf_period)
            self.schaff = SchaffTrendCycle(
                self.data,
                cycle=self.p.schaff_cycle,
                fast=self.p.schaff_fast,
                slow=self.p.schaff_slow
            )
        
        if self.p.debug:
            print(f"✅ DataCollectionStrategy initialized with {len(optimized_patch.indicator_registry)} indicators")
    
    def next(self):
        """Capture data every bar - NO TRADING."""
        capture_patch(self)
    
    def stop(self):
        """Called when backtest ends - print summary."""
        if self.p.debug:
            from backtrader.TransparencyPatch import print_patch
            print_patch(auto_export=False)


class NeuralDataPipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.loader = PolarsDataLoader()
        from data.feature_extractor import IndicatorFeatureExtractor
        self.feature_extractor = IndicatorFeatureExtractor(
            lookback_windows=config.get('lookback_windows', [5, 10, 20, 50, 100])
        )

    def prepare_training_data_optimized(self, pipeline, df: pl.DataFrame, prediction_horizon: int = 5):
        from rich.console import Console
        console = Console()
        console.print("[cyan]Preparing training data (OPTIMIZED, parallel)...[/cyan]")

        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        # Vectorized forward returns in Polars
        df = df.with_columns([
            ((pl.col('close').shift(-prediction_horizon) - pl.col('close')) / pl.col('close'))
            .alias('forward_return')
        ])
        console.print(f"   ✅ Calculated forward returns (horizon={prediction_horizon})")

        ohlcv_cols = ['bar','datetime','open','high','low','close','volume']
        indicator_cols = [c for c in df.columns if c not in ohlcv_cols]
        console.print(f"   ✅ Found {len(indicator_cols)} indicator features")

        # Fill numeric nulls once (Polars expression, multi-threaded)
        from polars import selectors as cs
        df = df.with_columns(cs.numeric().fill_null(0))

        # Build numpy arrays for indicators (read-only views)
        indicator_arrays = {c: df.get_column(c).to_numpy() for c in indicator_cols}

        seq_len = int(pipeline.config.get('seq_len', pipeline.config.get('seqlen', 100)))
        console.print(f"\n   Extracting features (seq_len={seq_len})...")

        total_rows = df.height
        valid_rows = total_rows - prediction_horizon
        start_idx = seq_len
        end_idx = valid_rows
        if end_idx <= start_idx:
            raise ValueError("Not enough rows to extract features with given seq_len and horizon")

        indices = list(range(start_idx, end_idx))

        # Determine expected feature dimension once using the first index
        def compute_one(i: int):
            current = {k: v[:i] for k, v in indicator_arrays.items()}
            feats = pipeline.feature_extractor.extract_all_features(current)
            return np.asarray(feats, dtype=np.float32).ravel()

        first = compute_one(indices[0])
        expected_dim = int(first.size)

        fe_params = dict(lookback_windows=self.config.get('lookback_windows', [5,10,20,50,100]))

        # Build batches
        chunk = int(self.config.get('fe_chunk', 1024))
        batches = [indices[i:i+chunk] for i in range(0, len(indices), chunk)]

        from concurrent.futures import ProcessPoolExecutor
        import os

        max_workers = int(self.config.get('fe_workers', max(1, os.cpu_count() - 1)))
        console.print(f"   🏎️ Parallelizing feature extraction with {max_workers} workers, chunk={chunk}")

        features_list = []
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=fe_worker_init,
            initargs=(indicator_arrays, indicator_cols, fe_params, expected_dim),
        ) as ex:
            # Submit in order and collect in order to preserve chronology
            futures = [ex.submit(fe_worker_batch, b) for b in batches]
            for fut in futures:
                rows = fut.result()
                features_list.extend(rows)
                console.print(f"   Expected feature dimension: {expected_dim}")

        # Parallel worker: re-create a fresh extractor with same params
        # Avoid sending the full object if it's heavy or not picklable
        fe_params = dict(lookback_windows=self.config.get('lookback_windows', [5,10,20,50,100]))

        def worker_init(_indicator_arrays, _indicator_cols, _fe_params):
            # Stash in globals for the worker
            global W_INDICATORS, W_KEYS, W_FE
            W_INDICATORS = _indicator_arrays
            W_KEYS = _indicator_cols
            from data.feature_extractor import IndicatorFeatureExtractor
            W_FE = IndicatorFeatureExtractor(**_fe_params)

        def worker_batch(idx_batch):
            out = []
            for i in idx_batch:
                current = {k: W_INDICATORS[k][:i] for k in W_KEYS}
                try:
                    feats = W_FE.extract_all_features(current)
                    f = np.asarray(feats, dtype=np.float32).ravel()
                    # pad/truncate in worker to reduce main-thread work
                    if f.size != expected_dim:
                        if f.size < expected_dim:
                            f = np.pad(f, (0, expected_dim - f.size))
                        else:
                            f = f[:expected_dim]
                    out.append(f)
                except Exception:
                    out.append(np.zeros(expected_dim, dtype=np.float32))
            return out

        # Build batches of indices
        import math, os
        chunk = int(self.config.get('fe_chunk', 1024))  # configurable
        batches = [indices[i:i+chunk] for i in range(0, len(indices), chunk)]

        # Use ProcessPoolExecutor for CPU-bound work
        from concurrent.futures import ProcessPoolExecutor, as_completed
        max_workers = int(self.config.get('fe_workers', max(1, os.cpu_count() - 1)))

        # Note: indicator_arrays are NumPy; supported by pickling, but large dicts incur overhead
        # For big runs, consider memory-mapping or a shared store keyed by filename paths.

        features_list = []
        console.print(f"   🏎️ Parallelizing feature extraction with {max_workers} workers, chunk={chunk}")
        with ProcessPoolExecutor(max_workers=max_workers,
                                initializer=worker_init,
                                initargs=(indicator_arrays, indicator_cols, fe_params)) as ex:
            # Submit all batches
            futures = [ex.submit(worker_batch, b) for b in batches]
            # Consume results in submission order to preserve index order
            for fut in futures:
                rows = fut.result()
                features_list.extend(rows)

        # Stack into final array
        features = np.vstack(features_list).astype(np.float32)

        # Align returns to features
        returns = df.get_column('forward_return')[seq_len:seq_len + len(features)].to_numpy().copy()
        np.nan_to_num(returns, copy=False)

        console.print(f"[green]✅ Feature extraction complete![/green]")
        console.print(f"   Shape: {features.shape}")
        console.print(f"   Feature dimension: {features.shape[1]}")
        console.print(f"   NaN count: {np.isnan(features).sum()}")
        console.print(f"   Inf count: {np.isinf(features).sum()}")

        timestamps = df.get_column('datetime')[seq_len:seq_len + len(features)].to_numpy() if 'datetime' in df.columns else None

        return {
            'features': features,
            'returns': returns[:len(features)],
            'feature_dim': features.shape[1],
            'timestamps': timestamps,
            'indicator_columns': indicator_cols
        }

    def prepare_training_data(
        self,
        df: pl.DataFrame,
        prediction_horizon: int = 5
    ) -> Dict:
        # Route to optimized path
        return self.prepare_training_data_optimized(self, df, prediction_horizon)

    def collect_data_from_backtrader(
        self,
        coin: str = 'BTC',
        interval: str = '4h',
        start_date: str = '2018-01-01',
        end_date: str = '2024-12-31',
        collateral: str = 'USDT',
        force_recollect: bool = False
    ) -> pl.DataFrame:
        from rich.console import Console
        console = Console()

        # Build a deterministic export filename
        export_dir = Path('neural_data')
        export_dir.mkdir(parents=True, exist_ok=True)
        export_stem = f'{coin}_{interval}_neural_data'
        export_parquet = export_dir / f'{export_stem}.parquet'

        # Fast path: reuse previously exported indicators
        if export_parquet.exists() and not force_recollect:
            console.print(f"\n📥 [cyan]Loading cached export: {export_parquet}[/cyan]")
            df_collected = pl.read_parquet(str(export_parquet))
            console.print(f"✅ [green]Loaded {len(df_collected):,} bars from cache[/green]")
            return df_collected  # Parquet is columnar and fast to read with Polars [web:177]

        # Otherwise, run Backtrader + TransparencyPatch once
        console.print("🔬 [bold cyan]Starting Neural Data Collection[/bold cyan]")
        console.print(f"   Symbol: {coin}/{collateral}")
        console.print(f"   Interval: {interval}")
        console.print(f"   Period: {start_date} → {end_date}")

        console.print("\n🔧 [yellow]Activating TransparencyPatch...[/yellow]")
        activate_patch(debug=False)

        cerebro = bt.Cerebro(oldbuysell=True, runonce=False, stdstats=False)

        console.print(f"\n📥 [cyan]Loading data for {coin}...[/cyan]")
        spec = DataSpec(
            symbol=coin,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            collateral=collateral
        )

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

        # Clean up
        del cerebro, data_feed
        import gc; gc.collect()

        return df_collected

# ==============================================================================
# ONE-COMMAND TRAINING PIPELINE
# ==============================================================================

def train_neural_system(
    coin: str = 'BTC',
    interval: str = '4h',
    start_date: str = '2018-01-01',
    end_date: str = '2024-12-31',
    collateral: str = 'USDT',
    config: Dict = None
):
    """
    🚀 ONE FUNCTION TO RULE THEM ALL
    
    Uses YOUR existing infrastructure:
    - PolarsDataLoader for data loading with caching
    - TransparencyPatch for indicator capture
    - Neural network for pattern learning
    
    Example:
        >>> train_neural_system(
        ...     coin='BTC',
        ...     interval='4h',
        ...     start_date='2020-01-01',
        ...     end_date='2024-12-31'
        ... )
    """
    from rich.console import Console
    from rich.panel import Panel
    
    console = Console()
    
    # Default config
    if config is None:
        config = {
            # Data collection
            'seq_len': 100,
            'prediction_horizon': 5,
            'lookback_windows': [5, 10, 20, 50, 100],
            
            # Model architecture
            'd_model': 256,
            'num_heads': 8,
            'num_layers': 6,
            'd_ff': 1024,
            'dropout': 0.1,
            'latent_dim': 8,
            
            # Training
            'batch_size': 32,
            'num_epochs': 100,
            'lr': 1e-4,
            'min_lr': 1e-6,
            'weight_decay': 1e-5,
            'grad_accum_steps': 4,
            'T_0': 10,
            'patience': 15,
            'save_every': 10,
            
            # Experiment tracking
            'use_wandb': True,
            'run_name': f'neural_{coin}_{interval}',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    
    console.print(Panel.fit(
        f"[bold cyan]NEURAL TRADING SYSTEM[/bold cyan]\n"
        f"[yellow]Automated Training Pipeline[/yellow]\n\n"
        f"Symbol: {coin}/{collateral}\n"
        f"Interval: {interval}\n"
        f"Period: {start_date} → {end_date}\n"
        f"Device: {config['device']}",
        title="🧠 Configuration",
        border_style="cyan"
    ))
    
    # Initialize pipeline
    pipeline = NeuralDataPipeline(config)
    
    # Step 1: Collect data from Backtrader (with TransparencyPatch)
    df = pipeline.collect_data_from_backtrader(
        coin=coin,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        collateral=collateral
    )
    
    # Step 2: Prepare training data
    training_data = pipeline.prepare_training_data(
        df,
        prediction_horizon=config['prediction_horizon']
    )
    
    # Step 3: Train neural model
    model_path = f'models/neural_{coin}_{interval}_{start_date}_{end_date}.pt'
    trainer, test_features, test_returns = pipeline.train_neural_model(
        training_data,
        save_path=model_path
    )
    
    console.print("\n" + "="*80)
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
        border_style="green"
    ))
    
    return pipeline, trainer, training_data


# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================

if __name__ == '__main__':
    
    # Example 1: Train on BTC 4h data
    pipeline, trainer, data = train_neural_system(
        coin='BTC',
        interval='4h',
        start_date='2017-01-01',
        end_date='2024-01-01',
        collateral='USDT'
    )
    
    # Example 2: Custom configuration
    custom_config = {
        'seq_len': 150,  # Longer sequences
        'prediction_horizon': 10,  # Predict further ahead
        'd_model': 512,  # Bigger model
        'num_layers': 8,  # Deeper network
        'batch_size': 16,  # Smaller batches
        'num_epochs': 200,  # More training
        'use_wandb': True,
        'run_name': 'btc_4h_deep',
        'device': 'cuda'
    }
    
    # DEBUG_config={ # ONLY USED FOR 4h CANDLE CROSS CHECK IF CODE WORKS
    #     'seq_len': 10,           # Very short
    #     'prediction_horizon': 1, # Predict 1 bar ahead
    #     'd_model': 256,  # Bigger model
    #     'num_heads': 8,
    #     'num_layers': 1,  # Deeper network
    #     'batch_size': 4,         # Tiny batches
    #     'num_epochs': 10,        # Quick test
    #     'lookback_windows': [5, 10],  # Minimal features
    #     'device': 'cuda'
    # }

    # pipeline, trainer, data = train_neural_system(
    #     coin='BTC',
    #     interval='4h',
    #     start_date='2020-01-01',
    #     end_date='2020-03-01',
    #     collateral='USDT',
    #     config=DEBUG_config

    # )

    # pipeline, trainer, data = train_neural_system(
    #     coin='BTC',
    #     interval='4h',
    #     start_date='2020-01-01',
    #     end_date='2024-12-31',
    #     config=custom_config
    # )
    
    # Example 3: Train on multiple timeframes (run separately)
    # for interval in ['1h', '4h', '1d']:
    #     train_neural_system(
    #         coin='BTC',
    #         interval=interval,
    #         start_date='2020-01-01',
    #         end_date='2024-12-31'
    #     )