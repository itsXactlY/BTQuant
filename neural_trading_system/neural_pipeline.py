# ==============================================================================
# NEURAL DATA COLLECTION - INTEGRATED WITH YOUR INFRASTRUCTURE
# ==============================================================================

import backtrader as bt
import polars as pl
import torch
import numpy as np
from typing import Dict

from backtrader.TransparencyPatch import activate_patch, capture_patch, export_data, optimized_patch
from backtrader.utils.backtest import PolarsDataLoader, DataSpec

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
    """
    Automated pipeline using YOUR existing infrastructure:
    PolarsDataLoader → Backtrader → TransparencyPatch → Neural Training
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
        collateral: str = 'USDT'
    ) -> pl.DataFrame:
        """
        Run backtrader with DataCollectionStrategy to harvest indicator data.
        Uses YOUR existing PolarsDataLoader infrastructure.
        """
        from rich.console import Console
        console = Console()
        
        console.print("🔬 [bold cyan]Starting Neural Data Collection[/bold cyan]")
        console.print(f"   Symbol: {coin}/{collateral}")
        console.print(f"   Interval: {interval}")
        console.print(f"   Period: {start_date} → {end_date}")
        
        # Activate transparency patch BEFORE running backtrader
        console.print("\n🔧 [yellow]Activating TransparencyPatch...[/yellow]")
        activate_patch(debug=False)
        
        # Initialize cerebro
        cerebro = bt.Cerebro(oldbuysell=True, runonce=False, stdstats=False)
        
        # Load data using YOUR PolarsDataLoader
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
        
        # Add data collection strategy (no trading, just capture)
        cerebro.addstrategy(
            DataCollectionStrategy,
            backtest=True,
            debug=False
        )
        
        # Set minimal broker settings (not needed for data collection, but required)
        cerebro.broker.setcash(10000)
        cerebro.broker.setcommission(commission=0.001)
        
        # Run backtest (pure data collection)
        console.print("\n📊 [bold green]Running Backtrader to collect indicator data...[/bold green]")
        cerebro.run()
        
        # Export collected data
        console.print("\n💾 [yellow]Exporting collected data...[/yellow]")
        df_collected = export_data(
            filename=f'{coin}_{interval}_neural_data',
            export_dir='neural_data'
        )
        
        console.print(f"\n✅ [bold green]Collection Complete![/bold green]")
        console.print(f"   Bars: {len(df_collected):,}")
        console.print(f"   Features: {len(df_collected.columns)}")
        console.print(f"   OHLCV columns: {len([c for c in df_collected.columns if c in ['open', 'high', 'low', 'close', 'volume']])}")
        console.print(f"   Indicator features: {len([c for c in df_collected.columns if c not in ['bar', 'datetime', 'open', 'high', 'low', 'close', 'volume']])}")
        
        # Clean up
        del cerebro, data_feed
        import gc
        gc.collect()
        
        return df_collected

    def prepare_training_data_optimized(self, pipeline, df, prediction_horizon: int = 5):
        """
        Optimized version with batch processing, progress tracking, AND CONSISTENT FEATURE SIZES.
        """
        from rich.console import Console
        from tqdm import tqdm
        
        console = Console()
        console.print("[cyan]Preparing training data (OPTIMIZED)...[/cyan]")
        
        df_pd = df.to_pandas()
        
        if 'close' not in df_pd.columns:
            raise ValueError("DataFrame must contain 'close' column")
        
        # Calculate forward returns
        close_prices = df_pd['close'].values
        returns = np.zeros(len(close_prices))
        for i in range(len(close_prices) - prediction_horizon):
            returns[i] = (close_prices[i + prediction_horizon] - close_prices[i]) / close_prices[i]
        
        console.print(f"   ✅ Calculated forward returns (horizon={prediction_horizon})")
        
        # Build indicator data dict
        ohlcv_cols = ['bar', 'datetime', 'open', 'high', 'low', 'close', 'volume']
        indicator_cols = [c for c in df_pd.columns if c not in ohlcv_cols]
        
        console.print(f"   ✅ Found {len(indicator_cols)} indicator features")
        
        indicator_data_full = {}
        for col in df_pd.columns:
            if col not in ['bar', 'datetime']:
                indicator_data_full[col] = df_pd[col].fillna(0).values
        
        seq_len = pipeline.config.get('seqlen', 100)
        console.print(f"\n   Extracting features (seq_len={seq_len})...")
        
        features_list = []
        expected_feature_dim = None  # Track expected dimension
        
        # Extract features with progress bar AND size validation
        with tqdm(total=len(df_pd) - seq_len, desc="Extracting features") as pbar:
            for i in range(seq_len, len(df_pd)):
                # Build current_data dict (slice up to current bar)
                current_data = {key: values[:i] for key, values in indicator_data_full.items()}
                
                try:
                    features = pipeline.feature_extractor.extract_all_features(current_data)
                    
                    # Convert to numpy array if needed
                    if not isinstance(features, np.ndarray):
                        features = np.array(features, dtype=np.float32)
                    
                    # Flatten if multi-dimensional
                    if features.ndim > 1:
                        features = features.flatten()
                    
                    # FIRST ITERATION: Set expected dimension
                    if expected_feature_dim is None:
                        expected_feature_dim = len(features)
                        console.print(f"   Expected feature dimension: {expected_feature_dim}")
                    
                    # VALIDATE SIZE
                    actual_dim = len(features)
                    if actual_dim != expected_feature_dim:
                        console.print(f"[yellow]⚠️  Size mismatch at bar {i}: got {actual_dim}, expected {expected_feature_dim}[/yellow]")
                        
                        # Pad or truncate to match expected size
                        if actual_dim < expected_feature_dim:
                            # Pad with zeros
                            features = np.pad(features, (0, expected_feature_dim - actual_dim), 
                                            mode='constant', constant_values=0)
                        else:
                            # Truncate
                            features = features[:expected_feature_dim]
                    
                    features_list.append(features)
                    
                except Exception as e:
                    console.print(f"[yellow]⚠️  Warning at bar {i}: {e}[/yellow]")
                    
                    # Use correctly sized fallback
                    if len(features_list) > 0:
                        # Use zeros with same shape as previous feature
                        features_list.append(np.zeros_like(features_list[-1]))
                    elif expected_feature_dim is not None:
                        # Use zeros with expected dimension
                        features_list.append(np.zeros(expected_feature_dim, dtype=np.float32))
                    else:
                        # Skip this bar if we don't know the dimension yet
                        console.print(f"[red]❌ Skipping bar {i} - no valid features yet[/red]")
                        continue
                
                pbar.update(1)
        
        # SAFE ARRAY CREATION with validation
        console.print("\n   Creating feature array...")
        
        if len(features_list) == 0:
            raise ValueError("No features extracted! Check your data and feature extractor.")
        
        # Debug: Check shapes before creating array
        console.print(f"   Total features extracted: {len(features_list)}")
        if len(features_list) > 0:
            sample_shapes = [f.shape if isinstance(f, np.ndarray) else len(f) for f in features_list[:5]]
            console.print(f"   Sample shapes (first 5): {sample_shapes}")
        
        # Verify all features have consistent shape
        inconsistent_indices = []
        first_shape = features_list[0].shape if isinstance(features_list[0], np.ndarray) else (len(features_list[0]),)
        
        for idx, f in enumerate(features_list):
            f_shape = f.shape if isinstance(f, np.ndarray) else (len(f),)
            if f_shape != first_shape:
                console.print(f"[red]❌ Inconsistent shape at index {idx}: {f_shape} != {first_shape}[/red]")
                inconsistent_indices.append(idx)
        
        if inconsistent_indices:
            console.print(f"[red]Found {len(inconsistent_indices)} inconsistent features. Fixing...[/red]")
            # Fix inconsistent entries by padding/truncating
            for idx in inconsistent_indices:
                f = features_list[idx]
                if len(f) < first_shape[0]:
                    features_list[idx] = np.pad(f, (0, first_shape[0] - len(f)), mode='constant')
                else:
                    features_list[idx] = f[:first_shape[0]]
        
        # NOW create the array - should work!
        try:
            # Use vstack for safety (handles 1D arrays better than np.array)
            features = np.vstack(features_list).astype(np.float32)
        except ValueError as e:
            console.print(f"[red]❌ Failed to create feature array: {e}[/red]")
            console.print(f"   features_list length: {len(features_list)}")
            if len(features_list) > 0:
                console.print(f"   Unique shapes: {set([f.shape for f in features_list])}")
            raise
        
        # Align returns with features
        returns = returns[seq_len:seq_len + len(features)]
        
        console.print(f"[green]✅ Feature extraction complete![/green]")
        console.print(f"   Shape: {features.shape}")
        console.print(f"   Feature dimension: {features.shape[1]}")
        console.print(f"   NaN count: {np.isnan(features).sum()}")
        console.print(f"   Inf count: {np.isinf(features).sum()}")
        
        # Get timestamps if available
        timestamps = None
        if 'datetime' in df_pd.columns:
            timestamps = df_pd['datetime'].values[seq_len:seq_len + len(features)]
        
        return {
            'features': features,
            'returns': returns,
            'feature_dim': features.shape[1],
            'timestamps': timestamps,
            'indicator_columns': indicator_cols
        }

    def prepare_training_data(
        self,
        df: pl.DataFrame,
        prediction_horizon: int = 5
    ) -> Dict:
        """Use the optimized version."""
        return self.prepare_training_data_optimized(self, df, prediction_horizon)

    def train_neural_model(
        self,
        training_data: Dict,
        save_path: str = 'best_model.pt'
    ):
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
        
        # Update config with feature dimension
        self.config['feature_dim'] = feature_dim
        
        # Fit scaler on training data
        train_end = int(len(features) * 0.7)
        console.print(f"\n   Fitting scaler on {train_end:,} training samples...")
        self.feature_extractor.fit_scaler(features[:train_end])
        
        # Normalize features
        console.print("   Normalizing features...")
        features_normalized = np.array([
            self.feature_extractor.transform(f) for f in features
        ])
        
        # Train/val/test split (chronological, no shuffle)
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
        
        # Create datasets
        train_dataset = TradingDataset(
            train_features, train_returns,
            seq_len=self.config['seq_len'],
            prediction_horizon=self.config.get('prediction_horizon', 5)
        )
        
        val_dataset = TradingDataset(
            val_features, val_returns,
            seq_len=self.config['seq_len'],
            prediction_horizon=self.config.get('prediction_horizon', 5)
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,  # Don't shuffle time series!
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Create model
        console.print("\n🏗️  [cyan]Building neural architecture...[/cyan]")
        model = create_model(feature_dim, self.config)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        console.print(f"   Total parameters:     {total_params:>12,}")
        console.print(f"   Trainable parameters: {trainable_params:>12,}")
        
        # Create trainer
        trainer = NeuralTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.config,
            device=self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Train
        console.print("\n🎯 [bold green]Starting training loop...[/bold green]")
        trainer.train(self.config.get('num_epochs', 100))
        console.print(f"[cyan]Train size: {len(train_dataset)}[/cyan]")
        console.print(f"[cyan]Val size:   {len(val_dataset)}[/cyan]")

        # Save feature extractor
        import pickle
        feature_extractor_path = save_path.replace('.pt', '_feature_extractor.pkl')
        with open(feature_extractor_path, 'wb') as f:
            pickle.dump(self.feature_extractor, f)
        
        console.print(f"\n✅ [bold green]Training complete![/bold green]")
        console.print(f"   Model saved to: {save_path}")
        console.print(f"   Feature extractor saved to: {feature_extractor_path}")
        
        return trainer, test_features, test_returns


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