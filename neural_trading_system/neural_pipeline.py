# neural_pipeline.py - COMPLETE FILE

import backtrader as bt
import polars as pl
import numpy as np
import torch
from pathlib import Path
from typing import Dict
import time

from backtrader.TransparencyPatch import activate_patch, capture_patch, export_data, optimized_patch
from backtrader.utils.backtest import PolarsDataLoader, DataSpec

# ============================================================================
# ALL INDICATOR IMPORTS
# ============================================================================

from backtrader.indicators.CyberCycle import CyberCycle
from backtrader.indicators.ElhersDecyclerOscillator import DecyclerOscillator
from backtrader.indicators.RoofingFilter import RoofingFilter
from backtrader.indicators.AdaptiveCyberCycle import AdaptiveCyberCycle
from backtrader.indicators.AdaptiveLaguerreFilter import AdaptiveLaguerreFilter
from backtrader.indicators.kama import AdaptiveMovingAverage as KAMA
from backtrader.indicators.hurst import HurstExponent
from backtrader.indicators.wavetrend import WaveTrend
from backtrader.indicators.DamianiVolatmeter import DamianiVolatmeter
from backtrader.indicators.SqueezeVolatility import SqueezeVolatility
from backtrader.indicators.StandarizedATR import StandarizedATR
from backtrader.indicators.RSX import RSX
from backtrader.indicators.qqemod import QQEMod as QQE
from backtrader.indicators.MesaAdaptiveMovingAverage import MAMA
from backtrader.indicators.TrendTriggerFactor import TrendTriggerFactor
from backtrader.indicators.rmi import RelativeMomentumIndex
from backtrader.indicators.SchaffTrendCycle import SchaffTrendCycle

from backtrader.indicators.SuperTrend import SuperTrend
from backtrader.indicators.VumanchuMarketCipher_A import VuManchCipherA
from backtrader.indicators.VumanchuMarketCipher_B import VuManchCipherB
from backtrader.indicators.WaddahAttarExplosion import WaddahAttarExplosion
from backtrader.indicators.vortex import Vortex
from backtrader.indicators.ultimateoscillator import UltimateOscillator
from backtrader.indicators.SuperSmoothFilter import SuperSmoothFilter
from backtrader.indicators.ButterWorth import Butterworth
from backtrader.indicators.ChaikinMoneyFlow import ChaikinMoneyFlow
from backtrader.indicators.ChaikinVolatility import ChaikinVolatility
from backtrader.indicators.Klingeroscillator import KlingerOscillator
from backtrader.indicators.LaguerreFilter import LaguerreFilter
from backtrader.indicators.MADR import MADRIndicator
from backtrader.indicators.MADRV2 import ModifiedMADR
from backtrader.indicators.SMAA import SSMA
from backtrader.indicators.SSLChannel import SSLChannel
from backtrader.indicators.SignalFiller import SignalFiller
from backtrader.indicators.TrendDirectionForceIndex import TrendDirectionForceIndex
from backtrader.indicators.VolumeOscillator import VolumeOscillator
from backtrader.indicators.WilliamsAligator import WilliamsAlligator
from backtrader.indicators.ASH import ASH
from backtrader.indicators.ASI import AccumulativeSwingIndex
from backtrader.indicators.AccumulativeSwingIndex import AccumulativeSwingIndex
from backtrader.indicators.FibonacciLevels import FibonacciLevels
from backtrader.indicators.HeikinAshi import HeikenAshi
from backtrader.indicators.iDecycler import iDecycler
from backtrader.indicators.iFisher import iFisher
from backtrader.indicators.iTrend import iTrend
from backtrader.indicators.lrsi import LaguerreRSI

from backtrader.indicators.tsi import TrueStrengthIndicator
from backtrader.indicators.hma import HullMovingAverage
from backtrader.indicators.zlema import ZeroLagExponentialMovingAverage
from backtrader.indicators.zlind import ZeroLagIndicator
from backtrader.indicators.kst import KnowSureThing
from backtrader.indicators.ols import OLS_Slope_InterceptN
from backtrader.indicators.psar import ParabolicSAR
from backtrader.indicators.dpo import DetrendedPriceOscillator
from backtrader.indicators.dv2 import DV2
from backtrader.indicators.awesomeoscillator import AwesomeOscillator
from backtrader.indicators.prettygoodoscillator import PrettyGoodOscillator


def find_latest_cache(cache_dir='neural_data/features'):
    """Find the most recent feature cache"""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return None
    
    cache_files = list(cache_path.glob('features_*.pkl'))
    if not cache_files:
        return None
    
    return str(max(cache_files, key=lambda p: p.stat().st_mtime))


class DataCollectionStrategy(bt.Strategy):
    """
    Complete data collection strategy with ALL 90+ indicators
    Runs ONLY to collect indicator data via TransparencyPatch. No trading.
    """
    
    params = dict(
        # === CYCLE ===
        use_cycle_signals=True,
        cyber_cycle_period=20,
        roofing_hp_period=48,
        roofing_ss_period=10,
        butterworth_period=20,
        supersmooth_period=10,
        
        # === REGIME ===
        use_regime_signals=True,
        hurst_period=100,
        
        # === VOLATILITY ===
        use_volatility_signals=True,
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
        
        # === MOMENTUM ===
        use_momentum_signals=True,
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
        
        # === TREND ===
        use_trend_signals=True,
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
        
        # === MARKET CIPHER ===
        vumanchu_a_period=10,
        vumanchu_b_period=13,
        waddah_fast=20,
        waddah_slow=40,
        waddah_channel=20,
        waddah_mult=2.0,
        waddah_sensitivity=150,
        
        # === VOLUME ===
        chaikin_mf_period=20,
        klinger_fast=34,
        klinger_slow=55,
        vol_osc_fast=5,
        vol_osc_slow=10,
        
        # === VORTEX ===
        vortex_period=14,
        
        # === OTHER ===
        madr_period=10,
        smaa_period=20,
        hma_period=20,
        zlema_period=20,
        psar_af=0.02,
        psar_max_af=0.2,
        dpo_period=20,
        
        # === CONFIG ===
        backtest=True,
        debug=False,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Standard indicators
        self.atr = bt.indicators.ATR(self.data, period=14)
        
        # ====================================================================
        # CYCLE INDICATORS
        # ====================================================================
        
        if self.p.use_cycle_signals:
            self.cyber_cycle = CyberCycle(self.data, period=self.p.cyber_cycle_period, plot=False)
            self.decycler = DecyclerOscillator(self.data, plot=False)
            self.roofing = RoofingFilter(self.data, hp_period=self.p.roofing_hp_period, ss_period=self.p.roofing_ss_period, plot=False)
            self.adaptive_cycle = AdaptiveCyberCycle(self.data, plot=False)
            self.butterworth = Butterworth(self.data, period=self.p.butterworth_period, plot=False)
            self.supersmooth = SuperSmoothFilter(self.data, period=self.p.supersmooth_period, plot=False)
            self.idecycler = iDecycler(self.data, plot=False)
            self.ifisher = iFisher(self.data, plot=False)
            self.itrend = iTrend(self.data, plot=False)

        # ====================================================================
        # REGIME DETECTION
        # ====================================================================
        
        if self.p.use_regime_signals:
            self.hurst = HurstExponent(self.data, period=self.p.hurst_period, plot=False)
        
        # ====================================================================
        # VOLATILITY INDICATORS
        # ====================================================================
        
        if self.p.use_volatility_signals:
            self.laguerre = AdaptiveLaguerreFilter(self.data, length=self.p.laguerre_length, plot=False)
            self.laguerrefilter = LaguerreFilter(self.data, plot=False)
            self.damiani = DamianiVolatmeter(self.data, atr_fast=self.p.damiani_atr_fast, std_fast=self.p.damiani_std_fast, atr_slow=self.p.damiani_atr_slow, std_slow=self.p.damiani_std_slow, thresh=self.p.damiani_thresh, plot=False)
            self.squeeze = SqueezeVolatility(self.data, period=self.p.squeeze_period, mult=self.p.squeeze_mult, period_kc=self.p.squeeze_period_kc, mult_kc=self.p.squeeze_mult_kc, plot=False)
            self.satr = StandarizedATR(self.data, atr_period=self.p.satr_atr_period, std_period=self.p.satr_std_period, plot=False)
            self.chaikin_vol = ChaikinVolatility(self.data, ema_period=self.p.chaikin_vol_period, roc_period=self.p.chaikin_vol_period, plot=False)
        
        # ====================================================================
        # MOMENTUM INDICATORS
        # ====================================================================
        
        if self.p.use_momentum_signals:
            self.rsx = RSX(self.data, length=self.p.rsx_period, plot=False)
            self.qqe = QQE(self.data, rsi_period=self.p.qqe_period, sf=self.p.qqe_fast, qqe=self.p.qqe_q, threshold=3, wi_period=14, plot=False)
            self.rmi = RelativeMomentumIndex(self.data, period=self.p.rmi_period, lookback=self.p.rmi_lookback, plot=False)
            self.wavetrend = WaveTrend(self.data, period=self.p.wavetrend_period, plot=False)
            self.lrsi = LaguerreRSI(self.data, period=self.p.lrsi_period, plot=False)
            self.tsi = TrueStrengthIndicator(self.data, period2=self.p.tsi_period_slow, period1=self.p.tsi_period_fast, plot=False)
            self.dv2 = DV2(self.data, period=self.p.dv2_period, plot=False)
            self.ultimate = UltimateOscillator(self.data, p1=self.p.ultimate_osc_period1, p2=self.p.ultimate_osc_period2, p3=self.p.ultimate_osc_period3, plot=False)
            self.awesome = AwesomeOscillator(self.data, plot=False)
            self.prettygood = PrettyGoodOscillator(self.data, plot=False)
        
        # ====================================================================
        # TREND INDICATORS
        # ====================================================================
        
        if self.p.use_trend_signals:
            self.kama = KAMA(self.data, period=self.p.kama_period, fast=self.p.kama_fast, slow=self.p.kama_slow, plot=False)
            self.mesa = MAMA(self.data, fast=self.p.mesa_fast, slow=self.p.mesa_slow, plot=False)
            self.ttf = TrendTriggerFactor(self.data, period=self.p.ttf_period, plot=False)
            self.schaff = SchaffTrendCycle(self.data, cycle=self.p.schaff_cycle, fast=self.p.schaff_fast, slow=self.p.schaff_slow, plot=False)
            self.supertrend = SuperTrend(self.data, period=self.p.supertrend_period, multiplier=self.p.supertrend_multiplier, plot=False)
            self.ssl = SSLChannel(self.data, period=self.p.ssl_period, plot=False)
            self.tdfi = TrendDirectionForceIndex(self.data, period=self.p.tdfi_period, plot=False)
            self.kst = KnowSureThing(self.data, rp1=self.p.kst_roc1, rp2=self.p.kst_roc2, rp3=self.p.kst_roc3, rp4=self.p.kst_roc4, plot=False)
            self.williams_alligator = WilliamsAlligator(self.data, jaw_period=self.p.williams_alligator_jaw, teeth_period=self.p.williams_alligator_teeth, lips_period=self.p.williams_alligator_lips, plot=False)
            self.hma = HullMovingAverage(self.data, period=self.p.hma_period, plot=False)
            self.zlema = ZeroLagExponentialMovingAverage(self.data, period=self.p.zlema_period, plot=False)
        
        # ====================================================================
        # MARKET CIPHER INDICATORS
        # ====================================================================
        
        self.vumanchu_a = VuManchCipherA(self.data, plot=False)
        self.vumanchu_b = VuManchCipherB(self.data, plot=False)
        self.waddah = WaddahAttarExplosion(self.data, fast=self.p.waddah_fast, slow=self.p.waddah_slow, channel=self.p.waddah_channel, mult=self.p.waddah_mult, sensitivity=self.p.waddah_sensitivity, plot=False)
        
        # ====================================================================
        # VOLUME INDICATORS
        # ====================================================================
        
        self.chaikin_mf = ChaikinMoneyFlow(self.data, period=self.p.chaikin_mf_period, plot=False)
        self.klinger = KlingerOscillator(self.data, fast=self.p.klinger_fast, slow=self.p.klinger_slow, plot=False)
        self.vol_osc = VolumeOscillator(self.data, shortlen=self.p.vol_osc_fast, longlen=self.p.vol_osc_slow, plot=False)
        
        # ====================================================================
        # VORTEX
        # ====================================================================
        
        self.vortex = Vortex(self.data, period=self.p.vortex_period, plot=False)
        
        # ====================================================================
        # MISCELLANEOUS INDICATORS
        # ====================================================================
        
        self.ash = ASH(self.data, plot=False)
        self.asi = AccumulativeSwingIndex(self.data, plot=False)
        self.accswing = AccumulativeSwingIndex(self.data, plot=False)
        self.madr = MADRIndicator(self.data, window=self.p.madr_period, plot=False)
        self.madrv2 = ModifiedMADR(self.data, plot=False)
        self.smaa = SSMA(self.data, period=self.p.smaa_period, plot=False)
        self.sigfiller = SignalFiller(self.data, plot=False)
        self.fib = FibonacciLevels(self.data, plot=False)
        self.heikinashi = HeikenAshi(self.data, plot=False)
        self.zlind = ZeroLagIndicator(self.data, plot=False)
        self.psar = ParabolicSAR(self.data, af=self.p.psar_af, afmax=self.p.psar_max_af, plot=False)
        self.dpo = DetrendedPriceOscillator(self.data, period=self.p.dpo_period, plot=False)
        
        # ====================================================================
        # STANDARD BACKTRADER INDICATORS
        # ====================================================================
        
        self.rsi = bt.indicators.RSI(self.data, period=14)
        self.rsi7 = bt.indicators.RSI(self.data, period=7)
        self.rsi21 = bt.indicators.RSI(self.data, period=21)
        self.macd = bt.indicators.MACD(self.data)
        self.bb = bt.indicators.BollingerBands(self.data, period=20, devfactor=2.0)
        self.stoch = bt.indicators.Stochastic(self.data, period=14)
        self.adx = bt.indicators.ADX(self.data, period=14)
        self.cci = bt.indicators.CCI(self.data, period=20)
        self.williams = bt.indicators.WilliamsR(self.data, period=14)
        self.aroon = bt.indicators.AroonOscillator(self.data, period=25)
        self.sma20 = bt.indicators.SMA(self.data, period=20)
        self.sma50 = bt.indicators.SMA(self.data, period=50)
        self.sma200 = bt.indicators.SMA(self.data, period=200)
        self.ema12 = bt.indicators.EMA(self.data, period=12)
        self.ema26 = bt.indicators.EMA(self.data, period=26)
        self.dema = bt.indicators.DEMA(self.data, period=20)
        self.tema = bt.indicators.TEMA(self.data, period=20)
        self.trix = bt.indicators.TRIX(self.data)
        self.mom = bt.indicators.Momentum(self.data, period=12)
        self.roc = bt.indicators.RateOfChange(self.data, period=12)

        
        if self.p.debug:
            print(f"✅ DataCollectionStrategy initialized with 90+ indicators")

    def next(self):
        """Capture all indicator data on every bar"""
        capture_patch(self)

    def stop(self):
        """Export all collected indicator data"""
        if self.p.debug:
            print("✅ Data collection complete")
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
            console.print(f"\n🔥 [cyan]Loading cached export: {export_parquet}[/cyan]")
            df_collected = pl.read_parquet(str(export_parquet))
            
            # ✅ FIX: Convert strings to datetime for comparison
            if 'datetime' in df_collected.columns:
                # Ensure datetime column is proper datetime type
                if df_collected['datetime'].dtype != pl.Datetime:
                    df_collected = df_collected.with_columns(
                        pl.col('datetime').str.to_datetime()
                    )
                
                # Convert string dates to datetime for filtering
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
        """
        🆕 COMPLETE: Optimized feature extraction with disk caching, high-signal filtering,
        and price data for exit management labels.
        
        Args:
            df: Polars DataFrame with OHLCV and indicators
            prediction_horizon: Bars to look ahead for labels
            force_cache_file: Path to specific cache file to use (optional)
            
        Returns:
            Dict with:
                - features: np.ndarray [N, feature_dim]
                - returns: np.ndarray [N]
                - prices: np.ndarray [N] - 🆕 ADDED for exit labels
                - feature_dim: int
                - timestamps: np.ndarray [N] (if available)
                - indicator_columns: List[str]
        """
        from rich.console import Console
        import pickle
        import hashlib
        console = Console()
        
        # Build cache key from data hash and config
        cache_dir = Path('neural_data/features')
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # ✅ NEW: First check if force_cache_file is specified
        if force_cache_file:
            cache_file = Path(force_cache_file)
            if cache_file.exists():
                console.print(f"[cyan]🔥 Force loading cache: {cache_file}[/cyan]")
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                console.print(f"[green]✅ Loaded! Shape: {cached['features'].shape}[/green]")
                
                # 🆕 Ensure prices are included
                if 'prices' not in cached and 'close' in df.columns:
                    console.print("[yellow]⚠️ Adding prices to cached data...[/yellow]")
                    seq_len = self.config.get('seq_len', 100)
                    prices = df.get_column('close').to_numpy()
                    cached['prices'] = prices[seq_len:seq_len + len(cached['features'])]
                
                return cached
            else:
                console.print(f"[red]❌ Cache file not found: {cache_file}[/red]")
        
        # ✅ NEW: Auto-detect existing cache files
        existing_caches = list(cache_dir.glob('features_*.pkl'))
        if existing_caches:
            # Use the most recent cache
            latest_cache = max(existing_caches, key=lambda p: p.stat().st_mtime)
            console.print(f"[yellow]🔍 Found existing cache: {latest_cache.name}[/yellow]")
            console.print(f"[yellow]   Use this cache? (it will skip feature extraction)[/yellow]")
            
            # Auto-use the cache
            console.print(f"[cyan]🔥 Auto-loading existing cache: {latest_cache}[/cyan]")
            with open(latest_cache, 'rb') as f:
                cached = pickle.load(f)
            console.print(f"[green]✅ Loaded! Shape: {cached['features'].shape}[/green]")
            
            # 🆕 Ensure prices are included
            if 'prices' not in cached and 'close' in df.columns:
                console.print("[yellow]⚠️ Adding prices to cached data...[/yellow]")
                seq_len = self.config.get('seq_len', 100)
                prices = df.get_column('close').to_numpy()
                cached['prices'] = prices[seq_len:seq_len + len(cached['features'])]
            
            return cached
        
        # Create cache hash (only if no existing cache found)
        data_hash = hashlib.md5(str(df.shape).encode()).hexdigest()[:16]
        config_str = f"seq{self.config.get('seq_len', 100)}_hor{prediction_horizon}"
        cache_file = cache_dir / f'features_{data_hash}_{config_str}.pkl'
        
        # Try to load cached features
        if cache_file.exists():
            console.print(f"[cyan]🔥 Loading cached features from {cache_file.name}[/cyan]")
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            console.print(f"[green]✅ Loaded cached features! Shape: {cached['features'].shape}[/green]")
            console.print(f"   Feature dimension: {cached['feature_dim']}")
            
            # 🆕 Ensure prices are included
            if 'prices' not in cached and 'close' in df.columns:
                console.print("[yellow]⚠️ Adding prices to cached data...[/yellow]")
                seq_len = self.config.get('seq_len', 100)
                prices = df.get_column('close').to_numpy()
                cached['prices'] = prices[seq_len:seq_len + len(cached['features'])]
            
            return cached
        
        console.print("[cyan]Preparing training data (will cache for next run)...[/cyan]")

        # ===== VALIDATION =====
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        # ===== 1) VECTORIZED FORWARD RETURNS =====
        df = df.with_columns([
            ((pl.col('close').shift(-prediction_horizon) - pl.col('close')) / pl.col('close')).alias('forward_return')
        ])
        console.print(f"   ✅ Calculated forward returns (horizon={prediction_horizon})")

        # ===== 2) EXTRACT PRICE DATA (🆕 FOR EXIT LABELS) =====
        prices = df.get_column('close').to_numpy()
        console.print(f"   ✅ Extracted {len(prices):,} price points")

        # ===== 3) INDICATOR COLUMNS =====
        ohlcv_cols = ['bar', 'datetime', 'open', 'high', 'low', 'close', 'volume']
        indicator_cols = [c for c in df.columns if c not in ohlcv_cols + ['forward_return']]
        console.print(f"   ✅ Found {len(indicator_cols)} indicator features")

        # ===== 4) FILL NULLS =====
        from polars import selectors as cs
        df = df.with_columns(cs.numeric().fill_null(0))

        # ===== 5) BUILD NUMPY ARRAYS =====
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

        # ✅ FAST (5 minutes)
        # Build indicator matrix (all data at once)
        indicator_matrix = df.select(indicator_cols).to_numpy().astype(np.float32)
        console.print(f"   📊 Indicator matrix: {indicator_matrix.shape}")

        # Vectorized feature extraction using sliding windows
        from numpy.lib.stride_tricks import sliding_window_view

        console.print(f"   🚀 Vectorized extraction (seq_len={seq_len})...")
        t0 = time.perf_counter()

        # Create sliding windows (NO LOOP!)
        n_samples, n_features = indicator_matrix.shape
        sequences = sliding_window_view(
            indicator_matrix,
            window_shape=(seq_len, n_features),
            axis=(0, 1)
        )

        # Flatten: (n_sequences, seq_len * n_features)
        features = sequences.reshape(sequences.shape[0], -1).astype(np.float32)

        dt = time.perf_counter() - t0
        console.print(f"   ✅ Extracted {len(features)} sequences in {dt:.2f}s")
        console.print(f"      Speed: {len(features) / dt:.1f} sequences/s")

        # 9. ALIGN RETURNS AND PRICES
        # ⚡ FIX: Ensure exact alignment
        n_sequences = len(features)
        returns = returns_array[seq_len:seq_len + n_sequences].copy()
        aligned_prices = prices[seq_len:seq_len + n_sequences].copy()

        # Trim to minimum length (safety for edge cases)
        min_len = min(len(features), len(returns), len(aligned_prices))
        features = features[:min_len]
        returns = returns[:min_len]
        aligned_prices = aligned_prices[:min_len]

        np.nan_to_num(returns, copy=False)

        console.print(f"   📏 Aligned: features={len(features)}, returns={len(returns)}")

        # 10. FILTER
        console.print("📊 Filtering for high-signal data...")
        signal_threshold = np.percentile(np.abs(returns), 20)
        high_signal_mask = np.abs(returns) >= signal_threshold

        # ✅ Now guaranteed to match
        features_filtered = features[high_signal_mask]
        returns_filtered = returns[high_signal_mask]
        prices_filtered = aligned_prices[high_signal_mask]
        
        console.print(f"   Signal threshold: {signal_threshold:.6f}")
        console.print(f"   Kept {len(features_filtered):,} / {len(features):,} bars ({len(features_filtered)/len(features)*100:.1f}%)")
        console.print(f"   Filtered returns - Mean: {np.mean(returns_filtered):.6f}, Std: {np.std(returns_filtered):.6f}")

        console.print(f"[green]✅ Complete! Shape: {features_filtered.shape}, NaN: {np.isnan(features_filtered).sum()}, Inf: {np.isinf(features_filtered).sum()}[/green]")

        # ===== 11) TIMESTAMPS =====
        timestamps = df.get_column('datetime')[seq_len:seq_len + len(features)].to_numpy() if 'datetime' in df.columns else None
        if timestamps is not None:
            timestamps = timestamps[high_signal_mask]

        # ===== 12) BUILD RESULT =====
        result = {
            'features': features_filtered,
            'returns': returns_filtered,
            'prices': prices_filtered,  # 🆕 ADDED PRICES
            'feature_dim': features_filtered.shape[1],
            'timestamps': timestamps,
            'indicator_columns': indicator_cols,
        }
        
        # ===== 13) CACHE RESULT =====
        console.print(f"[yellow]💾 Caching features to {cache_file.name}[/yellow]")
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        console.print(f"[green]✅ Cached! Next run will load in ~2 seconds[/green]")
        
        return result

    def train_neural_model(self, training_data: Dict, save_path: str = 'best_model.pt'):
        """
        🆕 ENHANCED: Train the neural network on collected data with exit management.
        """
        from rich.console import Console
        from torch.utils.data import DataLoader
        from training.trainer import NeuralTrainer, TradingDataset  # 🆕 Use v2
        from models.architecture import create_model  # 🆕 Use v2
        
        console = Console()
        console.print("\n🧠 [bold magenta]Starting Neural Network Training[/bold magenta]")
        
        features = training_data['features']
        returns = training_data['returns']
        prices = training_data['prices']  # 🆕 ADDED
        feature_dim = training_data['feature_dim']
        
        self.config['feature_dim'] = feature_dim
        
        # ===== FIT SCALER =====
        train_end = int(len(features) * 0.7)
        console.print(f"\n   Fitting scaler on {train_end:,} training samples...")
        self.feature_extractor.fit_scaler(features[:train_end])
        
        console.print("   Normalizing features...")
        features_normalized = np.array([self.feature_extractor.transform(f) for f in features])
        
        # ===== SPLIT DATA =====
        val_start = int(len(features) * 0.7)
        test_start = int(len(features) * 0.85)
        
        train_features = features_normalized[:val_start]
        train_returns = returns[:val_start]
        train_prices = prices[:val_start]  # 🆕 ADDED
        
        val_features = features_normalized[val_start:test_start]
        val_returns = returns[val_start:test_start]
        val_prices = prices[val_start:test_start]  # 🆕 ADDED
        
        test_features = features_normalized[test_start:]
        test_returns = returns[test_start:]
        test_prices = prices[test_start:]  # 🆕 ADDED
        
        console.print(f"\n   📊 Data Split:")
        console.print(f"      Train: {len(train_features):>8,} bars ({len(train_features)/len(features)*100:>5.1f}%)")
        console.print(f"      Val:   {len(val_features):>8,} bars ({len(val_features)/len(features)*100:>5.1f}%)")
        console.print(f"      Test:  {len(test_features):>8,} bars ({len(test_features)/len(features)*100:>5.1f}%)")
        
        # ===== CREATE DATASETS =====
        train_dataset = TradingDataset(
            train_features, 
            train_returns, 
            train_prices,  # 🆕 PASS PRICES
            seq_len=self.config['seq_len'], 
            prediction_horizon=self.config.get('prediction_horizon', 5)
        )
        val_dataset = TradingDataset(
            val_features, 
            val_returns, 
            val_prices,  # 🆕 PASS PRICES
            seq_len=self.config['seq_len'], 
            prediction_horizon=self.config.get('prediction_horizon', 5)
        )
        
        # ===== CREATE DATALOADERS =====
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.get('batch_size', 128), 
            shuffle=False,  # Keep temporal order
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
        
        # ===== BUILD MODEL =====
        console.print("\n🗂️  [cyan]Building neural architecture...[/cyan]")
        model = create_model(feature_dim, self.config)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        console.print(f"   Total parameters:     {total_params:>12,}")
        console.print(f"   Trainable parameters: {trainable_params:>12,}")
        
        # ===== CREATE TRAINER =====
        trainer = NeuralTrainer(
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            config=self.config, 
            device=self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # ===== TRAIN =====
        console.print("\n🎯 [bold green]Starting training loop...[/bold green]")
        trainer.train(self.config.get('num_epochs', 200))
        console.print(f"[cyan]Train size: {len(train_dataset)}[/cyan]")
        console.print(f"[cyan]Val size:   {len(val_dataset)}[/cyan]")

        # ===== SAVE FEATURE EXTRACTOR =====
        import pickle
        feature_extractor_path = save_path.replace('.pt', '_feature_extractor.pkl')
        with open(feature_extractor_path, 'wb') as f:
            pickle.dump(self.feature_extractor, f)
        
        # ===== SAVE TEST DATA FOR ANALYSIS =====
        test_data = {
            'features': test_features,
            'returns': test_returns,
            'prices': test_prices  # 🆕 ADDED
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


# ==============================================================================
# TRAINING ORCHESTRATION
# ==============================================================================

def train_neural_system(
    coin: str = 'BTC',
    interval: str = '4h',
    start_date: str = '2000-01-01',
    end_date: str = '2030-12-31',
    collateral: str = 'USDT',
    config: Dict = None,
):
    """
    🆕 ENHANCED: One-command training pipeline with exit management.
    """
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    if config is None:
        config = {
            # === ENHANCED ARCHITECTURE ===
            'seq_len': 100,
            'prediction_horizon': 5,
            'lookback_windows': [5, 10, 20, 50, 100, 200],
            'd_model': 256,
            'num_heads': 8,
            'num_layers': 6,
            'd_ff': 1024,
            'dropout': 0.15,
            'latent_dim': 16,
            
            # === ENHANCED TRAINING ===
            'batch_size': 16,
            'num_epochs': 200,
            'lr': 0.0003,
            'min_lr': 1e-7,
            'weight_decay': 1e-4,
            'grad_accum_steps': 2,
            'T_0': 20,
            'patience': 25,
            'save_every': 5,
            
            # === TRACKING ===
            'use_wandb': False,  # Set to True if you have wandb
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

    # ===== CREATE PIPELINE =====
    pipeline = NeuralDataPipeline(config)
    
    # ===== COLLECT DATA =====
    df = pipeline.collect_data_from_backtrader(
        coin=coin, 
        interval=interval, 
        start_date=start_date, 
        end_date=end_date, 
        collateral=collateral
    )

    # ===== FIND EXISTING CACHE =====
    latest_cache = find_latest_cache()
    if latest_cache:
        console.print(f"[cyan]🔍 Latest cache: {latest_cache}[/cyan]")
    else:
        console.print("[yellow]No feature cache found - will extract features[/yellow]")

    # ===== PREPARE TRAINING DATA =====
    training_data = pipeline.prepare_training_data(
        df, 
        prediction_horizon=config['prediction_horizon'],
        force_cache_file=latest_cache  # Auto-use latest
    )

    # ===== TRAIN MODEL =====
    model_path = f'models/exit_aware_{coin}_{interval}_{start_date}_{end_date}.pt'
    trainer, test_features, test_returns = pipeline.train_neural_model(training_data, save_path=model_path)

    # ===== COMPLETION MESSAGE =====
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

    return pipeline, trainer, training_data


if __name__ == '__main__':
    
    # 🆕 EXIT-AWARE CONFIGURATION
    exit_aware_config = {
        # === ARCHITECTURE - Balanced Performance ===
        'seq_len': 100,
        'prediction_horizon': 5,
        'lookback_windows': [5, 10, 20, 50, 100, 200],
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 1024,
        'dropout': 0.15,
        'latent_dim': 16,
        
        # === TRAINING ===
        'batch_size': 16,
        'num_epochs': 200,
        'lr': 0.0001, # 0.0003
        'min_lr': 1e-7,
        'weight_decay': 1e-4,
        'grad_accum_steps': 2,
        
        # === SCHEDULER ===
        'T_0': 20,
        'patience': 25,
        'save_every': 5,
        
        # === TRACKING ===
        'use_wandb': False,
        'run_name': 'exit_aware_btc_1h',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'best_model_path': 'models/best_exit_aware_model.pt',
    }

    train_neural_system(
        coin='BTC',
        interval='1h',
        start_date='2017-01-01',
        end_date='2025-01-01',
        collateral='USDT',
        config=exit_aware_config
    )