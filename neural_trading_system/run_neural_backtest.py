import backtrader as bt
import torch
import numpy as np
import pickle
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import sys
sys.path.append('.')

from backtrader.utils.backtest import PolarsDataLoader, DataSpec
from backtrader.TransparencyPatch import activate_patch, capture_patch, optimized_patch

console = Console()

class PerfectNeuralStrategy2(bt.Strategy):
    """
    PERFECT MODEL-ALIGNED BACKTEST STRATEGY
    
    FINAL FIX: Feature extraction now properly handles batch = list[dict]
    """
    
    params = dict(
        # === MODEL PATHS ===
        model_path='best_model.pt',
        feature_extractor_path='best_model_feature_extractor.pkl',
        
        # === MODEL CONFIGURATION ===
        seq_len=100,
        
        # === DENORMALIZATION ===
        return_scale=0.05,
        
        # === ENTRY/EXIT THRESHOLDS ===
        min_entry_prob=0.35,
        min_expected_return=0.005,
        max_exit_prob=0.55,
        max_expected_return=-0.003,
        
        # === POSITION SIZING ===
        position_size_mode='neural',
        fixed_position_size=0.15,
        max_position_size=0.30,
        min_position_size=0.05,
        
        # === RISK MANAGEMENT ===
        use_stops=True,
        stop_loss_atr_mult=2.0,
        take_profit_atr_mult=3.0,
        use_trailing_stop=True,
        trailing_stop_atr_mult=1.5,
        atr_period=14,
        
        # === PERFORMANCE ===
        prediction_interval=5,
        
        # === DEBUG ===
        debug=True,
        log_every=50,
    )

    def _init_indicators(self):
        """Initialize ALL indicators (same as training!)"""
        # Import all indicators
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
        
        # Initialize all indicators (matching training)
        self.cyber_cycle = CyberCycle(self.data, period=20, plot=False)
        self.decycler = DecyclerOscillator(self.data, plot=False)
        self.roofing = RoofingFilter(self.data, hp_period=48, ss_period=10, plot=False)
        self.hurst = HurstExponent(self.data, period=100, plot=False)
        self.adaptive_cycle = AdaptiveCyberCycle(self.data, plot=False)
        self.laguerre = AdaptiveLaguerreFilter(self.data, length=20, plot=False)
        self.damiani = DamianiVolatmeter(self.data, atr_fast=13, std_fast=20, 
                                        atr_slow=40, std_slow=100, thresh=1.4, plot=False)
        self.squeeze = SqueezeVolatility(self.data, period=20, mult=2, 
                                        period_kc=20, mult_kc=1.5, plot=False)
        self.satr = StandarizedATR(self.data, atr_period=14, std_period=20, plot=False)
        self.rsx = RSX(self.data, length=14, plot=False)
        self.qqe = QQE(self.data, period=6, fast=5, q=3.0, plot=False)
        self.rmi = RelativeMomentumIndex(self.data, period=20, lookback=5, plot=False)
        self.wavetrend = WaveTrend(self.data, period=10, plot=False)
        self.kama = KAMA(self.data, period=30, fast=2, slow=30, plot=False)
        self.mesa = MAMA(self.data, fast=20, slow=50, plot=False)
        self.ttf = TrendTriggerFactor(self.data, period=15, plot=False)
        self.schaff = SchaffTrendCycle(self.data, cycle=10, fast=23, slow=50, plot=False)


    def __init__(self):
        console.print(Panel("[bold cyan]Perfect Neural Strategy[/bold cyan] - Model-Aligned Backtest", 
                           style="cyan"))
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        console.print(f"🔧 Device: {self.device}")
        
        # Load model
        console.print(f"📦 Loading model from {self.p.model_path}")
        checkpoint = torch.load(self.p.model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        from neural_trading_system.models.architecture import create_model
        
        input_proj_weight = state_dict.get('input_projection.weight', None)
        if input_proj_weight is None:
            raise ValueError("Cannot find input_projection.weight in model state dict")
        
        feature_dim = input_proj_weight.shape[1]
        console.print(f"📊 Feature dimension: {feature_dim}")
        
        self.model = create_model(feature_dim=feature_dim, config={'seq_len': self.p.seq_len})
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        console.print("✅ Model loaded and set to eval mode")

        self._init_indicators()
        
        # Load feature extractor
        console.print(f"📦 Loading feature extractor from {self.p.feature_extractor_path}")
        with open(self.p.feature_extractor_path, 'rb') as f:
            self.feature_extractor = pickle.load(f)
        
        expected_feat_dim = self.feature_extractor.scaler.n_features_in_
        console.print(f"✅ Feature extractor loaded (dim={expected_feat_dim})")
        
        if expected_feat_dim != feature_dim:
            console.print(f"⚠️  [yellow]Warning: Feature dim mismatch! "
                         f"Extractor={expected_feat_dim}, Model={feature_dim}[/yellow]")
        
        # ATR indicator
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        
        # State tracking
        self.feature_buffer = []
        self.prediction_counter = 0
        self.last_prediction = None
        self.entry_bar = None
        self.stop_price = None
        self.target_price = None
        self.highest_price = None
        
        # Cache for optimization
        self._cached_indicator_keys = None
        self._last_indicator_arrays = None
        
        # Pre-allocate tensor
        self._input_tensor = torch.zeros(
            1, self.p.seq_len, feature_dim,
            dtype=torch.float32,
            device=self.device
        )
        
        self._warmup = self.p.seq_len + self.p.atr_period + 10
        
        # Stats
        self.stats = {
            'total_signals': 0,
            'trades_taken': 0,
            'trades_won': 0,
            'trades_lost': 0,
            'trades_exited': 0,
        }
        
        console.print(f"🔥 Strategy initialized - Warmup: {self._warmup} bars")
    
    def _extract_features(self):
        """
        FINAL FIX: Extract features from batch = list[dict]
        This matches your old run_neural_backtest.py implementation
        """
        try:
            batch = optimized_patch.current_batch
            
            if batch is None or len(batch) == 0:
                return None
            
            # Batch is a list of dicts: [{indicator: value, ...}, {...}, ...]
            # Convert to dict of arrays: {indicator: [val1, val2, ...], ...}
            
            if self._cached_indicator_keys is None:
                # Cache keys (exclude metadata)
                self._cached_indicator_keys = [
                    k for k in batch[0].keys()
                    if k not in ['bar', 'datetime']
                ]
            
            # Build indicator arrays
            indicator_arrays = {
                k: np.array([b.get(k, 0.0) for b in batch], dtype=np.float32)
                for k in self._cached_indicator_keys
            }
            
            self._last_indicator_arrays = indicator_arrays
            
            # Extract features using feature_extractor
            feats = self.feature_extractor.extract_all_features(indicator_arrays)
            f = np.asarray(feats, dtype=np.float32).ravel()
            
            # Enforce dimension
            expected_dim = self._input_tensor.shape[2]
            if f.size != expected_dim:
                if f.size < expected_dim:
                    f = np.pad(f, (0, expected_dim - f.size), mode='constant')
                else:
                    f = f[:expected_dim]
            
            # Scale
            f = self.feature_extractor.transform(f.reshape(1, -1)).ravel()
            
            return f
            
        except Exception as e:
            if self.p.debug:
                console.print(f"[red]Feature extraction error: {e}[/red]")
                import traceback
                console.print(f"[red]{traceback.format_exc()}[/red]")
            return None
    
    def _predict(self):
        """Run model prediction"""
        try:
            seq = np.array(self.feature_buffer, dtype=np.float32)
            self._input_tensor[0, :, :] = torch.from_numpy(seq)
            
            with torch.no_grad():
                out = self.model(self._input_tensor)
            
            # Extract outputs (no double sigmoid)
            entry_prob = out['entry_prob'].squeeze().item()
            exit_prob = out['exit_prob'].squeeze().item()
            
            # Denormalize expected_return
            exp_ret_norm = out['expected_return'].squeeze().item()
            exp_ret = exp_ret_norm * self.p.return_scale
            
            vol_forecast = out['volatility_forecast'].squeeze().item()
            pos_size = out['position_size'].squeeze().item()
            
            return {
                'entry_prob': entry_prob,
                'exit_prob': exit_prob,
                'expected_return': exp_ret,
                'expected_return_norm': exp_ret_norm,
                'volatility_forecast': vol_forecast,
                'position_size': pos_size,
            }
            
        except Exception as e:
            if self.p.debug:
                console.print(f"[red]Prediction error: {e}[/red]")
                import traceback
                console.print(f"[red]{traceback.format_exc()}[/red]")
            return None
    
    def next(self):
        """Main strategy logic"""
        capture_patch(self)
        
        if len(self) < self._warmup:
            return
        
        # Extract features EVERY bar
        features = self._extract_features()
        if features is None:
            return
        
        # Maintain consecutive buffer
        self.feature_buffer.append(features)
        if len(self.feature_buffer) > self.p.seq_len:
            self.feature_buffer.pop(0)
        
        if len(self.feature_buffer) < self.p.seq_len:
            return
        
        # Predict
        should_predict = (self.prediction_counter % self.p.prediction_interval == 0)
        self.prediction_counter += 1
        
        if should_predict:
            self.last_prediction = self._predict()
        
        if self.last_prediction is None:
            return
        
        # Debug logging
        if self.p.debug and len(self) % self.p.log_every == 0:
            pred = self.last_prediction
            console.print(
                f"[{len(self):5d}] "
                f"entry={pred['entry_prob']:.3f} "
                f"exit={pred['exit_prob']:.3f} "
                f"exp_ret={pred['expected_return']:+.4f} "
                f"size={pred['position_size']:.3f}"
            )
        
        # Trading logic
        if not self.position:
            self._check_entry(self.last_prediction)
        else:
            self._check_exit(self.last_prediction)
    
    def _check_entry(self, pred):
        """Entry logic - use params"""
        entry_prob = pred['entry_prob']
        exp_ret = pred['expected_return']
        
        # Use params instead of hardcoded
        if entry_prob < self.p.min_entry_prob:
            return
        if exp_ret < self.p.min_expected_return:
            return
        
        size_pct = 0.20
        available_cash = self.broker.getcash()
        price = self.data.close[0]
        size = (available_cash * size_pct) / price
        
        self.buy(size=size)
        self.entry_bar = len(self)
        self.stats['trades_taken'] += 1
        
        if self.p.debug:
            console.print(
                f"🚀 ENTRY - Prob: {entry_prob:.3f} | "
                f"ExpRet: {exp_ret:+.4f} | Size: {size_pct:.1%}"
            )

    def _check_exit(self, pred):
        """Exit logic - use params"""  
        exit_prob = pred['exit_prob']
        
        if exit_prob > self.p.max_exit_prob:
            # Get entry price before closing
            entry_price = self.position.price if self.position else None
            exit_price = self.data.close[0]
            pnl_pct = ((exit_price - entry_price) / entry_price) if entry_price else 0
            
            self.close()
            self.stats['trades_exited'] += 1
            
            # Track win/loss
            if pnl_pct > 0:
                self.stats['trades_won'] += 1
            else:
                self.stats['trades_lost'] += 1
            
            if self.p.debug:
                console.print(
                    f"🛑 EXIT - Exit prob: {exit_prob:.2f} | "
                    f"P&L: {pnl_pct:+.2%}"
                )

    def notify_order(self, order):
        pass
    
    def notify_trade(self, trade):
        pass
    
    def stop(self):
        """Final statistics"""
        win_rate = (self.stats['trades_won'] / self.stats['trades_taken'] 
                   if self.stats['trades_taken'] > 0 else 0.0)
        
        table = Table(title="📊 Final Statistics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Signals", str(self.stats['total_signals']))
        table.add_row("Trades Taken", str(self.stats['trades_taken']))
        table.add_row("Trades Won", str(self.stats['trades_won']))
        table.add_row("Trades Lost", str(self.stats['trades_lost']))
        table.add_row("Win Rate", f"{win_rate:.1%}")
        table.add_row("Final Value", f"${self.broker.getvalue():.2f}")
        
        console.print(table)


import torch
import numpy as np
import random

# Force determinism
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # ~24mb bigger GPU memory footprint or ':16:8' for less memory (!) may limit overall performance (!)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def run_backtest(
    coin='BTC',
    interval='4h',
    start_date='2025-01-01',
    end_date='2030-12-31',
    collateral='USDT',
    init_cash=10000,
    model_path='best_model.pt',
    plot=True
):
    """Run neural strategy backtest."""
    
    console.print(Panel.fit(
        f"[bold cyan]NEURAL STRATEGY BACKTEST[/bold cyan]\n"
        f"[yellow]{coin}/{collateral} - {interval}[/yellow]\n\n"
        f"Period: {start_date} → {end_date}\n"
        f"Initial Cash: ${init_cash:,}\n"
        f"Model: {model_path}",
        title="🧠 Backtest",
        border_style="cyan"
    ))
    
    # Initialize cerebro
    cerebro = bt.Cerebro(oldbuysell=True, runonce=False, stdstats=False)
    
    # Load data
    console.print(f"\n📥 [cyan]Loading data for {coin}...[/cyan]")
    
    loader = PolarsDataLoader()
    spec = DataSpec(
        symbol=coin,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        collateral=collateral
    )
    
    df = loader.load_data(spec, use_cache=True)
    data_feed = loader.make_backtrader_feed(df, spec)
    
    console.print(f"✅ [green]Loaded {len(df):,} bars[/green]")
    
    cerebro.adddata(data_feed)
    
    # Add strategy
    import json

    # Load the created config
    with open('neural_trading_system/models/model_config.json', 'r') as f:
        config = json.load(f)

    backtest_params = config['backtest_requirements']

    # Use config values instead of hardcoded!
    cerebro.addstrategy(
        PerfectNeuralStrategy2,
        model_path='neural_trading_system/models/best_model.pt',
        feature_extractor_path='neural_trading_system/models/neural_BTC_4h_2017-01-01_2024-01-01_feature_extractor.pkl',
        return_scale=backtest_params['critical_parameters']['return_scale'],  # ← FROM CONFIG
        min_entry_prob=backtest_params['recommended_thresholds']['min_entry_prob'],  # ← FROM CONFIG
        min_expected_return=backtest_params['recommended_thresholds']['min_expected_return'],  # ← FROM CONFIG
        max_exit_prob=backtest_params['recommended_thresholds']['max_exit_prob'],  # ← FROM CONFIG
        prediction_interval=5,
        debug=True
    )
    
    # Broker settings
    cerebro.broker.setcash(init_cash)
    cerebro.broker.setcommission(commission=0.001)
    
    # Analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    # Run
    console.print("\n📊 [bold green]Running backtest...[/bold green]")
    results = cerebro.run()
    strat = results[0]
    
    # Results
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - init_cash) / init_cash * 100
    
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
    drawdown = strat.analyzers.drawdown.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    
    # Print results
    console.print("\n" + "="*80)
    console.print(Panel.fit(
        "[bold cyan]BACKTEST RESULTS[/bold cyan]",
        border_style="cyan"
    ))
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="yellow")
    
    table.add_row("Initial Cash", f"${init_cash:,.2f}")
    table.add_row("Final Value", f"${final_value:,.2f}")
    table.add_row("Total Return", f"{total_return:.2f}%")
    table.add_row("Sharpe Ratio", f"{sharpe:.3f}" if sharpe else "N/A")
    table.add_row("Max Drawdown", f"{drawdown.get('max', {}).get('drawdown', 0):.2f}%")
    table.add_row("", "")
    table.add_row("Total Trades", str(trades.get('total', {}).get('total', 0)))
    table.add_row("Won Trades", str(trades.get('won', {}).get('total', 0)))
    table.add_row("Lost Trades", str(trades.get('lost', {}).get('total', 0)))
    
    if trades.get('total', {}).get('total', 0) > 0:
        win_rate = trades.get('won', {}).get('total', 0) / trades.get('total', {}).get('total', 1) * 100
        table.add_row("Win Rate", f"{win_rate:.2f}%")
    
    console.print(table)
    console.print("="*80)
    
    # Plot
    if plot:
        console.print("\n📊 [cyan]Generating plot...[/cyan]")
        cerebro.plot(style='candlestick', barup='green', bardown='red')
    
    return final_value


if __name__ == '__main__':
    run_backtest(
        coin='BTC',
        interval='4h',
        start_date='2020-01-01',  # More recent training window
        end_date='2024-01-01',
        collateral='USDT',
        init_cash=10000,
        model_path='neural_trading_system/models/best_model.pt', # TODO fix pathing
        plot=True
    )
    