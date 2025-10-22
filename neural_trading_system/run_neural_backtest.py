#!/usr/bin/env python3
"""
Backtest the trained neural trading strategy.
"""

import backtrader as bt
import torch
import numpy as np
import pickle
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Import your infrastructure
import sys
sys.path.append('.')

from backtrader.utils.backtest import PolarsDataLoader, DataSpec
from backtrader.TransparencyPatch import activate_patch, capture_patch, optimized_patch

console = Console()


class NeuralTradingStrategy(bt.Strategy):
    """
    Backtrader strategy using trained neural network.
    """

    params = dict(
        model_path='best_model.pt',
        feature_extractor_path='best_model_feature_extractor.pkl',
        seq_len=100,
        confidence_threshold=0.6,
        min_expected_return=0.01,
        position_size_mode='neural',  # 'neural' or 'fixed'
        fixed_position_size=0.3,
        use_trailing_stop=True,

        # ===== GLOBAL SETTINGS =====
        position_size=0.30,
        atr_period=14,

        # ===== ENTRY SETTINGS =====
        min_vote_threshold=3,
        require_trend_filter=True,

        # ===== EXIT SETTINGS =====
        stop_loss_atr=1.5,
        take_profit_atr=3.0,
        trailing_stop_atr=1.0,
        enable_trailing=True,
        max_bars_in_trade=3000,

        # ===== CATEGORY WEIGHTS =====
        weight_cycle=1.0,
        weight_regime=1.0,
        weight_volatility=1.0,
        weight_momentum=1.0,
        weight_trend=1.0,

        # ===== CYCLE INDICATORS (Strategy 1) =====
        cycle_period=20,
        roofing_hp_period=48,
        roofing_ss_period=10,

        # ===== REGIME INDICATORS (Strategy 2) =====
        hurst_period=100,
        hurst_trending_threshold=0.55,
        hurst_ranging_threshold=0.45,

        # ===== VOLATILITY INDICATORS (Strategy 3) =====
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

        # ===== MOMENTUM INDICATORS (Strategy 4) =====
        rsx_period=14,
        rsx_oversold=30,
        rsx_overbought=70,
        qqe_period=6,           # CHANGED: was qqe_rsi_period
        qqe_fast=5,             # CHANGED: was qqe_sf
        qqe_q=3.0,              # CHANGED: was qqe_wilders_period
        rmi_period=20,
        rmi_lookback=5,         # CHANGED: was rmi_momentum
        wavetrend_period=10,    # CHANGED: only one period param
        wavetrend_oversold=-60,

        # ===== TREND INDICATORS (Strategy 5) =====
        kama_period=30,
        kama_fast=2,
        kama_slow=30,
        mesa_fast=20,           # CHANGED: was mesa_fastlimit (converted to period)
        mesa_slow=50,           # CHANGED: was mesa_slowlimit (converted to period)
        ttf_period=15,
        schaff_cycle=10,        # CHANGED: was schaff_period
        schaff_fast=23,         # CHANGED: was schaff_fastperiod
        schaff_slow=50,         # CHANGED: was schaff_slowperiod

        # ===== FEATURE TOGGLES =====
        use_cycle_signals=True,
        use_regime_signals=True,
        use_volatility_signals=True,
        use_momentum_signals=True,
        use_trend_signals=True,

        debug=True
    )


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self._last_indicator_arrays = None
        self._warmup = max(self.p.seq_len, 50)  # don’t over-warm for short runs
        
        # Activate transparency patch
        activate_patch(debug=self.p.debug)
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Load feature extractor
        with open(self.p.feature_extractor_path, 'rb') as f:
            self.feature_extractor = pickle.load(f)
        
        # ATR for stops
        self.atr = bt.indicators.ATR(self.data, period=14)
        
        # Initialize all indicators (same as DataCollectionStrategy)
        self._init_indicators()
        
        # Feature buffer
        self.feature_buffer = []
        
        # Trade tracking
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.highest_price = None
        
        # Stats
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        self._warmup = 200
        
        if self.p.debug:
            print(f"✅ Neural strategy initialized")
    
    def _load_model(self):
        checkpoint = torch.load(self.p.model_path, map_location=('cuda' if torch.cuda.is_available() else 'cpu'))
        config = checkpoint.get('config', {})
        from neural_trading_system.models.architecture import NeuralTradingModel
        model = NeuralTradingModel(
            feature_dim=config.get('feature_dim', 500),
            d_model=config.get('d_model', 256),
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 6),
            d_ff=config.get('d_ff', 1024),
            dropout=0.0,
            latent_dim=config.get('latent_dim', 8),
            seq_len=self.p.seq_len
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.config = config  # make available to extractor path
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device).eval()
        # cache the expected dim for extractor
        self._expected_dim = int(config.get('feature_dim', 0) or 0)
        return model

    def _init_indicators(self):
        """Initialize all indicators for data collection."""
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
        
        # Initialize (minimal for example)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)

        # ===== CATEGORY 1: CYCLE INDICATORS =====
        if self.p.use_cycle_signals:
            self.cyber_cycle = CyberCycle(self.data, period=self.p.cycle_period)
            self.decycler = DecyclerOscillator(self.data)
            self.roofing = RoofingFilter(
                self.data, 
                hp_period=self.p.roofing_hp_period, 
                ss_period=self.p.roofing_ss_period
            )

        # ===== CATEGORY 2: REGIME INDICATORS =====
        if self.p.use_regime_signals:
            self.hurst = HurstExponent(self.data, period=self.p.hurst_period)
            self.adaptive_cycle = AdaptiveCyberCycle(self.data)

        # ===== CATEGORY 3: VOLATILITY INDICATORS =====
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

        # ===== CATEGORY 4: MOMENTUM INDICATORS =====
        if self.p.use_momentum_signals:
            # RSX uses 'length' not 'period'
            self.rsx = RSX(self.data, length=self.p.rsx_period)
            
            # QQE uses 'period', 'fast', 'q'
            self.qqe = QQE(
                self.data, 
                period=self.p.qqe_period,
                fast=self.p.qqe_fast,
                q=self.p.qqe_q
            )
            
            # RMI uses 'period', 'lookback'
            self.rmi = RelativeMomentumIndex(
                self.data, 
                period=self.p.rmi_period, 
                lookback=self.p.rmi_lookback
            )
            
            # WaveTrend uses only 'period'
            self.wavetrend = WaveTrend(
                self.data, 
                period=self.p.wavetrend_period
            )

        # ===== CATEGORY 5: TREND INDICATORS =====
        if self.p.use_trend_signals:
            self.kama = KAMA(
                self.data, 
                period=self.p.kama_period, 
                fast=self.p.kama_fast, 
                slow=self.p.kama_slow
            )
            
            # MAMA uses 'fast', 'slow' (periods, not limits)
            self.mesa = MAMA(
                self.data,
                fast=self.p.mesa_fast,
                slow=self.p.mesa_slow
            )
            
            self.ttf = TrendTriggerFactor(self.data, period=self.p.ttf_period)
            
            # SchaffTrendCycle uses 'cycle', 'fast', 'slow'
            self.schaff = SchaffTrendCycle(
                self.data, 
                cycle=self.p.schaff_cycle, 
                fast=self.p.schaff_fast, 
                slow=self.p.schaff_slow
            )
    
    def next(self):
        """Main strategy logic."""
        
        # Capture current bar
        capture_patch(self)
        
        # Warmup
        if len(self) < self._warmup:
            return
        
        # Extract features
        features = self._extract_features()
        
        if features is None:
            return
        
        self.feature_buffer.append(features)
        
        if len(self.feature_buffer) > self.p.seq_len:
            self.feature_buffer = self.feature_buffer[-self.p.seq_len:]
        
        if len(self.feature_buffer) < self.p.seq_len:
            return
        
        # Neural prediction
        predictions = self._predict()
        
        if predictions is None:
            return
        
        # Trading logic
        if not self.position:
            self._check_entry(predictions)
        else:
            self._check_exit(predictions)
    
    def _extract_features(self):
        try:
            # Build full-history arrays per indicator key
            if optimized_patch.current_batch:
                keys = [k for k in optimized_patch.current_batch[0].keys() if k not in ['bar', 'datetime']]
                indicator_arrays = {k: np.asarray([b.get(k, 0.0) for b in optimized_patch.current_batch], dtype=np.float32) for k in keys}
                self._last_indicator_arrays = indicator_arrays  # cache last non-empty
            else:
                if self._last_indicator_arrays is None:
                    if self.p.debug:
                        print("current_batch empty")
                    return None
                indicator_arrays = self._last_indicator_arrays  # reuse last arrays

            feats = self.feature_extractor.extract_all_features(indicator_arrays)
            f = np.asarray(feats, dtype=np.float32).ravel()

            # Enforce training width before scaling
            expected_dim = int(self._expected_dim or f.size)
            if f.size != expected_dim:
                if f.size < expected_dim:
                    f = np.pad(f, (0, expected_dim - f.size))
                else:
                    f = f[:expected_dim]

            # Scale with the saved scaler (2-D in)
            f = self.feature_extractor.transform(f.reshape(1, -1)).ravel()

            # Optional one-time sanity log
            if self.p.debug and not hasattr(self, "_logged_dim"):
                print(f"Live feature width: {f.size} (expected {expected_dim})")
                self._logged_dim = True

            return f
        except Exception as e:
            if self.p.debug:
                print(f"Error extracting features: {e}")
                import traceback; traceback.print_exc()
            return None

    def _predict(self):
        try:
            seq = np.array(self.feature_buffer)  # [T, F]
            x = torch.as_tensor(seq, dtype=torch.float32, device=next(self.model.parameters()).device).unsqueeze(0)
            with torch.no_grad():
                out = self.model(x)

            # Handle dict/tuple/tensor outputs; take logits then sigmoid for decisions
            if isinstance(out, dict):
                entry_logit = out.get('entry_prob', torch.tensor(0.0, device=x.device))
                exit_logit  = out.get('exit_prob',  torch.tensor(0.0, device=x.device))
                exp_ret     = out.get('expected_return', torch.tensor(0.0, device=x.device))
                vol_fc      = out.get('volatility_forecast', torch.tensor(0.0, device=x.device))
                pos_size    = out.get('position_size', torch.tensor(self.p.fixed_position_size, device=x.device))
            elif isinstance(out, (list, tuple)):
                entry_logit = out[0]
                exit_logit  = out[1] if len(out) > 1 else torch.zeros_like(entry_logit)
                exp_ret, vol_fc, pos_size = torch.tensor(0.0, device=x.device), torch.tensor(0.0, device=x.device), torch.tensor(self.p.fixed_position_size, device=x.device)
            else:
                entry_logit = out
                exit_logit  = torch.zeros_like(entry_logit)
                exp_ret, vol_fc, pos_size = torch.tensor(0.0, device=x.device), torch.tensor(0.0, device=x.device), torch.tensor(self.p.fixed_position_size, device=x.device)

            entry_prob = torch.sigmoid(entry_logit).item()
            exit_prob  = torch.sigmoid(exit_logit).item()
            pos_size   = float(np.clip(pos_size.detach().cpu().item(), 0.0, 1.0))

            return {
                'entry_prob': entry_prob,
                'exit_prob': exit_prob,
                'expected_return': float(exp_ret.detach().cpu().item()),
                'volatility_forecast': float(vol_fc.detach().cpu().item()),
                'position_size': pos_size,
            }
        except Exception as e:
            if self.p.debug:
                print(f"Error in prediction: {e}")
                import traceback; traceback.print_exc()
            return None

    def _check_entry(self, predictions):
        """Check if should enter trade."""
        entry_prob = predictions['entry_prob']
        expected_return = predictions['expected_return']
        neural_size = predictions['position_size']
        
        # Entry conditions
        if (entry_prob > self.p.confidence_threshold and
            expected_return > self.p.min_expected_return):
            
            # Calculate position size
            if self.p.position_size_mode == 'neural':
                size_pct = neural_size
            else:
                size_pct = self.p.fixed_position_size
            
            available_cash = self.broker.getcash()
            size = (available_cash * size_pct) / self.data.close[0]
            
            # Set stops
            self.stop_price = self.data.close[0] - (self.atr[0] * self.p.stop_loss_atr)
            self.target_price = self.data.close[0] + (self.atr[0] * 3.0)
            
            self.buy(size=size)
            
            if self.p.debug:
                print(f"🚀 ENTRY - Prob: {entry_prob:.2f}, ExpRet: {expected_return:.2%}, Size: {size_pct:.2%}")
    
    def _check_exit(self, predictions):
        """Check if should exit trade."""
        exit_prob = predictions['exit_prob']
        
        # Update trailing stop
        if self.p.use_trailing_stop:
            if self.data.high[0] > self.highest_price:
                self.highest_price = self.data.high[0]
            
            trailing_stop = self.highest_price - (self.atr[0] * self.p.trailing_stop_atr)
            self.stop_price = max(self.stop_price, trailing_stop)
        
        # Exit conditions
        stop_hit = self.data.close[0] <= self.stop_price
        target_hit = self.data.close[0] >= self.target_price
        neural_exit = exit_prob > 0.7
        
        if stop_hit or target_hit or neural_exit:
            self.close()
            
            if self.p.debug:
                reason = "Stop" if stop_hit else "Target" if target_hit else "Neural"
                print(f"🛑 EXIT - {reason}")
    
    def notify_order(self, order):
        """Track order execution."""
        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_price = order.executed.price
                self.highest_price = order.executed.price
            elif order.issell():
                pnl = order.executed.price - self.entry_price
                pnl_pct = (pnl / self.entry_price) * 100
                
                self.trade_count += 1
                if pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                if self.p.debug:
                    print(f"Trade #{self.trade_count} - P&L: {pnl_pct:.2f}%")


def run_backtest(
    coin='BTC',
    interval='4h',
    start_date='2024-01-01',
    end_date='2024-12-31',
    collateral='USDT',
    init_cash=10000,
    model_path='best_model.pt',
    feature_extractor_path='best_model_feature_extractor.pkl',
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
    cerebro.addstrategy(
        NeuralTradingStrategy,
        model_path=model_path,
        feature_extractor_path=feature_extractor_path,
        confidence_threshold=0.6,
        position_size_mode='neural',
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
        start_date='2024-01-01',
        end_date='2024-12-31',
        collateral='USDT',
        init_cash=10000,
        model_path='neural_trading_system/models/best_model.pt', # TODO fix pathing
        feature_extractor_path='neural_trading_system/models/neural_BTC_4h_2017-01-01_2024-01-01_feature_extractor.pkl', # TODO fix naming
        plot=True
    )