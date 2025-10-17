import math
import numpy as np
from backtrader.utils.backtest import backtest, bulk_backtest
from backtrader.strategies.base import BaseStrategy, bt, OrderTracker, datetime
from backtrader.indicators.MesaAdaptiveMovingAverage import MAMA

# Enhanced QQE Indicator with Smoothed Crossover Signals
class QQEIndicator(bt.Indicator):
    params = (
        ("period", 14),
        ("fast", 5),
        ("q", 4.236),
        ("smoothing", 7),
        ("debug", False)
    )
    lines = ("qqe_line", "qqe_smoothed", "qqe_signal", "qqe_trend")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Core QQE calculation
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.fast)
        
        # Improved DAR calculation with proper smoothing
        rsi_ma = bt.indicators.EMA(self.rsi, period=int(self.p.period/2))
        dar_period = int((self.p.period * 2) - 1)
        self.dar = bt.indicators.EMA(bt.If(self.atr > 0, self.atr * self.p.q, 0), period=dar_period)
        
        # Enhanced QQE line calculation
        self.lines.qqe_line = self.rsi + bt.If(self.rsi > rsi_ma, self.dar, -self.dar)
        
        # Smoothed versions for signal generation
        self.lines.qqe_smoothed = bt.indicators.EMA(self.lines.qqe_line, period=self.p.smoothing)
        self.lines.qqe_signal = bt.indicators.EMA(self.lines.qqe_smoothed, period=self.p.smoothing * 2)
        
        # Trend determination (above/below 50 level)
        self.lines.qqe_trend = bt.If(self.lines.qqe_smoothed > 50, 1, -1)
    
    def get_signal(self):
        """More permissive QQE signals"""
        if len(self.lines.qqe_smoothed) < 2:
            return 0
            
        # Crossover signals
        bull_cross = (self.lines.qqe_smoothed[-1] < self.lines.qqe_signal[-1] and 
                    self.lines.qqe_smoothed[0] > self.lines.qqe_signal[0])
        bear_cross = (self.lines.qqe_smoothed[-1] > self.lines.qqe_signal[-1] and 
                    self.lines.qqe_smoothed[0] < self.lines.qqe_signal[0])
        
        # Level-based signals
        bull_level = self.lines.qqe_smoothed[0] > 50 and self.lines.qqe_smoothed[-1] <= 50
        bear_level = self.lines.qqe_smoothed[0] < 50 and self.lines.qqe_smoothed[-1] >= 50
        
        # Simplified logic
        if bull_cross or bull_level or self.lines.qqe_smoothed[0] > 60:
            return 1
        elif bear_cross or bear_level or self.lines.qqe_smoothed[0] < 40:
            return -1
        else:
            return 0

# Enhanced Volume Oscillator with Advanced Signal Processing
class VolumeOscillator(bt.Indicator):
    params = (
        ("short_period", 14),
        ("long_period", 28),
        ("smooth_period", 7),
        ("signal_period", 14),
        ("debug", False)
    )
    lines = ("osc", "smoothed", "signal", "trend", "momentum")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Basic volume oscillator calculation
        short_vol = bt.indicators.SMA(self.data.volume, period=self.p.short_period)
        long_vol = bt.indicators.SMA(self.data.volume, period=self.p.long_period)
        
        # Raw oscillator (percentage difference)
        self.lines.osc = ((short_vol - long_vol) / (long_vol + 1e-6)) * 100
        
        # Smoothed oscillator for cleaner signals
        self.lines.smoothed = bt.indicators.EMA(self.lines.osc, period=self.p.smooth_period)
        
        # Signal line (longer smoothing for crossovers)
        self.lines.signal = bt.indicators.EMA(self.lines.smoothed, period=self.p.signal_period)
        
        # Volume trend (positive/negative momentum)
        self.lines.trend = bt.If(self.lines.smoothed > self.lines.signal, 1, -1)
        
        # Volume momentum (rate of change)
        self.lines.momentum = bt.indicators.RateOfChange(self.lines.smoothed, period=5)

    def get_signal(self):
        """More permissive volume signals"""
        if len(self.lines.smoothed) < 2:
            return 0
            
        # Crossover detection
        bull_cross = (self.lines.smoothed[-1] < self.lines.signal[-1] and 
                    self.lines.smoothed[0] > self.lines.signal[0])
        bear_cross = (self.lines.smoothed[-1] > self.lines.signal[-1] and 
                    self.lines.smoothed[0] < self.lines.signal[0])
        
        # Simple regime-based signals
        if bull_cross or (self.lines.smoothed[0] > self.lines.signal[0] and self.lines.smoothed[0] > 0):
            return 1
        elif bear_cross or (self.lines.smoothed[0] < self.lines.signal[0] and self.lines.smoothed[0] < 0):
            return -1
        else:
            return 0

# Hull Moving Average
class HMA(bt.Indicator):
    lines = ('hma',)
    params = dict(period=20)
    plotinfo = dict(plot=True, subplot=False)
    
    def __init__(self):
        p = self.p.period
        self.wma_half = bt.ind.WeightedMovingAverage(self.data, period=int(p / 2))
        self.wma_full = bt.ind.WeightedMovingAverage(self.data, period=p)
        self.raw_hma = 2 * self.wma_half - self.wma_full
        self.lines.hma = bt.ind.WeightedMovingAverage(self.raw_hma, period=int(math.sqrt(p)))

# Zero Lag Indicator
class ZeroLag(bt.Indicator):
    lines = ('zerolag',)
    params = dict(period=20)
    plotinfo = dict(plot=True, subplot=False)
    
    def __init__(self):
        self.hma = HMA(self.data, period=self.p.period)
        self.hma_sma = bt.ind.SMA(self.hma, period=int(self.p.period / 2))
        self.lines.zerolag = 2 * self.hma - self.hma_sma

# Sine Weighted Moving Average
class SineWeightedMA(bt.Indicator):
    lines = ('sine_wma',)
    params = dict(period=20)
    plotinfo = dict(plot=True, subplot=False)
    
    def __init__(self):
        self.lines.sine_wma = bt.ind.SMA(self.data, period=self.p.period)

# Enhanced Strategy with Quantitative Multi-Indicator Signals
class QuantitativeMultiIndicatorStrategy(BaseStrategy):
    params = dict(
        # Zero Lag and Moving Average parameters
        zl_period=14,
        sine_period=20,
        sma_filter=200,
        mom_period=14,
        mom_lookback=3,
        
        # QQE parameters
        qqe_period=10,
        qqe_fast=5,
        qqe_q=4.236,
        qqe_smoothing=5,
        
        # Volume Oscillator parameters
        vol_short=14,
        vol_long=28,
        vol_smooth=5,
        vol_signal=10,
        
        # Risk management - IMPROVED
        take_profit=8,           # Increased from 2%
        stop_loss=4,             # Tighter stop
        dca_deviation=2.5,       # Wider spacing
        percent_sizer=0.03,      # More aggressive (3% per trade)
        
        # Signal filtering - RELAXED
        use_volatility_filter=False,
        adx_threshold=15,
        min_signal_strength=0.2,  # Much more permissive

        fast=13,
        slow=17,
        
        debug=True,
        backtest=True
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Price-based indicators
        self.zero_lag = ZeroLag(self.data, period=self.p.zl_period, plot=False)
        self.sine_wma = SineWeightedMA(self.zero_lag, period=self.p.sine_period, plot=False)
        self.sma_200 = bt.ind.SMA(self.data, period=self.p.sma_filter, plot=True)
        self.momentum = bt.ind.Momentum(self.data, period=self.p.mom_period, plot=True)
        self.zl_crossover = bt.ind.CrossOver(self.zero_lag, self.sine_wma, plot=True)

        # SMACROSSMESAADAPTIVEPRIME
        self.sma17 = bt.ind.SMA(period=17)
        self.sma47 = bt.ind.SMA(period=47)
        self.mama = MAMA(self.data, fast=self.p.fast, slow=self.p.slow)
        self.sma_crossover = bt.ind.CrossOver(self.sma17, self.sma47)
        
        self.DCA = True
        
        # Enhanced indicators
        self.qqe = QQEIndicator(
            self.data, 
            period=self.p.qqe_period,
            fast=self.p.qqe_fast,
            q=self.p.qqe_q,
            smoothing=self.p.qqe_smoothing,
            subplot=True
        )
        
        self.volosc = VolumeOscillator(
            self.data,
            short_period=self.p.vol_short,
            long_period=self.p.vol_long,
            smooth_period=self.p.vol_smooth,
            signal_period=self.p.vol_signal,
            subplot=False
        )
        
        # Volatility and trend strength filters
        self.atr = bt.indicators.ATR(self.data, period=14, plot=False)
        self.adx = bt.indicators.ADX(self.data, period=14, plot=False)
        
        # Position tracking
        self.active_orders = []
        self.entry_prices = []
        self.sizes = []
        self.buy_executed = False
        self.signal_history = []
        self.last_dca_bar = -999  # Track last DCA to avoid multiple per bar
    
    def _determine_size(self):
        available_cash = self.broker.get_cash()
        size = (available_cash * self.p.percent_sizer) / self.data.close[0]
        return max(size, 0.001)
    
    def reset_position_state(self):
        self.active_orders = []
        self.entry_prices = []
        self.sizes = []
        self.buy_executed = False
        self.last_dca_bar = -999
    
    def get_signal(self):
        """Simplified signal generation"""
        if len(self.momentum) < self.p.mom_lookback:
            return 0
            
        # Basic trend confirmation
        price_above_sma = self.data.close[0] > self.sma_200[0]
        
        # Momentum check (but not too strict)
        momentum_positive = self.momentum[0] > 0
        
        # Get individual signals
        qqe_signal = self.qqe.get_signal()
        vol_signal = self.volosc.get_signal()
        
        # Zero lag crossover
        zl_signal = 0
        if self.zl_crossover > 0:
            zl_signal = 1
        elif self.zl_crossover < 0:
            zl_signal = -1
        
        # Count positive signals
        positive_signals = sum([
            1 if qqe_signal > 0 else 0,
            1 if vol_signal > 0 else 0,
            1 if zl_signal > 0 else 0,
            1 if momentum_positive else 0
        ])
        
        negative_signals = sum([
            1 if qqe_signal < 0 else 0,
            1 if vol_signal < 0 else 0,
            1 if zl_signal < 0 else 0
        ])
        
        # SIMPLIFIED: Just need 2+ positive signals + trend
        if positive_signals >= 2 and price_above_sma:
            return 1
        elif negative_signals >= 2:
            return -1
        else:
            return 0

    def buy_or_short_condition(self):
        """Initial entry - SIMPLIFIED"""
        if self.buy_executed:
            return
            
        signal = self.get_signal()
        mama_bullish = self.mama.lines.MAMA > self.mama.lines.FAMA
        
        # MUCH SIMPLER: Either strong signal OR crossover with MAMA
        entry_trigger = (signal > 0) or (self.zl_crossover > 0 and mama_bullish)
        
        if entry_trigger:
            size = self._determine_size()
            order_tracker = OrderTracker(
                entry_price=self.data.close[0],
                size=size,
                take_profit_pct=self.params.take_profit,
                symbol=getattr(self, 'symbol', self.p.asset),
                order_type="BUY",
                backtest=self.params.backtest
            )
            order_tracker.order_id = f"order_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            self.active_orders.append(order_tracker)
            self.entry_prices.append(self.data.close[0])
            self.sizes.append(size)
            self.order = self.buy(size=size, exectype=bt.Order.Market)
            
            if self.p.debug:
                print(f"[{len(self)} BUY] Entry: {size:.4f} @ ${self.data.close[0]:.2f}")
            
            if not hasattr(self, 'first_entry_price') or self.first_entry_price is None:
                self.first_entry_price = self.data.close[0]
            
            self.buy_executed = True
            self.calc_averages()

    def dca_or_short_condition(self):
        """DCA logic - FIXED"""
        if not self.buy_executed or not self.entry_prices:
            return
        
        # Prevent multiple DCAs in same bar
        if self.last_dca_bar == len(self):
            return
            
        # Check if price dropped enough for DCA
        price_drop = (self.entry_prices[-1] - self.data.close[0]) / self.entry_prices[-1] * 100
        
        if price_drop >= self.params.dca_deviation:
            size = self._determine_size()
            order_tracker = OrderTracker(
                entry_price=self.data.close[0],
                size=size,
                take_profit_pct=self.params.take_profit,
                symbol=getattr(self, 'symbol', self.p.asset),
                order_type="BUY",
                backtest=self.params.backtest
            )
            order_tracker.order_id = f"order_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            self.active_orders.append(order_tracker)
            self.entry_prices.append(self.data.close[0])
            self.sizes.append(size)
            self.order = self.buy(size=size, exectype=bt.Order.Market)
            self.last_dca_bar = len(self)
            
            if self.p.debug:
                print(f"[{len(self)} DCA] Adding: {size:.4f} @ ${self.data.close[0]:.2f} (drop: {price_drop:.2f}%)")
            
            self.calc_averages()

    def sell_or_cover_condition(self):
        """Exit logic with stop loss"""
        if not self.active_orders or not self.buy_executed:
            return
            
        current_price = self.data.close[0]
        orders_to_remove = []
        
        # Check each order for TP or SL
        for idx, order in enumerate(self.active_orders):
            # Take profit
            if current_price >= order.take_profit_price:
                self.order = self.sell(size=order.size, exectype=bt.Order.Market)
                profit_pct = ((current_price / order.entry_price) - 1) * 100
                if self.p.debug:
                    print(f"[{len(self)} TP] Exit: {order.size:.4f} @ ${current_price:.2f} (+{profit_pct:.2f}%)")
                order.close_order(current_price)
                orders_to_remove.append(idx)
            
            # Stop loss
            elif current_price <= order.entry_price * (1 - self.params.stop_loss / 100):
                self.order = self.sell(size=order.size, exectype=bt.Order.Market)
                loss_pct = ((current_price / order.entry_price) - 1) * 100
                if self.p.debug:
                    print(f"[{len(self)} SL] Exit: {order.size:.4f} @ ${current_price:.2f} ({loss_pct:.2f}%)")
                order.close_order(current_price)
                orders_to_remove.append(idx)
        
        # Remove closed orders
        for idx in sorted(orders_to_remove, reverse=True):
            self.active_orders.pop(idx)
        
        # Update tracking
        if orders_to_remove:
            self.entry_prices = [order.entry_price for order in self.active_orders]
            self.sizes = [order.size for order in self.active_orders]
            
            if not self.active_orders:
                self.reset_position_state()
            else:
                self.calc_averages()
    
    def next(self):
        """Main strategy loop - PROPER ORDER"""
        # 1. Check exits first
        self.sell_or_cover_condition()
        
        # 2. Then check entries
        if not self.buy_executed:
            self.buy_or_short_condition()
        else:
            self.dca_or_short_condition()

    def stop(self):
        if self.p.debug:
            print("\n" + "="*50)
            print("ENHANCED STRATEGY RESULTS")
            print("="*50)
            print(f"Final Portfolio Value: ${self.broker.getvalue():.2f}")
            print(f"Final Cash: ${self.broker.get_cash():.2f}")
            print(f"Total Return: {((self.broker.getvalue() / 1000) - 1) * 100:.2f}%")
            if hasattr(self, 'signal_history') and self.signal_history:
                avg_signal = np.mean([abs(s) for s in self.signal_history if s != 0])
                print(f"Average Signal Strength: {avg_signal:.3f}")
            print("="*50 + "\n")


# Run Backtest
if __name__ == '__main__':
    try:
        backtest(
            QuantitativeMultiIndicatorStrategy,
            coin='BTC',
            collateral='USDT',
            start_date="2016-01-01",
            interval="1h",
            init_cash=1000,
            plot=True,
            quantstats=False,
            debug=True,
        )
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()