import math
from backtrader.utils.backtest import backtest, bulk_backtest
from backtrader.strategies.base import BaseStrategy, bt
from backtrader.indicators.MesaAdaptiveMovingAverage import MAMA

class QQEIndicator(bt.Indicator):
    params = (
        ("period", 14),
        ("fast", 5),
        ("q", 4.236),
        ("smoothing", 7),
    )
    lines = ("qqe_line", "qqe_smoothed", "qqe_signal", "qqe_trend")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.fast)
        rsi_ma = bt.indicators.EMA(self.rsi, period=int(self.p.period/2))
        dar_period = int((self.p.period * 2) - 1)
        self.dar = bt.indicators.EMA(bt.If(self.atr > 0, self.atr * self.p.q, 0), period=dar_period)
        self.lines.qqe_line = self.rsi + bt.If(self.rsi > rsi_ma, self.dar, -self.dar)
        self.lines.qqe_smoothed = bt.indicators.EMA(self.lines.qqe_line, period=self.p.smoothing)
        self.lines.qqe_signal = bt.indicators.EMA(self.lines.qqe_smoothed, period=self.p.smoothing * 2)
        self.lines.qqe_trend = bt.If(self.lines.qqe_smoothed > 50, 1, -1)
    
    def get_signal(self):
        if len(self.lines.qqe_smoothed) < 2:
            return 0
        bull_cross = (self.lines.qqe_smoothed[-1] < self.lines.qqe_signal[-1] and 
                    self.lines.qqe_smoothed[0] > self.lines.qqe_signal[0])
        bear_cross = (self.lines.qqe_smoothed[-1] > self.lines.qqe_signal[-1] and 
                    self.lines.qqe_smoothed[0] < self.lines.qqe_signal[0])
        bull_level = self.lines.qqe_smoothed[0] > 50 and self.lines.qqe_smoothed[-1] <= 50
        bear_level = self.lines.qqe_smoothed[0] < 50 and self.lines.qqe_smoothed[-1] >= 50
        
        if bull_cross or bull_level or self.lines.qqe_smoothed[0] > 60:
            return 1
        elif bear_cross or bear_level or self.lines.qqe_smoothed[0] < 40:
            return -1
        return 0

class VolumeOscillator(bt.Indicator):
    params = (
        ("short_period", 14),
        ("long_period", 28),
        ("smooth_period", 7),
        ("signal_period", 14),
    )
    lines = ("osc", "smoothed", "signal", "trend", "momentum")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        short_vol = bt.indicators.SMA(self.data.volume, period=self.p.short_period)
        long_vol = bt.indicators.SMA(self.data.volume, period=self.p.long_period)
        
        self.lines.osc = ((short_vol - long_vol) / (long_vol + 1e-6)) * 100
        self.lines.smoothed = bt.indicators.EMA(self.lines.osc, period=self.p.smooth_period)
        self.lines.signal = bt.indicators.EMA(self.lines.smoothed, period=self.p.signal_period)
        self.lines.trend = bt.If(self.lines.smoothed > self.lines.signal, 1, -1)
        self.lines.momentum = bt.indicators.RateOfChange(self.lines.smoothed, period=5)

    def get_signal(self):
        if len(self.lines.smoothed) < 2:
            return 0
        bull_cross = (self.lines.smoothed[-1] < self.lines.signal[-1] and 
                    self.lines.smoothed[0] > self.lines.signal[0])
        bear_cross = (self.lines.smoothed[-1] > self.lines.signal[-1] and 
                    self.lines.smoothed[0] < self.lines.signal[0])

        if bull_cross or (self.lines.smoothed[0] > self.lines.signal[0] and self.lines.smoothed[0] > 0):
            return 1
        elif bear_cross or (self.lines.smoothed[0] < self.lines.signal[0] and self.lines.smoothed[0] < 0):
            return -1
        return 0

class HMA(bt.Indicator):
    lines = ('hma',)
    params = dict(period=20)
    
    def __init__(self):
        p = self.p.period
        self.wma_half = bt.ind.WeightedMovingAverage(self.data, period=int(p / 2))
        self.wma_full = bt.ind.WeightedMovingAverage(self.data, period=p)
        self.raw_hma = 2 * self.wma_half - self.wma_full
        self.lines.hma = bt.ind.WeightedMovingAverage(self.raw_hma, period=int(math.sqrt(p)))

class ZeroLag(bt.Indicator):
    lines = ('zerolag',)
    params = dict(period=20)
    
    def __init__(self):
        self.hma = HMA(self.data, period=self.p.period)
        self.hma_sma = bt.ind.SMA(self.hma, period=int(self.p.period / 2))
        self.lines.zerolag = 2 * self.hma - self.hma_sma

class SineWeightedMA(bt.Indicator):
    lines = ('sine_wma',)
    params = dict(period=20)
    
    def __init__(self):
        self.lines.sine_wma = bt.ind.SMA(self.data, period=self.p.period)

class SineWeightZeroLagQQEVolMesaAdaptive(BaseStrategy):
    params = dict(
        # Zero Lag and Moving Average parameters
        zl_period=14,
        sine_period=17,
        sma_filter=197,
        mom_period=14,
        mom_lookback=3,
        
        # QQE parameters
        qqe_period=13,           # Was 14 - faster signals
        qqe_fast=5,
        qqe_q=4.236,
        qqe_smoothing=5,         # Was 7 - less smoothing = more signals
        
        # Volume Oscillator parameters
        vol_short=14,
        vol_long=28,
        vol_smooth=5,            # Was 7 - less smoothing
        vol_signal=10,           # Was 14 - faster signal line
        
        # MESA Adaptive parameters
        fast=13,
        slow=17,
        
        # Risk management
        take_profit=0.7,           # 1% take profit
        dca_deviation=4.5,       # DCA when price drops 4.5%
        percent_sizer=0.0225,    # 2.25% of available cash per trade
        
        # Execution
        debug=True,
        backtest=True
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Indicators only - BaseStrategy handles all position tracking
        self.zero_lag = ZeroLag(self.data, period=self.p.zl_period, plot=False)
        self.sine_wma = SineWeightedMA(self.zero_lag, period=self.p.sine_period, plot=False)
        self.sma_200 = bt.ind.SMA(self.data, period=self.p.sma_filter, plot=True)
        self.fmomentum = bt.ind.Momentum(self.data, period=self.p.mom_period, plot=True)
        self.crossover = bt.ind.CrossOver(self.zero_lag, self.sine_wma, plot=True)
        
        self.sma17 = bt.ind.SMA(period=17)
        self.sma47 = bt.ind.SMA(period=47)
        self.mama = MAMA(self.data, fast=self.p.fast, slow=self.p.slow)
        self.mesa_cross = bt.ind.CrossOver(self.sma17, self.sma47)
        self.smomentum = bt.ind.Momentum(period=42)
        
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
            subplot=True
        )
        
        self.DCA = True
    
    def get_signal(self):
        """Consolidated signal logic"""
        momentum_trending_up = all(
            self.fmomentum[0] > self.fmomentum[-i]
            for i in range(1, min(self.p.mom_lookback + 1, len(self.fmomentum)))
            if len(self.fmomentum) > i
        ) and all(
            self.smomentum[0] > self.smomentum[-i]
            for i in range(1, min(self.p.mom_lookback + 1, len(self.smomentum)))
            if len(self.smomentum) > i
        )
        
        price_above_sma = self.data.close[0] > self.sma_200[0]
        
        positive_signals = sum([
            self.qqe.get_signal() > 0,
            self.volosc.get_signal() > 0,
            self.crossover > 0,
            momentum_trending_up
        ])
        
        negative_signals = sum([
            self.qqe.get_signal() < 0,
            self.volosc.get_signal() < 0,
            self.crossover < 0
        ])

        if positive_signals >= 4 and price_above_sma:
            return 1
        elif negative_signals >= 3 or not price_above_sma:
            return -1
        return 0

    def buy_or_short_condition(self):
        """Entry logic - BaseStrategy calls this when no position"""
        signal = self.get_signal()
        
        if (signal > 0 and 
            self.crossover > 0 and 
            self.fmomentum > self.smomentum and 
            self.mama.lines.MAMA > self.mama.lines.FAMA):
            
            self.create_order(action='BUY')  # BaseStrategy handles everything!
            return True
        
        return False

    def dca_or_short_condition(self):
        """DCA logic - BaseStrategy calls this when position exists and DCA=True"""
        signal = self.get_signal()
        
        if (self.entry_prices and 
            self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_deviation / 100) and
            signal > 0 and 
            self.crossover > 0 and 
            self.fmomentum > self.smomentum and 
            self.mama.lines.MAMA > self.mama.lines.FAMA):
            
            self.create_order(action='BUY')  # BaseStrategy handles everything!
            return True
        
        return False

    def sell_or_cover_condition(self):
        """Exit logic - BaseStrategy calls this when position exists"""
        current_price = self.data.close[0]
        
        for order_tracker in list(self.active_orders):
            if current_price >= order_tracker.take_profit_price:
                self.close_order(order_tracker)  # BaseStrategy handles everything!
                return True
        
        return False


