from .base import BaseStrategy, bt, OrderTracker
from backtrader.indicators.SuperTrend import SuperTrend
from backtrader.indicators.RSX import RSX
from datetime import datetime

class SuperSTrend_Scalp(BaseStrategy):
    params = (
        ("dca_deviation", 1.5),
        ("take_profit", 2),
        ('percent_sizer', 0.25),
        ('trailing_stop_pct', 0.4),
        ("adx_period", 13),
        ("adx_strength", 31),
        ("di_period", 14),
        ("adxth", 25),
        ("st_fast", 2),
        ('st_fast_multiplier', 3),
        ("st_slow", 6),
        ('st_slow_multiplier', 7),
        ('rsxlen', 14),
        ('debug', False),
        ("backtest", None)
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.adx = bt.indicators.ADX(self.data, period=self.p.adx_period, plot=False)
        self.plusDI = bt.indicators.PlusDI(self.data, period=self.p.di_period, plot=False)
        self.minusDI = bt.indicators.MinusDI(self.data, period=self.p.di_period, plot=False)
        self.supertrend_fast = SuperTrend(period=self.p.st_fast, multiplier=self.p.st_fast_multiplier, plotname='SuperTrend Fast: ', plot=False)
        self.supertrend_slow = SuperTrend(period=self.p.st_slow, multiplier=self.p.st_slow_multiplier, plotname='SuperTrend Slow: ', plot=False)
        self.supertrend_uptrend_signal = bt.indicators.CrossOver(self.supertrend_fast, self.supertrend_slow, plot=False)
        self.rsx = RSX(self.data, length=self.p.rsxlen, plot=False)
        self.DCA = True
        self.peak = 0

    def buy_or_short_condition(self):
        if (self.adx[0] >= self.params.adxth and
            self.minusDI[0] > self.params.adxth and
            self.plusDI[0] < self.params.adxth and
            self.supertrend_uptrend_signal):
            self.create_order(action='BUY')
            return True
        return False

    def dca_or_short_condition(self):
        self.peak = max(self.peak, self.data.close[0])
        if (self.entry_prices and 
            self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_deviation / 100) and
            self.adx[0] >= self.params.adxth and
            self.minusDI[0] > self.params.adxth and
            self.plusDI[0] < self.params.adxth and
            self.supertrend_uptrend_signal):
            self.create_order(action='BUY')
            return True
        return False

    def sell_or_cover_condition(self):
        current_price = self.data.close[0]
        for order_tracker in list(self.active_orders):
            if current_price >= order_tracker.take_profit_price:
                self.close_order(order_tracker)
                return True
        return False