from .base import BaseStrategy, bt, OrderTracker, datetime
from backtrader.indicators.MesaAdaptiveMovingAverage import MAMA

class SMA_Cross_MESAdaptivePrime(BaseStrategy):
    params = (
        ('fast', 13),
        ('slow', 37),
        ('dca_deviation', 1.5),
        ('take_profit', 2),
        ('percent_sizer', 0.01),
        ('debug', False),
        ("backtest", None)
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sma17 = bt.ind.SMA(period=17)
        self.sma47 = bt.ind.SMA(period=47)
        self.mama = MAMA(self.data, fast=self.p.fast, slow=self.p.slow)
        self.crossover = bt.ind.CrossOver(self.sma17, self.sma47)
        self.momentum = bt.ind.Momentum(period=42)
        self.DCA = True

    def buy_or_short_condition(self):
        if (self.crossover > 0 and 
            self.momentum > 0 and 
            self.mama.lines.MAMA > self.mama.lines.FAMA):
            self.create_order(action='BUY')
            return True
        return False

    def dca_or_short_condition(self):
        if (self.entry_prices and 
            self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_deviation / 100) and
            self.crossover > 0 and 
            self.momentum > 0 and 
            self.mama.lines.MAMA > self.mama.lines.FAMA):
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