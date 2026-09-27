"""Auto-generated strategy: E2E_Momentum_RSI
Source hypothesis: p
Description: d
Indicators: ['ema', 'rsi', 'atr']
Entry conditions: ['self.ema_fast[0] > self.ema_slow[0]']
Exit conditions: ['self.ema_fast[0] < self.ema_slow[0]']
"""
from backtrader.strategies.base import BaseStrategy, bt


class E2E_Momentum_RSI(BaseStrategy):
    """d"""

    params = (
        ("stop_atr_mult", 2.0),
        ("rsi_period", 14),
        ("ema_period", 21),
        ("sma_period", 20),
        ("atr_period", 14),
        ("ema_fast_period", 12),
        ("ema_slow_period", 26),
        ("percent_sizer", 0.95),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rsi = bt.ind.RSI(self.data.close, period=self.p.rsi_period)
        self.atr = bt.ind.ATR(self.data, period=self.p.atr_period)
        self.ema = bt.ind.EMA(self.data.close, period=self.p.ema_period)
        self.ema_fast = bt.ind.EMA(self.data.close, period=self.p.ema_fast_period)
        self.ema_slow = bt.ind.EMA(self.data.close, period=self.p.ema_slow_period)

    def buy_or_short_condition(self):
        if self.buy_executed:
            return False
        if (self.self.ema_fast[0][0] > self.self.ema_slow[0][0]):
            self.create_order(action="BUY")
            return True
        return False

    def sell_or_cover_condition(self):
        if not self.active_orders:
            return False
        fired = False
        if (self.self.ema_fast[0][0] < self.self.ema_slow[0][0]):
            fired = True
        if (not fired and self.p.stop_atr_mult > 0
                and hasattr(self, "atr") and self.first_entry_price):
            stop_price = (self.first_entry_price
                          - self.atr[0] * self.p.stop_atr_mult)
            if self.data.close[0] <= stop_price:
                fired = True
        if fired:
            for tracker in list(self.active_orders):
                self.close_order(tracker)
            return True
        return False
