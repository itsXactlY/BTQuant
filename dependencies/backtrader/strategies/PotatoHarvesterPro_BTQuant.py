from .base import BaseStrategy, bt

class PotatoHarvesterPro(BaseStrategy):
    params = (
        ('centre_period', 21),
        ('atr_period', 21),
        ('levels', 5),
        ('mult_start', 1.0),
        ('mult_step', 0.5),
        ('take_profit', 1.5),
        ('stop_loss', 1.2),
        ('percent_sizer', 0.1),      # Added: 10% position size
        ('debug', False),
    )

    def __init__(self):
        super().__init__()
        self.centre_ema = bt.indicators.EMA(self.data.close, period=self.p.centre_period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        m_vals = [self.p.mult_start + self.p.mult_step * i for i in range(self.p.levels)]
        self.band_top = [self.centre_ema + self.atr * m for m in m_vals]
        self.band_bot = [self.centre_ema - self.atr * m for m in m_vals]
        self.buy_cross = [bt.indicators.CrossOver(self.data.close, b) for b in self.band_bot]
        self.sell_cross = [bt.indicators.CrossOver(t, self.data.close) for t in self.band_top]

    def buy_or_short_condition(self):
        for i, c in enumerate(self.buy_cross):
            if c[0] > 0:
                self.create_order(action='BUY')
                if self.p.debug: print(f"BUY level {i+1}")
                return True
        return False

    def sell_or_cover_condition(self):
        for ot in list(self.active_orders):
            sl = ot.entry_price * (1 - self.p.stop_loss / 100)
            tp = ot.entry_price * (1 + self.p.take_profit / 100)
            if self.data.close[0] <= sl or self.data.close[0] >= tp:
                self.close_order(ot)
                return True
        return False

    def next(self):
        self.sell_or_cover_condition()
        super().next()