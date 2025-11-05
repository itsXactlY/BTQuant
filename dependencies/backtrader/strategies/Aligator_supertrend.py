from .base import BaseStrategy
from backtrader.indicators.SuperTrend import SuperTrend
from backtrader.indicators.WilliamsAligator import WilliamsAlligator

class AliG_STrend(BaseStrategy):
    params = \
    (
        ('dca_threshold', 1.5),
        ('take_profit', 2),
        ('percent_sizer', 0.045), # 0.01 -> 1%
        ('premium', 0.003),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supertrend = SuperTrend(plotname='Supertrend: ', plot=True)
        self.alligator = WilliamsAlligator(plotname='WilliamsAlligator: ', plot=False)
        self.DCA = True

    def buy_or_short_condition(self):
        if self.supertrend.lines.super_trend[0] > 0 and \
            self.alligator.lines.jaw[0] > 0 and \
            self.alligator.lines.teeth[0] > 0 and \
            self.alligator.lines.lips[0] > 0:
            self.create_order(action='BUY')
            return True
        return False
            
    def dca_or_short_condition(self):
        if self.entry_prices and \
            self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_threshold / 100) and \
            self.buy_executed and self.supertrend.lines.super_trend[0] > 0 and \
            self.alligator.lines.jaw[0] > 0 and \
            self.alligator.lines.teeth[0] > 0 and \
            self.alligator.lines.lips[0] > 0:
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

    def next(self):
        super().next()
        if not self.p.backtest:
            self.report_positions()