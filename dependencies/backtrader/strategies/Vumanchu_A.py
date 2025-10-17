from .base import BaseStrategy, bt, OrderTracker, datetime
from backtrader.indicators.VumanchuMarketCipher_A import VuManchCipherA

class VuManchCipher_A(BaseStrategy):
    params = (
        ('take_profit', 2),
        ('percent_sizer', 0.02),
        ('dca_deviation', 1.5),
        ('debug', False),
        ('backtest', None),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cipher = VuManchCipherA(subplot=True)
        self.DCA = True

    def buy_or_short_condition(self):
        if self.cipher.lines.blue_triangle[0] > 0:
            self.create_order(action='BUY')
            return True
        return False

    def dca_or_short_condition(self):
        if (self.entry_prices and 
            self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_deviation / 100) and
            self.cipher.lines.blue_triangle[0] > 0):
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