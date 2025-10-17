from .base import BaseStrategy, bt, OrderTracker, datetime
from backtrader.indicators.VumanchuMarketCipher_B import VuManchCipherB

class VuManchCipher_B(BaseStrategy):
    params = (
        ('take_profit', 3),
        ('percent_sizer', 0.075),
        ('dca_deviation', 4),
        ('ssma_period', 17),
        ('smoothing', 0.5),
        ('sensitivity', 0.3),
        ('debug', False),
        ('backtest', None),
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.market_cipher = VuManchCipherB()
        self.DCA = True

    def buy_or_short_condition(self):
        if self.market_cipher.lines.wtCrossUp[0] and self.market_cipher.lines.wtOversold[0]:
            self.create_order(action='BUY')
            return True
        return False

    def dca_or_short_condition(self):
        if (self.entry_prices and 
            self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_deviation / 100) and
            self.market_cipher.lines.wtCrossUp[0] and 
            self.market_cipher.lines.wtOversold[0]):
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
            dt = self.datas[0].datetime.datetime(0)
            print(f'Realtime: {datetime.now()} processing candle date: {dt}, with {self.data.close[0]}')

    def stop(self):
        super().stop()
        if hasattr(self, 'active_orders'):
            self.report_positions()
        else:
            print("No active orders at the end of the backtest.")