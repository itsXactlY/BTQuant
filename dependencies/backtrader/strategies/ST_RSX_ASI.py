from backtrader.indicators.RSX import RSX
from backtrader.indicators.AccumulativeSwingIndex import AccumulativeSwingIndex
from backtrader.indicators.SuperTrend import SuperTrend
from backtrader.strategies.base import BaseStrategy, bt

class STrend_RSX_AccumulativeSwingIndex(BaseStrategy):
    params = (
        ('stlen', 7),
        ('stmult', 7.0),
        ('rsxlen', 14),
        ("dca_deviation", 1.5),
        ("take_profit", 2),
        ('percent_sizer', 0.01),
        ('trailing_stop_pct', 0.2),
        ('debug', False),
        ('backtest', None)
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.DCA = True
        self.peak = 0
        self.asi_short = AccumulativeSwingIndex(period=7, plot=True)
        self.asi_long = AccumulativeSwingIndex(period=14, plot=True)
        self.rsx = RSX(self.data, length=self.p.rsxlen, plot=False)
        self.sttrend = SuperTrend(self.data, period=self.p.stlen, multiplier=self.p.stmult, plot=True)
        self.stLong = bt.ind.CrossOver(self.data.close, self.sttrend, plot=True)

    def buy_or_short_condition(self):
        if (self.stLong and 
            self.asi_short[0] > self.asi_short[-1] and 
            self.asi_short[0] > 5 and 
            self.asi_long[0] > self.asi_long[-1] and 
            self.rsx[0] < 30):
            self.create_order(action='BUY')
            return True
        return False

    def dca_or_short_condition(self):
        self.peak = max(self.peak, self.data.close[0])
        if (self.entry_prices and 
            self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_deviation / 100) and
            self.stLong and 
            self.asi_short[0] > self.asi_short[-1] and 
            self.asi_short[0] > 5 and 
            self.asi_long[0] > self.asi_long[-1] and 
            self.rsx[0] < 30):
            self.create_order(action='BUY')
            return True
        return False

    def sell_or_cover_condition(self):
        current_price = self.data.close[0]
        orders_to_remove = []
        
        for order_tracker in list(self.active_orders):
            if current_price >= order_tracker.take_profit_price:
                # Trailing stop logic with RSX
                if hasattr(self, 'rsx') and self.rsx[0] > 70:
                    if not hasattr(order_tracker, 'peak'):
                        order_tracker.peak = current_price
                    else:
                        order_tracker.peak = max(order_tracker.peak, current_price)
                    
                    trailing_trigger = order_tracker.peak * (1 - self.p.trailing_stop_pct / 100)
                    
                    if current_price < trailing_trigger:
                        if (round(current_price, 9) < round(order_tracker.entry_price, 9) or 
                            round(current_price, 9) < round(order_tracker.take_profit_price, 9)):
                            continue
                        
                        self.close_order(order_tracker)
                        order_tracker.peak = 0
                        return True
                else:
                    self.close_order(order_tracker)
                    return True
        return False