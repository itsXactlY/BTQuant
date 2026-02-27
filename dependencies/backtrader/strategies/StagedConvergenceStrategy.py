from .base import BaseStrategy, bt, OrderTracker
from datetime import datetime

class StagedConvergenceStrategy(BaseStrategy):
    params = (
        ('fast1', 5),
        ('slow1', 50),
        ('fast2', 10),
        ('slow2', 100),
        ('fast3', 20),
        ('slow3', 200),
        ('hold1', 50),
        ('hold2', 35),
        ('hold3', 0),
        ('dca_threshold', 3),
        ('take_profit', 4),
        ('percent_sizer', 0.0015),
        ('stop_loss', 20),
        ('debug', True),
        ('backtest', None),
        ('use_stoploss', False),
    )

    def __init__(self):
        super().__init__()
        self.sma1_fast = bt.indicators.SMA(period=self.params.fast1)
        self.sma1_slow = bt.indicators.SMA(period=self.params.slow1)
        self.sma2_fast = bt.indicators.SMA(period=self.params.fast2)
        self.sma2_slow = bt.indicators.SMA(period=self.params.slow2)
        self.sma3_fast = bt.indicators.SMA(period=self.params.fast3)
        self.sma3_slow = bt.indicators.SMA(period=self.params.slow3)
        
        self.hold_counter1 = 0
        self.hold_counter2 = 0
        self.hold_counter3 = 0
        self.DCA = True

    def buy_or_short_condition(self):
        buy_signal = (self.sma1_fast[0] > self.sma1_slow[0] and
                      self.sma2_fast[0] > self.sma2_slow[0] and
                      self.sma3_fast[0] > self.sma3_slow[0])
        
        if buy_signal and self.hold_counter1 == 0 and self.hold_counter2 == 0 and self.hold_counter3 == 0:
            self.create_order(action='BUY')
            return True
        return False

    def dca_or_short_condition(self):
        buy_signal = (self.sma1_fast[0] > self.sma1_slow[0] and
                      self.sma2_fast[0] > self.sma2_slow[0] and
                      self.sma3_fast[0] > self.sma3_slow[0])
        
        if buy_signal and self.hold_counter1 == 0 and self.hold_counter2 == 0 and self.hold_counter3 == 0:
            self.create_order(action='BUY')
            return True
        return False

    def check_stop_loss(self):
        """Custom stop loss implementation"""
        if self.buy_executed and self.stop_loss_price is not None:
            current_price = self.data.close[0]
            if current_price <= self.stop_loss_price:
                if self.p.debug:
                    print(f'STOP LOSS TRIGGERED {self.stop_loss_price:.12f}')
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
        # Update holding periods
        if self.hold_counter1 > 0:
            self.hold_counter1 -= 1
        if self.hold_counter2 > 0:
            self.hold_counter2 -= 1
        if self.hold_counter3 > 0:
            self.hold_counter3 -= 1

        # Reset holding periods when signal changes
        if (self.sma1_fast[0] > self.sma1_slow[0]) != (self.sma1_fast[-1] > self.sma1_slow[-1]):
            self.hold_counter1 = self.params.hold1
        if (self.sma2_fast[0] > self.sma2_slow[0]) != (self.sma2_fast[-1] > self.sma2_slow[-1]):
            self.hold_counter2 = self.params.hold2

        if self.p.use_stoploss:
            self.check_stop_loss()
        
        # Call parent's next() to execute strategy logic
        super().next()