from .base import BaseStrategy, bt, OrderTracker, datetime

class AliG_STrend(BaseStrategy):
    params = (
        ('dca_threshold', 7.5),
        ('take_profit', 25),
        ('percent_sizer', 0.3),
        ('debug', True),
        ('backtest', None)
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.DCA = True

    def buy_or_short_condition(self):
        size = self._determine_size()
        self.order = self.buy(size=size, exectype=bt.Order.Market, transmit=True)
        self.enqueue_web3order('buy', amount=size)
        
        order_tracker = self.create_order(action='BUY', size=size)
        return True

    def dca_or_short_condition(self):
        if self.entry_prices and self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_threshold / 100):
            size = self._determine_size()
            self.order = self.buy(size=size, exectype=bt.Order.Market, transmit=True)
            self.enqueue_web3order('buy', amount=size)
            
            self.create_order(action='BUY', size=size)
            return True
        return False
    
    def sell_or_cover_condition(self):
        current_price = self.data.close[0]
        for order_tracker in list(self.active_orders):
            if current_price >= order_tracker.take_profit_price:
                self.order = self.sell(size=order_tracker.size, exectype=bt.Order.Market, transmit=True)
                self.enqueue_web3order('sell', amount=order_tracker.size)
                self.close_order(order_tracker)
                return True
        return False

    def next(self):
        super().next()
        if not self.p.backtest and self.p.debug:
            if hasattr(self, 'active_orders') and self.print_counter % 15 == 0:
                self.report_positions()
            dt = self.datas[0].datetime.datetime(0)
            print(f'Realtime: {datetime.now()} processing candle date: {dt}, with {self.data.close[0]}')

    def stop(self):
        super().stop()
        if hasattr(self, 'active_orders'):
            for order in self.active_orders:
                print(f"Open Order - Entry: {order.entry_price}, Size: {order.size}, TP: {order.take_profit_price}")
            self.report_positions()
        else:
            print("No active orders at the end of the backtest.")