from .base import BaseStrategy, bt, OrderTracker, datetime
from backtrader.indicators.MesaAdaptiveMovingAverage import MAMA

class SMA_Cross_MESAdaptivePrime(BaseStrategy, bt.SignalStrategy):
    params = (
        ('fast', 13),
        ('slow', 37),
        ('dca_deviation', 1.5),
        ('take_profit_percent', 1),
        ('percent_sizer', 0.045), # 4.5%
        ('debug', False),
        ("backtest", None)
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Simple Moving Averages
        self.sma17 = bt.ind.SMA(period=17)
        self.sma47 = bt.ind.SMA(period=47)
        # Mesa Adaptive Moving Average
        self.mama = MAMA(self.data, fast=self.p.fast, slow=self.p.slow)
        # Crossover signal
        self.crossover = bt.ind.CrossOver(self.sma17, self.sma47)
        # Momentum
        self.momentum = bt.ind.Momentum(period=42)
        self.DCA = True

    def buy_or_short_condition(self):
        if not self.buy_executed and not self.conditions_checked:
            if self.crossover > 0 and self.momentum > 0 and self.mama.lines.MAMA > self.mama.lines.FAMA:
                size = self._determine_size()
                order_tracker = OrderTracker(
                    entry_price=self.data.close[0],
                    size=size,
                    take_profit_pct=self.params.take_profit,
                    symbol=getattr(self, 'symbol', self.p.asset),
                    order_type="BUY",
                    backtest=self.params.backtest
                )
                order_tracker.order_id = f"order_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                if not hasattr(self, 'active_orders'):
                    self.active_orders = []
                    
                self.active_orders.append(order_tracker)
                self.entry_prices.append(self.data.close[0])
                self.sizes.append(size)
                self.order = self.buy(size=size, exectype=bt.Order.Market)
                if self.p.debug:
                    print(f"Buy order placed: {size} at {self.data.close[0]}")
                if not self.buy_executed:
                    if not hasattr(self, 'first_entry_price') or self.first_entry_price is None:
                        self.first_entry_price = self.data.close[0]
                    self.buy_executed = True
                self.calc_averages()
        self.conditions_checked = True

    def dca_or_short_condition(self):
        if self.entry_prices and self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_deviation / 100):
            if self.buy_executed and not self.conditions_checked:
                if self.crossover > 0 and self.momentum > 0 and self.mama.lines.MAMA > self.mama.lines.FAMA:  
                    size = self._determine_size()
                    order_tracker = OrderTracker(
                        entry_price=self.data.close[0],
                        size=size,
                        take_profit_pct=self.params.take_profit,
                        symbol=getattr(self, 'symbol', self.p.asset),
                        order_type="BUY",
                        backtest=self.params.backtest
                    )
                    order_tracker.order_id = f"order_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    if not hasattr(self, 'active_orders'):
                        self.active_orders = []
                        
                    self.active_orders.append(order_tracker)
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(size)
                    self.order = self.buy(size=size, exectype=bt.Order.Market)
                    if self.p.debug:
                        print(f"Buy order placed: {size} at {self.data.close[0]}")
                    if not self.buy_executed:
                        if not hasattr(self, 'first_entry_price') or self.first_entry_price is None:
                            self.first_entry_price = self.data.close[0]
                        self.buy_executed = True
                    self.calc_averages()
        self.conditions_checked = True

    def sell_or_cover_condition(self):
        if hasattr(self, 'active_orders') and self.active_orders and self.buy_executed:
            current_price = self.data.close[0]
            orders_to_remove = []

            for idx, order in enumerate(self.active_orders):
                if current_price >= order.take_profit_price:
                    self.order = self.sell(size=order.size, exectype=bt.Order.Market)
                    if self.p.debug:
                        print(f"TP hit: Selling {order.size} at {current_price} (entry: {order.entry_price})")
                    order.close_order(current_price)
                    orders_to_remove.append(idx)
            for idx in sorted(orders_to_remove, reverse=True):
                removed_order = self.active_orders.pop(idx)
                profit_pct = ((current_price / removed_order.entry_price) - 1) * 100
                if self.p.debug:
                    print(f"Order removed: {profit_pct:.2f}% profit")
            if orders_to_remove:
                self.entry_prices = [order.entry_price for order in self.active_orders]
                self.sizes = [order.size for order in self.active_orders]
                if not self.active_orders:
                    self.reset_position_state()
                    self.buy_executed = False
                else:
                    self.calc_averages()
        self.conditions_checked = True

