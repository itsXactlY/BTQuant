from .base import BaseStrategy, bt, OrderTracker
from backtrader.indicators.SuperTrend import SuperTrend
from backtrader.indicators.RSX import RSX
from datetime import datetime

class SuperSTrend_Scalp(BaseStrategy):
    params = (
        ("dca_deviation", 1.5),
        ("take_profit", 2),
        ('percent_sizer', 0.25),
        ('trailing_stop_pct', 0.4),
        # Trend Strenght
        ("adx_period", 13),
        ("adx_strength", 31),
        ("di_period", 14),
        ("adxth", 25),

        # Supertrends
        ("st_fast", 2),
        ('st_fast_multiplier', 3),
        ("st_slow", 6),
        ('st_slow_multiplier', 7),

        # RSX
        ('rsxlen', 14),

        ('debug', False),
        ("backtest", None)
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.adx = bt.indicators.ADX(self.data, period=self.p.adx_period, plot=False)
        self.plusDI = bt.indicators.PlusDI(self.data, period=self.p.di_period, plot=False)
        self.minusDI = bt.indicators.MinusDI(self.data, period=self.p.di_period, plot=False)
        self.supertrend_fast = SuperTrend(period=self.p.st_fast, multiplier=self.p.st_fast_multiplier, plotname='SuperTrend Fast: ', plot=False)
        self.supertrend_slow = SuperTrend(period=self.p.st_slow, multiplier=self.p.st_slow_multiplier, plotname='SuperTrend Slow: ', plot=False)
        self.supertrend_uptrend_signal = bt.indicators.CrossOver(self.supertrend_fast, self.supertrend_slow, plot=False)
        self.rsx = RSX(self.data, length=self.p.rsxlen, plot=False)
        
        self.DCA = True
        self.peak = 0
        self.buy_executed = False
        self.conditions_checked = False

    def buy_or_short_condition(self):
        if (
            self.adx[0] >= self.params.adxth and \
            self.minusDI[0] > self.params.adxth and \
            self.plusDI[0] < self.params.adxth and \
            self.supertrend_uptrend_signal
        ):
            if not self.buy_executed:
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
        self.peak = max(self.peak, self.data.close[0])
        if (
            self.adx[0] >= self.params.adxth and \
            self.minusDI[0] > self.params.adxth and \
            self.plusDI[0] < self.params.adxth and \
            self.supertrend_uptrend_signal
        ):
            if self.entry_prices and self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_deviation / 100):
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
