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
        ('take_profit', 4),  # 2% take profi
        ('percent_sizer', 0.15), # 0.01 -> 1%
        ('stop_loss', 20),   # 20% stop loss
        ('debug', False),
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

        self.entry_price = None
        self.buy_executed = False
        self.conditions_checked = False
        self.DCA = True

    def buy_or_short_condition(self):
        buy_signal = (self.sma1_fast[0] > self.sma1_slow[0] and
                      self.sma2_fast[0] > self.sma2_slow[0] and
                      self.sma3_fast[0] > self.sma3_slow[0])

        if buy_signal and self.hold_counter1 == 0 and self.hold_counter2 == 0 and self.hold_counter3 == 0:
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
        buy_signal = (self.sma1_fast[0] > self.sma1_slow[0] and
                      self.sma2_fast[0] > self.sma2_slow[0] and
                      self.sma3_fast[0] > self.sma3_slow[0])
        if buy_signal and self.hold_counter1 == 0 and self.hold_counter2 == 0 and self.hold_counter3 == 0:
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

    def check_stop_loss(self):
        if self.buy_executed and self.stop_loss_price is not None:
            current_price = self.data.close[0]
            if current_price <= self.stop_loss_price:                
                
                # TODO :: Rework Stoploss logic
                # if self.params.backtest == False:
                #     self.rabbit.send_jrr_close_request(exchange=self.exchange, account=self.account, asset=self.asset)
                # elif self.params.backtest == True:
                #     self.close()
                
                print(f'STOP LOSS TRIGGERED {self.stop_loss_price:.12f}')
                
                # self.reset_position_state()
                self.conditions_checked = True
                return True
        return False

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

        if self.p.use_stoploss == True:
            self.check_stop_loss()
        BaseStrategy.next(self)

        self.conditions_checked = False

