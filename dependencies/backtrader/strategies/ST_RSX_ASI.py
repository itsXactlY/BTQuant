from backtrader.indicators.RSX import RSX
from backtrader.indicators.AccumulativeSwingIndex import AccumulativeSwingIndex
from backtrader.indicators.SuperTrend import SuperTrend
<<<<<<< HEAD
from backtrader.strategies.base import BaseStrategy, bt
=======
from .base import BaseStrategy, bt, OrderTracker, datetime
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented

class STrend_RSX_AccumulativeSwingIndex(BaseStrategy):
    params = (
        ('stlen', 7),
        ('stmult', 7.0),
        ('rsxlen', 14),
<<<<<<< HEAD
        ("dca_deviation", 1.5),
        ("take_profit", 2),
        ('percent_sizer', 0.01),
        ('trailing_stop_pct', 0.2),
        ('debug', False),
        ('backtest', None)
    )

=======
        ("dca_deviation", 2),
        ("take_profit", 8),
        ('percent_sizer', 0.1), # 0.1 - 10%
        ('trailing_stop_pct', 1.5),
        ('backtest', None)
    )
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.DCA = True
        self.peak = 0
<<<<<<< HEAD
=======
        
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented
        self.asi_short = AccumulativeSwingIndex(period=7, plot=True)
        self.asi_long = AccumulativeSwingIndex(period=14, plot=True)
        self.rsx = RSX(self.data, length=self.p.rsxlen, plot=False)
        self.sttrend = SuperTrend(self.data, period=self.p.stlen, multiplier=self.p.stmult, plot=True)
        self.stLong = bt.ind.CrossOver(self.data.close, self.sttrend, plot=True)

    def buy_or_short_condition(self):
<<<<<<< HEAD
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
=======
        if not self.buy_executed and not self.conditions_checked:
            if self.stLong and self.asi_short[0] > self.asi_short[-1] and self.asi_short[0] > 5 and self.asi_long[0] > self.asi_long[-1] and self.rsx[0] < 30:

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
        if self.buy_executed and not self.conditions_checked:
            if self.buy_executed:
                self.peak = max(self.peak, self.data.close[0])
            if self.stLong and self.asi_short[0] > self.asi_short[-1] and self.asi_short[0] > 5 and self.asi_long[0] > self.asi_long[-1] and self.rsx[0] < 30:
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

    '''This function here is an "one of its kind", with including trailing stop mechanism.'''
    def sell_or_cover_condition(self):
        if hasattr(self, 'active_orders') and self.active_orders and self.buy_executed:
            current_price = self.data.close[0]
            orders_to_remove = []

            for idx, order in enumerate(self.active_orders):
                # Regular take profit condition
                if current_price >= order.take_profit_price:
                    # Check for trailing stop condition if RSX is available and > 70
                    if hasattr(self, 'rsx') and self.rsx[0] > 70:
                        # Update peak price for this order
                        if not hasattr(order, 'peak'):
                            order.peak = current_price
                        else:
                            order.peak = max(order.peak, current_price)
                        
                        # Calculate trailing trigger
                        trailing_trigger = order.peak * (1 - self.p.trailing_stop_pct / 100)
                        
                        # Only sell if trailing stop is triggered
                        if current_price < trailing_trigger:
                            # Additional safety check - we dont want to sell below entry or take profit
                            if (round(current_price, 9) < round(order.entry_price, 9) or 
                                round(current_price, 9) < round(order.take_profit_price, 9)):
                                continue
                            
                            # Execute the trailing stop sell
                            self.order = self.sell(size=order.size, exectype=bt.Order.Market)
                            if self.p.debug:
                                print(f"Trailing Stop: Selling {order.size} at {current_price} (entry: {order.entry_price}, peak: {order.peak})")

                            order.close_order(current_price)
                            orders_to_remove.append(idx)
                            
                            # Reset peak for this specific order
                            order.peak = 0
                    else:
                        # Doing regular take profit without trailing
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
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented
