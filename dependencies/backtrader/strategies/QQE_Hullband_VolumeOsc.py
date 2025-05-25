from .base import BaseStrategy, bt, OrderTracker, datetime
from numpy import isnan

class VolumeOscillator(bt.Indicator):
    lines = ('short', 'long', 'osc')
    params = (
        ('shortlen', 5),
        ('longlen', 10),
        ('debug', False),
        )

    def __init__(self):
        # self.addminperiod(self.p.longlen)
        shortlen, longlen = self.params.shortlen, self.params.longlen
        self.lines.short = bt.indicators.ExponentialMovingAverage(self.data.volume, period=shortlen)
        self.lines.long = bt.indicators.ExponentialMovingAverage(self.data.volume, period=longlen)

    def next(self):
        try:
            if self.p.debug:
                # Log the volume data and EMAs to check for issues
                print(f"Volume: {self.data.volume[0]}, Short EMA: {self.lines.short[0]}, Long EMA: {self.lines.long[0]}")

            # Check if the volume data or EMAs are None, zero, or NaN to avoid division by zero
            if not (self.lines.long[0] and self.lines.long[0] > 0 and not isnan(self.lines.long[0])):
                print(f"Invalid volume data detected: Long EMA is {self.lines.long[0]}. Setting oscillator to 0.")
                self.lines.osc[0] = 0
            else:
                # Calculate oscillator only when valid values exist
                self.lines.osc[0] = (self.lines.short[0] - self.lines.long[0]) / self.lines.long[0] * 100
        except Exception as e:
            print(f"Error calculating Volume Oscillator: {e}")
            self.lines.osc[0] = 0

class QQEIndicator(bt.Indicator):
    params = (
        ("period", 6),
        ("fast", 5),
        ("q", 3.0),
        ("debug", False)
    )
    lines = ("qqe_line",)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.fast)
        self.dar = bt.If(self.atr > 0, bt.indicators.EMA(self.atr - self.p.q, period=int((self.p.period * 2) - 1)), 0)
        self.lines.qqe_line = bt.If(self.rsi > 0, self.rsi + self.dar, 0)

    def next(self):
        # check if ATR is not zero to avoid division by zero errors
        if self.atr[0] == 0:
            print("ATR is zero, skipping this iteration to avoid division by zero.")
            return

        # check if RSI and DAR are valid before computing the QQE line
        if self.rsi[0] != 0 and self.dar[0] != 0:
            self.lines.qqe_line[0] = self.rsi[0] + self.dar[0]
        else:
            self.lines.qqe_line[0] = 0
        
        if self.p.debug:
            print(f"RSI: {self.rsi[0]}, DAR: {self.dar[0]}, ATR: {self.atr[0]}, QQE: {self.lines.qqe_line[0]}")

class QQE_Example(BaseStrategy):
    params = (
        ("ema_length", 20),
        ('hull_length', 53),
        ('take_profit', 4),
        ('dca_deviation', 4),
        ('percent_sizer', 0.01),
        ('debug', False),
    )

    def __init__(self, **kwargs):
        print('Initialized QQE')
        super().__init__(**kwargs)
        self.qqe = QQEIndicator(self.data)
        self.hma = bt.indicators.HullMovingAverage(self.data, period=self.p.hull_length)
        self.ema = bt.indicators.EMA(self.data.close, period=self.params.ema_length)
        self.volosc = VolumeOscillator(self.data)
        self.DCA = True
        self.conditions_checked = False

    def buy_or_short_condition(self):
        if not self.buy_executed and not self.conditions_checked:
            if (self.qqe.qqe_line[-1] > 0) and \
               (self.data.close[-1] < self.hma[0]) and \
               (self.volosc.osc[-1] < self.volosc.lines.short[0]):

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
                if (self.qqe.qqe_line[-1] > 0) and (self.data.close[-1] < self.hma[0]) and (self.volosc.osc[-1] < self.volosc.lines.short[0]): 
                    
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
