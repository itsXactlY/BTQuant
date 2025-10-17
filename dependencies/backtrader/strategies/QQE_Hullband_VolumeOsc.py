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


class QuantitativeMultiIndicatorDCAStrategy(bt.Strategy):
    params = dict(
        # Zero Lag and Moving Average parameters
        zl_period=14,
        sine_period=20,
        sma_filter=200,
        mom_period=14,
        mom_lookback=3,
        
        # QQE parameters
        qqe_period=10,
        qqe_fast=5,
        qqe_q=4.236,
        qqe_smoothing=5,
        
        # Volume Oscillator parameters
        vol_short=14,
        vol_long=28,
        vol_smooth=5,
        vol_signal=10,
        
        # Risk management
        take_profit=25,
        stop_loss=12,
        percent_sizer=0.95,
        dca_enabled=True,
        dca_deviation=1.5,      # Price drop % to trigger next DCA layer
        dca_max_layers=5,       # Max DCA entries
        dca_size_factor=0.5,    # Each DCA order is fraction of previous
        debug=False,
        backtest=True
    )

    def __init__(self):
        # Core indicators
        self.zero_lag = ZeroLag(self.data, period=self.p.zl_period, plot=False)
        self.sine_wma = SineWeightedMA(self.zero_lag, period=self.p.sine_period, plot=False)
        self.sma_200 = bt.ind.SMA(self.data, period=self.p.sma_filter, plot=True)
        self.momentum = bt.ind.Momentum(self.data, period=self.p.mom_period, plot=True)
        self.crossover = bt.ind.CrossOver(self.zero_lag, self.sine_wma, plot=True)

        self.qqe = QQEIndicator(
            self.data, 
            period=self.p.qqe_period,
            fast=self.p.qqe_fast,
            q=self.p.qqe_q,
            smoothing=self.p.qqe_smoothing,
            subplot=True
        )

        self.volosc = VolumeOscillator(
            self.data,
            short_period=self.p.vol_short,
            long_period=self.p.vol_long,
            smooth_period=self.p.vol_smooth,
            signal_period=self.p.vol_signal,
            subplot=False
        )

        # DCA tracking
        self.active_orders: List[OrderTracker] = []
        self.buy_executed = False

        BuySellArrows(self.data0, barplot=True)

    def _determine_size(self, prev_size=None):
        """Determine position size for initial or DCA entries"""
        available_cash = self.broker.get_cash()
        if prev_size is None:
            size = (available_cash * self.p.percent_sizer) / self.data.close[0]
        else:
            size = prev_size * self.p.dca_size_factor
        return max(size, 0.001)

    def get_signal(self):
        """Composite indicator signal"""
        momentum_up = all(
            self.momentum[0] > self.momentum[-i]
            for i in range(1, min(self.p.mom_lookback + 1, len(self.momentum)))
            if len(self.momentum) > i
        )
        price_above_sma = self.data.close[0] > self.sma_200[0]
        qqe_signal = self.qqe.get_signal()
        vol_signal = self.volosc.get_signal()
        zl_signal = 0
        if self.crossover > 0:
            zl_signal = 1
        elif self.crossover < 0:
            zl_signal = -1

        pos_signals = sum([qqe_signal > 0, vol_signal > 0, zl_signal > 0, momentum_up])
        neg_signals = sum([qqe_signal < 0, vol_signal < 0, zl_signal < 0])

        if pos_signals >= 2 and price_above_sma:
            return 1
        elif neg_signals >= 2 or not price_above_sma:
            return -1
        return 0

    def _add_dca_order(self, base_order=None):
        """Place DCA order below previous entry"""
        prev_price = base_order.entry_price if base_order else self.data.close[0]
        size = self._determine_size(prev_order.size if base_order else None)

        order_tracker = OrderTracker(
            entry_price=self.data.close[0],
            size=size,
            take_profit_pct=self.p.take_profit,
            symbol=getattr(self, 'symbol', 'ASSET'),
            order_type="BUY",
            backtest=self.p.backtest
        )

        self.active_orders.append(order_tracker)
        self.buy(size=size, exectype=bt.Order.Market)

        if self.p.debug:
            print(f"DCA buy: {size} at {self.data.close[0]} (prev: {prev_price})")

    def next(self):
        signal = self.get_signal()
        current_price = self.data.close[0]

        # INITIAL BUY
        if not self.buy_executed and signal > 0:
            self._add_dca_order()
            self.buy_executed = True
            return

        # DCA LOGIC
        if self.p.dca_enabled and self.active_orders:
            last_order = self.active_orders[-1]
            layers = len(self.active_orders)
            target_price = last_order.entry_price * (1 - self.p.dca_deviation / 100)

            if current_price <= target_price and layers < self.p.dca_max_layers:
                self._add_dca_order(last_order)

        # SELL / TAKE PROFIT / STOP LOSS
        orders_to_remove = []
        for idx, order in enumerate(self.active_orders):
            # Take profit
            if current_price >= order.take_profit_price:
                self.sell(size=order.size, exectype=bt.Order.Market)
                if self.p.debug:
                    print(f"TP hit: {((current_price/order.entry_price)-1)*100:.2f}% at {current_price}")
                order.close_order(current_price)
                orders_to_remove.append(idx)

            # Stop loss
            elif current_price <= order.entry_price * (1 - self.p.stop_loss/100):
                self.sell(size=order.size, exectype=bt.Order.Market)
                if self.p.debug:
                    print(f"SL hit: {((current_price/order.entry_price)-1)*100:.2f}% at {current_price}")
                order.close_order(current_price)
                orders_to_remove.append(idx)

            # Signal-based exit
            elif signal < -0.5:
                self.sell(size=order.size, exectype=bt.Order.Market)
                if self.p.debug:
                    print(f"Signal exit at {current_price}")
                order.close_order(current_price)
                orders_to_remove.append(idx)

        # Remove closed orders
        for idx in sorted(orders_to_remove, reverse=True):
            self.active_orders.pop(idx)

        if not self.active_orders:
            self.buy_executed = False

    def stop(self):
        if self.p.debug:
            print(f"\n=== DCA STRATEGY RESULTS ===")
            print(f"Final Portfolio Value: ${self.broker.getvalue():.2f}")
            print(f"Final Cash: ${self.broker.get_cash():.2f}")
            print(f"Total DCA layers executed: {len(self.active_orders)}")
