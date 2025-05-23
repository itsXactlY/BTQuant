from .base import BaseStrategy, bt, OrderTracker
from datetime import datetime


class CCI(bt.Indicator):
    lines = ('cci',)
    params = (('period', 20), ('safediv', True))

    def __init__(self):
        super(CCI, self).__init__()
        self.addminperiod(self.p.period)
        self.ma = bt.indicators.SMA(self.data.close, period=self.p.period)

    def next(self):
        mean_dev = sum(abs(self.data.close[-i] - self.ma[-i]) for i in range(self.p.period)) / self.p.period
        if mean_dev > 1e-8:  # Avoid division by very small numbers
            self.lines.cci[0] = (self.data.close[0] - self.ma[0]) / (0.015 * mean_dev)
        else:
            self.lines.cci[0] = 0

class APO(bt.Indicator): # Absolute Price Oscillator
    lines = ('apo',)
    params = (('fast', 12), ('slow', 26), ('safediv', True))

    def __init__(self):
        super(APO, self).__init__()
        self.fast_ma = bt.indicators.EMA(self.data, period=self.p.fast)
        self.slow_ma = bt.indicators.EMA(self.data, period=self.p.slow)

    def next(self):
        self.lines.apo[0] = self.fast_ma[0] - self.slow_ma[0]

class MFI(bt.Indicator):
    lines = ('mfi',)
    params = (('period', 14), ('safediv', True))

    def __init__(self):
        super(MFI, self).__init__()
        self.addminperiod(self.p.period)

    def next(self):
        typical_price = (self.data.high + self.data.low + self.data.close) / 3
        money_flow = typical_price * self.data.volume

        positive_flow = sum(money_flow[-i] for i in range(1, self.p.period + 1) if typical_price[-i] > typical_price[-i-1])
        negative_flow = sum(money_flow[-i] for i in range(1, self.p.period + 1) if typical_price[-i] < typical_price[-i-1])

        if negative_flow < 1e-8:  # Avoid division by very small numbers
            self.lines.mfi[0] = 100
        elif positive_flow < 1e-8:
            self.lines.mfi[0] = 0
        else:
            money_ratio = positive_flow / negative_flow
            self.lines.mfi[0] = 100 - (100 / (1 + money_ratio))

class Stochastic_Generic(bt.Indicator):
    '''
    This generic indicator doesn't assume the data feed has the components
    ``high``, ``low`` and ``close``. It needs three data sources passed to it,
    which whill considered in that order. (following the OHLC standard naming)
    '''
    lines = ('k', 'd', 'dslow',)
    params = dict(
        pk=14,
        pd=3,
        pdslow=3,
        movav=bt.ind.SMA,
        slowav=None,
    )

    def __init__(self):
        # Get highest from period k from 1st data
        highest = bt.ind.Highest(self.data0, period=self.p.pk)
        # Get lowest from period k from 2nd data
        lowest = bt.ind.Lowest(self.data1, period=self.p.pk)

        # Apply the formula to get raw K
        kraw = 100.0 * (self.data2 - lowest) / (highest - lowest)

        # The standard k in the indicator is a smoothed versin of K
        self.l.k = k = self.p.movav(kraw, period=self.p.pd)

        # Smooth k => d
        slowav = self.p.slowav or self.p.movav  # chose slowav
        self.l.d = slowav(k, period=self.p.pdslow)

class Enhanced_MACD_ADX(BaseStrategy):
    params = (
        ("dca_threshold", 1.5),
        ("take_profit", 2),
        ('stop_loss', 20),
        ('percent_sizer', 0.05),
        ("macd_period_me1", 11),
        ("macd_period_me2", 23),
        ("macd_period_signal", 7),
        ("adx_period", 13),
        ("adx_strength", 31),
        ("di_period", 14),
        ("adxth", 25),
        
        ("momentum_period", 14),
        ("rsi_period", 14),
        ("stoch_period", 14),
        ("cci_period", 20),
        ("trix_period", 15),
        ("apo_fast", 12),
        ("apo_slow", 26),
        ("cmf_period", 20),
        
        ('debug', False),
        ("backtest", None),
        ('use_stoploss', False),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Existing indicators
        self.macd = bt.indicators.MACD(self.data.close, period_me1=self.p.macd_period_me1,
                                      period_me2=self.p.macd_period_me2, period_signal=self.p.macd_period_signal, plot=True)
        self.adx = bt.indicators.ADX(self.data, period=self.p.adx_period, plot=True)
        self.plusDI = bt.indicators.PlusDI(self.data, period=self.p.di_period, plot=True)
        self.minusDI = bt.indicators.MinusDI(self.data, period=self.p.di_period, plot=True)
        
        # New indicators
        self.momentum = bt.indicators.Momentum(self.data, period=self.p.momentum_period, plot=True)
        self.rsi = bt.indicators.RSI(self.data, period=self.p.rsi_period, plot=True)
        d0 = bt.ind.EMA(self.data.high, period=14)
        d1 = bt.ind.EMA(self.data.low, period=14)
        d2 = bt.ind.EMA(self.data.close, period=14)
        self.stoch = Stochastic_Generic(d0, d1, d2)
        self.cci = bt.indicators.CCI(self.data, period=self.p.cci_period, plot=True)
        self.trix = bt.indicators.Trix(self.data, period=self.p.trix_period, plot=True)
        self.apo = bt.indicators.APO(self.data, plot=True)
        
        self.DCA = True

    def buy_or_short_condition(self):
        if (self.macd.lines.macd[0] > 0 and
            self.adx[0] >= self.p.adxth and
            self.minusDI[0] > self.p.adxth and
            self.plusDI[0] < self.p.adxth and
            self.momentum[0] > 0 and
            self.rsi[0] < 70 and
            self.stoch.lines.k < 80 and
            self.cci[0] > -100 and
            self.trix[0] > 0 and
            self.apo[0] > 0
            ):

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
        if self.position and (self.macd.lines.macd[0] > 0 and
            self.adx[0] >= self.p.adxth and
            self.minusDI[0] > self.p.adxth and
            self.plusDI[0] < self.p.adxth and
            self.momentum[0] > 0 and
            self.rsi[0] < 70 and
            self.stoch.lines.k < 80 and
            self.cci[0] > -100 and
            self.trix[0] > 0 and
            self.apo[0] > 0):


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
