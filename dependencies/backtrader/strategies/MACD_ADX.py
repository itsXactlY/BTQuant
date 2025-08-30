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
        ("dca_threshold", 3.5),
        ("take_profit", 7),
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


    ''' 
    TODO DI logic is inverted for longs. For long entries you want +DI > -DI, not the opposite.
        Some lines lack [0] indexing (e.g., self.stoch.lines.k < 80). That can silently misbehave.
        DCA: you have dca_threshold in params but don’t use it; DCA condition is identical to the entry condition.
        Consider adding MFI (Money Flow Index) for better volume analysis.'''
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



class Enhanced_MACD_ADX2(BaseStrategy):
    params = (
        ("dca_threshold", 3.5),         # not % anymore – we’ll use ATR-based adds; keep this if you still want % adds
        ("take_profit", 7),
        ('stop_loss', 20),
        ('percent_sizer', 0.05),

        ("macd_period_me1", 11),
        ("macd_period_me2", 23),
        ("macd_period_signal", 7),
        ("adx_period", 13),
        ("di_period", 14),
        ("adxth", 25),

        # New core tuning knobs
        ("breakout_period", 20),        # Donchian breakout
        ("atr_period", 14),
        ("ema_fast", 20),
        ("ema_slow", 50),
        ("ema_trend", 200),
        ("vol_window", 20),
        ("vol_mult", 1.3),              # breakout vol confirm

        # DCA / pyramiding
        ("use_dca", True),
        ("max_adds", 3),
        ("add_cooldown", 5),            # bars between adds
        ("dca_atr_mult", 1.0),          # add if pullback >= this * ATR from last add or avg entry
        ("add_on_ema_touch", True),     # add if touch EMA20 during trend

        # Exits
        ("use_trailing_stop", False),
        ("trail_atr_mult", 3.0),        # Chandelier style
        ("init_sl_atr_mult", 1.25),     # initial stop below breakout low
        ("move_to_breakeven_R", 1.0),   # when >=1R, bump stop to BE

        # Oscillators you already had
        ("momentum_period", 14),
        ("rsi_period", 14),
        ("stoch_period", 14),
        ("cci_period", 20),
        ("trix_period", 15),

        ('debug', False),
        ("backtest", None),
        ('use_stoploss', True),
    )
    '''
    Parameter starting points (good first pass)

    breakout_period = 20
    adxth = 20–25 (25 is common)
    vol_mult = 1.3–1.8 (higher for less noise)
    init_sl_atr_mult = 1.0–1.5
    trail_atr_mult = 2.5–3.5 (lower = tighter)
    dca_atr_mult = 0.8–1.2
    max_adds = 2–4
    add_cooldown = 3–8 bars
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Core trend and volatility
        self.ema_fast = bt.ind.EMA(self.data.close, period=self.p.ema_fast)
        self.ema_slow = bt.ind.EMA(self.data.close, period=self.p.ema_slow)
        self.ema_trend = bt.ind.EMA(self.data.close, period=self.p.ema_trend)
        self.atr = bt.ind.ATR(self.data, period=self.p.atr_period)
        self.vol_sma = bt.ind.SMA(self.data.volume, period=self.p.vol_window)

        # Donchian breakout (use previous bar’s upper to avoid same-bar lookahead)
        self.dc_high = bt.ind.Highest(self.data.high, period=self.p.breakout_period)
        self.dc_low = bt.ind.Lowest(self.data.low, period=self.p.breakout_period)

        # Existing indicators
        self.macd = bt.indicators.MACD(self.data.close,
                                       period_me1=self.p.macd_period_me1,
                                       period_me2=self.p.macd_period_me2,
                                       period_signal=self.p.macd_period_signal, plot=True)
        self.adx = bt.indicators.ADX(self.data, period=self.p.adx_period, plot=True)
        self.plusDI = bt.indicators.PlusDI(self.data, period=self.p.di_period, plot=True)
        self.minusDI = bt.indicators.MinusDI(self.data, period=self.p.di_period, plot=True)

        self.momentum = bt.indicators.Momentum(self.data, period=self.p.momentum_period, plot=True)
        self.rsi = bt.indicators.RSI(self.data, period=self.p.rsi_period, plot=True)

        # Simple stoch using built-ins (safer indexing)
        self.stoch = bt.ind.Stochastic(self.data, period=self.p.stoch_period)
        self.cci = bt.indicators.CCI(self.data, period=self.p.cci_period, plot=True)
        self.trix = bt.indicators.Trix(self.data, period=self.p.trix_period, plot=True)

        # Internal state
        self.DCA = self.p.use_dca
        self.n_adds = 0
        self.last_add_bar = -999999
        self.breakout_low = None
        self.trail_stop = None
        self.avg_entry = None

    # ---------- Helpers
    def trend_ok(self):
        # Strong trend: EMAs stacked, +DI > -DI, ADX above threshold and rising
        ema_stack = self.ema_fast[0] > self.ema_slow[0] > self.ema_trend[0]
        di_ok = self.plusDI[0] > self.minusDI[0]  # fixed: +DI must dominate for longs
        adx_ok = self.adx[0] >= self.p.adxth and self.adx[0] >= self.adx[-1]
        return ema_stack and di_ok and adx_ok

    def breakout_up(self):
        # Close crosses above yesterday's Donchian upper and volume confirms, avoid huge extension
        if len(self.data) < self.p.breakout_period + 2:
            return False
        prior_upper = self.dc_high[-1]
        crossed = self.data.close[-1] <= prior_upper and self.data.close[0] > prior_upper
        vol_ok = self.data.volume[0] > self.p.vol_mult * max(self.vol_sma[0], 1e-8)
        not_stretched = (self.data.close[0] - prior_upper) <= 1.0 * self.atr[0]
        return crossed and vol_ok and not_stretched

    def momentum_ok(self):
        return (self.macd.macd[0] > self.macd.signal[0] and self.macd.macd[0] > 0 and
                self.momentum[0] > 0 and self.rsi[0] < 70 and
                self.stoch.percK[0] < 80 and self.cci[0] > -100 and self.trix[0] > 0)

    def update_trailing_stop(self):
        if not self.position:
            self.trail_stop = None
            return
        # Highest close since entry can be approximated via a Highest on close starting at entry; for simplicity, we roll a max
        if not hasattr(self, 'run_high'):
            self.run_high = self.data.close[0]
        self.run_high = max(self.run_high, self.data.close[0])

        chandelier = self.run_high - self.p.trail_atr_mult * self.atr[0]
        if self.trail_stop is None:
            self.trail_stop = chandelier
        else:
            self.trail_stop = max(self.trail_stop, chandelier)  # ratchet only upward

        # Move to breakeven after 1R
        if self.avg_entry and self.breakout_low:
            risk_per_share = self.avg_entry - self.breakout_low
            if risk_per_share > 0:
                R = (self.data.close[0] - self.avg_entry) / risk_per_share
                if R >= self.p.move_to_breakeven_R:
                    self.trail_stop = max(self.trail_stop, self.avg_entry)

    def can_add(self):
        return (self.DCA and self.position and
                self.n_adds < self.p.max_adds and
                (len(self) - self.last_add_bar) >= self.p.add_cooldown and
                self.trend_ok())

    # ---------- Entry / DCA / Exit conditions
    def buy_or_short_condition(self):
        self.conditions_checked = True

        if self.position:
            return

        if self.trend_ok() and self.breakout_up() and self.momentum_ok():
            # Place initial long
            size = self._determine_size()
            order_tracker = OrderTracker(
                entry_price=self.data.close[0],
                size=size,
                take_profit_pct=self.params.take_profit,    # keep if OrderTracker uses it
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
                print(f"Buy (breakout) {size} @ {self.data.close[0]}")

            if not self.buy_executed:
                if not hasattr(self, 'first_entry_price') or self.first_entry_price is None:
                    self.first_entry_price = self.data.close[0]
                self.buy_executed = True

            # Set initial risk reference: breakout_low is prior Donchian low or last swing
            self.breakout_low = self.dc_low[-1]
            # Optional: initial stop as a virtual line; implement sell in sell_or_cover_condition
            init_stop = self.breakout_low - self.p.init_sl_atr_mult * self.atr[0]
            self.trail_stop = init_stop
            self.run_high = self.data.close[0]
            self.n_adds = 0
            self.last_add_bar = len(self)
            self.calc_averages()         # updates self.avg_entry (your helper)

    def dca_or_short_condition(self):
        self.conditions_checked = True
        if not self.can_add():
            return

        # Pullback-based add: price retraces to EMA20 or falls >= dca_atr_mult * ATR from last add/avg
        touch_ema = self.p.add_on_ema_touch and (self.data.low[0] <= self.ema_fast[0])
        atr_pullback = False
        if self.avg_entry:
            atr_pullback = (self.avg_entry - self.data.close[0]) >= (self.p.dca_atr_mult * self.atr[0])

        if (touch_ema or atr_pullback) and self.momentum_ok():
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
                print(f"DCA add {self.n_adds+1}/{self.p.max_adds}: {size} @ {self.data.close[0]}")

            self.n_adds += 1
            self.last_add_bar = len(self)
            self.calc_averages()

    def sell_or_cover_condition(self):
        self.conditions_checked = True
        if not hasattr(self, 'active_orders'):
            self.active_orders = []

        if not self.position:
            # Clean up state
            if self.active_orders:
                self.active_orders = []
            self.trail_stop = None
            self.run_high = None
            self.n_adds = 0
            return

        current_price = self.data.close[0]

        # Update trailing logic
        if self.p.use_trailing_stop:
            self.update_trailing_stop()

        # 1) Hard/Trailing stop
        if self.p.use_trailing_stop and self.trail_stop is not None and current_price <= self.trail_stop:
            # Close all
            total_size = sum(o.size for o in self.active_orders) if self.active_orders else self.position.size
            self.order = self.sell(size=total_size, exectype=bt.Order.Market)
            if self.p.debug:
                print(f"Trailing stop hit @ {current_price}, closing {total_size}")
            # Clear and reset state
            self.active_orders = []
            self.reset_position_state()
            self.buy_executed = False
            self.trail_stop = None
            self.run_high = None
            self.n_adds = 0
            return

        # 2) Take profit per sub-order (if you want to keep this behavior)
        orders_to_remove = []
        for idx, order in enumerate(self.active_orders):
            if current_price >= order.take_profit_price:
                self.order = self.sell(size=order.size, exectype=bt.Order.Market)
                if self.p.debug:
                    print(f"TP hit: Selling {order.size} @ {current_price} (entry: {order.entry_price})")
                order.close_order(current_price)
                orders_to_remove.append(idx)

        for idx in sorted(orders_to_remove, reverse=True):
            removed_order = self.active_orders.pop(idx)
            if self.p.debug:
                profit_pct = ((current_price / removed_order.entry_price) - 1) * 100
                print(f"Order removed: {profit_pct:.2f}% profit")

        if orders_to_remove:
            self.entry_prices = [order.entry_price for order in self.active_orders]
            self.sizes = [order.size for order in self.active_orders]
            if not self.active_orders:
                self.reset_position_state()
                self.buy_executed = False
                self.trail_stop = None
                self.run_high = None
                self.n_adds = 0
            else:
                self.calc_averages()


class Enhanced_MACD_ADX3(BaseStrategy):
    params = (
        # Positioning
        ('percent_sizer', 0.05),

        # Core trend indicators
        ("macd_period_me1", 11),
        ("macd_period_me2", 23),
        ("macd_period_signal", 7),
        ("adx_period", 13),
        ("di_period", 14),
        ("adxth", 25),

        # Breakout and volatility
        ("breakout_period", 20),     # Donchian breakout window
        ("atr_period", 14),
        ("ema_fast", 20),
        ("ema_slow", 50),
        ("ema_trend", 200),
        ("vol_window", 20),
        ("vol_mult", 1.3),           # breakout volume confirmation
        ("stretch_atr_mult", 1.0),   # max close above breakout <= this * ATR

        # Oscillators (filters)
        ("momentum_period", 14),
        ("rsi_period", 14),
        ("stoch_period", 14),
        ("cci_period", 20),
        ("trix_period", 15),

        # DCA / pyramiding
        ("use_dca", True),
        ("max_adds", 7),
        ("add_cooldown", 50),         # bars between adds
        ("dca_atr_mult", 1.0),       # add if pullback >= this * ATR from avg entry
        ("add_on_ema_touch", True),  # add on EMA20 touch during trend

        # Exits
        ("take_profit", 0.5),          # per-suborder TP (still supported)
        ('use_stoploss', False),
        ("use_trailing_stop", False),

        # Gentler trailing stop defaults
        ("trail_mode", "ema_band"),  # 'ema_band' | 'chandelier' | 'donchian' | 'pivot'
        ("trail_atr_mult", 4.5),     # chandelier multiple (if used)
        ("ema_band_mult", 2.25),     # EMA20 - k*ATR
        ("donchian_trail_period", 55),
        ("pivot_left", 2),           # pivot low confirmation (L/R bars)
        ("pivot_right", 2),

        ("init_sl_atr_mult", 1.25),  # initial stop below structure
        ("trail_arm_R", 2.0),        # arm trailing after >= R
        ("trail_arm_bars", 10),      # or after N bars
        ("trail_update_every", 3),   # recalc trail every N bars
        ("move_to_breakeven_R", 1.5),

        # Runtime
        ("backtest", False),
        ('debug', False),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Trend + vol
        self.ema_fast = bt.ind.EMA(self.data.close, period=self.p.ema_fast)
        self.ema_slow = bt.ind.EMA(self.data.close, period=self.p.ema_slow)
        self.ema_trend = bt.ind.EMA(self.data.close, period=self.p.ema_trend)
        self.atr = bt.ind.ATR(self.data, period=self.p.atr_period)
        self.vol_sma = bt.ind.SMA(self.data.volume, period=self.p.vol_window)

        # Donchian breakout and slow exit
        self.dc_high = bt.ind.Highest(self.data.high, period=self.p.breakout_period)
        self.dc_low = bt.ind.Lowest(self.data.low, period=self.p.breakout_period)
        self.dc_exit = bt.ind.Lowest(self.data.low, period=self.p.donchian_trail_period)

        # ADX/DI + momentum filters
        self.adx = bt.ind.ADX(self.data, period=self.p.adx_period, plot=True)
        self.plusDI = bt.ind.PlusDI(self.data, period=self.p.di_period, plot=True)
        self.minusDI = bt.ind.MinusDI(self.data, period=self.p.di_period, plot=True)

        self.macd = bt.ind.MACD(self.data.close,
                                period_me1=self.p.macd_period_me1,
                                period_me2=self.p.macd_period_me2,
                                period_signal=self.p.macd_period_signal,
                                plot=True)
        self.momentum = bt.ind.Momentum(self.data, period=self.p.momentum_period, plot=True)
        self.rsi = bt.ind.RSI(self.data, period=self.p.rsi_period, plot=True)
        self.stoch = bt.ind.Stochastic(self.data, period=self.p.stoch_period)
        self.cci = bt.ind.CCI(self.data, period=self.p.cci_period, plot=True)
        self.trix = bt.ind.Trix(self.data, period=self.p.trix_period, plot=True)

        # State
        self.DCA = self.p.use_dca
        self.n_adds = 0
        self.last_add_bar = -10**9
        self.breakout_low = None

        self.trail_stop = None
        self.init_stop = None
        self.run_high = None
        self.entry_bar = None
        self.trail_armed = False
        self.last_trail_update = -10**9
        self.initial_risk = None
        self.last_pivot_low = None

        # Ensure lists exist
        if not hasattr(self, 'active_orders'):
            self.active_orders = []
        if not hasattr(self, 'entry_prices'):
            self.entry_prices = []
        if not hasattr(self, 'sizes'):
            self.sizes = []
        if not hasattr(self, 'buy_executed'):
            self.buy_executed = False

    # ----------------------- Helpers -----------------------
    def _compute_avg_entry(self):
        # Average price weighted by size from active orders or local lists
        try:
            if self.active_orders:
                tot = sum(o.size for o in self.active_orders)
                return sum(o.entry_price * o.size for o in self.active_orders) / tot if tot else None
        except Exception:
            pass
        if self.entry_prices and self.sizes:
            tot = sum(self.sizes)
            return sum(p * s for p, s in zip(self.entry_prices, self.sizes)) / tot if tot else None
        return None

    def trend_ok(self):
        ema_stack = self.ema_fast[0] > self.ema_slow[0] > self.ema_trend[0]
        di_ok = self.plusDI[0] > self.minusDI[0]
        adx_ok = self.adx[0] >= self.p.adxth and self.adx[0] >= self.adx[-1]
        return ema_stack and di_ok and adx_ok

    def breakout_up(self):
        # cross above prior Donchian high with vol confirmation and limited ATR stretch
        if len(self.data) < self.p.breakout_period + 2:
            return False
        prior_upper = self.dc_high[-1]
        crossed = self.data.close[-1] <= prior_upper and self.data.close[0] > prior_upper
        vol_ok = self.data.volume[0] > self.p.vol_mult * max(self.vol_sma[0], 1e-8)
        not_stretched = (self.data.close[0] - prior_upper) <= self.p.stretch_atr_mult * self.atr[0]
        return crossed and vol_ok and not_stretched

    def momentum_ok(self):
        return (self.macd.macd[0] > self.macd.signal[0] and self.macd.macd[0] > 0 and
                self.momentum[0] > 0 and self.rsi[0] < 70 and
                self.stoch.percK[0] < 80 and self.cci[0] > -100 and self.trix[0] > 0)

    def can_add(self):
        return (self.DCA and self.position and
                self.n_adds < self.p.max_adds and
                (len(self) - self.last_add_bar) >= self.p.add_cooldown and
                self.trend_ok())

    def _maybe_arm_trail(self):
        if self.trail_armed or self.entry_bar is None:
            return
        bars_in_trade = len(self) - self.entry_bar
        R = 0.0
        avg_entry = self._compute_avg_entry()
        if avg_entry and self.initial_risk and self.initial_risk > 0:
            R = (self.data.close[0] - avg_entry) / self.initial_risk
        if (bars_in_trade >= self.p.trail_arm_bars) or (R >= self.p.trail_arm_R):
            self.trail_armed = True

    def _update_pivot_low(self):
        # Simple 5-bar pivot low: 2 left, pivot at -2, 2 right
        if len(self.data) < 5:
            return
        c = self.data.low[-2]
        if (c < self.data.low[-3] and c < self.data.low[-4] and
            c < self.data.low[-1] and c < self.data.low[0]):
            self.last_pivot_low = c

    def update_trailing_stop(self):
        if not self.position:
            self.trail_stop = None
            return

        # Keep run_high for chandelier
        self.run_high = max(self.run_high or self.data.high[0], self.data.high[0])

        # Arm trailing later
        self._maybe_arm_trail()

        # Before armed: keep initial stop, nudge to BE at later R if desired
        if not self.trail_armed:
            avg_entry = self._compute_avg_entry()
            if (avg_entry and self.initial_risk and
                (self.data.close[0] - avg_entry) / self.initial_risk >= self.p.move_to_breakeven_R):
                self.trail_stop = max(self.trail_stop or -1e18, avg_entry)
            return

        # Throttle trail updates
        if (len(self) - self.last_trail_update) < self.p.trail_update_every:
            return

        candidate = None
        if self.p.trail_mode == "ema_band":
            candidate = float(self.ema_fast[0] - self.p.ema_band_mult * self.atr[0])

        elif self.p.trail_mode == "chandelier":
            candidate = float(self.run_high - self.p.trail_atr_mult * self.atr[0])

        elif self.p.trail_mode == "donchian":
            candidate = float(self.dc_exit[0])

        elif self.p.trail_mode == "pivot":
            self._update_pivot_low()
            if self.last_pivot_low is not None:
                candidate = float(self.last_pivot_low - 0.5 * self.atr[0])

        if candidate is not None:
            new_stop = max(self.trail_stop or -1e18, candidate, self.init_stop or -1e18)
            avg_entry = self._compute_avg_entry()
            if (avg_entry and self.initial_risk and
                (self.data.close[0] - avg_entry) / self.initial_risk >= self.p.move_to_breakeven_R):
                new_stop = max(new_stop, avg_entry)
            self.trail_stop = new_stop
            self.last_trail_update = len(self)

    # ----------------------- Entry / DCA / Exit -----------------------
    def buy_or_short_condition(self):
        self.conditions_checked = True
        if self.position:
            return

        if self.trend_ok() and self.breakout_up() and self.momentum_ok():
            size = self._determine_size()
            order_tracker = OrderTracker(
                entry_price=self.data.close[0],
                size=size,
                take_profit_pct=self.p.take_profit,
                symbol=getattr(self, 'symbol', getattr(self.p, 'asset', None)),
                order_type="BUY",
                backtest=self.p.backtest
            )
            order_tracker.order_id = f"order_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            if not hasattr(self, 'active_orders'):
                self.active_orders = []
            if not hasattr(self, 'entry_prices'):
                self.entry_prices = []
            if not hasattr(self, 'sizes'):
                self.sizes = []

            self.active_orders.append(order_tracker)
            self.entry_prices.append(self.data.close[0])
            self.sizes.append(size)
            self.order = self.buy(size=size, exectype=bt.Order.Market)

            if self.p.debug:
                print(f"Buy (breakout) {size} @ {self.data.close[0]}")

            if not self.buy_executed:
                if not hasattr(self, 'first_entry_price') or self.first_entry_price is None:
                    self.first_entry_price = self.data.close[0]
                self.buy_executed = True

            # Initial risk references
            self.breakout_low = self.dc_low[-1]
            self.init_stop = float(self.breakout_low - self.p.init_sl_atr_mult * self.atr[0])
            self.trail_stop = self.init_stop
            self.run_high = self.data.high[0]
            self.entry_bar = len(self)
            self.trail_armed = False
            self.n_adds = 0
            self.last_add_bar = len(self)

            # Update averages/risk
            if hasattr(self, 'calc_averages'):
                self.calc_averages()
            avg_entry = self._compute_avg_entry()
            self.initial_risk = max(1e-8, (avg_entry - self.init_stop)) if avg_entry else None

    def dca_or_short_condition(self):
        self.conditions_checked = True
        if not self.can_add():
            return

        avg_entry = self._compute_avg_entry()
        touch_ema = self.p.add_on_ema_touch and (self.data.low[0] <= self.ema_fast[0])
        atr_pullback = False
        if avg_entry:
            atr_pullback = (avg_entry - self.data.close[0]) >= (self.p.dca_atr_mult * self.atr[0])

        if (touch_ema or atr_pullback) and self.momentum_ok():
            size = self._determine_size()
            order_tracker = OrderTracker(
                entry_price=self.data.close[0],
                size=size,
                take_profit_pct=self.p.take_profit,
                symbol=getattr(self, 'symbol', getattr(self.p, 'asset', None)),
                order_type="BUY",
                backtest=self.p.backtest
            )
            order_tracker.order_id = f"order_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            self.active_orders.append(order_tracker)
            self.entry_prices.append(self.data.close[0])
            self.sizes.append(size)
            self.order = self.buy(size=size, exectype=bt.Order.Market)

            if self.p.debug:
                print(f"DCA add {self.n_adds+1}/{self.p.max_adds}: {size} @ {self.data.close[0]}")

            self.n_adds += 1
            self.last_add_bar = len(self)

            if hasattr(self, 'calc_averages'):
                self.calc_averages()
            # Keep initial stop as is; trailing logic will manage ratchet
            avg_entry = self._compute_avg_entry()
            # Recompute initial risk only if init_stop changed; otherwise keep

    def sell_or_cover_condition(self):
        self.conditions_checked = True

        if not hasattr(self, 'active_orders'):
            self.active_orders = []

        if not self.position:
            # Clean state if needed
            self.active_orders = []
            self.trail_stop = None
            self.init_stop = None
            self.run_high = None
            self.entry_bar = None
            self.trail_armed = False
            self.n_adds = 0
            return

        current_price = self.data.close[0]

        # Update trailing stop first
        if self.p.use_trailing_stop:
            self.update_trailing_stop()

        # Hard/Trailing stop hit? (use low to catch intrabar/gaps)
        if self.p.use_trailing_stop and self.trail_stop is not None and self.data.low[0] <= self.trail_stop:
            total_size = sum(o.size for o in self.active_orders) if self.active_orders else self.position.size
            self.order = self.sell(size=total_size, exectype=bt.Order.Market)
            if self.p.debug:
                print(f"Trailing stop hit @ {self.data.low[0]} <= {self.trail_stop} (close {current_price}) - closing {total_size}")

            # Close and clear all tracked orders
            for o in self.active_orders:
                o.close_order(self.data.low[0])
            self.active_orders = []
            if hasattr(self, 'reset_position_state'):
                self.reset_position_state()
            self.buy_executed = False

            # Reset trail state
            self.trail_stop = None
            self.init_stop = None
            self.run_high = None
            self.entry_bar = None
            self.trail_armed = False
            self.n_adds = 0
            return

        # Per-suborder take-profits (optional; keep if you like partials)
        orders_to_remove = []
        for idx, order in enumerate(self.active_orders):
            if current_price >= order.take_profit_price:
                self.order = self.sell(size=order.size, exectype=bt.Order.Market)
                if self.p.debug:
                    print(f"TP hit: Selling {order.size} @ {current_price} (entry {order.entry_price}, TP {order.take_profit_price})")
                order.close_order(current_price)
                orders_to_remove.append(idx)

        # Remove closed orders and recalc averages
        for idx in sorted(orders_to_remove, reverse=True):
            removed_order = self.active_orders.pop(idx)
            if self.p.debug:
                profit_pct = ((current_price / removed_order.entry_price) - 1) * 100
                print(f"Order removed: {profit_pct:.2f}% profit")

        if orders_to_remove:
            self.entry_prices = [o.entry_price for o in self.active_orders]
            self.sizes = [o.size for o in self.active_orders]
            if not self.active_orders:
                if hasattr(self, 'reset_position_state'):
                    self.reset_position_state()
                self.buy_executed = False
                # Reset trail state
                self.trail_stop = None
                self.init_stop = None
                self.run_high = None
                self.entry_bar = None
                self.trail_armed = False
                self.n_adds = 0
            else:
                if hasattr(self, 'calc_averages'):
                    self.calc_averages()