from backtrader.utils.backtest import backtest

# ========== Custom Indicators ==========
# from backtrader.indicators.MesaAdaptiveMovingAverage import MAMA
# from backtrader.indicators.ChaikinMoneyFlow import ChaikinMoneyFlow
# from backtrader.indicators.ChaikinVolatility import ChaikinVolatility
# from backtrader.indicators.MADR import MADRIndicator
# ========================================


# from backtrader.strategies.NearestNeighbors_RationalQuadraticKernel import NRK as strategy
# from backtrader.strategies.ST_RSX_ASI import STrend_RSX_AccumulativeSwingIndex as strategy
# from backtrader.strategies.Order_Chain_Kioseff_Trading import Order_Chain_Kioseff_Trading
# from backtrader.strategies.Vumanchu_A import VuManchCipher_A
# from backtrader.strategies.Vumanchu_B import VuManchCipher_B

from backtrader.strategies.base import BaseStrategy, bt, np, OrderTracker, datetime
from collections import deque
import math

class OrderChain1sX(bt.Indicator):
    """
    1s OrderChain with extra context:
      - oc_ema / oc_z: EMA of signed volume flow and its z-score
      - vol_z: volume z-score (impulse filter)
      - rng_z: range (high-low) z-score (impulse filter)
      - wick_ratio, pos_in_bar
    """
    lines = ('oc_ema','oc_z','wick_ratio','rv1s','vol_z','rng_z','pos_in_bar',
             'long_mom','short_mom','long_rev','short_rev')
    plotinfo = dict(subplot=True)
    params = (
        ('ema_tau_s', 12),
        ('z_window', 120),
        ('z_entry', 2.5),
        ('wick_min', 0.25),   # min wick for reversion
        ('wick_max', 0.60),   # max wick for momentum (avoid exhaustion bars)
        ('rv_cap', 0.010),    # 1s realized vol cap (1.0%) for alts; tighten for majors
        ('vol_window', 120),
        ('rng_window', 120),
        ('vol_z_min', 0.5),
        ('rng_z_min', 0.5),
        ('eps', 1e-12),
    )

    def __init__(self):
        self.prev_close = None
        self.oc_ema_val = 0.0
        self.alpha = 1.0 - math.exp(-1.0 / max(1, self.p.ema_tau_s))
        self._zbuf = deque(maxlen=self.p.z_window)
        self._volbuf = deque(maxlen=self.p.vol_window)
        self._rngbuf = deque(maxlen=self.p.rng_window)

    def next(self):
        c = float(self.data.close[0])
        o = float(self.data.open[0])
        h = float(self.data.high[0])
        l = float(self.data.low[0])
        v = float(self.data.volume[0]) if not math.isnan(self.data.volume[0]) else 0.0

        # Signed flow with conviction
        if self.prev_close is None:
            direction = 0.0
        else:
            dc = c - self.prev_close
            direction = 1.0 if dc > 0 else (-1.0 if dc < 0 else 0.0)
        rng = max(self.p.eps, h - l)
        dir_weight = abs(c - o) / rng if rng > 0 else 0.0
        conviction = 0.5 + 0.5 * dir_weight
        flow = direction * v * conviction

        # EMA and z
        self.oc_ema_val = (1 - self.alpha) * self.oc_ema_val + self.alpha * flow
        self.lines.oc_ema[0] = self.oc_ema_val

        self._zbuf.append(self.oc_ema_val)
        if len(self._zbuf) >= 2:
            m = float(np.mean(self._zbuf))
            s = float(np.std(self._zbuf)) + self.p.eps
            oc_z = (self.oc_ema_val - m) / s
        else:
            oc_z = 0.0
        self.lines.oc_z[0] = oc_z

        # Volume/Range z-scores
        self._volbuf.append(v)
        self._rngbuf.append(rng)
        vol_z = 0.0
        rng_z = 0.0
        if len(self._volbuf) > 10:
            vm = float(np.mean(self._volbuf)); vs = float(np.std(self._volbuf)) + self.p.eps
            vol_z = (v - vm) / vs
        if len(self._rngbuf) > 10:
            rm = float(np.mean(self._rngbuf)); rs = float(np.std(self._rngbuf)) + self.p.eps
            rng_z = (rng - rm) / rs
        self.lines.vol_z[0] = vol_z
        self.lines.rng_z[0] = rng_z

        # Context
        wick_up = max(0.0, h - max(o, c))
        wick_dn = max(0.0, min(o, c) - l)
        wick_ratio = max(wick_up, wick_dn) / (rng + self.p.eps)
        self.lines.wick_ratio[0] = wick_ratio
        prev_c = self.prev_close if self.prev_close is not None else c
        rv1s = abs(c - prev_c) / max(prev_c, self.p.eps)
        self.lines.rv1s[0] = rv1s
        pos_in_bar = (c - l) / (rng + self.p.eps)  # 0..1
        self.lines.pos_in_bar[0] = pos_in_bar

        # Gating
        impulse_ok = (vol_z >= self.p.vol_z_min) and (rng_z >= self.p.rng_z_min) and (rv1s <= self.p.rv_cap)

        # Momentum signals (follow oc_z direction, avoid giant wicks)
        long_mom  = (oc_z >= self.p.z_entry)   and impulse_ok and (wick_ratio <= self.p.wick_max) and (pos_in_bar > 0.5)
        short_mom = (oc_z <= -self.p.z_entry)  and impulse_ok and (wick_ratio <= self.p.wick_max) and (pos_in_bar < 0.5)

        # Reversion signals (fade oc_z extremes with big wicks near extremes)
        long_rev  = (oc_z <= -self.p.z_entry)  and impulse_ok and (wick_ratio >= self.p.wick_min) and (pos_in_bar <= 0.25)
        short_rev = (oc_z >= self.p.z_entry)   and impulse_ok and (wick_ratio >= self.p.wick_min) and (pos_in_bar >= 0.75)

        self.lines.long_mom[0]  = 1 if long_mom else 0
        self.lines.short_mom[0] = -1 if short_mom else 0
        self.lines.long_rev[0]  = 1 if long_rev else 0
        self.lines.short_rev[0] = -1 if short_rev else 0

        self.prev_close = c


class OrderChainZ_LongShort_X_A(BaseStrategy):
    """
    Bi-directional scalp with momentum/reversion modes, DCA, TP/SL/time-stop, cooldown.
    """
    params = (
        # PnL targets
        ('take_profit', 0.8),      # percent per leg; try 0.5-1.0% on alts, 0.2-0.6% on majors
        ('stop_bps', 30),          # 30 bps = 0.30% hard stop (None to disable)
        ('time_stop_s', 3),        # 1-5s recommended for 1s scalps
        ('percent_sizer', 0.02),
        ('dca_deviation', 0.8),    # ~0.5-1.0% for alts; 0.3-0.6% majors
        ('max_legs', 3),           # max number of legs including first
        ('cooldown_s', 2),         # min seconds between flat->new entry
        ('allow_shorts', True),
        ('debug', False),
        ('backtest', None),

        # Indicator params (exposed for tuning)
        ('ema_tau_s', 12),
        ('z_window', 120),
        ('z_entry', 2.8),
        ('wick_min', 0.35),
        ('wick_max', 0.55),
        ('rv_cap', 0.012),
        ('vol_window', 120),
        ('rng_window', 120),
        ('vol_z_min', 0.75),
        ('rng_z_min', 0.75),

        # Mode: "momentum" or "reversion"
        ('entry_mode', "reversion"),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ind = OrderChain1sX(
            ema_tau_s=self.p.ema_tau_s, z_window=self.p.z_window, z_entry=self.p.z_entry,
            wick_min=self.p.wick_min, wick_max=self.p.wick_max, rv_cap=self.p.rv_cap,
            vol_window=self.p.vol_window, rng_window=self.p.rng_window,
            vol_z_min=self.p.vol_z_min, rng_z_min=self.p.rng_z_min, subplot=True
        )

        self.DCA = True
        self.buy_executed = False
        self.side = None              # 'LONG' or 'SHORT'
        self.order_open_times = []
        self.last_flat_time = None

    def _get_datetime(self, dt_value):
        """Convert datetime value to proper datetime object"""
        if isinstance(dt_value, float):
            return bt.num2date(dt_value)
        return dt_value

    def _get_current_time(self):
        """Get current time as datetime object"""
        return self._get_datetime(self.data.datetime[0])

    # helpers
    def _tp_price(self, entry, side_long):
        tp = float(self.params.take_profit) / 100.0
        return entry * (1 + tp) if side_long else entry * (1 - tp)

    def _sl_price(self, entry, side_long):
        if self.p.stop_bps is None:
            return None
        bps = float(self.p.stop_bps) / 10000.0
        return entry * (1 - bps) if side_long else entry * (1 + bps)

    def _cooldown_ok(self):
        if self.last_flat_time is None or self.p.cooldown_s is None:
            return True
        
        current_time = self._get_current_time()
        last_time = self._get_datetime(self.last_flat_time)
        
        elapsed = (current_time - last_time).total_seconds()
        return elapsed >= self.p.cooldown_s

    def _open_leg(self, side_long):
        size = self._determine_size()
        order_type = "BUY" if side_long else "SELL"
        price = self.data.close[0]
        ot = OrderTracker(
            entry_price=price, size=size, take_profit_pct=self.params.take_profit,
            symbol=getattr(self, 'symbol', self.p.asset), order_type=order_type, backtest=self.params.backtest
        )
        ot.order_id = f"order_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        if not hasattr(self, 'active_orders'):
            self.active_orders = []
        self.active_orders.append(ot)
        self.entry_prices.append(price)
        self.sizes.append(size)

        if side_long:
            self.order = self.buy(size=size, exectype=bt.Order.Market)
            self.side = 'LONG'
        else:
            self.order = self.sell(size=size, exectype=bt.Order.Market)
            self.side = 'SHORT'

        # Store current time as datetime object
        self.order_open_times.append(self._get_current_time())
        self.buy_executed = True
        if not hasattr(self, 'first_entry_price') or self.first_entry_price is None:
            self.first_entry_price = price
        self.calc_averages()

        if self.p.debug:
            mode = self.p.entry_mode
            print(f"[ENTER {self.side}/{mode}] size={size} price={price} z={self.ind.lines.oc_z[0]:.2f} "
                  f"volZ={self.ind.lines.vol_z[0]:.2f} rngZ={self.ind.lines.rng_z[0]:.2f} wick={self.ind.lines.wick_ratio[0]:.2f}")

    # entries
    def buy_or_short_condition(self):
        if self.buy_executed:
            self.conditions_checked = True
            return
        if not self._cooldown_ok():
            self.conditions_checked = True
            return

        long_sig = False
        short_sig = False
        if self.p.entry_mode.lower().startswith('mom'):
            long_sig = self.ind.lines.long_mom[0] > 0
            short_sig = self.ind.lines.short_mom[0] < 0
        else:
            long_sig = self.ind.lines.long_rev[0] > 0
            short_sig = self.ind.lines.short_rev[0] < 0

        if long_sig:
            self._open_leg(True)
        elif short_sig and self.p.allow_shorts:
            self._open_leg(False)

        self.conditions_checked = True

    # DCA
    def dca_or_short_condition(self):
        if not (self.buy_executed and self.DCA and hasattr(self, 'active_orders') and self.active_orders):
            self.conditions_checked = True
            return
        if len(self.active_orders) >= int(self.p.max_legs):
            self.conditions_checked = True
            return

        price = self.data.close[0]
        last_entry = self.entry_prices[-1]
        dev = float(self.params.dca_deviation) / 100.0

        if self.side == 'LONG':
            cond_price = price <= last_entry * (1 - dev)
            sig_ok = (self.ind.lines.long_mom[0] > 0) if self.p.entry_mode.startswith('mom') else (self.ind.lines.long_rev[0] > 0)
            if cond_price and sig_ok:
                self._open_leg(True)
        else:  # SHORT
            cond_price = price >= last_entry * (1 + dev)
            sig_ok = (self.ind.lines.short_mom[0] < 0) if self.p.entry_mode.startswith('mom') else (self.ind.lines.short_rev[0] < 0)
            if cond_price and sig_ok:
                self._open_leg(False)

        self.conditions_checked = True

    # exits
    def sell_or_cover_condition(self):
        if not (hasattr(self, 'active_orders') and self.active_orders and self.buy_executed):
            self.conditions_checked = True
            return

        price = self.data.close[0]
        current_time = self._get_current_time()
        side_long = (self.side == 'LONG')
        to_remove = []

        for idx, order in enumerate(self.active_orders):
            entry = order.entry_price
            tp = getattr(order, 'take_profit_price', None)
            if tp is None:
                tp = self._tp_price(entry, side_long)
            sl = self._sl_price(entry, side_long)

            # TP
            if (side_long and price >= tp) or ((not side_long) and price <= tp):
                if side_long:
                    self.order = self.sell(size=order.size, exectype=bt.Order.Market)
                else:
                    self.order = self.buy(size=order.size, exectype=bt.Order.Market)
                order.close_order(price)
                to_remove.append(idx)
                if self.p.debug:
                    print(f"[TP {self.side}] exit {order.size} @ {price} (entry {entry})")
                continue

            # SL
            if sl is not None:
                if (side_long and price <= sl) or ((not side_long) and price >= sl):
                    if side_long:
                        self.order = self.sell(size=order.size, exectype=bt.Order.Market)
                    else:
                        self.order = self.buy(size=order.size, exectype=bt.Order.Market)
                    order.close_order(price)
                    to_remove.append(idx)
                    if self.p.debug:
                        print(f"[SL {self.side}] exit {order.size} @ {price} (entry {entry})")
                    continue

            # Time stop per leg
            if self.p.time_stop_s is not None and idx < len(self.order_open_times):
                open_time = self._get_datetime(self.order_open_times[idx])
                held = (current_time - open_time).total_seconds()
                if held >= self.p.time_stop_s:
                    if side_long:
                        self.order = self.sell(size=order.size, exectype=bt.Order.Market)
                    else:
                        self.order = self.buy(size=order.size, exectype=bt.Order.Market)
                    order.close_order(price)
                    to_remove.append(idx)
                    if self.p.debug:
                        print(f"[TIME {self.side}] exit {order.size} @ {price} after {held:.2f}s")

        # book-keeping
        for idx in sorted(to_remove, reverse=True):
            self.active_orders.pop(idx)
            if idx < len(self.order_open_times):
                self.order_open_times.pop(idx)

        if self.active_orders:
            self.entry_prices = [o.entry_price for o in self.active_orders]
            self.sizes = [o.size for o in self.active_orders]
            self.calc_averages()
        else:
            self.reset_position_state()
            self.buy_executed = False
            self.side = None
            self.order_open_times = []
            self.last_flat_time = current_time

        self.conditions_checked = True

# from .base import BaseStrategy, bt, np, OrderTracker, datetime


if __name__ == '__main__':
    try:
        backtest(
            OrderChainZ_LongShort_X_A,
            coin='1000CAT_1s',
            collateral='USDT',
            start_date="2025-01-01", 
            end_date="2025-01-04", 
            interval="1s",
            init_cash=1000,
            plot=True, 
            quantstats=False
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()