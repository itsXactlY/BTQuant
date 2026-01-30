from .base import BaseStrategy, bt, OrderTracker
<<<<<<< HEAD
import math

# --------------------- Strategy (MTF + Risk + optional Short) ---------------------
# class Enhanced_MACD_ADX3(BaseStrategy):
#     params = (
#         ('risk_per_trade_pct', 0.0025),
#         ('max_leverage', 2.0),
#         ('min_qty', 0.0),
#         ('qty_step', 1.0),
#         ('price_tick', None),
#         ('round_prices', True),
#         ('can_short', False),
#         ('short_regime_mode', 'neutral'),
#         ('rsi_oversold', 25),
#         ('use_htf', True),
#         ('tf5m_breakout_period', 55),
#         ('tf15m_adx_period', 14),
#         ('tf15m_ema_fast', 50),
#         ('tf15m_ema_slow', 200),
#         ('tf60m_ema_fast', 50),
#         ('tf60m_ema_slow', 200),
#         ('ema_fast', 20),
#         ('ema_slow', 50),
#         ('ema_trend', 200),
#         ('atr_period', 14),
#         ('rsi_overheat', 75),
#         ('adxth', 20),
#         ('confirm_bars', 2),
#         ('max_stretch_atr_mult', 1.0),
#         ('atr_stop_mult', 2.5),
#         ('use_trailing_stop', True),
#         ('trail_mode', 'chandelier'),
#         ('trail_atr_mult', 4.0),
#         ('ema_band_mult', 2.0),
#         ('donchian_trail_period', 55),
#         ('close_based_stop', True),
#         ('move_to_breakeven_R', 1.0),
#         ('trail_update_every', 2),
#         ('max_bars_in_trade', 6*60),
#         ('reentry_cooldown_bars', 5),
#         ('use_pyramiding', True),
#         ('max_adds', 2),
#         ('add_cooldown', 20),
#         ('add_atr_mult', 1.0),
#         ('add_min_R', 1.0),
#         ('take_profit', 4.0),
#         ('use_volume_filter', False),
#         ('volume_filter_mult', 1.2),
#         ('use_regime_long', True),
#         ('use_trend_long', True),
#         ('use_regime_short', True),
#         ('use_trend_short', True),
#         ('regime_mode_long', 'ema'),      # 'ema' | 'price_vs_slow' | 'off'
#         ('regime_mode_short', 'neutral'), # 'ema' | 'neutral' | 'off'
#         ('backtest', True),
#         ('debug', False),
#     )

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.d1  = self.datas[0]
#         self.d5  = self.datas[1] if len(self.datas) > 1 else self.d1
#         self.d15 = self.datas[2] if len(self.datas) > 2 else self.d5
#         self.d60 = self.datas[3] if len(self.datas) > 3 else self.d15

#         self.atr1 = bt.ind.ATR(self.d1, period=self.p.atr_period)
#         self.ema1_fast = bt.ind.EMA(self.d1.close, period=self.p.ema_fast)
#         self.ema1_slow = bt.ind.EMA(self.d1.close, period=self.p.ema_slow)
#         self.ema1_trend = bt.ind.EMA(self.d1.close, period=self.p.ema_trend)
#         self.rsi1 = bt.ind.RSI(self.d1, period=14)
#         self.vsma1 = bt.ind.SMA(self.d1.volume, period=20) if self.p.use_volume_filter else None

#         self.atr5 = bt.ind.ATR(self.d5, period=self.p.atr_period)
#         self.dc_high5 = bt.ind.Highest(self.d5.high, period=self.p.tf5m_breakout_period)
#         self.dc_low5  = bt.ind.Lowest(self.d5.low,  period=self.p.tf5m_breakout_period)
#         self.dc_exit5_low  = bt.ind.Lowest(self.d5.low,  period=self.p.donchian_trail_period)
#         self.dc_exit5_high = bt.ind.Highest(self.d5.high, period=self.p.donchian_trail_period)

#         self.adx15 = bt.ind.ADX(self.d15, period=self.p.tf15m_adx_period)
#         self.plusDI15 = bt.ind.PlusDI(self.d15, period=self.p.tf15m_adx_period)
#         self.minusDI15 = bt.ind.MinusDI(self.d15, period=self.p.tf15m_adx_period)
#         self.ema15_fast = bt.ind.EMA(self.d15.close, period=self.p.tf15m_ema_fast)
#         self.ema15_slow = bt.ind.EMA(self.d15.close, period=self.p.tf15m_ema_slow)

#         self.ema60_fast = bt.ind.EMA(self.d60.close, period=self.p.tf60m_ema_fast)
#         self.ema60_slow = bt.ind.EMA(self.d60.close, period=self.p.tf60m_ema_slow)

#         self.entry_bar = None
#         self.trail_stop = None
#         self.init_stop = None
#         self.initial_risk = None
#         self.run_high = None
#         self.run_low = None
#         self.last_trail_update = -10**9
#         self.n_adds = 0
#         self.last_add_bar = -10**9
#         self.last_exit_bar = -10**9

#         self.active_orders, self.entry_prices, self.sizes = [], [], []
#         self.block_counts = dict(regime=0, trend=0, breakout=0, s_regime=0, s_trend=0, breakdown=0)

#     def start(self):
#         if self.p.debug:
#             names = ['Ticks', 'MicroSec', 'Seconds', 'Minutes', 'Days', 'Weeks', 'Months', 'Years']
#             for i, d in enumerate(self.datas):
#                 name = getattr(d, '_name', f'data{i}')
#                 tf = getattr(d.p, 'timeframe', getattr(d, '_timeframe', None))
#                 comp_p = getattr(d.p, 'compression', None)
#                 comp_attr = getattr(d, '_compression', None)
#                 tfstr = names[int(tf)] if isinstance(tf, int) and 0 <= tf < len(names) else str(tf)
#                 print(f"Data{i} {name} -> TF={tfstr} p.comp={comp_p} attr._comp={comp_attr}")

#     def stop(self):
#         if self.p.debug:
#             print("Blocks:", self.block_counts)

#     def _equity(self): return self.broker.getvalue()
#     def _round_qty(self, size):
#         # Use realistic crypto steps; default to 0.001 if not provided
#         step = float(self.p.qty_step) if self.p.qty_step else 0.001
#         if step <= 0:
#             step = 0.001

#         q = math.floor(size / step) * step
#         # If positive but floored to zero, place one step (so small equity still trades)
#         if q <= 0 and size > 0:
#             q = step

#         # Enforce exchange minimum
#         if self.p.min_qty and q < self.p.min_qty:
#             return 0.0
#         return q

#     def _risk_based_size(self, entry, stop):
#         eq = self._equity()
#         risk = eq * self.p.risk_per_trade_pct
#         dist = max(1e-8, abs(entry - stop))
#         # Cap by leverage
#         size_float = min(risk / dist, (eq * self.p.max_leverage) / max(entry, 1e-8))
#         return self._round_qty(size_float)

#     def _round_price(self, price):
#         if not (self.p.round_prices and self.p.price_tick): return float(price)
#         tick = float(self.p.price_tick);  return round(price / tick) * tick
#     def _avg_entry(self):
#         if self.entry_prices and self.sizes:
#             tot = sum(self.sizes);  return sum(p*s for p,s in zip(self.entry_prices, self.sizes))/tot if tot else None
#         return None
#     def _R(self):
#         ae = self._avg_entry()
#         if not (ae and self.initial_risk and self.initial_risk > 0): return 0.0
#         if self.position.size > 0: return (self.d1.close[0] - ae) / self.initial_risk
#         if self.position.size < 0: return (ae - self.d1.close[0]) / self.initial_risk
#         return 0.0

#     def _enough_history(self):
#         if len(self.d5)  <= max(self.p.tf5m_breakout_period, self.p.donchian_trail_period) + 2: return False
#         if len(self.d15) <= max(self.p.tf15m_adx_period, self.p.tf15m_ema_slow) + 2: return False
#         if len(self.d60) <= self.p.tf60m_ema_slow + 2: return False
#         if len(self.d1)  <= max(self.p.ema_trend, self.p.atr_period) + 2: return False
#         return True

#     def regime_ok_long(self):
#         if not self.p.use_regime_long or self.p.regime_mode_long == 'off':
#             return True
#         if self.p.regime_mode_long == 'price_vs_slow':
#             return self.d60.close[0] > self.ema60_slow[0]
#         return self.ema60_fast[0] > self.ema60_slow[0]

#     def regime_ok_short(self):
#         if not self.p.can_short or not self.p.use_regime_short or self.p.regime_mode_short == 'off':
#             return True
#         if self.p.regime_mode_short == 'neutral':
#             return True
#         return self.ema60_fast[0] < self.ema60_slow[0]

#     def trend_ok_long(self):
#         if not self.p.use_trend_long:
#             return True
#         return (self.adx15[0] >= self.p.adxth and self.plusDI15[0] > self.minusDI15[0]
#                 and self.ema15_fast[0] > self.ema15_slow[0] and self.ema1_fast[0] > self.ema1_slow[0])

#     def trend_ok_short(self):
#         if not self.p.use_trend_short:
#             return True
#         return (self.adx15[0] >= self.p.adxth and self.minusDI15[0] > self.plusDI15[0]
#                 and self.ema15_fast[0] < self.ema15_slow[0] and self.ema1_fast[0] < self.ema1_slow[0])

#     def breakout_up(self):
#         if len(self.d5) < 2 or len(self.d1) < self.p.confirm_bars + 2: return False
#         level = float(self.dc_high5[-1])
#         confirmed = all(self.d1.close[-i] > level for i in range(self.p.confirm_bars, 0, -1))
#         stretched = (self.d1.close[0] - level) > self.p.max_stretch_atr_mult * float(self.atr5[0])
#         if (not confirmed) or stretched or (self.rsi1[0] >= self.p.rsi_overheat): return False
#         if self.p.use_volume_filter and self.vsma1 is not None:
#             if self.d1.volume[0] <= self.p.volume_filter_mult * max(self.vsma1[0], 1e-8): return False
#         return True

#     def breakdown_down(self):
#         if len(self.d5) < 2 or len(self.d1) < self.p.confirm_bars + 2: return False
#         level = float(self.dc_low5[-1])
#         confirmed = all(self.d1.close[-i] < level for i in range(self.p.confirm_bars, 0, -1))
#         stretched = (level - self.d1.close[0]) > self.p.max_stretch_atr_mult * float(self.atr5[0])
#         if (not confirmed) or stretched or (self.rsi1[0] <= self.p.rsi_oversold): return False
#         if self.p.use_volume_filter and self.vsma1 is not None:
#             if self.d1.volume[0] <= self.p.volume_filter_mult * max(self.vsma1[0], 1e-8): return False
#         return True

#     def _update_trailing_stop(self):
#         if not self.position:
#             self.trail_stop=None; return
#         if self.position.size > 0:
#             self.run_high = max(self.run_high or self.d1.high[0], self.d1.high[0])
#         else:
#             self.run_low  = min(self.run_low  or self.d1.low[0],  self.d1.low[0])
#         if (len(self) - self.last_trail_update) < self.p.trail_update_every:
#             return

#         candidate = None
#         if self.p.trail_mode == "chandelier":
#             candidate = float((self.run_high - self.p.trail_atr_mult * self.atr5[0]) if self.position.size>0
#                               else (self.run_low + self.p.trail_atr_mult * self.atr5[0]))
#         elif self.p.trail_mode == "ema_band":
#             candidate = float((self.ema1_fast[0] - self.p.ema_band_mult * self.atr1[0]) if self.position.size>0
#                               else (self.ema1_fast[0] + self.p.ema_band_mult * self.atr1[0]))
#         elif self.p.trail_mode == "donchian":
#             candidate = float(self.dc_exit5_low[0] if self.position.size>0 else self.dc_exit5_high[0])

#         if candidate is not None:
#             candidate = max(candidate, self.init_stop or -1e18) if self.position.size>0 else min(candidate, self.init_stop or 1e18)
#             ae = self._avg_entry()
#             if ae and self._R() >= self.p.move_to_breakeven_R:
#                 candidate = max(candidate, ae) if self.position.size>0 else min(candidate, ae)
#             self.trail_stop = candidate if self.trail_stop is None else (
#                 max(self.trail_stop, candidate) if self.position.size>0 else min(self.trail_stop, candidate)
#             )
#             self.last_trail_update = len(self)

#     def _stop_hit(self):
#         if self.trail_stop is None: return False
#         if self.p.close_based_stop:
#             return (self.d1.close[0] <= self.trail_stop) if self.position.size>0 else (self.d1.close[0] >= self.trail_stop)
#         else:
#             return (self.d1.low[0]   <= self.trail_stop) if self.position.size>0 else (self.d1.high[0]  >= self.trail_stop)

#     def _enter_long(self):
#         entry = float(self._round_price(self.d1.close[0]))
#         init_stop = self._round_price(entry - self.p.atr_stop_mult * float(self.atr5[0]))
#         size = self._risk_based_size(entry, init_stop)
#         if size <= 0: return
#         tp = self._round_price(entry * (1 + self.p.take_profit/100.0))
#         self.active_orders.append(type('Leg', (), dict(entry_price=entry, size=size, take_profit_price=tp, dir=+1)))
#         self.entry_prices.append(entry); self.sizes.append(size)
#         self.buy(size=size, exectype=bt.Order.Market)
#         self.init_stop = init_stop; self.trail_stop = init_stop
#         self.initial_risk = max(1e-8, entry - init_stop)
#         self.run_high = self.d1.high[0]; self.run_low=None
#         self.entry_bar = len(self); self.n_adds=0; self.last_add_bar = len(self)

#     def _enter_short(self):
#         entry = float(self._round_price(self.d1.close[0]))
#         init_stop = self._round_price(entry + self.p.atr_stop_mult * float(self.atr5[0]))
#         size = self._risk_based_size(entry, init_stop)
#         if size <= 0: return
#         tp = self._round_price(entry * (1 - self.p.take_profit/100.0))
#         self.active_orders.append(type('Leg', (), dict(entry_price=entry, size=size, take_profit_price=tp, dir=-1)))
#         self.entry_prices.append(entry); self.sizes.append(size)
#         self.sell(size=size, exectype=bt.Order.Market)
#         self.init_stop = init_stop; self.trail_stop = init_stop
#         self.initial_risk = max(1e-8, init_stop - entry)
#         self.run_low = self.d1.low[0]; self.run_high=None
#         self.entry_bar = len(self); self.n_adds=0; self.last_add_bar = len(self)

#     def _can_pyramid(self):
#         if not (self.p.use_pyramiding and self.position): return False
#         if self.n_adds >= self.p.max_adds: return False
#         if (len(self) - self.last_add_bar) < self.p.add_cooldown: return False
#         if self._R() < self.p.add_min_R: return False
#         if self.position.size > 0:
#             return self.d1.close[0] >= ((self.run_high or self.d1.high[0]) + self.p.add_atr_mult * float(self.atr5[0]))
#         else:
#             return self.d1.close[0] <= ((self.run_low  or self.d1.low[0])  - self.p.add_atr_mult * float(self.atr5[0]))

#     def _do_pyramid(self):
#         entry = float(self._round_price(self.d1.close[0]))
#         stop = float(self.trail_stop or (entry - self.p.atr_stop_mult * float(self.atr5[0])) if self.position.size>0
#                      else self.trail_stop or (entry + self.p.atr_stop_mult * float(self.atr5[0])))
#         size = self._round_qty(self._risk_based_size(entry, stop) / 2.0)
#         if size <= 0: return
#         if self.position.size > 0:
#             tp = self._round_price(entry * (1 + self.p.take_profit/100.0))
#             self.active_orders.append(type('Leg', (), dict(entry_price=entry, size=size, take_profit_price=tp, dir=+1)))
#             self.entry_prices.append(entry); self.sizes.append(size)
#             self.buy(size=size, exectype=bt.Order.Market)
#         else:
#             tp = self._round_price(entry * (1 - self.p.take_profit/100.0))
#             self.active_orders.append(type('Leg', (), dict(entry_price=entry, size=size, take_profit_price=tp, dir=-1)))
#             self.entry_prices.append(entry); self.sizes.append(size)
#             self.sell(size=size, exectype=bt.Order.Market)
#         self.n_adds += 1; self.last_add_bar = len(self)

#     def _exit_all(self, reason):
#         qty = sum(l.size for l in self.active_orders) if self.active_orders else abs(self.position.size)
#         if self.position.size > 0: self.sell(size=qty, exectype=bt.Order.Market)
#         elif self.position.size < 0: self.buy(size=qty, exectype=bt.Order.Market)
#         self.active_orders.clear(); self.entry_prices.clear(); self.sizes.clear()
#         self.trail_stop=None; self.init_stop=None; self.initial_risk=None
#         self.run_high=None; self.run_low=None; self.entry_bar=None
#         self.n_adds=0; self.last_trail_update=-10**9; self.last_add_bar=-10**9
#         self.last_exit_bar = len(self)

#     def prenext(self):
#         if not self._enough_history():
#             return

#     def next(self):
#         if not self._enough_history():
#             return
#         if (len(self) - self.last_exit_bar) < self.p.reentry_cooldown_bars:
#             pass

#         if self.position:
#             if self.p.max_bars_in_trade and self.entry_bar and (len(self) - self.entry_bar) >= self.p.max_bars_in_trade:
#                 self._exit_all(f"TimeStop {self.p.max_bars_in_trade}"); return
#             self._update_trailing_stop()
#             if self._stop_hit(): self._exit_all("TrailStop"); return

#             price = self.d1.close[0]
#             to_remove = []
#             for idx, leg in enumerate(self.active_orders):
#                 if leg.dir > 0 and price >= leg.take_profit_price:
#                     self.sell(size=leg.size, exectype=bt.Order.Market); to_remove.append(idx)
#                 elif leg.dir < 0 and price <= leg.take_profit_price:
#                     self.buy(size=leg.size, exectype=bt.Order.Market);  to_remove.append(idx)
#             for idx in reversed(to_remove):
#                 self.active_orders.pop(idx); self.entry_prices.pop(idx); self.sizes.pop(idx)
#             if not self.active_orders: self._exit_all("All TPs"); return

#             if self._can_pyramid(): self._do_pyramid()
#             return

#         # Flat → entries
#         if self.p.use_htf:
#             if not self.regime_ok_long(): self.block_counts['regime'] += 1
#             elif not self.trend_ok_long(): self.block_counts['trend'] += 1
#             elif not self.breakout_up(): self.block_counts['breakout'] += 1
#             else:
#                 self._enter_long(); return

#             if self.p.can_short:
#                 if not self.regime_ok_short(): self.block_counts['s_regime'] += 1
#                 elif not self.trend_ok_short(): self.block_counts['s_trend'] += 1
#                 elif not self.breakdown_down(): self.block_counts['breakdown'] += 1
#                 else:
#                     self._enter_short(); return
#         else:
#             if self.ema1_fast[0] > self.ema1_slow[0] and self.breakout_up(): self._enter_long(); return
#             if self.p.can_short and self.ema1_fast[0] < self.ema1_slow[0] and self.breakdown_down(): self._enter_short(); return



class Enhanced_MACD_ADX4(BaseStrategy):
    params = (
        # Fallback-Budget (falls Risikosize zu klein → min-fill via qty_step)
        ('percent_sizer', 0.05),

        # Risk / Sizing
        ('risk_per_trade_pct', 0.0025),
        ('max_leverage', 10.0),
        ('min_qty', 0.0),
        ('qty_step', 1.0),          # z. B. BTC/ETH: 0.001
        ('price_tick', None),
        ('round_prices', True),

        # Shorts
        ('can_short', False),
        ('short_regime_mode', 'neutral'),  # 'neutral' (=immer erlaubt) | 'ema' (=nur wenn 60m-EMA down)
        ('rsi_oversold', 25),

        # Timeframes: data0=1m, data1=5m, data2=15m, data3=60m
        ('use_htf', True),
        ('tf5m_breakout_period', 55),
        ('tf15m_adx_period', 14),
        ('tf15m_ema_fast', 50),
        ('tf15m_ema_slow', 200),
        ('tf60m_ema_fast', 50),
        ('tf60m_ema_slow', 200),

        # 1m Baseline für Overheat/Infos
        ('ema_fast', 20),
        ('ema_slow', 50),
        ('ema_trend', 200),
        ('atr_period', 14),
        ('rsi_overheat', 75),

        # Entry-/Stretch-/Stops
        ('adxth', 20),
        ('confirm_bars', 2),
        ('max_stretch_atr_mult', 1.0),
        ('atr_stop_mult', 2.5),
        ('take_profit', 4.0),      # % pro Leg

        # Trailing optional
        ('use_trailing_stop', False),
        ('trail_mode', 'chandelier'),  # chandelier | ema_band | donchian
        ('trail_atr_mult', 4.0),
        ('ema_band_mult', 2.0),
        ('donchian_trail_period', 55),
        ('close_based_stop', True),
        ('move_to_breakeven_R', 1.0),
        ('trail_update_every', 2),
        ('max_bars_in_trade', 6*60),
        ('reentry_cooldown_bars', 5),

        # Winner-Pyramiding optional
        ('use_pyramiding', False),
        ('max_adds', 0),
        ('add_cooldown', 20),
        ('add_atr_mult', 1.0),
        ('add_min_R', 1.0),

        # Volume-Filter (safe)
        ('use_volume_filter', False),
        ('volume_filter_mult', 1.2),

        # Regime/Trend Filter
        ('use_regime_long', True),
        ('use_trend_long', True),
        ('use_regime_short', True),
        ('use_trend_short', True),
        ('regime_mode_long', 'ema'),      # 'ema' | 'price_vs_slow' | 'off'
        ('regime_mode_short', 'neutral'), # 'ema' | 'neutral' | 'off'

        ('backtest', True),
        ('debug', True),
=======
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
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
<<<<<<< HEAD
        self.data = self.datas[0]
        self.d1  = self.datas[0]
        self.d5  = self.datas[1] if len(self.datas) > 1 else self.d1
        self.d15 = self.datas[2] if len(self.datas) > 2 else self.d5
        self.d60 = self.datas[3] if len(self.datas) > 3 else self.d15

        # 1m
        self.atr1 = bt.ind.ATR(self.d1, period=self.p.atr_period)
        self.ema1_fast = bt.ind.EMA(self.d1.close, period=self.p.ema_fast)
        self.ema1_slow = bt.ind.EMA(self.d1.close, period=self.p.ema_slow)
        self.ema1_trend = bt.ind.EMA(self.d1.close, period=self.p.ema_trend)
        self.rsi1 = bt.ind.RSI(self.d1, period=14)
        self.vsma1 = bt.ind.SMA(self.d1.volume, period=20) if self.p.use_volume_filter else None

        # 5m
        self.atr5 = bt.ind.ATR(self.d5, period=self.p.atr_period)
        self.dc_high5 = bt.ind.Highest(self.d5.high, period=self.p.tf5m_breakout_period)
        self.dc_low5  = bt.ind.Lowest(self.d5.low,  period=self.p.tf5m_breakout_period)
        self.dc_exit5_low  = bt.ind.Lowest(self.d5.low,  period=self.p.donchian_trail_period)
        self.dc_exit5_high = bt.ind.Highest(self.d5.high, period=self.p.donchian_trail_period)

        # 15m/60m
        self.adx15 = bt.ind.ADX(self.d15, period=self.p.tf15m_adx_period)
        self.plusDI15 = bt.ind.PlusDI(self.d15, period=self.p.tf15m_adx_period)
        self.minusDI15 = bt.ind.MinusDI(self.d15, period=self.p.tf15m_adx_period)
        self.ema15_fast = bt.ind.EMA(self.d15.close, period=self.p.tf15m_ema_fast)
        self.ema15_slow = bt.ind.EMA(self.d15.close, period=self.p.tf15m_ema_slow)
        self.ema60_fast = bt.ind.EMA(self.d60.close, period=self.p.tf60m_ema_fast)
        self.ema60_slow = bt.ind.EMA(self.d60.close, period=self.p.tf60m_ema_slow)

        # State
        self.entry_bar = None
        self.trail_stop = None
        self.init_stop = None
        self.initial_risk = None
        self.run_high = None
        self.run_low = None
        self.last_trail_update = -10**9
        self.n_adds = 0
        self.last_add_bar = -10**9
        self.last_exit_bar = -10**9

=======

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
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented
        if not hasattr(self, 'active_orders'):
            self.active_orders = []
        if not hasattr(self, 'entry_prices'):
            self.entry_prices = []
        if not hasattr(self, 'sizes'):
            self.sizes = []
<<<<<<< HEAD

        self.block_counts = dict(regime=0, trend=0, breakout=0, s_regime=0, s_trend=0, breakdown=0)

    def start(self):
        if self.p.debug:
            names = ['Ticks', 'MicroSec', 'Seconds', 'Minutes', 'Days', 'Weeks', 'Months', 'Years']
            for i, d in enumerate(self.datas):
                name = getattr(d, '_name', f'data{i}')
                tf = getattr(d.p, 'timeframe', getattr(d, '_timeframe', None))
                comp_p = getattr(d.p, 'compression', None)
                comp_attr = getattr(d, '_compression', None)
                tfstr = names[int(tf)] if isinstance(tf, int) and 0 <= tf < len(names) else str(tf)
                print(f"Data{i} {name} -> TF={tfstr} p.comp={comp_p} attr._comp={comp_attr}")

    def stop(self):
        if self.p.debug:
            print("Blocks:", self.block_counts)

    # --------- Rounding / Sizing ---------
    def _round_qty(self, size):
        step = float(self.p.qty_step) if self.p.qty_step else 0.001
        if step <= 0:
            step = 0.001
        q = math.floor(size / step) * step
        if q <= 0 and size > 0:
            q = step
        if self.p.min_qty and q < self.p.min_qty:
            return 0.0
        return q

    def _round_price(self, price):
        if not (self.p.round_prices and self.p.price_tick):
            return float(price)
        tick = float(self.p.price_tick)
        if tick <= 0:
            return float(price)
        return round(price / tick) * tick

    def _risk_based_size(self, entry, stop):
        eq = self.broker.getvalue()
        risk = eq * self.p.risk_per_trade_pct
        dist = max(1e-8, abs(entry - stop))
        size_float = min(risk / dist, (eq * self.p.max_leverage) / max(entry, 1e-8))
        return self._round_qty(size_float)

    def _volume_filter_ok(self):
        if not self.p.use_volume_filter or self.vsma1 is None:
            return True
        try:
            base = float(self.vsma1[0])
        except IndexError:
            return False
        base = max(base, 1e-8)
        return self.d1.volume[0] > self.p.volume_filter_mult * base

    # --------- MTF Guards ---------
    def _enough_history(self):
        if len(self.d5)  <= max(self.p.tf5m_breakout_period, self.p.donchian_trail_period) + 2: return False
        if len(self.d15) <= max(self.p.tf15m_adx_period, self.p.tf15m_ema_slow) + 2: return False
        if len(self.d60) <= self.p.tf60m_ema_slow + 2: return False
        if len(self.d1)  <= max(self.p.ema_trend, self.p.atr_period) + 2: return False
        if self.p.use_volume_filter and self.vsma1 is not None:
            try: _ = self.vsma1[0]
            except IndexError: return False
        return True

    def regime_ok_long(self):
        if not self.p.use_regime_long or self.p.regime_mode_long == 'off':
            return True
        if self.p.regime_mode_long == 'price_vs_slow':
            return self.d60.close[0] > self.ema60_slow[0]
        return self.ema60_fast[0] > self.ema60_slow[0]

    def regime_ok_short(self):
        if not self.p.can_short or not self.p.use_regime_short or self.p.regime_mode_short == 'off':
            return True
        if self.p.regime_mode_short == 'neutral':
            return True
        # 'ema' short-mode: 60m down
        return self.ema60_fast[0] < self.ema60_slow[0]

    def trend_ok_long(self):
        if not self.p.use_trend_long:
            return True
        return (self.adx15[0] >= self.p.adxth and self.plusDI15[0] > self.minusDI15[0]
                and self.ema15_fast[0] > self.ema15_slow[0] and self.ema1_fast[0] > self.ema1_slow[0])

    def trend_ok_short(self):
        if not self.p.use_trend_short:
            return True
        return (self.adx15[0] >= self.p.adxth and self.minusDI15[0] > self.plusDI15[0]
                and self.ema15_fast[0] < self.ema15_slow[0] and self.ema1_fast[0] < self.ema1_slow[0])

    def breakout_up(self):
        if len(self.d5) < 2 or len(self.d1) < self.p.confirm_bars + 2: return False
        level = float(self.dc_high5[-1])
        confirmed = all(self.d1.close[-i] > level for i in range(self.p.confirm_bars, 0, -1))
        stretched = (self.d1.close[0] - level) > self.p.max_stretch_atr_mult * float(self.atr5[0])
        if (not confirmed) or stretched or (self.rsi1[0] >= self.p.rsi_overheat): return False
        if not self._volume_filter_ok(): return False
        return True

    def breakdown_down(self):
        if len(self.d5) < 2 or len(self.d1) < self.p.confirm_bars + 2: return False
        level = float(self.dc_low5[-1])
        confirmed = all(self.d1.close[-i] < level for i in range(self.p.confirm_bars, 0, -1))
        stretched = (level - self.d1.close[0]) > self.p.max_stretch_atr_mult * float(self.atr5[0])
        if (not confirmed) or stretched or (self.rsi1[0] <= self.p.rsi_oversold): return False
        if not self._volume_filter_ok(): return False
        return True

    # --------- Trailing ---------
    def _update_trailing_stop(self):
        if not self.position:
            self.trail_stop=None; return

        if self.position.size > 0:
            self.run_high = max(self.run_high or self.d1.high[0], self.d1.high[0])
        else:
            self.run_low  = min(self.run_low  or self.d1.low[0],  self.d1.low[0])

        if (len(self) - self.last_trail_update) < self.p.trail_update_every:
            return

        candidate = None
        if self.p.trail_mode == "chandelier":
            if self.position.size > 0:
                candidate = float(self.run_high - self.p.trail_atr_mult * self.atr5[0])
            else:
                candidate = float(self.run_low + self.p.trail_atr_mult * self.atr5[0])
        elif self.p.trail_mode == "ema_band":
            if self.position.size > 0:
                candidate = float(self.ema1_fast[0] - self.p.ema_band_mult * self.atr1[0])
            else:
                candidate = float(self.ema1_fast[0] + self.p.ema_band_mult * self.atr1[0])
        elif self.p.trail_mode == "donchian":
            candidate = float(self.dc_exit5_low[0] if self.position.size > 0 else self.dc_exit5_high[0])

        if candidate is not None:
            candidate = max(candidate, self.init_stop or -1e18) if self.position.size>0 else min(candidate, self.init_stop or 1e18)
            ae = self._avg_entry()
            if ae and self._R() >= self.p.move_to_breakeven_R:
                candidate = max(candidate, ae) if self.position.size>0 else min(candidate, ae)
            self.trail_stop = candidate if self.trail_stop is None else (
                max(self.trail_stop, candidate) if self.position.size>0 else min(self.trail_stop, candidate)
            )
            self.last_trail_update = len(self)

    def _stop_hit(self):
        if self.trail_stop is None: return False
        if self.p.close_based_stop:
            return (self.d1.close[0] <= self.trail_stop) if self.position.size>0 else (self.d1.close[0] >= self.trail_stop)
        else:
            return (self.d1.low[0]   <= self.trail_stop) if self.position.size>0 else (self.d1.high[0]  >= self.trail_stop)

    # --------- Entries ---------
    def _enter_long(self):
        entry = float(self._round_price(self.d1.close[0]))
        init_stop = self._round_price(entry - self.p.atr_stop_mult * float(self.atr5[0]))
        size = self._risk_based_size(entry, init_stop)
        if size <= 0:
            # Fallback auf percent_sizer
            cash = float(self.broker.getcash())
            budget = cash * float(self.p.percent_sizer or 0.0)
            size = self._round_qty(budget / max(entry, 1e-8))
            if size <= 0:
                return
        tp = self._round_price(entry * (1 + self.p.take_profit/100.0))

        self.entry_prices.append(entry)
        self.sizes.append(size)
        self.active_orders.append(dict(entry=entry, size=size, tp=tp, dir=+1))

        self.buy(size=size, exectype=bt.Order.Market)
        self.init_stop = init_stop
        self.trail_stop = init_stop
        self.initial_risk = max(1e-8, entry - init_stop)
        self.run_high = self.d1.high[0]
        self.run_low  = None
        self.entry_bar = len(self)
        self.last_add_bar = len(self)

        if self.p.debug:
            print(f"ENTER LONG {size} @ {entry} | SL={init_stop} | TP={tp}")

    def _enter_short(self):
        if not self.p.can_short:
            return
        entry = float(self._round_price(self.d1.close[0]))
        init_stop = self._round_price(entry + self.p.atr_stop_mult * float(self.atr5[0]))
        size = self._risk_based_size(entry, init_stop)
        if size <= 0:
            cash = float(self.broker.getcash())
            budget = cash * float(self.p.percent_sizer or 0.0)
            size = self._round_qty(budget / max(entry, 1e-8))
            if size <= 0:
                return
        tp = self._round_price(entry * (1 - self.p.take_profit/100.0))

        self.entry_prices.append(entry)
        self.sizes.append(size)
        self.active_orders.append(dict(entry=entry, size=size, tp=tp, dir=-1))

        self.sell(size=size, exectype=bt.Order.Market)
        self.init_stop = init_stop
        self.trail_stop = init_stop
        self.initial_risk = max(1e-8, init_stop - entry)
        self.run_low  = self.d1.low[0]
        self.run_high = None
        self.entry_bar = len(self)
        self.last_add_bar = len(self)

        if self.p.debug:
            print(f"ENTER SHORT {size} @ {entry} | SL={init_stop} | TP={tp}")

    def _can_pyramid(self):
        if not (self.p.use_pyramiding and self.position):
            return False
        if self.n_adds >= self.p.max_adds:
            return False
        if (len(self) - self.last_add_bar) < self.p.add_cooldown:
            return False
        if self._R() < self.p.add_min_R:
            return False
        if self.position.size > 0:
            return self.d1.close[0] >= ((self.run_high or self.d1.high[0]) + self.p.add_atr_mult * float(self.atr5[0]))
        else:
            return self.d1.close[0] <= ((self.run_low  or self.d1.low[0])  - self.p.add_atr_mult * float(self.atr5[0]))

    def _do_pyramid(self):
        entry = float(self._round_price(self.d1.close[0]))
        if self.position.size > 0:
            stop = float(self.trail_stop or entry - self.p.atr_stop_mult * float(self.atr5[0]))
        else:
            stop = float(self.trail_stop or entry + self.p.atr_stop_mult * float(self.atr5[0]))
        size = self._round_qty(self._risk_based_size(entry, stop) / 2.0)
        if size <= 0:
            return

        if self.position.size > 0:
            tp = self._round_price(entry * (1 + self.p.take_profit/100.0))
            self.active_orders.append(dict(entry=entry, size=size, tp=tp, dir=+1))
            self.entry_prices.append(entry)
            self.sizes.append(size)
            self.buy(size=size, exectype=bt.Order.Market)
        else:
            tp = self._round_price(entry * (1 - self.p.take_profit/100.0))
            self.active_orders.append(dict(entry=entry, size=size, tp=tp, dir=-1))
            self.entry_prices.append(entry)
            self.sizes.append(size)
            self.sell(size=size, exectype=bt.Order.Market)
        self.n_adds += 1
        self.last_add_bar = len(self)
        if self.p.debug:
            print(f"PYRAMID add #{self.n_adds} {size} @ {entry}")

    # ---------- Helper: OrderTracker für Long/Short bauen ----------
    def _make_tracker(self, entry: float, size: float, direction: int):
        # direction: +1 (long), -1 (short)
        order_type = "BUY" if direction > 0 else "SELL"
        tracker = OrderTracker(
            entry_price=entry,
            size=size,
            take_profit_pct=self.params.take_profit,
            symbol=getattr(self, 'symbol', self.p.asset),
            order_type=order_type,
            backtest=self.params.backtest
        )
        # OrderTracker berechnet standardmäßig Long-TP. Für Shorts TP anpassen:
        if order_type != "BUY":
            tp = entry * (1 - self.p.take_profit / 100.0)
            tracker.take_profit_price = self._round_price(tp)
        else:
            # optional runden
            tracker.take_profit_price = self._round_price(tracker.take_profit_price)
        return tracker

    # --------- Flat → Entries (BaseStrategy ruft diese Methoden in next() auf) ---------
    def buy_or_short_condition(self):
        if not self._enough_history():
            self.conditions_checked = True
            return False

        if not self.position:
            if self.p.use_htf:
                # Long
                if self.regime_ok_long() and self.trend_ok_long() and self.breakout_up():
                    entry = float(self.d1.close[0])
                    stop  = float(entry - self.p.atr_stop_mult * float(self.atr5[0]))
                    size  = self._risk_based_size(entry, stop)
                    if size <= 0:
                        cash = float(self.broker.getcash())
                        budget = cash * float(self.p.percent_sizer or 0.0)
                        size = self._round_qty(budget / max(entry, 1e-8))
                        if size <= 0:
                            self.conditions_checked = True
                            return False

                    tracker = self._make_tracker(entry, size, +1)
                    self.active_orders.append(tracker)
                    self.entry_prices.append(entry)
                    self.sizes.append(size)

                    self.buy(size=size, exectype=bt.Order.Market)

                    self.init_stop = self._round_price(stop)
                    self.trail_stop = self.init_stop
                    self.initial_risk = max(1e-8, entry - self.init_stop)
                    self.run_high = self.d1.high[0]; self.run_low=None
                    self.entry_bar = len(self); self.last_add_bar = len(self)

                    if self.p.debug:
                        print(f"ENTER LONG {size} @ {entry} | SL={self.init_stop} | TP={tracker.take_profit_price}")

                    self.first_entry_price = entry if not self.buy_executed else self.first_entry_price
                    self.buy_executed = True
                    self.DCA = True
                    self.conditions_checked = True
                    return False

                # Short
                if self.p.can_short and self.regime_ok_short() and self.trend_ok_short() and self.breakdown_down():
                    entry = float(self.d1.close[0])
                    stop  = float(entry + self.p.atr_stop_mult * float(self.atr5[0]))
                    size  = self._risk_based_size(entry, stop)
                    if size <= 0:
                        cash = float(self.broker.getcash())
                        budget = cash * float(self.p.percent_sizer or 0.0)
                        size = self._round_qty(budget / max(entry, 1e-8))
                        if size <= 0:
                            self.conditions_checked = True
                            return False

                    tracker = self._make_tracker(entry, size, -1)
                    self.active_orders.append(tracker)
                    self.entry_prices.append(entry)
                    self.sizes.append(size)

                    self.sell(size=size, exectype=bt.Order.Market)

                    self.init_stop = self._round_price(stop)
                    self.trail_stop = self.init_stop
                    self.initial_risk = max(1e-8, self.init_stop - entry)
                    self.run_low = self.d1.low[0]; self.run_high=None
                    self.entry_bar = len(self); self.last_add_bar = len(self)

                    if self.p.debug:
                        print(f"ENTER SHORT {size} @ {entry} | SL={self.init_stop} | TP={tracker.take_profit_price}")

                    self.first_entry_price = entry if not self.buy_executed else self.first_entry_price
                    self.buy_executed = True
                    self.DCA = True
                    self.conditions_checked = True
                    return False

            else:
                # Fallback ohne MTF
                entry = float(self.d1.close[0])
                # Long
                if self.ema1_fast[0] > self.ema1_slow[0] and self.breakout_up():
                    stop  = float(entry - self.p.atr_stop_mult * float(self.atr5[0]))
                    size  = self._risk_based_size(entry, stop)
                    if size <= 0:
                        cash = float(self.broker.getcash())
                        budget = cash * float(self.p.percent_sizer or 0.0)
                        size = self._round_qty(budget / max(entry, 1e-8))
                    if size > 0:
                        tracker = self._make_tracker(entry, size, +1)
                        self.active_orders.append(tracker)
                        self.entry_prices.append(entry)
                        self.sizes.append(size)
                        self.buy(size=size, exectype=bt.Order.Market)
                        self.init_stop = self._round_price(stop)
                        self.trail_stop = self.init_stop
                        self.initial_risk = max(1e-8, entry - self.init_stop)
                        self.run_high = self.d1.high[0]; self.run_low=None
                        self.entry_bar = len(self); self.last_add_bar = len(self)
                        self.first_entry_price = entry if not self.buy_executed else self.first_entry_price
                        self.buy_executed = True; self.DCA = True

                # Short
                if self.p.can_short and self.ema1_fast[0] < self.ema1_slow[0] and self.breakdown_down():
                    stop  = float(entry + self.p.atr_stop_mult * float(self.atr5[0]))
                    size  = self._risk_based_size(entry, stop)
                    if size <= 0:
                        cash = float(self.broker.getcash())
                        budget = cash * float(self.p.percent_sizer or 0.0)
                        size = self._round_qty(budget / max(entry, 1e-8))
                    if size > 0:
                        tracker = self._make_tracker(entry, size, -1)
                        self.active_orders.append(tracker)
                        self.entry_prices.append(entry)
                        self.sizes.append(size)
                        self.sell(size=size, exectype=bt.Order.Market)
                        self.init_stop = self._round_price(stop)
                        self.trail_stop = self.init_stop
                        self.initial_risk = max(1e-8, self.init_stop - entry)
                        self.run_low = self.d1.low[0]; self.run_high=None
                        self.entry_bar = len(self); self.last_add_bar = len(self)
                        self.first_entry_price = entry if not self.buy_executed else self.first_entry_price
                        self.buy_executed = True; self.DCA = True

        self.conditions_checked = True
        return False

    def dca_or_short_condition(self):
        # Winner-Add (optional, wenn use_pyramiding=True)
        if self.p.use_pyramiding and self.position and self._can_pyramid():
            entry = float(self.d1.close[0])
            if self.position.size > 0:
                stop = float(self.trail_stop or entry - self.p.atr_stop_mult * float(self.atr5[0]))
                direction = +1
            else:
                stop = float(self.trail_stop or entry + self.p.atr_stop_mult * float(self.atr5[0]))
                direction = -1
            size = self._round_qty(self._risk_based_size(entry, stop) / 2.0)
            if size > 0:
                tracker = self._make_tracker(entry, size, direction)
                self.active_orders.append(tracker)
                self.entry_prices.append(entry)
                self.sizes.append(size)
                if direction > 0:
                    self.buy(size=size, exectype=bt.Order.Market)
                else:
                    self.sell(size=size, exectype=bt.Order.Market)
                self.n_adds += 1
                self.last_add_bar = len(self)
                if self.p.debug:
                    print(f"PYRAMID add #{self.n_adds} {size} @ {entry}")
        self.conditions_checked = True
        return False

    def sell_or_cover_condition(self):
        if not self.active_orders:
            self.conditions_checked = True
            return False

        current = float(self.d1.close[0])
        to_remove = []

        # Take Profits je Leg
        for idx, o in enumerate(self.active_orders):
            if o.order_type == "BUY":
                if current >= o.take_profit_price:
                    self.sell(size=o.size, exectype=bt.Order.Market)
                    o.close_order(current)
                    to_remove.append(idx)
                    if self.p.debug:
                        prof = (current / o.entry_price - 1) * 100
                        print(f"TP LONG: -{o.size} @ {current} (+{prof:.2f}%)")
            else:
                if current <= o.take_profit_price:
                    self.buy(size=o.size, exectype=bt.Order.Market)
                    o.close_order(current)
                    to_remove.append(idx)
                    if self.p.debug:
                        prof = (1 - current / o.entry_price) * 100
                        print(f"TP SHORT: +{o.size} @ {current} (+{prof:.2f}%)")

        # Trailing optional
        if self.p.use_trailing_stop:
            self._update_trailing_stop()
            if self._stop_hit():
                # Close nach Richtung
                if self.position.size > 0:
                    qty = sum(o.size for o in self.active_orders if o.order_type == "BUY")
                    if qty > 0:
                        self.sell(size=qty, exectype=bt.Order.Market)
                else:
                    qty = sum(o.size for o in self.active_orders if o.order_type != "BUY")
                    if qty > 0:
                        self.buy(size=qty, exectype=bt.Order.Market)
                # alle offenen tracker schließen
                to_remove = list(range(len(self.active_orders)))

        # Remove + State-Update
        if to_remove:
            for i in reversed(to_remove):
                self.active_orders.pop(i)
                self.entry_prices.pop(i)
                self.sizes.pop(i)

            if not self.active_orders:
                self.reset_position_state()
                self.buy_executed = False
            else:
                self.calc_averages()

        self.conditions_checked = True
        return False




from typing import Dict, Any, List, Tuple, Optional, Callable
class VectorMACD_ADX(bt.Strategy):
    params = (
        # Sizing
        ('percent_sizer', 0.05),
        ('risk_per_trade_pct', 0.0025),
        ('max_leverage', 2.0),
        ('min_qty', 0.0),
        ('qty_step', 0.001),      # FIX: realistic default
        ('price_tick', 0.1),      # FIX: realistic default
        ('round_prices', True),

        # Shorts
        ('can_short', False),
        ('regime_mode_long', 'ema'),      # 'ema' | 'price_vs_slow'
        ('regime_mode_short', 'neutral'), # 'neutral' | 'ema'
        ('rsi_oversold', 25),

        # Feature periods (must match feature builder)
        ('use_htf', True),
        ('tf5m_breakout_period', 55),
        ('tf15m_adx_period', 14),
        ('tf15m_ema_fast', 50),
        ('tf15m_ema_slow', 200),
        ('tf60m_ema_fast', 50),
        ('tf60m_ema_slow', 200),

        # 1m baseline
        ('ema_fast', 20),
        ('ema_slow', 50),
        ('ema_trend', 200),
        ('atr_period', 14),
        ('rsi_overheat', 75),

        # Entries/Stops/TP
        ('adxth', 20),
        ('confirm_bars', 2),
        ('max_stretch_atr_mult', 1.0),
        ('atr_stop_mult', 2.5),
        ('take_profit', 4.0),      # % per leg

        # Trailing
        ('use_trailing_stop', True),
        ('trail_mode', 'chandelier'),  # chandelier | ema_band | donchian
        ('trail_atr_mult', 4.0),
        ('ema_band_mult', 2.0),
        ('donchian_trail_period', 55),  # already baked in features
        ('close_based_stop', True),
        ('move_to_breakeven_R', 1.0),
        ('trail_update_every', 2),
        ('max_bars_in_trade', 6*60),
        ('reentry_cooldown_bars', 5),

        # Pyramiding
        ('use_pyramiding', False),
        ('max_adds', 0),
        ('add_cooldown', 20),
        ('add_atr_mult', 1.0),
        ('add_min_R', 1.0),

        # Volume filter
        ('use_volume_filter', False),
        ('volume_filter_mult', 1.2),

        ('backtest', True),
        ('debug', False),
    )

    def __init__(self):
        self.d = self.datas[0]
        # State
        self.entry_bar = None
        self.trail_stop = None
        self.init_stop = None
        self.initial_risk = None
        self.run_high = None
        self.run_low = None
        self.last_trail_update = -10**9
        self.n_adds = 0
        self.last_add_bar = -10**9
        self.last_exit_bar = -10**9

        self.active_orders = []
        self.entry_prices = []
        self.sizes = []

    # ---------- Sizing / Rounding ----------
    def _round_qty(self, size: float) -> float:
        step = float(self.p.qty_step) if self.p.qty_step else 0.001
        if step <= 0:
            step = 0.001
        q = math.floor(max(0.0, float(size)) / step) * step  # round DOWN only
        if self.p.min_qty and q < float(self.p.min_qty):
            return 0.0
        return q

    def _round_price(self, price: float) -> float:
        if not (self.p.round_prices and self.p.price_tick):
            return float(price)
        tick = float(self.p.price_tick)
        if tick <= 0:
            return float(price)
        return round(price / tick) * tick

    def _risk_based_size(self, entry: float, stop: float) -> float:
        eq = float(self.broker.getvalue())
        risk = eq * float(self.p.risk_per_trade_pct or 0.0)
        dist = max(1e-8, abs(float(entry) - float(stop)))
        s_risk = risk / dist if dist > 0 else 0.0
        s_lev = (eq * float(self.p.max_leverage)) / max(float(entry), 1e-8)
        s_raw = max(0.0, min(s_risk, s_lev))
        s = self._round_qty(s_raw)
        if s <= 0.0:
            return 0.0
        # Clamp to leverage post-rounding
        max_units = (eq * float(self.p.max_leverage)) / max(float(entry), 1e-8)
        if s > max_units:
            step = float(self.p.qty_step) if self.p.qty_step else 0.001
            s = math.floor(max_units / step) * step
            if self.p.min_qty and s < float(self.p.min_qty):
                s = 0.0
        return s

    # ---------- Helpers ----------
    def _avg_entry(self) -> Optional[float]:
        if not self.active_orders:
            return None
        total = sum(o["size"] for o in self.active_orders)
        if total <= 0:
            return None
        return sum(o["entry"] * o["size"] for o in self.active_orders) / total

    def _R(self) -> float:
        if self.initial_risk is None or self.initial_risk <= 0:
            return 0.0
        px = float(self.d.close[0])
        ae = self._avg_entry() or px
        if self.position.size > 0:
            return (px - ae) / self.initial_risk
        elif self.position.size < 0:
            return (ae - px) / self.initial_risk
        return 0.0

    def _volume_filter_ok(self) -> bool:
        if not self.p.use_volume_filter:
            return True
        base = float(self.d.vsma1[0]) if not math.isnan(float(self.d.vsma1[0])) else None
        if base is None or base <= 0:
            return False
        return float(self.d.volume[0]) > self.p.volume_filter_mult * base

    # ---------- Trailing ----------
    def _update_trailing_stop(self):
=======
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
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented
        if not self.position:
            self.trail_stop = None
            return

<<<<<<< HEAD
        if self.position.size > 0:
            self.run_high = max(self.run_high or float(self.d.high[0]), float(self.d.high[0]))
        else:
            self.run_low  = min(self.run_low or float(self.d.low[0]), float(self.d.low[0]))

=======
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
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented
        if (len(self) - self.last_trail_update) < self.p.trail_update_every:
            return

        candidate = None
<<<<<<< HEAD
        if self.p.trail_mode == "chandelier":
            if self.position.size > 0:
                candidate = float((self.run_high or float(self.d.high[0])) - self.p.trail_atr_mult * float(self.d.atr5[0]))
            else:
                candidate = float((self.run_low  or float(self.d.low[0]))  + self.p.trail_atr_mult * float(self.d.atr5[0]))
        elif self.p.trail_mode == "ema_band":
            if self.position.size > 0:
                candidate = float(self.d.ema1_fast[0] - self.p.ema_band_mult * float(self.d.atr1[0]))
            else:
                candidate = float(self.d.ema1_fast[0] + self.p.ema_band_mult * float(self.d.atr1[0]))
        elif self.p.trail_mode == "donchian":
            candidate = float(self.d.dc_exit5_low[0] if self.position.size > 0 else self.d.dc_exit5_high[0])

        if candidate is not None:
            # lock to at least init stop
            candidate = max(candidate, self.init_stop or -1e18) if self.position.size > 0 else min(candidate, self.init_stop or 1e18)
            ae = self._avg_entry()
            if ae and self._R() >= self.p.move_to_breakeven_R:
                candidate = max(candidate, ae) if self.position.size > 0 else min(candidate, ae)
            self.trail_stop = candidate if self.trail_stop is None else (
                max(self.trail_stop, candidate) if self.position.size > 0 else min(self.trail_stop, candidate)
            )
            self.last_trail_update = len(self)

    def _stop_hit(self) -> bool:
        if self.trail_stop is None:
            return False
        if self.p.close_based_stop:
            return (self.d.close[0] <= self.trail_stop) if self.position.size > 0 else (self.d.close[0] >= self.trail_stop)
        else:
            return (self.d.low[0] <= self.trail_stop) if self.position.size > 0 else (self.d.high[0] >= self.trail_stop)

    # ---------- Conditions ----------
    def regime_ok_long(self) -> bool:
        if self.p.regime_mode_long == 'price_vs_slow':
            return bool(self.d.close[0] > self.d.ema60_slow[0])
        return bool(self.d.ema60_fast[0] > self.d.ema60_slow[0])

    def regime_ok_short(self) -> bool:
        if not self.p.can_short:
            return False
        if self.p.regime_mode_short == 'neutral':
            return True
        return bool(self.d.ema60_fast[0] < self.d.ema60_slow[0])

    def trend_ok_long(self) -> bool:
        return bool(self.d.adx[0] >= self.p.adxth and self.d.plus_di[0] > self.d.minus_di[0]
                    and self.d.ema15_fast[0] > self.d.ema15_slow[0] and self.d.ema1_fast[0] > self.d.ema1_slow[0])

    def trend_ok_short(self) -> bool:
        return bool(self.d.adx[0] >= self.p.adxth and self.d.minus_di[0] > self.d.plus_di[0]
                    and self.d.ema15_fast[0] < self.d.ema15_slow[0] and self.d.ema1_fast[0] < self.d.ema1_slow[0])

    def breakout_up(self) -> bool:
        if self.p.use_volume_filter and not self._volume_filter_ok():
            return False
        if float(self.d.rsi[0]) >= self.p.rsi_overheat:
            return False
        stretched = (float(self.d.close[0]) - float(self.d.dc_high5_prev[0])) > self.p.max_stretch_atr_mult * float(self.d.atr5[0])
        return bool(self.d.breakout_up[0] and not stretched)

    def breakdown_down(self) -> bool:
        if self.p.use_volume_filter and not self._volume_filter_ok():
            return False
        if float(self.d.rsi[0]) <= self.p.rsi_oversold:
            return False
        stretched = (float(self.d.dc_low5_prev[0]) - float(self.d.close[0])) > self.p.max_stretch_atr_mult * float(self.d.atr5[0])
        return bool(self.d.breakdown_down[0] and not stretched)

    # ---------- Orders ----------
    def _enter_long(self):
        entry = float(self._round_price(self.d.close[0]))
        init_stop = self._round_price(entry - self.p.atr_stop_mult * float(self.d.atr5[0]))
        size = self._risk_based_size(entry, init_stop)
        if size <= 0:
            cash = float(self.broker.getcash())
            budget = cash * float(self.p.percent_sizer or 0.0)
            step = float(self.p.qty_step) if self.p.qty_step else 0.001
            size = math.floor((budget / max(entry, 1e-8)) / step) * step
            if self.p.min_qty and size < float(self.p.min_qty):
                size = 0.0
            if size <= 0.0:
                return
        tp = self._round_price(entry * (1 + self.p.take_profit / 100.0))

        self.active_orders.append(dict(entry=entry, size=size, tp=tp, dir=+1))
        self.buy(size=size, exectype=bt.Order.Market)
        self.init_stop = init_stop
        self.trail_stop = init_stop
        self.initial_risk = max(1e-8, entry - init_stop)
        self.run_high = float(self.d.high[0]); self.run_low = None
        self.entry_bar = len(self); self.last_add_bar = len(self)

        if self.p.debug:
            console.print(f"ENTER LONG {size} @ {entry} | SL={init_stop} | TP={tp}")

    def _enter_short(self):
        if not self.p.can_short:
            return
        entry = float(self._round_price(self.d.close[0]))
        init_stop = self._round_price(entry + self.p.atr_stop_mult * float(self.d.atr5[0]))
        size = self._risk_based_size(entry, init_stop)
        if size <= 0:
            cash = float(self.broker.getcash())
            budget = cash * float(self.p.percent_sizer or 0.0)
            step = float(self.p.qty_step) if self.p.qty_step else 0.001
            size = math.floor((budget / max(entry, 1e-8)) / step) * step
            if self.p.min_qty and size < float(self.p.min_qty):
                size = 0.0
            if size <= 0.0:
                return
        tp = self._round_price(entry * (1 - self.p.take_profit / 100.0))

        self.active_orders.append(dict(entry=entry, size=size, tp=tp, dir=-1))
        self.sell(size=size, exectype=bt.Order.Market)
        self.init_stop = init_stop
        self.trail_stop = init_stop
        self.initial_risk = max(1e-8, init_stop - entry)
        self.run_low = float(self.d.low[0]); self.run_high = None
        self.entry_bar = len(self); self.last_add_bar = len(self)

        if self.p.debug:
            console.print(f"ENTER SHORT {size} @ {entry} | SL={init_stop} | TP={tp}")

    def _can_pyramid(self) -> bool:
        if not (self.p.use_pyramiding and self.position):
            return False
        if self.n_adds >= self.p.max_adds:
            return False
        if (len(self) - self.last_add_bar) < self.p.add_cooldown:
            return False
        if self._R() < self.p.add_min_R:
            return False
        if self.position.size > 0:
            return float(self.d.close[0]) >= ((self.run_high or float(self.d.high[0])) + self.p.add_atr_mult * float(self.d.atr5[0]))
        else:
            return float(self.d.close[0]) <= ((self.run_low  or float(self.d.low[0]))  - self.p.add_atr_mult * float(self.d.atr5[0]))

    def _do_pyramid(self):
        entry = float(self._round_price(self.d.close[0]))
        stop = float(self.trail_stop or (entry - self.p.atr_stop_mult * float(self.d.atr5[0]) if self.position.size > 0
                                         else entry + self.p.atr_stop_mult * float(self.d.atr5[0])))
        size = self._round_qty(self._risk_based_size(entry, stop) / 2.0)
        if size <= 0:
            return
        if self.position.size > 0:
            tp = self._round_price(entry * (1 + self.p.take_profit/100.0))
            self.active_orders.append(dict(entry=entry, size=size, tp=tp, dir=+1))
            self.buy(size=size, exectype=bt.Order.Market)
        else:
            tp = self._round_price(entry * (1 - self.p.take_profit/100.0))
            self.active_orders.append(dict(entry=entry, size=size, tp=tp, dir=-1))
            self.sell(size=size, exectype=bt.Order.Market)
        self.n_adds += 1
        self.last_add_bar = len(self)
        if self.p.debug:
            console.print(f"PYRAMID add #{self.n_adds} {size} @ {entry}")

    def _take_profits_and_trail(self):
        if not self.active_orders:
            return
        current = float(self.d.close[0])
        to_remove = []

        # Per-leg TP
        for idx, o in enumerate(self.active_orders):
            if o["dir"] > 0 and current >= o["tp"]:
                self.sell(size=o["size"], exectype=bt.Order.Market)
                to_remove.append(idx)
                if self.p.debug:
                    prof = (current / o["entry"] - 1) * 100
                    console.print(f"TP LONG: -{o['size']} @ {current} (+{prof:.2f}%)")
            elif o["dir"] < 0 and current <= o["tp"]:
                self.buy(size=o["size"], exectype=bt.Order.Market)
                to_remove.append(idx)
                if self.p.debug:
                    prof = (1 - current / o["entry"]) * 100
                    console.print(f"TP SHORT: +{o['size']} @ {current} (+{prof:.2f}%)")

        # Trailing stop
        if self.p.use_trailing_stop:
            self._update_trailing_stop()
            if self._stop_hit():
                qty = sum(o["size"] for o in self.active_orders if (o["dir"] > 0 and self.position.size > 0) or (o["dir"] < 0 and self.position.size < 0))
                if qty > 0:
                    if self.position.size > 0:
                        self.sell(size=qty, exectype=bt.Order.Market)
                    else:
                        self.buy(size=qty, exectype=bt.Order.Market)
                to_remove = list(range(len(self.active_orders)))

        if to_remove:
            for i in reversed(to_remove):
                self.active_orders.pop(i)
                self.sizes = [o["size"] for o in self.active_orders]
                self.entry_prices = [o["entry"] for o in self.active_orders]

            if not self.active_orders:
                self._reset_position_state()

    def _reset_position_state(self):
        self.entry_bar = None
        self.trail_stop = None
        self.init_stop = None
        self.initial_risk = None
        self.run_high = None
        self.run_low = None
        self.last_trail_update = -10**9
        self.n_adds = 0
        self.last_add_bar = -10**9

    def next(self):
        # Flat -> Entry
        if not self.position:
            # Basic cooldown after exit
            if (len(self) - self.last_exit_bar) < self.p.reentry_cooldown_bars:
                return

            if self.p.use_htf:
                if self.regime_ok_long() and self.trend_ok_long() and self.breakout_up():
                    self._enter_long()
                elif self.p.can_short and self.regime_ok_short() and self.trend_ok_short() and self.breakdown_down():
                    self._enter_short()
            else:
                # Fallback: only use 1m features and breakout
                if self.d.ema1_fast[0] > self.d.ema1_slow[0] and self.breakout_up():
                    self._enter_long()
                elif self.p.can_short and self.d.ema1_fast[0] < self.d.ema1_slow[0] and self.breakdown_down():
                    self._enter_short()
        else:
            # Manage position
            self._take_profits_and_trail()
            # Optional pyramiding
            if self.p.use_pyramiding and self._can_pyramid():
                self._do_pyramid()

    def notify_trade(self, trade):
        if trade.isclosed:
            self.last_exit_bar = len(self)
=======
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
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented
