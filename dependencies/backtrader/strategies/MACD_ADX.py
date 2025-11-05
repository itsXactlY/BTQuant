from .base import BaseStrategy, bt, OrderTracker
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
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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

        if not hasattr(self, 'active_orders'):
            self.active_orders = []
        if not hasattr(self, 'entry_prices'):
            self.entry_prices = []
        if not hasattr(self, 'sizes'):
            self.sizes = []

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
        if not self.position:
            self.trail_stop = None
            return

        if self.position.size > 0:
            self.run_high = max(self.run_high or float(self.d.high[0]), float(self.d.high[0]))
        else:
            self.run_low  = min(self.run_low or float(self.d.low[0]), float(self.d.low[0]))

        if (len(self) - self.last_trail_update) < self.p.trail_update_every:
            return

        candidate = None
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
