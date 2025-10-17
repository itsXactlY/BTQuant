# Cheers, u lazy fucks.

import math, os
from backtrader.utils.backtest import backtest
from backtrader.strategies.base import BaseStrategy, bt
from backtrader.indicators.MesaAdaptiveMovingAverage import MAMA

# ---- Lightweight QQE (faster than multiple nested EMAs) ----
class FastQQE(bt.Indicator):
    """
    Fast QQE-ish indicator:
      - RSI base
      - small "dar" volatility term via ATR * q
      - one smoothing EMA + signal EMA
    Minimal lines to reduce overhead.
    """
    params = dict(period=13, fast=5, q=4.236, smoothing=5, eps=1e-9)
    lines = ("smoothed", "signal",)

    def __init__(self):
        p = self.p
        self.rsi = bt.ind.RSI(self.data.close, period=p.period)
        self.rsi_ma = bt.ind.EMA(self.rsi, period=max(2, int(p.period // 2)))

        # ATR-based amplitude (dar). avoid bt.If nesting: multiply then EMA
        self.atr = bt.ind.ATR(self.data, period=max(2, p.fast))
        self.dar_input = self.atr * p.q
        dar_period = max(3, (p.period * 2) - 1)
        self.dar = bt.ind.EMA(self.dar_input, period=dar_period)

        # Build directional qqe_line using a tiny conditional (one bt.If only)
        # qqe_line = rsi + sign * dar
        self.qqe_line = self.rsi + bt.If(self.rsi > self.rsi_ma, self.dar, -self.dar)

        # Smooth + signal
        self.lines.smoothed = bt.ind.EMA(self.qqe_line, period=p.smoothing)
        self.lines.signal = bt.ind.EMA(self.lines.smoothed, period=max(3, p.smoothing * 2))

    def get_signal(self):
        """Return +1 / -1 / 0 with minimal indexing and tolerant thresholds."""
        if len(self.lines.smoothed) < 2:
            return 0
        s0 = float(self.lines.smoothed[0])
        s1 = float(self.lines.smoothed[-1])
        sig0 = float(self.lines.signal[0])
        sig1 = float(self.lines.signal[-1])

        cross_up = (s1 < sig1) and (s0 > sig0)
        cross_down = (s1 > sig1) and (s0 < sig0)

        # level thresholds - slightly more tolerant than strict 50/60/40
        if cross_up or s0 > 55:
            return 1
        if cross_down or s0 < 45:
            return -1
        return 0

# ---- Lightweight Volume Oscillator ----
class FastVolOsc(bt.Indicator):
    params = dict(short_period=14, long_period=28, smooth_period=5, signal_period=10, eps=1e-9)
    lines = ("osc", "smoothed", "signal",)

    def __init__(self):
        p = self.p
        short_vol = bt.ind.SMA(self.data.volume, period=max(2, p.short_period))
        long_vol = bt.ind.SMA(self.data.volume, period=max(3, p.long_period))

        # Avoid costly operations: simple numeric expression
        self.lines.osc = (short_vol - long_vol) / (long_vol + p.eps) * 100.0
        self.lines.smoothed = bt.ind.EMA(self.lines.osc, period=max(2, p.smooth_period))
        self.lines.signal = bt.ind.EMA(self.lines.smoothed, period=max(2, p.signal_period))

    def get_signal(self):
        if len(self.lines.smoothed) < 2:
            return 0
        s0 = float(self.lines.smoothed[0])
        s1 = float(self.lines.smoothed[-1])
        sig0 = float(self.lines.signal[0])
        sig1 = float(self.lines.signal[-1])

        cross_up = (s1 < sig1) and (s0 > sig0)
        cross_down = (s1 > sig1) and (s0 < sig0)

        if cross_up or (s0 > sig0 and s0 > 0):
            return 1
        if cross_down or (s0 < sig0 and s0 < 0):
            return -1
        return 0

# ---- Keep HMA implementation but simplified and reused ----
class HMA(bt.Indicator):
    lines = ("hma",)
    params = dict(period=20)

    def __init__(self):
        p = self.p.period
        half = max(1, int(p / 2))
        sqrtp = max(1, int(math.sqrt(p)))
        self.wma_half = bt.ind.WeightedMovingAverage(self.data, period=half)
        self.wma_full = bt.ind.WeightedMovingAverage(self.data, period=p)
        raw = 2.0 * self.wma_half - self.wma_full
        self.lines.hma = bt.ind.WeightedMovingAverage(raw, period=sqrtp)

class ZeroLag(bt.Indicator):
    lines = ("zerolag",)
    params = dict(period=14)
    def __init__(self):
        h = HMA(self.data, period=self.p.period)
        self.lines.zerolag = 2 * h - bt.ind.SMA(h, period=max(2, int(self.p.period / 2)))

# ---- Optimized Strategy class: same order flow but faster indicator usage ----
class FastSineWeightZeroLagQQEVolMesaAdaptive(BaseStrategy):
    params = dict(
        # indicator params (kept similar)
        zl_period=14,
        sine_period=17,
        sma_filter=197,
        mom_lookback=3,

        qqe_period=13,
        qqe_fast=5,
        qqe_q=4.236,
        qqe_smoothing=5,

        vol_short=14,
        vol_long=28,
        vol_smooth=5,
        vol_signal=10,

        fast=13, slow=17,   # MAMA params (left unchanged)

        # Risk mgmt
        take_profit=0.7,
        dca_deviation=2.5,
        percent_sizer=0.01,

        debug=False,
        backtest=True,
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        p = self.p
        # Precomputed / lightweight indicators
        self.zero_lag = ZeroLag(self.data, period=p.zl_period)
        # reuse zero_lag as input for sine_wma (kept simple)
        self.sine_wma = bt.ind.SMA(self.zero_lag, period=p.sine_period)

        # long SMA filter (200-ish)
        self.sma200 = bt.ind.SMA(self.data, period=p.sma_filter)

        # fastqqe & volosc (optimized)
        self.qqe = FastQQE(self.data, period=p.qqe_period, fast=p.qqe_fast, q=p.qqe_q, smoothing=p.qqe_smoothing)
        self.volosc = FastVolOsc(self.data, short_period=p.vol_short, long_period=p.vol_long,
                                 smooth_period=p.vol_smooth, signal_period=p.vol_signal)

        # simple crossover (zero_lag / sine_wma)
        self.crossover = bt.ind.CrossOver(self.zero_lag, self.sine_wma)

        # MAMA (left as before since it's heavier but necessary)
        self.mama = MAMA(self.data, fast=p.fast, slow=p.slow)

        # Momentum via close deltas (no Momentum indicator)
        # We'll compute lookback deltas in get_signal to keep memory light

        # DCA enabled by default (keeps original semantics)
        self.DCA = True

    # ---------- Lightweight helper: price delta function ----------
    def _price_delta(self, period):
        """Return simple price delta: close[0] - close[-period] (0 if insufficient length)"""
        if len(self.data.close) <= period:
            return 0.0
        return float(self.data.close[0] - self.data.close[-period])

    # ---------- Consolidated signal with relaxed thresholds ----------
    def get_signal(self):
        """Combine signals (fast, tolerant)"""
        # small, fast checks so this is cheap per-bar
        qqe_sig = self.qqe.get_signal()
        vol_sig = self.volosc.get_signal()
        cross_sig = 1 if self.crossover[0] > 0 else (-1 if self.crossover[0] < 0 else 0)

        # momentum trend using close deltas (cheap)
        fm = self._price_delta(14)
        sm = self._price_delta(42)
        momentum_trending_up = (fm > 0 and sm > 0) and (fm > sm)

        price_above_sma = float(self.data.close[0]) > float(self.sma200[0])

        # Count positive and negative signals (looser than original: require 3 positives instead of 4)
        positives = sum([qqe_sig > 0, vol_sig > 0, cross_sig > 0, momentum_trending_up])
        negatives = sum([qqe_sig < 0, vol_sig < 0, cross_sig < 0])

        # Decision: more tolerant thresholds to avoid permanent neutrality
        if positives >= 3 and price_above_sma:
            return 1
        if negatives >= 2 and not price_above_sma:
            return -1
        # price filter fallback: if price is below sma strongly -> negative
        if not price_above_sma and negatives >= 1:
            return -1
        return 0

    # ---------- Entry ----------
    def buy_or_short_condition(self):
        sig = self.get_signal()
        mama_dir = float(self.mama.lines.MAMA[0]) - float(self.mama.lines.FAMA[0]) if hasattr(self.mama.lines, 'FAMA') else 0.0
        if sig > 0 and self.crossover[0] > 0 and mama_dir > 1e-6:
            self.create_order(action='BUY')
            return True
        return False

    # ---------- DCA ----------
    def dca_or_short_condition(self):
        if not self.entry_prices:
            return False
        last_entry = self.entry_prices[-1]
        price = float(self.data.close[0])
        threshold = last_entry * (1 - (self.params.dca_deviation / 100.0))
        if price < threshold:
            if self.get_signal() > 0 and self.crossover[0] > 0:
                self.create_order(action='BUY')
                return True
        return False

    # ---------- Exit ----------
    def sell_or_cover_condition(self):
        price = float(self.data.close[0])
        # Iterate active orders, close if TP reached (fast)
        for order_tracker in list(self.active_orders):
            # When order_tracker.take_profit_price exists, compare
            if getattr(order_tracker, 'take_profit_price', None) is not None and price >= order_tracker.take_profit_price:
                self.close_order(order_tracker)
                return True
        return False

if __name__ == '__main__':
    if os.path.exists("order_tracker.csv"):
        os.remove("order_tracker.csv")
    try:
        backtest(
            FastSineWeightZeroLagQQEVolMesaAdaptive,
            coin='BTC',
            collateral='USDT',
            start_date="2017-01-01",
            # end_date="2024-09-01",
            interval="1h",
            init_cash=1000,
            plot=False,
            quantstats=False,
            debug=False,
        )
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()