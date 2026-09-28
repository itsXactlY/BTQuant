#!/usr/bin/env python3
"""
Native matplotlib backtest plot — a real window, not a browser.

Reproduces the classic multi-panel layout in one Figure:
    price panel   candles + VWAP / VAH / VAL / POC bands
    momentum      rate-of-change oscillator
    volume        own panel
    balance       0/1 regime flag

Run:
    python3 Examples/Plot_Native_Matplotlib.py
        opens an interactive Tk window — no browser, no HTML, no plotly

    python3 Examples/Plot_Native_Matplotlib.py --save fig.png
        renders the identical figure to a file (Agg, non-blocking)

Data is deterministic synthetic OHLCV: runs offline and always identical.
Indicator math is plain Python over LineBuffer.get(ago, size), so it does not
depend on the auto-binding quirks of the vendored backtrader.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dependencies"))

import numpy as np
import polars as pl

import backtrader as bt
from backtrader.plot import PlotScheme  # noqa: E402  (after the sys.path fix)


# ---------------------------------------------------------------- synthetic feed
def make_feed(n=1400, seed=7):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.011, n)
    drift = np.linspace(0.0, 0.9, n) / n
    close = 24000.0 * np.exp(np.cumsum(steps + drift))

    spread = np.abs(rng.normal(0, 0.004, n)) + 0.001
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = np.abs(rng.normal(900, 260, n)) + 60
    volume[::210] *= 1.9  # session bursts so the volume panel has shape

    start = np.datetime64("2023-01-01T00:00:00")
    stamps = (start + np.arange(n) * np.timedelta64(8, "h")).astype("datetime64[ms]")

    return pl.DataFrame(
        {
            "datetime": [str(s) for s in stamps],
            "open": open_.tolist(),
            "high": high.tolist(),
            "low": low.tolist(),
            "close": close.tolist(),
            "volume": volume.tolist(),
        }
    )


# ---------------------------------------------------------------- helpers
def window(line, size):
    """Last <=size finite values of `line`, oldest first."""
    out = [float(x) for x in line.get(0, size)]
    return [x for x in out if x == x]  # drop NaN


def mean_std(vals):
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(vals) / n
    var = sum((x - m) ** 2 for x in vals) / n
    return m, math.sqrt(var)


# ---------------------------------------------------------------- indicators
class ValueBands(bt.Indicator):
    """Anchored VWAP + value-area envelope (VAH/VAL) + POC, on the price panel.

    vwap  cumsum(typical*vol) / cumsum(vol)   -> anchored VWAP
    poc   vwap
    vah   vwap + 1.2 * stddev(close, W)
    val   vwap - 1.2 * stddev(close, W)

    Per-line colour/linestyle/legend text comes from `plotlines` plus
    `plotlinelabels`. This backtrader does NOT read an indicator's plot()
    method for styling (plot/plot.py:420-455), so that hook is deliberately absent.
    """

    lines = ("vwap", "vah", "val", "poc")
    params = (("window", 48),)
    plotinfo = dict(subplot=False, plotscale=1.0, plotorder=99, plotlinelabels=True)
    plotlines = dict(
        vwap=dict(_name="VWAP", color="purple", linestyle="-", linewidth=1.7),
        vah=dict(_name="VAH", color="red", linestyle="--", linewidth=1.0),
        val=dict(_name="VAL", color="green", linestyle="--", linewidth=1.0),
        poc=dict(_name="POC", color="steelblue", linestyle="-.", linewidth=1.0),
    )

    def __init__(self):
        self._cum_pv = 0.0
        self._cum_v = 0.0
        self.addminperiod(self.p.window)

    def next(self):
        h, l, c, v = (float(x[0]) for x in (self.data.high, self.data.low,
                                           self.data.close, self.data.volume))
        self._cum_pv += ((h + l + c) / 3.0) * v
        self._cum_v += v
        vwap = self._cum_pv / self._cum_v if self._cum_v > 0 else float("nan")
        self.lines.vwap[0] = vwap

        _m, sd = mean_std(window(self.data.close, self.p.window))
        if vwap != vwap or sd != sd:
            self.lines.poc[0] = self.lines.vah[0] = self.lines.val[0] = float("nan")
            return
        self.lines.poc[0] = vwap
        self.lines.vah[0] = vwap + 1.2 * sd
        self.lines.val[0] = vwap - 1.2 * sd


class Momentum(bt.Indicator):
    """Rate-of-change oscillator, own row."""

    lines = ("mom",)
    params = (("period", 14),)
    plotinfo = dict(subplot=True)
    plotlines = dict(mom=dict(_name="Momentum", color="darkturquoise", linewidth=1.2))

    def __init__(self):
        self.addminperiod(self.p.period + 1)

    def next(self):
        back = float(self.data.close.get(-self.p.period)[0])
        cur = float(self.data.close[0])
        self.lines.mom[0] = 100.0 * (cur - back) / back if back else float("nan")


class Balance(bt.Indicator):
    """0/1 regime flag: fast mean above slow mean. Own row."""

    lines = ("balance",)
    params = (("fast", 24), ("slow", 48),)
    plotinfo = dict(subplot=True)
    plotlines = dict(balance=dict(_name="Balance", color="darkorange",
                                  linewidth=1.0, drawstyle="steps-post"))

    def __init__(self):
        self.addminperiod(self.p.slow)

    def next(self):
        slow_vals = window(self.data.close, self.p.slow)
        if len(slow_vals) < self.p.slow:
            self.lines.balance[0] = float("nan")
            return
        fast = sum(slow_vals[-self.p.fast:]) / self.p.fast
        slow = sum(slow_vals) / len(slow_vals)
        self.lines.balance[0] = 1.0 if fast > slow else 0.0


class SMA(bt.Indicator):
    """Rolling mean of close, not plotted (signal helper)."""

    lines = ("sma",)
    params = (("period", 20),)
    plotinfo = dict(subplot=False, plot=False)

    def __init__(self):
        self.addminperiod(self.p.period)

    def next(self):
        vals = window(self.data.close, self.p.period)
        self.lines.sma[0] = sum(vals) / len(vals) if vals else float("nan")


class Demo(bt.Strategy):
    lines = ("crossover",)

    def __init__(self):
        self.bands = ValueBands(window=48)
        self.mom = Momentum(period=14)
        self.bal = Balance(fast=24, slow=48)
        self._fast = SMA(period=20)
        self._slow = SMA(period=60)
        self._prev_slow = float("nan")

    def next(self):
        f, s = float(self._fast.sma[0]), float(self._slow.sma[0])
        if math.isnan(f) or math.isnan(s):
            self.lines.crossover[0] = 0.0
            return

        signal = 0.0
        if not math.isnan(self._prev_slow):
            if f > s and self._prev_slow <= s:
                signal = 1.0
            elif f < s and self._prev_slow >= s:
                signal = -1.0
        self._prev_slow = s
        self.lines.crossover[0] = signal

        if signal > 0 and not self.position and self.bal.balance[0] > 0.5:
            self.buy()
        elif signal < 0 and self.position:
            self.close()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", metavar="PNG", help="write the figure to a file instead of showing a window")
    ap.add_argument("--bars", type=int, default=1400)
    ap.add_argument("--rows-major", type=int, default=3, help="price panel height share")
    ap.add_argument("--rows-minor", type=int, default=1, help="height share per indicator panel")
    ap.add_argument("--width", type=float, default=19.2, help="figure width in inches")
    ap.add_argument("--height", type=float, default=10.8, help="figure height in inches")
    ap.add_argument("--dpi", type=int, default=100, help="figure dpi (19.2in x 100dpi = 1920px)")
    args = ap.parse_args()

    if args.save:
        import matplotlib
        matplotlib.use("Agg", force=True)

    # backtrader's Plot.newfig() calls mpyplot.figure() with no figsize, so the
    # width/height/dpi of cerebro.plot() never reach the window. rcParams is the
    # only thing that actually sizes the figure.
    import matplotlib
    matplotlib.rcParams["figure.figsize"] = (args.width, args.height)
    matplotlib.rcParams["figure.dpi"] = args.dpi

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(bt.feeds.PolarsData(dataname=make_feed(args.bars)))
    cerebro.addstrategy(Demo)
    cerebro.broker.setcash(100_000.0)
    cerebro.broker.setcommission(commission=0.00075)
    cerebro.run()

    scheme = PlotScheme()
    scheme.style = "candle"
    scheme.barup = "#3b8f5a"
    scheme.bardown = "#c0392b"
    scheme.barupfill = scheme.bardownfill = True
    scheme.volume = True
    scheme.voloverlay = False           # own panel, like the reference layout
    scheme.volup = "#7aa6c2"
    scheme.voldown = "#c0392b"
    scheme.grid = True
    scheme.tickrotation = 15
    scheme.rowsmajor = args.rows_major
    scheme.rowsminor = args.rows_minor
    scheme.legendindloc = "upper left"
    scheme.legenddataloc = "upper left"
    scheme.fmt_x_ticks = "%b %d, %H:%M"

    cerebro.plot(scheme=scheme, width=19.2, height=10.8, dpi=110)

    if args.save:
        import matplotlib.pyplot as plt
        for num in plt.get_fignums():
            plt.figure(num).savefig(args.save, dpi=args.dpi)
        print("geschrieben:", os.path.abspath(args.save))
    else:
        print("Backend:", matplotlib.get_backend())
        print("Fenster offen — schliessen mit dem X oben rechts.")


if __name__ == "__main__":
    main()
