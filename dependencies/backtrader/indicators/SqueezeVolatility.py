import backtrader as bt
import numpy as np
from scipy import stats

class SqueezeVolatility(bt.Indicator):
    # Clone of the TradingBear Squeeze Volatility Indicator

    lines = ('hist', 'sqz', 'y', 'ma')

    params = (
        ('period', 10),
        ('mult', 2),
        ('period_kc', 10),
        ('mult_kc', 1.5),
        ('movav', bt.ind.SMA)
    )

    plotinfo = dict(
        plot=True,
        plotname='Squeeze Volatility',
        subplot=True,
        plotlinelabels=True)

    plotlines = dict(sqz=dict(_plotskip=False, ),
                     y=dict(_plotskip=True, ),
                     ma=dict(_plotskip=True, ),
                     )

    def __init__(self):

        self.addminperiod(self.p.period_kc*2)
        bands = bt.indicators.BollingerBands(self.data, period=self.p.period, devfactor=self.p.mult)
        self.l.ma = ma = self.p.movav(self.data,period=self.p.period_kc)
        atr = bt.indicators.ATR(self.data,period=self.p.period_kc)
        rma = self.p.movav(atr.atr, period=self.p.period_kc)
        uKC = ma + rma*self.p.mult_kc
        lKC = ma - rma*self.p.mult_kc

        bool = bt.And(bands.bot>lKC, bands.top<uKC)

        self.l.sqz = bt.If(bool,0.0,1.0)

    def prenext(self):

        self.l.y[0] = 0.0#self.data.close[0]

    def next(self):

        h = max(self.data.high.get(size=self.p.period_kc))
        l = min(self.data.low.get(size=self.p.period_kc))
        av1 = (h+l)/2
        av2 = (av1 + self.l.ma[0])/2
        self.l.y[0] = self.data.close[0] - av2

        # Perform Linear Regression
        x = np.arange(0,self.p.period_kc,1)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, self.l.y.get(size=self.p.period_kc))
        self.l.hist[0] = intercept + slope * (self.p.period_kc - 1)
