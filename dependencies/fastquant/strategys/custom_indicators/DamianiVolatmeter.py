import backtrader as bt

class DamianiVolatmeter(bt.Indicator):

    lines = ('v','t','aF','sS','aS','sS')

    params = (
        ('atr_fast', 13),
        ('std_fast', 20),
        ('atr_slow', 40),
        ('std_slow', 100),
        ('thresh', 1.4),
        ('lag_supress', True)
    )

    plotinfo = dict(
        plot=True,
        plotname='Damiani Volatmeter',
        subplot=True,
        plotlinelabels=True)

    plotlines = dict(aF=dict(_plotskip=True, ),
                     aS=dict(_plotskip=True, ),
                     sF=dict(_plotskip=True, ),
                     sS=dict(_plotskip=True, ),
                     )

    def __init__(self):

        self.lag_s = 0.5
        self.l.aF = bt.indicators.AverageTrueRange(self.data,period=self.p.atr_fast)
        self.l.sF = bt.indicators.StandardDeviation(self.data.close,period=self.p.std_fast)

        self.l.aS = bt.indicators.AverageTrueRange(self.data, period=self.p.atr_slow)
        self.l.sS = bt.indicators.StandardDeviation(self.data.close, period=self.p.std_slow)

    def prenext(self):

        self.l.v[0] = 0.0050

    def next(self):

        s1 = self.l.v[-1]
        s3 = self.l.v[-3]

        self.l.v[0] = self.l.aF[0]/self.l.aS[0] + self.lag_s*(s1-s3) if self.p.lag_supress else self.l.aF[0]/self.l.aS[0]
        anti_thresh = self.l.sF[0]/self.l.sS[0]
        self.l.t[0] = self.p.thresh - anti_thresh