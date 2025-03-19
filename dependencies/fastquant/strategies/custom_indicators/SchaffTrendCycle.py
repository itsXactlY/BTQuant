import backtrader as bt

class SchaffTrendCycle(bt.Indicator):

    lines = ('schaff','macd','f1','f2','pf')

    params = (
        ('fast', 23),
        ('slow', 50),
        ('cycle', 10),
        ('factor', 0.5)
    )

    plotinfo = dict(
        plot=True,
        plotname='Schaff Trend Cycle',
        subplot=True,
        plotlinelabels=True)

    plotlines = dict(macd=dict(_plotskip=True, ),
                     f1=dict(_plotskip=True, ),
                     f2=dict(_plotskip=True, ),
                     pf=dict(_plotskip=True, ),
                     )

    def __init__(self):
        # Plot horizontal Line
        self.plotinfo.plotyhlines = [25,75]


        self.addminperiod(self.p.slow)
        self.l.macd = bt.indicators.MACD(self.data,period_me1=self.p.fast,period_me2=self.p.slow)

    def prenext(self):

        self.l.f1[0] = self.data.close[0]
        self.l.pf[0] = self.data.open[0]
        self.l.f2[0] = self.data.high[0]
        self.l.schaff[0] = self.data.low[0]

    def next(self):

        v1 = min(self.l.macd.get(size=self.p.cycle))
        v2 = max(self.l.macd.get(size=self.p.cycle))-v1

        self.l.f1[0] = 100*(self.l.macd[0]-v1)/v2 if v2 > 0 else self.l.f1[-1]
        self.l.pf[0] = self.l.pf[-1] + (self.p.factor*(self.l.f1[0]-self.l.pf[-1]))

        v3 = min(self.l.pf.get(size=self.p.cycle))
        v4 = max(self.l.pf.get(size=self.p.cycle))-v3

        self.l.f2[0] = 100*(self.l.pf[0]-v3)/v4 if v4 > 0 else self.l.f2[-1]
        self.l.schaff[0] = self.l.schaff[-1] + (self.p.factor*(self.l.f2[0]-self.l.schaff[-1]))
