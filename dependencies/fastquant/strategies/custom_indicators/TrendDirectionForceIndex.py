import backtrader as bt
import numpy as np

class TrendDirectionForceIndex(bt.Indicator):

    lines=('mma','smma','tdf','ntdf')

    params = (
        ('period', 13),
    )

    plotlines = dict(mma=dict(_plotskip=True, ),
                     smma=dict(_plotskip=True,),
                     tdf=dict(_plotskip=True,)
                     )

    plotinfo = dict(
        plot=True,
        plotname='Trend Force and Direction Index',
        subplot=True,
        plotlinelabels=True)

    def __init__(self):

        self.addminperiod(self.p.period*3)
        self.l.mma = bt.indicators.EMA(self.data.close*1000,period=self.p.period)
        self.l.smma = bt.indicators.EMA(self.l.mma,period=self.p.period)

    def prenext(self):

        impetmma = self.l.mma[0] - self.l.mma[-1]
        impetsmma = self.l.smma[0] - self.l.smma[-1]
        divma = abs(self.l.mma[0] - self.l.smma[0])
        averimpet = (impetmma+impetsmma)/2
        pow = averimpet**3
        self.l.tdf[0] = divma * pow

    def next(self):

        impetmma = self.l.mma[0] - self.l.mma[-1]
        impetsmma = self.l.smma[0] - self.l.smma[-1]
        divma = abs(self.l.mma[0] - self.l.smma[0])
        averimpet = (impetmma+impetsmma)/2
        pow = averimpet**3

        self.l.tdf[0] = divma * pow
        self.l.ntdf[0] = self.l.tdf[0]/(max(np.absolute(self.l.tdf.get(size=self.p.period*3))))
