import backtrader as bt
import numpy as np

class iFisher(bt.Indicator):

    lines = ('ifisher','scaled','smoothed')
    params = (('scaling',5.0),('period',20),('smoothing',5))

    plotlines = dict(scaled=dict(_plotskip=True, ),
                     smoothed=dict(_plotskip=True, ))

    plotinfo = dict(
        plot=True,
        plotname='Inverse Fisher Transform',
        subplot=True,
        plotlinelabels=True)

    def __init__(self):

        self.addminperiod(self.p.period)

        hi = bt.indicators.Highest(self.data, period=self.p.period)
        lo = bt.indicators.Lowest(self.data, period=self.p.period)

        # Calculate Rescaling

        self.lines.scaled = s = (2*self.p.scaling)*((self.data-lo)/(hi-lo))-self.p.scaling
        self.lines.smoothed = ss = bt.indicators.SMA(s, period=self.p.smoothing)

    def next(self):
        self.lines.ifisher[0] = (np.exp(2*self.l.smoothed[0])-1)/(np.exp(2*self.l.smoothed[0])+1)
