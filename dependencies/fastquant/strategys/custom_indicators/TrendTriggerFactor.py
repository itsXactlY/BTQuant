import backtrader as bt

class TrendTriggerFactor(bt.Indicator):

    lines = ('ttf',)
    params = (
        ('period', 20),
    )

    plotinfo = dict(
        plot=True,
        plotname='Trend Trigger Factor',
        subplot=True,
        plotlinelabels=True)

    def __init__(self):
        self.addminperiod(2*self.p.period+1)

    def next(self):
        p = self.p.period+1
        Hi = self.data.high.get(ago=0,size=self.p.period)
        Lo = self.data.low.get(ago=0,size=self.p.period)
        lagHi = self.data.high.get(ago=-p,size=self.p.period)
        lagLo = self.data.low.get(ago=-p,size=self.p.period)

        bp = max(Hi) - min(lagLo)
        sp = max(lagHi) - min(Lo)
        self.l.ttf[0] = 100*(bp-sp)/(0.5*(bp+sp))