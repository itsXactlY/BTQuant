import backtrader as bt


class SSLChannel(bt.Indicator):
    lines = ('ssld', 'sslu')
    params = (('period', 30),)
    plotinfo = dict(
        plot=True,
        plotname='SSL Channel',
        subplot=False,
        plotlinelabels=True)

    def _plotlabel(self):
        return [self.p.period]

    def __init__(self):
        self.addminperiod(self.p.period)
        self.hma_lo = bt.indicators.SmoothedMovingAverage(self.data.low, period=self.p.period)
        self.hma_hi = bt.indicators.SmoothedMovingAverage(self.data.high, period=self.p.period)

    def next(self):
        hlv = 1 if self.data.close > self.hma_hi[0] else -1
        if hlv == -1:
            self.lines.ssld[0] = self.hma_hi[0]
            self.lines.sslu[0] = self.hma_lo[0]

        elif hlv == 1:
            self.lines.ssld[0] = self.hma_lo[0]
            self.lines.sslu[0] = self.hma_hi[0]