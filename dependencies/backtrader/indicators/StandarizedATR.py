import backtrader as bt

class StandarizedATR(bt.Indicator):

    lines = ('natr',)

    params = (
        ('atr_period', 14),
        ('std_period', 20),
        ('movav', bt.ind.SMA),
    )

    plotinfo = dict(
        plot=True,
        plotname='Standardized ATR',
        subplot=True,
        plotlinelabels=True)

    def __init__(self):

        atr = bt.indicators.ATR(self.data,period=self.p.atr_period)
        satr = self.p.movav(atr,period=self.p.std_period)
        self.stdev = bt.indicators.StandardDeviation(self.data,period=self.p.std_period,movav=self.p.movav)

        self.l.natr = satr/self.stdev