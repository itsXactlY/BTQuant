import backtrader as bt

class ChaikinMoneyFlow(bt.Indicator):
    lines = ('money_flow',)
    params = (
        ('period', 20),
    )

    plotlines = dict(
        money_flow=dict(
            _name='CMF',
            color='green',
            alpha=0.50
        )
    )

    def __init__(self):
        # Let the indicator get enough data
        self.addminperiod(self.p.period)

        # Plot horizontal Line
        self.plotinfo.plotyhlines = [0]

        # Aliases to avoid long lines
        c = self.data.close
        h = self.data.high
        l = self.data.low
        v = self.data.volume

        self.data.ad = bt.If(bt.Or(bt.And(c == h, c == l), h == l), 0, ((2 * c - l - h) / (h - l)) * v)
        self.lines.money_flow = bt.indicators.SumN(self.data.ad, period=self.p.period) / bt.indicators.SumN(
            self.data.volume, period=self.p.period)