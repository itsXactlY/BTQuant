import backtrader as bt

class AccumulativeSwingIndex(bt.Indicator):
    lines = ('asi',)
    params = (
        ('period', 14),
    )

    def __init__(self):
        super(AccumulativeSwingIndex, self).__init__()
        self.addminperiod(self.params.period)
        self.lines.asi = bt.indicators.WMA(
        self.data.close - self.data.open,
        period=self.params.period
        ) + bt.indicators.SumN(
        self.data.high - self.data.low,
        period=self.params.period
        )