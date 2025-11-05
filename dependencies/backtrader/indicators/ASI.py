import backtrader as bt

class AccumulativeSwingIndex(bt.Indicator):
    lines = ('asi',)
    params = (
        ('period', 14),  # Period for the ASI calculation
    )

    def __init__(self):
        super(AccumulativeSwingIndex, self).__init__()

        # Calculate ASI using a custom function
        self.addminperiod(self.params.period)
        self.lines.asi = bt.indicators.WMA(
            self.datas[0].close - self.datas[0].open,
            period=self.params.period
            ) + bt.indicators.SumN(
                self.datas[0].high - self.datas[0].low,
                period=self.params.period
            )
