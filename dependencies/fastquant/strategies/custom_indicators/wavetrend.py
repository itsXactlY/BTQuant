import backtrader as bt

class WaveTrend(bt.Indicator):
    lines = ('WT1', 'WT2',)
    params = (
        ('period', 10),
    )

    def __init__(self):
        HL2 = (self.data.high + self.data.low) / 2
        EMA1 = bt.ind.EMA(HL2, period=self.params.period)
        EMA2 = bt.ind.EMA(EMA1, period=self.params.period)
        D = EMA1 - EMA2
        self.lines.WT1 = bt.ind.EMA(D, period=self.params.period)
        self.lines.WT2 = bt.ind.EMA(self.lines.WT1, period=4)