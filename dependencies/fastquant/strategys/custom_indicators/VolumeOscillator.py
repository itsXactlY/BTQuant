import backtrader as bt

class VolumeOscillator(bt.Indicator):
    lines = ('osc',)
    params = (('shortlen', 5),
            ('longlen', 10))
    
    def __init__(self):
        shortlen, longlen = self.params.shortlen, self.params.longlen
        self.lines.short = bt.indicators.ExponentialMovingAverage(self.data.volume, period=shortlen)
        self.lines.long = bt.indicators.ExponentialMovingAverage(self.data.volume, period=longlen)
        self.lines.osc = (self.lines.short - self.lines.long) / self.lines.long * 100

    def next(self):
        self.osc[0] = (self.lines.short[0] - self.lines.long[0]) / self.lines.long[0] * 100