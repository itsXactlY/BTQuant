import backtrader as bt

class WilliamsAlligator(bt.Indicator):
    alias = ('Alligator',)
    lines = ('jaw', 'teeth', 'lips')

    params = (
        ('jaw_period', 13),
        ('teeth_period', 8),
        ('lips_period', 5),
        ('jaw_offset', 8),
        ('teeth_offset', 5),
        ('lips_offset', 3),
    )

    plotinfo = dict(subplot=False)

    def __init__(self):
        # Calculate moving averages with shift applied to the entire indicator
        jaw_sma = bt.indicators.SimpleMovingAverage(self.data, period=self.params.jaw_period, plot=True)
        teeth_sma = bt.indicators.SimpleMovingAverage(self.data, period=self.params.teeth_period, plot=True)
        lips_sma = bt.indicators.SimpleMovingAverage(self.data, period=self.params.lips_period, plot=True)

        self.lines.jaw = jaw_sma(-self.params.jaw_offset)
        self.lines.teeth = teeth_sma(-self.params.teeth_offset)
        self.lines.lips = lips_sma(-self.params.lips_offset)