import backtrader as bt

class FibonacciLevels(bt.Indicator):
    lines = ('fib23', 'fib38', 'fib50', 'fib61', 'fib78')
    params = (('period', 14),)

    def __init__(self):
        self.addminperiod(self.p.period)
        self.highest = bt.indicators.Highest(self.data.high, period=self.p.period)
        self.lowest = bt.indicators.Lowest(self.data.low, period=self.p.period)

    def next(self):
        range = self.highest[0] - self.lowest[0]
        self.lines.fib23[0] = self.highest[0] - 0.236 * range
        self.lines.fib38[0] = self.highest[0] - 0.382 * range
        self.lines.fib50[0] = self.highest[0] - 0.5 * range
        self.lines.fib61[0] = self.highest[0] - 0.618 * range
        self.lines.fib78[0] = self.highest[0] - 0.786 * range