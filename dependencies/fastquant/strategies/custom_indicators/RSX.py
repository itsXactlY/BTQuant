import backtrader as bt
import backtrader.indicators as btind

class Accumulator(bt.Indicator):
    lines = ('acc',)
    params = (('seed', 1),)

    def __init__(self, condition, func):
        self.condition = condition
        self.func = func

    def next(self):
        if len(self) == 1:
            self.lines.acc[0] = self.p.seed
        else:
            if self.condition:
                self.lines.acc[0] = self.func(self.lines.acc[-1])
            else:
                self.lines.acc[0] = self.lines.acc[-1]


class RSX(bt.Indicator):
    lines = ('rsx',)
    params = (('length', 14), ('src', 'close'))

    def __init__(self):
        self.addminperiod(self.p.length)
        self.f8 = 100 * self.data.close
        self.f10 = self.f8(-1)
        self.v8 = self.f8 - self.f10

        self.f18 = 3 / (self.p.length + 2)
        self.f20 = 1 - self.f18

        self.f28 = btind.EMA(self.v8, period=self.p.length)
        self.f30 = btind.EMA(self.f28, period=self.p.length)
        self.vC = self.f28 * 1.5 - self.f30 * 0.5

        self.f38 = btind.EMA(self.vC, period=self.p.length)
        self.f40 = btind.EMA(self.f38, period=self.p.length)
        self.v10 = self.f38 * 1.5 - self.f40 * 0.5

        self.f48 = btind.EMA(self.v10, period=self.p.length)
        self.f50 = btind.EMA(self.f48, period=self.p.length)
        self.v14 = self.f48 * 1.5 - self.f50 * 0.5

        self.f58 = btind.EMA(abs(self.v8), period=self.p.length)
        self.f60 = btind.EMA(self.f58, period=self.p.length)
        self.v18 = self.f58 * 1.5 - self.f60 * 0.5

        self.f68 = btind.EMA(self.v18, period=self.p.length)
        self.f70 = btind.EMA(self.f68, period=self.p.length)
        self.v1C = self.f68 * 1.5 - self.f70 * 0.5

        self.f78 = btind.EMA(self.v1C, period=self.p.length)
        self.f80 = btind.EMA(self.f78, period=self.p.length)
        self.v20 = self.f78 * 1.5 - self.f80 * 0.5

        self.f88 = btind.If(self.f80 == 0, self.p.length - 1, 5)
        self.f90 = Accumulator(condition=self.f88 == self.f88(-1), func=lambda x: x + 1)

        self.v4 = btind.If(bt.And(self.f88 < self.f90, self.v20 > 0), (self.v14 / self.v20 + 1) * 50, 50)
        self.lines.rsx = btind.If(self.v4 > 100, 100, btind.If(self.v4 < 0, 0, self.v4))
