import backtrader as bt

class iTrend(bt.indicators.PeriodN):

    lines = ('itrend','trigger',)

    params = (('period', 20),)

    plotinfo = dict(
        plot=True,
        plotname='iTrend',
        subplot=False,
        plotlinelabels=True)

    def __init__(self):
        self.alpha = 2.0/(1+self.p.period)
        self.addminperiod(self.p.period)

    def prenext(self):
        self.l.itrend[0] = (self.data[0] + 2*self.data[-1] + self.data[-2])/4

    def next(self):

        it1 = self.lines.itrend[-1]
        it2 = self.lines.itrend[-2]
        p = self.data[0]
        p1 = self.data[-1]
        p2 = self.data[-2]
        a = self.alpha

        self.lines.itrend[0] = (a - (a/2)**2)*p + (a**2/2)*p1 - (a - 3*a**2/4)*p2 \
            + 2*(1-a)*it1 - ((1-a)**2)*it2

        self.lines.trigger[0] = 2*self.l.itrend[0] - self.l.itrend[-2]

class iTrend(bt.indicators.PeriodN):

    lines = ('itrend','trigger',)

    params = (('period', 20),)

    plotinfo = dict(
        plot=True,
        plotname='iTrend',
        subplot=False,
        plotlinelabels=True)

    def __init__(self):
        self.alpha = 2.0/(1+self.p.period)
        self.addminperiod(self.p.period)

    def prenext(self):
        self.l.itrend[0] = (self.data[0] + 2*self.data[-1] + self.data[-2])/4

    def next(self):

        it1 = self.lines.itrend[-1]
        it2 = self.lines.itrend[-2]
        p = self.data[0]
        p1 = self.data[-1]
        p2 = self.data[-2]
        a = self.alpha

        self.lines.itrend[0] = (a - (a/2)**2)*p + (a**2/2)*p1 - (a - 3*a**2/4)*p2 \
            + 2*(1-a)*it1 - ((1-a)**2)*it2

        self.lines.trigger[0] = 2*self.l.itrend[0] - self.l.itrend[-2]