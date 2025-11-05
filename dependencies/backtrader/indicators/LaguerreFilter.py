import backtrader as bt

class LaguerreFilter(bt.Indicator):

    lines = ('filter', 'p', 'L0', 'L1', 'L2', 'L3')

    params = (
        ('period', 48),
    )

    plotinfo = dict(
        plot=True,
        plotname='Laguerre Filter',
        subplot=False,
        plotlinelabels=True)

    plotlines = dict(p=dict(_plotskip=True, ),
                     L0=dict(_plotskip=True, ),
                     L1=dict(_plotskip=True, ),
                     L2=dict(_plotskip=True, ),
                     L3=dict(_plotskip=True, ),
                     )

    def __init__(self):
        self.addminperiod(self.p.period)
        self.alpha = 2/(self.p.period+1)
        self.l.p = (self.data.high + self.data.low)/2

    def prenext(self):
        price = self.l.p[0]
        self.l.L0[0] = price
        self.l.L1[0] = price
        self.l.L2[0] = price
        self.l.L3[0] = price
        self.l.filter[0] = price


    def next(self):
        a = self.alpha
        p = self.l.p
        self.l.L0[0] = a*p[0] + (1-a)*self.l.L0[-1]
        self.l.L1[0] = -(1 - a) * self.l.L0[0] + self.l.L0[-1] + (1 - a) * self.l.L1[-1]
        self.l.L2[0] = -(1 - a) * self.l.L1[0] + self.l.L1[-1] + (1 - a) * self.l.L2[-1]
        self.l.L3[0] = -(1 - a) * self.l.L2[0] + self.l.L2[-1] + (1 - a) * self.l.L3[-1]
        self.l.filter[0] = (self.l.L0[0] + 2*self.l.L1[0] + 2*self.l.L2[0] + self.l.L3[0])/6
