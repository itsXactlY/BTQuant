import backtrader as bt

class CyberCycle(bt.indicators.PeriodN):

    lines = ('cycle',
             'smooth',
             'trigger',
             )

    plotlines = dict(smooth=dict(_plotskip=True, ))

    plotinfo = dict(
        plot=True,
        plotname='Cyber Cycle',
        subplot=True,
        plotlinelabels=True)

    params = (('period',30),
              )

    def __init__(self):

        self.addminperiod(self.p.period)
        self.alpha = 2.0/(1.0 + self.p.period)

    def prenext(self):

        self.l.cycle[0] = (self.data[0] - 2*self.data[-1] + self.data[-2])/4
        self.l.smooth[0] = (self.data[0] + 2*self.data[-1] \
                            + 2*self.data[-2] + self.data[-3])/6

    def next(self):
        a = self.alpha
        self.l.smooth[0] = (self.data[0] + 2*self.data[-1] \
                            + 2*self.data[-2] + self.data[-3])/6

        self.l.cycle[0] = ((1 - 0.5*a)**2)*(self.l.smooth[0] - 2*self.l.smooth[-1] + self.l.smooth[-2]) \
            + 2*(1-a)*self.l.cycle[-1] - ((1-a)**2)*self.l.cycle[-2]
        self.l.trigger[0] = self.l.cycle[-1]