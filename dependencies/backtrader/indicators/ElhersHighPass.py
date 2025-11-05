import backtrader as bt
import numpy as np

class ElhersHighPass(bt.Indicator):

    lines = ('hp',)

    params = (
        ('period',48),
    )

    plotinfo = dict(
        plot=True,
        plotname='Elhers High Pass',
        subplot=True,
        plotlinelabels=True)

    def __init__(self):
        self.addminperiod(10)

    def deg(self,arg):
        return np.deg2rad(arg)

    def prenext(self):
        c = self.data[0]
        c1 = self.data[-1]
        c2 = self.data[-2]
        a1 = (np.cos(self.deg(.707*360/self.p.period)) + np.sin(self.deg(.707*360/self.p.period))-1)/np.cos(self.deg(.707*360/self.p.period))
        self.l.hp[0] = ((1 - a1/2)**2)*(c - 2*c1 + c2)

    def next(self):
        c = self.data[0]      # Changed from self.data.close[0]
        c1 = self.data[-1]    # Changed from self.data.close[-1]
        c2 = self.data[-2]    # Changed from self.data.close[-2]
        a1 = (np.cos(self.deg(.707*360/self.p.period)) + np.sin(self.deg(.707*360/self.p.period))-1)/np.cos(self.deg(.707*360/self.p.period))
        self.l.hp[0] = ((1 - a1/2)**2)*(c - 2*c1 + c2) + 2*(1-a1)*self.l.hp[-1] - ((1-a1)**2)*self.l.hp[-2]
