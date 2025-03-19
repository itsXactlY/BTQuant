import backtrader as bt
import numpy as np

class SuperSmoothFilter(bt.Indicator):

    lines = ('filter',)

    params = (
        ('period', 10),
    )

    plotinfo = dict(
        plot=True,
        plotname='Super Smooth Filter',
        subplot=True,
        plotlinelabels=True)

    def __init__(self):
        self.addminperiod(10)

    def prenext(self):
        self.l.filter[0] = (self.data[0]+self.data[-1])/2

    def next(self):

        a1 = np.exp(-1.414*3.14159/self.p.period)
        b1 = 2*a1*np.cos(np.deg2rad(1.414*180/self.p.period))
        c2 = b1
        c3 = -a1*a1
        c1 = 1 - c2 - c3
        self.l.filter[0] = c1*(self.data[0]+self.data[-1])/2 + c2*self.l.filter[-1] + c3*self.l.filter[-2]
