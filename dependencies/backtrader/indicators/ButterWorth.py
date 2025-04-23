import backtrader as bt
import numpy as np

class Butterworth(bt.Indicator):

    lines = ('butter', 'p')

    params = (
        ('period', 48),
        ('poles', 2)
    )

    plotlines = dict(p=dict(_plotskip=True, ),
                     )

    plotinfo = dict(
        plot=True,
        plotname='Butterworth Filter',
        subplot=False,
        plotlinelabels=True)

    def __init__(self):

        self.addminperiod(10)

        if self.p.poles == 2:
            self.a1 = np.exp(-1.414*3.14159/self.p.period)
            self.b1 = 2*self.a1*np.cos(np.deg2rad(1.414*180/self.p.period))
            self.c2 = self.b1
            self.c3 = -self.a1**2
            self.c1 = (1-self.b1+self.a1**2)/4

        elif self.p.poles == 3:
            self.a1 = np.exp(-3.14159 / self.p.period)
            self.b1 = 2 * self.a1 * np.cos(np.deg2rad(1.738 * 180 / self.p.period))
            self.c1 = self.a1 ** 2
            self.c2 = self.b1 + self.c1
            self.c3 = -(self.c1 + self.b1 * self.c1)
            self.c4 = self.c1 ** 2
            self.c1 = (1 - self.b1 + self.c1) * (1 - self.c1) / 8

        else:
            raise ValueError()

        self.l.p = (self.data.high + self.data.low)/2

    def prenext(self):

        self.l.butter[0] = self.l.p[0]

    def next(self):
        p = self.l.p
        if self.p.poles == 2:
            self.l.butter[0] = self.c1*(p[0]+2*p[-1]+p[-2]) + \
                                self.c2*self.l.butter[-1] + \
                                self.c3*self.l.butter[-2]
        elif self.p.poles == 3:
            self.l.butter[0] = self.c1 * (p[0] + 3 * p[-1] + 3 * p[-2] + p[-3]) + \
                                self.c2 * self.l.butter[-1] + \
                                self.c3 * self.l.butter[-2] + \
                                self.c4 * self.l.butter[-3]