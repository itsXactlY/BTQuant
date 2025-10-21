import backtrader as bt
import numpy as np
class AdaptiveLaguerreFilter(bt.Indicator):

    lines = ('filter', 'p', 'L0', 'L1', 'L2', 'L3')

    params = (
        ('length', 20),
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
        self.addminperiod(60)
        self.l.p = (self.data.high + self.data.low) / 2

    def prenext(self):
        self.l.filter[0] = self.l.p[0]
        self.l.L0[0] = self.l.p[0]
        
        # Only access previous bars if they exist
        if len(self) >= 2:
            self.l.L1[0] = self.l.p[-1]
        else:
            self.l.L1[0] = self.l.p[0]
            
        if len(self) >= 3:
            self.l.L2[0] = self.l.p[-2]
            self.l.L3[0] = self.l.p[-2]  # This also needs fixing
        else:
            self.l.L2[0] = self.l.p[0]
            self.l.L3[0] = self.l.p[0]

    def next(self):
        p = self.l.p
        diff = [abs(self.l.p[-i]-self.l.filter[-(i+1)]) for i in range(self.p.length)]
        HH = diff[0]
        LL = diff[0]
        for i in range(self.p.length):
            if diff[i] > HH:
                HH = diff[i]
            if diff[i] < LL:
                LL = diff[i]
        data = [(i - LL)/(HH - LL) for i in diff]
        if HH - LL != 0.0:
            a = np.median(data)

        self.l.L0[0] = a*p[0] + (1-a)*self.l.L0[-1]
        self.l.L1[0] = -(1 - a) * self.l.L0[0] + self.l.L0[-1] + (1 - a) * self.l.L1[-1]
        self.l.L2[0] = -(1 - a) * self.l.L1[0] + self.l.L1[-1] + (1 - a) * self.l.L2[-1]
        self.l.L3[0] = -(1 - a) * self.l.L2[0] + self.l.L2[-1] + (1 - a) * self.l.L3[-1]
        self.l.filter[0] = (self.l.L0[0] + 2*self.l.L1[0] + 2*self.l.L2[0] + self.l.L3[0])/6
