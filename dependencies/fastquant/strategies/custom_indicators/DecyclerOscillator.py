import backtrader as bt
import numpy as np

class DecyclerOscillator(bt.Indicator):

    lines = ('osc','decycle', 'hp')

    params = (
        ('hp_period', 48),
    )

    plotlines = dict(decycle=dict(_plotskip=True, ),
                     hp=dict(_plotskip=True, ),
                     )

    plotinfo = dict(
        plot=True,
        plotname='Elhers Decycler Oscillator',
        subplot=True,
        plotlinelabels=True)

    def filter(self,a,set1,set2):

        filt = ((1-a)**2)*(set1[0]-2*set1[-1]+set1[-2])+2*(1-a)*set2[-1] - ((1-a)**2)*set2[-2]
        return filt

    def __init__(self):
        self.addminperiod(20)
        self.b = np.deg2rad(0.707*360/self.p.hp_period)
        self.a1 = (np.cos(self.b)+np.sin(self.b)-1)/np.cos(self.b)
        self.a2 = (np.cos(self.b/2)+np.sin(self.b/2)-1)/np.cos(self.b)

    def prenext(self):

        self.l.hp[0] = 0.0
        self.l.osc[0] = 0.0
        self.l.decycle[0] = self.data.close[-1]

    def next(self):

        self.l.hp[0] = self.filter(self.a1, self.data.close, self.l.hp)
        self.l.decycle[0] = self.data.close[0] - self.l.hp[0]

        self.l.osc[0] = self.filter(self.a2, self.l.decycle, self.l.osc)