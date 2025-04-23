import backtrader as bt
import numpy as np

class MAMA(bt.Indicator):

    lines = ('p', 'S', 'D', 'mp', 'Q1', 'I1', 'Q2', 'I2', 'jI', 'jQ', 'Re', 'Im', 'phi', 'smoothPeriod',
            'MAMA','FAMA')

    params = (
        ('fast', 20),
        ('slow', 50),
    )

    plotlines = dict(p=dict(_plotskip=True, ),
                    S=dict(_plotskip=True, ),
                    D=dict(_plotskip=True, ),
                    mp=dict(_plotskip=True, ),
                    Q1=dict(_plotskip=True, ),
                    I1=dict(_plotskip=True, ),
                    Q2=dict(_plotskip=True, ),
                    I2=dict(_plotskip=True, ),
                    jI=dict(_plotskip=True, ),
                    jQ=dict(_plotskip=True, ),
                    Re=dict(_plotskip=True, ),
                    Im=dict(_plotskip=True, ),
                    phi=dict(_plotskip=True, ),
                    smoothPeriod=dict(_plotskip=True, )
                    )

    plotinfo = dict(
        plot=True,
        plotname='Mesa Adaptive Moving Average',
        subplot=False,
        plotlinelabels=True)

    def hilbertTransform(self,data):
        a, b, c, d = [0.0962, 0.5769, 0.075, 0.54]
        d0 = data[0]
        d2 = data[-2]
        d4 = data[-4]
        d6 = data[-6]
        hilbert = a*d0 + b*d2 - b*d4 - a*d6
        return hilbert*(c*self.l.mp[0] + d)

    def smoother(self,data):
        return 0.2*data[0] + 0.8*data[-1]

    def deg(self,arg):
        return np.rad2deg(arg)

    def __init__(self):
        self.fast = 2/(self.p.fast+1)
        self.slow = 2/(self.p.slow+1)
        self.addminperiod(40)
        self.l.p = (self.data.high + self.data.low)/2

    def prenext(self):
        # Variable Initialization
        self.l.Q1[0] = 0.0
        self.l.Q2[0] = 0.0
        self.l.I1[0] = 0.0
        self.l.I2[0] = 0.0
        self.l.jQ[0] = 0.0
        self.l.jI[0] = 0.0
        self.l.Im[0] = 0.0
        self.l.Re[0] = 0.0
        self.l.phi[0] = 0.0
        self.l.smoothPeriod[0] = 0.0
        self.l.mp[0] = 1.0
        self.l.MAMA[0] = 0.0
        self.l.FAMA[0] = 0.0
        self.l.S[0] = (4 * self.l.p[0] + 3 * self.l.p[-1] + 2 * self.l.p[-2] + 3 * self.l.p[-3])/10
        self.l.D[0] = self.hilbertTransform(self.l.S)

    def next(self):
        self.l.mp[0] = 0.0

        self.l.S[0] = (4*self.l.p[0] + 3*self.l.p[-1] + 2*self.l.p[-2]+ 3*self.l.p[-3])/10

        # Smooth and Detrend
        self.l.D[0] = self.hilbertTransform(self.l.S)

        # Inphase and Quadrature
        self.l.Q1[0] = self.hilbertTransform(self.l.D)
        self.l.I1[0] = self.l.D[-3]

        # Phase Advancement of Inphase and Quadrature by 90 degrees
        self.l.jI[0] = self.hilbertTransform(self.l.I1)
        self.l.jQ[0] = self.hilbertTransform(self.l.Q1)

        # Phasor Addition - 3 Bar Averaging
        self.l.I2[0] = self.l.I1[0] - self.l.jQ[0]
        self.l.Q2[0] = self.l.Q1[0] - self.l.jI[0]

        # Smoothing of I and Q
        self.l.I2[0] = self.smoother(self.l.I2)
        self.l.Q2[0] = self.smoother(self.l.Q2)

        # Hemodyne Discriminator
        self.l.Re[0] = self.l.I2[0]*self.l.I2[-1] + self.l.Q2[0]*self.l.Q2[-1]
        self.l.Im[0] = self.l.I2[0]*self.l.Q2[-1] + self.l.Q2[0]*self.l.I2[-1]

        self.l.Re[0] = self.smoother(self.l.Re)
        self.l.Im[0] = self.smoother(self.l.Im)

        if self.l.Im[0] != 0.0 and self.l.Re[0] != 0.0:
            self.l.mp[0] = 360/(self.deg(np.arctan(self.l.Im[0]/self.l.Re[0])))
        if self.l.mp[0] > 1.5*self.l.mp[-1]:
            self.l.mp[0] = 1.5*self.l.mp[-1]
        if self.l.mp[0] < (2.0/3)*self.l.mp[-1]:
            self.l.mp[0] = (2.0/3)*self.l.mp[-1]
        if self.l.mp[0] < 6:
            self.l.mp[0] = 6.0
        if self.l.mp[0] > 50:
            self.l.mp[0] = 50.0
        self.l.mp[0] = self.smoother(self.l.mp)
        self.l.smoothPeriod[0] = (1.0/3)*self.l.mp[0] + (2.0/3)*self.l.smoothPeriod[-1]

        if self.l.I1[0] != 0.0:
            self.l.phi[0] = self.deg(np.arctan(self.l.Q1[0]/self.l.I1[0]))
        dphi = self.l.phi[-1] - self.l.phi[0]

        if dphi < 1:
            dphi = 1.0

        alpha = self.p.fast/dphi

        if alpha < self.slow:
            alpha = self.slow
        if alpha > self.fast:
            alpha = self.fast

        self.l.MAMA[0] = alpha*self.l.p[0] + (1-alpha)*self.l.MAMA[-1]
        self.l.FAMA[0] = 0.5*alpha*self.l.MAMA[0] + (1-0.5*alpha)*self.l.FAMA[-1]
