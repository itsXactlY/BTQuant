import backtrader as bt

class SignalFiller(bt.Indicator):

    lines = ('signal',)

    def nexstart(self):
        self.l.signal[0] = 0.0

    def next(self):

        if self.data[0] != 0:
            self.l.signal[0] = self.data[0]
        else:
            self.l.signal[0] = self.l.signal[-1]

class NormalizedVolume(bt.Indicator):

    lines = ('ma','nv')

    params = (
        ('movav', bt.ind.SMA),
        ('period', 5),
    )

    plotinfo = dict(
        plot=True,
        plotname='Normalized Volume',
        subplot=True,
        plotlinelabels=True)

    plotlines = dict(ma=dict(_plotskip=True, ),
                     )

    def __init__(self):

        self.l.ma = self.p.movav(self.data.volume,period=self.p.period)
        #self.l.nv = 100*(self.data.volume/self.l.ma)

    def next(self):

        self.l.nv[0] = 100*self.data.volume[0]/self.l.ma[0]