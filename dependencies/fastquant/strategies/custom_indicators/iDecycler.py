import backtrader as bt
from custom_indicators.DecyclerOscillator import DecyclerOscillator
from custom_indicators.iFisher import iFisher

class iDecycler(bt.Indicator):

    lines = ('idosc',)

    params = (
        ('hp_period', 48),
        ('smooth', 2)
    )

    plotlines = dict(decycle=dict(_plotskip=True, ),
                     hp=dict(_plotskip=True, ),
                     )

    plotinfo = dict(
        plot=True,
        plotname='iDecycler',
        subplot=True,
        plotlinelabels=True)

    def __init__(self):

        osc = DecyclerOscillator(self.data,hp_period=self.p.hp_period)
        self.l.idosc = iFisher(osc.osc,smoothing=self.p.smooth)