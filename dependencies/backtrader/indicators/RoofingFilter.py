import backtrader as bt
<<<<<<< HEAD
from .ElhersHighPass import ElhersHighPass
from .SuperSmoothFilter import SuperSmoothFilter
from .iFisher import iFisher
=======
from fastquant.strategies.custom_indicators.ElhersHighPass import ElhersHighPass
from fastquant.strategies.custom_indicators.SuperSmoothFilter import SuperSmoothFilter
from fastquant.strategies.custom_indicators.iFisher import iFisher
>>>>>>> ralphy/agent-1-1769660237987-g2tgw2-complete-vulkan-initialization-sequence-documented

class RoofingFilter(bt.Indicator):

    lines = ('roof','iroof')

    params = (
        ('hp_period', 48),
        ('ss_period', 10),
        ('smooth', 2)
    )

    plotinfo = dict(
        plot=True,
        plotname='Elhers Roofing Filter',
        subplot=True,
        plotlinelabels=True)

    def __init__(self):
        self.addminperiod(10)

        hp = ElhersHighPass(self.data,period=self.p.hp_period)
        self.l.roof = SuperSmoothFilter(hp,period=self.p.ss_period)
        self.l.iroof = iFisher(self.l.roof, smoothing=self.p.smooth)