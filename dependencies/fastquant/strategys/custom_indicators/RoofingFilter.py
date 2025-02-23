import backtrader as bt
from fastquant.strategys.custom_indicators.ElhersHighPass import ElhersHighPass
from fastquant.strategys.custom_indicators.SuperSmoothFilter import SuperSmoothFilter
from fastquant.strategys.custom_indicators.iFisher import iFisher

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