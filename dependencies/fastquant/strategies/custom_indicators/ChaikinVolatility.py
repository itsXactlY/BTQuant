import backtrader as bt

class ChaikinVolatility(bt.Indicator):
    params=dict(ema_period=10,
                roc_period=10)
    lines=('cvi',)
    plotinfo = dict(
        plot=True,
        plotname='Chaikin Volatility Index',
        subplot=True,
        plotlinelabels=True)


    def __init__(self):
        price = self.data.high - self.data.low
        ema = bt.indicators.EMA(price,period=self.p.ema_period)
        self.l.cvi = bt.indicators.RateOfChange(ema.ema,period=self.p.roc_period)
