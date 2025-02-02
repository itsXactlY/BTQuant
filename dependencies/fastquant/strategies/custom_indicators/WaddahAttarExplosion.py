import backtrader as bt

class WaddahAttarExplosion(bt.Indicator):
    lines = ('macd', 'utrend', 'dtrend', 'exp', 'dead')

    params = (
        ('sensitivity', 150),
        ('fast', 20),
        ('slow', 40),
        ('channel', 20),
        ('mult', 2.0),
        ('dead', 3.7)

    )

    plotlines = dict(macd=dict(_plotskip=True, ),
                     )

    plotinfo = dict(
        plot=True,
        plotname='Waddah Attar Explosion',
        subplot=True,
        plotlinelabels=True)

    def __init__(self):
        # Plot horizontal Line

        self.l.macd = bt.indicators.MACD(self.data,period_me1=self.p.fast,period_me2=self.p.slow).macd
        boll = bt.indicators.BollingerBands(self.data,period=self.p.channel, devfactor=self.p.mult)

        t1 = (self.l.macd(0)-self.l.macd(-1))*self.p.sensitivity
        self.l.exp = boll.top - boll.bot

        self.l.utrend = bt.If(t1 >= 0, t1, 0.0)
        self.l.dtrend = bt.If(t1 < 0, -1.0*t1, 0.0)
        self.l.dead = bt.indicators.AverageTrueRange(self.data,period=50).atr*self.p.dead
