import backtrader as bt

class HeikenAshi(bt.Indicator):

    lines=('open','high','low','close','signal')

    plotlines = dict(open=dict(_plotskip=True, ),
                     high=dict(_plotskip=True,),
                     low=dict(_plotskip=True,),
                     close=dict(_plotskip=True,))
    plotinfo = dict(
        plot=True,
        plotname='Hieken Ashi Candles',
        subplot=False,
        plotlinelabels=True)


    def __init__(self):
        self.addminperiod=2

    def next(self):
        self.l.open[0] = o = (self.data.open[-1] + self.data.close[-1])/2.0
        self.l.close[0] = c = (self.data.open[0]+self.data.close[0]+self.data.high[0]+self.data.close[0])/4
        self.l.high[0] = max(self.data.high[0], self.data.open[0], self.data.close[0])
        self.l.low[0] = min(self.data.low[0], self.data.open[0], self.data.close[0])
        self.l.signal[0] = -1.0 if c < o else 1.0
