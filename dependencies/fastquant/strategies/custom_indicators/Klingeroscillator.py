import backtrader as bt

class KlingerOscillator(bt.Indicator):

    lines = ('sig', 'kvo')

    params = (('fast', 34), ('slow', 55), ('signal', 13))

    plotinfo = dict(
        plot=True,
        plotname='Klinger Oscillator',
        subplot=True,
        plotlinelabels=True)

    def __init__(self):
        self.plotinfo.plotyhlines = [0]
        self.addminperiod(55)

        self.data.hlc3 = (self.data.high + self.data.low + self.data.close) / 3
        # This works - Note indexing should be () rather than []
        # See: https://www.backtrader.com/docu/concepts.html#lines-delayed-indexing
        self.data.sv = bt.If((self.data.hlc3(0) - self.data.hlc3(-1)) / self.data.hlc3(-1) >= 0, self.data.volume,
                             -self.data.volume)
        self.lines.kvo = bt.indicators.EMA(self.data.sv, period=self.p.fast) - bt.indicators.EMA(self.data.sv,
                                                                                                    period=self.p.slow)
        self.lines.sig = bt.indicators.EMA(self.lines.kvo, period=self.p.signal)
