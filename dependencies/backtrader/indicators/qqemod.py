import backtrader as bt

class QQE(bt.Indicator):
    lines = ('qqe_line', 'rsi_ma', 'long_band', 'short_band', 'trend', 'qqe_oscillator')
    params = (
        ('rsi_period', 6),
        ('sf', 5),
        ('qqe', 3.0),
        ('threshold', 3),
        ('wi_period', 14),  # Wilders Period
    )

    plotlines = dict(
        qqe_oscillator=dict(_plotskip=False),
        long_band=dict(_plotskip=False),
        short_band=dict(_plotskip=False),
        trend=dict(_plotskip=False),
        qqe_line=dict(_plotskip=False),
        rsi_ma=dict(_plotskip=False)
    )

    plotinfo = dict(plot=True, subplot=True)

    def __init__(self):
        # RSI
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.rsi_ma = bt.indicators.EMA(self.rsi, period=self.p.sf)

        # ATR RSI
        self.atr_rsi = abs(self.rsi_ma(-1) - self.rsi_ma)
        self.ma_atr_rsi = bt.indicators.EMA(self.atr_rsi, period=self.p.wi_period)
        self.dar = bt.indicators.EMA(self.ma_atr_rsi, period=self.p.wi_period) * self.p.qqe

        # Bands
        self.long_band = bt.indicators.Max((self.rsi_ma - self.dar), (self.long_band(-1)))
        self.short_band = bt.indicators.Min((self.rsi_ma + self.dar), (self.short_band(-1)))

        # Crossover indicators
        self.cross_long = bt.indicators.CrossOver(self.rsi_ma, self.long_band)
        self.cross_short = bt.indicators.CrossOver(self.rsi_ma, self.short_band)

        # Zero line
        self.plotlines.qqe_oscillator._plotvalue = 50
        self.plotlines.qqe_oscillator._plotmaster = self.data

    def next(self):
        if self.rsi_ma[0] > self.long_band[-1] and self.rsi_ma[-1] > self.long_band[-1]:
            self.long_band[0] = max(self.long_band[-1], self.rsi_ma[0] - self.dar[0])
        else:
            self.long_band[0] = self.rsi_ma[0] - self.dar[0]

        if self.rsi_ma[0] < self.short_band[-1] and self.rsi_ma[-1] < self.short_band[-1]:
            self.short_band[0] = min(self.short_band[-1], self.rsi_ma[0] + self.dar[0])
        else:
            self.short_band[0] = self.rsi_ma[0] + self.dar[0]

        # Update trend based on crossovers
        if self.cross_short[0] == 1:
            self.trend[0] = 1
        elif self.cross_long[0] == -1:
            self.trend[0] = -1
        else:
            self.trend[0] = self.trend[-1]

        # Update QQE Line
        self.qqe_line[0] = self.long_band[0] if self.trend[0] == 1 else self.short_band[0]

        # Calculate QQE Oscillator
        self.qqe_oscillator[0] = self.rsi_ma[0] - 50