import backtrader as bt

class ModifiedMADR(bt.Indicator):
    lines = ('madr', 'buy_signal', 'sell_signal')
    params = (('window', 42), ('threshold_multiplier', 7))

    def __init__(self):
        self.addminperiod(self.params.window)
        self.src = self.data.high
        self.rolling_mean = bt.indicators.ExponentialMovingAverage(self.src, period=self.params.window)
        self.deviation = self.src - self.rolling_mean
        self.madr = self.deviation / self.rolling_mean
        self.madr_smoothed = bt.indicators.ExponentialMovingAverage(self.madr, period=self.params.window // 2)
        self.dynamic_threshold = bt.indicators.StandardDeviation(self.madr_smoothed, period=self.params.window) * self.params.threshold_multiplier

    def next(self):
        if len(self) >= self.params.window:
            self.lines.madr[0] = self.madr_smoothed[0]
            self.lines.buy_signal[0] = int(self.madr_smoothed[0] < -self.dynamic_threshold[0])
            self.lines.sell_signal[0] = int(self.madr_smoothed[0] > self.dynamic_threshold[0])