import backtrader as bt

class MADRIndicator(bt.Indicator):
    params = (
        ("window", 14),
        ("atr_window", 14),
        ("atr_multiplier", 0.45),
    )

    lines = ("madr", "buy_signal", "sell_signal")

    def __init__(self):
        rolling_mean = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.window)
        deviation = self.data.close - rolling_mean
        atr_value = bt.indicators.ATR(self.data, period=self.params.atr_window)

        self.lines.madr = deviation / rolling_mean
        self.lines.buy_signal = (self.lines.madr < -self.params.atr_multiplier * atr_value)
        self.lines.sell_signal = (self.lines.madr > self.params.atr_multiplier * atr_value * 0.95)
