import backtrader as bt

class QQE(bt.Indicator):
    params = (
        ("period", 6),
        ("fast", 5),
        ("q", 3.0),
        ("debug", False)
    )
    lines = ("qqe_line",)

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.fast)
        self.dar = bt.If(self.atr > 0, bt.indicators.EMA(self.atr - self.p.q, period=int((self.p.period * 2) - 1)), 0)
        self.lines.qqe_line = bt.If(self.rsi > 0, self.rsi + self.dar, 0)

    def next(self):
        # check if ATR is not zero to avoid division by zero errors
        if self.atr[0] == 0:
            print("ATR is zero, skipping this iteration to avoid division by zero.")
            return

        # check if RSI and DAR are valid before computing the QQE line
        if self.rsi[0] != 0 and self.dar[0] != 0:
            self.lines.qqe_line[0] = self.rsi[0] + self.dar[0]
        else:
            self.lines.qqe_line[0] = 0
        
        if self.p.debug:
            print(f"RSI: {self.rsi[0]}, DAR: {self.dar[0]}, ATR: {self.atr[0]}, QQE: {self.lines.qqe_line[0]}")
