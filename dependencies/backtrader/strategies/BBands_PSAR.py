from .base import BaseStrategy, bt

class BBands_PSAR(BaseStrategy):
    params = (('period', 20), ('bb_mult', 2.0), ('psar_af', 0.02), ('psar_afmax', 0.2),)

    def __init__(self):
        super().__init__()
        self.bb = bt.indicators.BollingerBands(self.data.close, period=self.p.period, devfactor=self.p.bb_mult)
        self.psar = bt.indicators.ParabolicSAR(af=self.p.psar_af, afmax=self.p.psar_afmax)

    def buy_or_short_condition(self):
        # LONG entry: price ≤ lower BB (oversold) AND PSAR below price (bullish)
        if self.data.close[0] <= self.bb.bot[0] and self.psar.psar[0] < self.data.close[0]:
            self.create_order(action='BUY')
            return True
        # SHORT entry: price ≥ upper BB (overbought) AND PSAR above price (bearish)
        if self.data.close[0] >= self.bb.top[0] and self.psar.psar[0] > self.data.close[0]:
            self.create_order(action='SELL')
            return True
        return False

    def sell_or_cover_condition(self):
        if not self.in_position:
            return False
        # Exit LONG: price ≥ upper BB or PSAR above price
        if self.buy_executed and (self.data.close[0] >= self.bb.top[0] or self.psar.psar[0] >= self.data.close[0]):
            self.create_order(action='SELL')
            return True
        # Exit SHORT: price ≤ lower BB or PSAR below price
        if self.short_executed and (self.data.close[0] <= self.bb.bot[0] or self.psar.psar[0] <= self.data.close[0]):
            self.create_order(action='BUY')
            return True
        return False