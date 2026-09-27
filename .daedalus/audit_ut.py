
import sys; sys.path.insert(0, "/home/alca/projects/PubBTQuant")
from autonomous_agency.strategy_factory import _find_uninitialised_attrs as f

hollow = """
class S:
    def __init__(self, params):
        self.period = params.get("period", 20)
    def next(self):
        macd = self.ema(self.data.close, 12)
        return (self.data.close > self.bb_lower)
"""
good = """
class S:
    def __init__(self, params):
        self.period = params.get("period", 20)
        self.ema = bt.indicators.EMA(self.data.close, period=self.period)
        self.bb_lower = self.indicators.BollingerBandsLower(period=20)
    def next(self):
        if self.data.close > self.bb_lower:
            return self.ema > 0
        return False
"""
print("HOLLOW ->", f(hollow))
print("GOOD   ->", f(good))
