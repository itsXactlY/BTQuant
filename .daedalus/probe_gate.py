"""Does the strengthened gate reject a permanently dead strategy?

BAD  = the real MACD file from the library: rsi[0] > sma[0], never true.
GOOD = a hand-written dual-SMA crossover, same shape, same API.
The gate must reject BAD with NoOrderPlaced and pass GOOD.
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/alca/projects/PubBTQuant")

from autonomous_agency.strategy_factory import StrategyFactory

BAD = Path("autonomous_agency/strategies/MACD_Momentum_Reversal_Strategy_20260927_071323.py")

GOOD = '''
import backtrader as bt
from backtrader.strategies.base import BaseStrategy


class GoodDualSma(BaseStrategy):
    params = (("fast", 10), ("slow", 30), ("size", 0.95),)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fast = bt.ind.SMA(period=self.p.fast)
        self.slow = bt.ind.SMA(period=self.p.slow)

    def next(self):
        if len(self.data) < self.p.slow + 2:
            return
        if not self.position:
            if self.fast[0] > self.slow[0] and self.fast[-1] <= self.slow[-1]:
                self.buy(size=self.p.size * self.broker.getvalue() / self.data.close[0])
        elif self.fast[0] < self.slow[0]:
            self.close()
'''

NOBARS = '''
import backtrader as bt
from backtrader.strategies.base import BaseStrategy


class NoNextAtAll(BaseStrategy):
    params = ()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ema = bt.ind.EMA(period=20)
        self.rsi = bt.ind.RSI(period=14)

    def helper(self):
        return self.rsi[0] > self.ema[0]
'''

BOOMS = '''
import backtrader as bt
from backtrader.strategies.base import BaseStrategy


class BadKwargs(BaseStrategy):
    params = ()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rsi = bt.ind.RSI(period=14, _ma="ema")

    def next(self):
        if self.rsi[0] < 30:
            self.buy()
'''


def main():
    f = StrategyFactory()
    cases = [("BAD  dead RSI>SMA", BAD.read_text()),
             ("GOOD dual SMA", GOOD),
             ("NO   no next()", NOBARS),
             ("BOOM bad kwarg", BOOMS)]
    for label, code in cases:
        err = f._smoke_run_generated_code(code)
        verdict = "REJECTED" if err else "passed"
        print(f"{label:20} {verdict:9} {(err or '')[:110]}")


if __name__ == "__main__":
    main()
