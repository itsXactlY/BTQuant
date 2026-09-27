import logging, sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
from autonomous_agency.strategy_factory import StrategyFactory
sf = StrategyFactory.__new__(StrategyFactory)
import logging as _l
sf.logger = _l.getLogger("smoke")
bad = open('autonomous_agency/strategies/RSI_Dip_Reversion_Probe_20260927_075438.py').read()
print("BAD  ->", sf._smoke_run_generated_code(bad))
good = '''
import backtrader as bt
from backtrader.strategies.base import BaseStrategy, bt

class GoodProbe(BaseStrategy):
    params = (("rsi_period", 14),)
    def __init__(self):
        super().__init__()
        self.rsi = bt.ind.RSI(period=self.p.rsi_period)
    def buy_or_short_condition(self):
        return self.rsi < 30
    def exit_condition(self):
        return self.rsi > 60
'''
print("GOOD ->", sf._smoke_run_generated_code(good))
