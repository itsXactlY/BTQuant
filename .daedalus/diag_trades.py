"""Diagnostic: why is win_rate ~1% and are trades == bars?"""
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING)

import backtrader as bt
import pandas as pd
from autonomous_agency.strategy_factory import StrategyFactory
from autonomous_agency.ai_interface import StrategyHypothesis

PARQUET = (
    "/home/alca/projects/PubBTQuant/.btq_cache/BTC_1m_USDT_2ee8fdb96a31.parquet"
)

df = pd.read_parquet(PARQUET)
print("=== frame ===")
print("rows (bars):", len(df))
print(df.head(3))
print(df.tail(2))

h = StrategyHypothesis(
    id="diag-001",
    name="Bollinger Band Mean Reversion on Volatility Compression",
    description="Fade the edge of a squeeze when volatility expands",
    indicators=["rsi", "atr"],
    entry_conditions=["condition1"],
    exit_conditions=["condition2"],
    parameters={"fast": 10, "slow": 30},
    rationale="diagnostic",
    mathematical_beauty_score=0.5,
    expected_regime="trending",
    risk_profile="moderate",
)
strat = StrategyFactory().generate_strategy(h)

frames = bt.feeds.PandasData(dataname=df, datetime=None, open=-1, high=-2, low=-3,
                             close=-4, volume=-5, openinterest=-1)


class Probe(bt.Strategy):
    def __init__(self):
        self.p = strat.parameters()
        for k, v in self.p.items():
            setattr(self.p, k, v)
        strat._strategy_name = "probe"
        self.__dict__.update(
            {k: v for k, v in vars(self).items() if k != "p"}
        )
        self._init_from(strat)
        # count bar-level actions
        self.n_buy = 0
        self.n_sell = 0


# simpler: just use the real strategy and attach a spy
spy = {"buys": 0, "sells": 0, "closes": 0}
orig_create = strat.create_order
orig_close = strat.close_all


def create(action=None, *a, **k):
    if action == "BUY":
        spy["buys"] += 1
    elif action == "SELL":
        spy["sells"] += 1
    return orig_create(action, *a, **k)


def close_all(*a, **k):
    spy["closes"] += 1
    return orig_close(*a, **k)


strat.create_order = create
strat.close_all = close_all

cerebro = bt.Cerebro()
cerebro.adddata(frames)
cerebro.addstrategy(strat)
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades2")
res = cerebro.run()

print("\n=== order-level spy over", len(df), "bars ===")
print(spy)

ta = res[0].analyzers.trades.get_analysis()
total = ta.get("total", {})
print("\n=== TradeAnalyzer total ===")
for k, v in total.items():
    if isinstance(v, dict):
        print(f"  {k}: total={v.get('total')} pnet={v.get('pnet')}")
    else:
        print(f"  {k}: {v}")
print("\n=== long ===")
print(ta.get("long", {}).get("total", {}).get("total"), "closed:",
      ta.get("long", {}).get("total", {}).get("closed"))
print("=== short ===")
print(ta.get("short", {}).get("total", {}).get("total"))
print("\n=== streak ===", ta.get("streak"))
print("\n=== pnl ===", ta.get("pnl"))
