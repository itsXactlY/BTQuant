"""Independent verification: do the LLM-generated strategies actually trade?

The generation gate runs a 400-bar smoke run and rejects code that places no
order. A gate that passes code which then sits at 0 trades on real data is worse
than no gate, so every accepted file is re-run here on real MSSQL-cache bars and
the trade count is reported. This script trusts nothing from the factory: it
imports each file, loads the strategy class, runs it, and counts.

Usage: verify_generated.py [results.json ...]
"""
import importlib.util
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, "/home/alca/projects/PubBTQuant")

import polars as pl

import backtrader as bt

CACHE = Path("/home/alca/projects/PubBTQuant/.btq_cache/mssql")


def load_spec(path: Path):
    name = f"gen_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # must be in sys.modules before exec: the generated code's import of
    # BaseStrategy resolves relative back to this module
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    for nm in dir(mod):
        obj = getattr(mod, nm)
        if (isinstance(obj, type) and issubclass(obj, bt.Strategy)
                and obj is not bt.Strategy and not nm.startswith("_")
                and "Base" not in nm):
            return obj
    return None


def bars(coin: str, tf: str, n: int) -> pl.DataFrame:
    p = CACHE / f"{coin}_{tf}.parquet"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    return pl.read_parquet(p).head(n)


class Counters(bt.Strategy):
    """Base with only the counters -- no trading logic of our own."""
    params = (("percent_sizer", 0.95),)

    def __init__(self, **kw):
        super().__init__(**kw)
        self.n_orders = 0
        self.n_buy = 0
        self.n_sell = 0

    def create_order(self, action='BUY', size=None, price=None):
        self.n_orders += 1
        if str(action).upper().startswith("B"):
            self.n_buy += 1
        else:
            self.n_sell += 1
        return super().create_order(action=action, size=size, price=price)

    def stop(self):
        pass


class TradeCounter(bt.analyzers.Analyzer):
    """Count orders and closed trades from notifications.

    `strat.stats.btstats` does not exist in this fork (ItemCollection has no
    such attribute), so the counts come from notify_order/notify_trade.

    Status codes are backtrader's, NOT the intuition: Created=0, Submitted=1,
    Accepted=2, Partial=3, **Completed=4**, Canceled=5, Expired=6, Margin=7,
    Rejected=8. Reading 4 as "Margin" makes every filled order look rejected.
    """
    FILLED = 4
    BAD = (5, 6, 7, 8)          # canceled, expired, margin, rejected

    def __init__(self):
        self.orders = 0
        self.buys = 0
        self.sells = 0
        self.rejected = 0
        self.margin = 0
        self.closed = 0

    def notify_order(self, order):
        if order.status == self.FILLED:
            self.orders += 1
            if order.isbuy():
                self.buys += 1
            else:
                self.sells += 1
        elif order.status in self.BAD:
            self.rejected += 1
            if order.status == 7:       # Margin
                self.margin += 1

    def notify_trade(self, trade):
        if trade.isclosed:
            self.closed += 1


def run(klass, df, **params):
    cerebro = bt.Cerebro(stdstats=False, runonce=False)
    cerebro.adddata(bt.feeds.PolarsData(dataname=df))
    cerebro.broker.setcash(100_000.0)
    cerebro.addstrategy(klass, **params)
    cerebro.addanalyzer(TradeCounter, _name="tc")
    strat = cerebro.run()[0]
    return strat, strat.analyzers.tc


def param_names(klass):
    """Param names of a strategy class.

    In this fork `klass.params` is not iterable at class level -- the metaclass
    consumes the tuple. `_getpairs()` is the supported accessor.
    """
    p = getattr(klass, "params", None)
    if p is None:
        return set()
    for accessor in ("_getpairs", "params"):
        fn = getattr(p, accessor, None)
        if fn is None:
            continue
        try:
            pairs = fn() if accessor == "_getpairs" else fn
            return {x[0] for x in pairs}
        except (TypeError, AttributeError):
            continue
    try:
        return {x[0] for x in p}
    except TypeError:
        return set()


def verify(path: Path, coin="BTCUSDT", tf="1h", n=1500):
    klass = load_spec(path)
    if klass is None:
        return {"error": "no bt.Strategy subclass in file"}
    df = bars(coin, tf, n)
    base_params = param_names(klass)
    kw = {}
    if "backtest" not in base_params:
        kw["backtest"] = True
    kw["percent_sizer"] = 0.95
    try:
        strat, tc = run(klass, df, **kw)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {str(e)[:140]}",
                "tb": traceback.format_exc()[-400:]}
    return {
        "bars": len(df),
        "orders": tc.orders,
        "buys": tc.buys,
        "sells": tc.sells,
        "rejected": tc.rejected,
        "margin_calls": tc.margin,
        "closed_trades": tc.closed,
        "final_value": round(float(strat.broker.getvalue()), 2),
    }


BATCH_NAMES = [
    "RSI_Dip_Reversion", "Dual_EMA_Trend", "Bollinger_Reversion",
    "MACD_Momentum", "Stochastic_Crossover", "ATR_Breakout",
    "ADX_DI_Crossover", "RSI_Momentum_Regime",
]


def newest_batch_files():
    """Newest file per batch hypothesis, read off disk.

    Not from the results JSON: a single-hypothesis re-run overwrites that file
    and drops the paths of everything that landed earlier.
    """
    d = Path("/home/alca/projects/PubBTQuant/autonomous_agency/strategies")
    out = []
    for stem in BATCH_NAMES:
        cands = sorted(d.glob(f"{stem}_*.py"), key=lambda p: p.stat().st_mtime)
        if cands:
            out.append(str(cands[-1]))
        else:
            print(f"  (no file for {stem})")
    return out


def main():
    args = sys.argv[1:]
    n = 1500
    coin, tf = "BTCUSDT", "1h"
    files = []
    i = 0
    while i < len(args):
        if args[i] == "--bars":
            n = int(args[i + 1]); i += 2
        elif args[i] == "--coin":
            coin = args[i + 1]; i += 2
        elif args[i] == "--tf":
            tf = args[i + 1]; i += 2
        else:
            files.append(args[i]); i += 1
    if not files:
        files = newest_batch_files()
    rows = []
    for f in files:
        p = Path(f)
        res = verify(p, coin=coin, tf=tf, n=n)
        res["name"] = p.stem
        rows.append(res)
        if "error" in res:
            print(f"  FAIL {p.stem[:52]:54s} {res['error'][:80]}")
        else:
            rate = res['orders'] / max(res['bars'], 1) * 1000
            print(f"  ok   {p.stem[:44]:46s} {res['orders']:>4} ord "
                  f"({res['buys']}B/{res['sells']}S) {res['closed_trades']:>4} closed "
                  f"{res['rejected']:>3} rej | {rate:5.1f}/1k bars | "
                  f"val={res['final_value']}")
    out = Path("/home/alca/projects/PubBTQuant/.daedalus/verify_generated.json")
    out.write_text(json.dumps(rows, indent=1))
    n_ok = sum(1 for r in rows if r.get("closed_trades", 0) > 0)
    print(f"\n{n_ok}/{len(rows)} strategies closed at least one trade on "
          f"{coin} {tf} x {n} bars")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
