"""Throughput benchmark: how many bars/s does one real generated strategy cost?

The grid size is decided by this number, not by a guess. Runs the agency's own
cerebro setup (analyzers + strategy class loading) so the figure is the real one.
"""
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, "/home/alca/projects/PubBTQuant")

from autonomous_agency.backtester import AutomatedBacktester
from autonomous_agency.strategy_factory import GeneratedStrategy
from autonomous_agency import mssql_store as ms

STRATEGY = sys.argv[1] if len(sys.argv) > 1 else None


def pick_strategy() -> Path:
    files = sorted(Path("autonomous_agency/strategies").glob("*.py"), key=lambda p: p.stat().st_size)
    files = [f for f in files if f.stat().st_size > 3000]
    if STRATEGY:
        return Path(STRATEGY)
    return files[len(files) // 2]


def class_name_of(path: Path) -> str:
    import ast
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for b in node.bases:
                if isinstance(b, ast.Name) and "Strategy" in b.id:
                    return node.name
                if isinstance(b, ast.Attribute) and "Strategy" in b.attr:
                    return node.name
    raise SystemExit(f"no Strategy class in {path}")


def main():
    path = pick_strategy()
    gs = GeneratedStrategy(
        hypothesis_id="bench", strategy_name=path.stem, code_path=str(path.resolve()),
        class_name=class_name_of(path), parameters={}, indicators=[],
        generated_at="2026-09-27", validation_status="ok", codegen_path="llm",
    )
    print(f"strategy: {path.name} ({path.stat().st_size} B), class={gs.class_name}")

    bt_engine = AutomatedBacktester()
    jobs = [("BTCUSDT", "1h", None), ("BTCUSDT", "1m", 500_000), ("BTCUSDT", "1m", None)]
    for symbol, tf, cap in jobs:
        cfg = {"source": "parquet", "path": str(ms.cache_path(symbol, tf)),
               "initial_cash": 100_000, "commission": 0.001}
        t0 = time.time()
        import polars as pl
        t0d = time.time()
        feed = bt_engine._load_data_feeds(cfg)
        t_data = time.time() - t0d
        rows = pl.scan_parquet(cfg["path"]).select(pl.len()).collect().item()
        if cap:
            rows = min(rows, cap)
        del feed

        cfg["max_bars"] = cap
        t0 = time.time()
        try:
            res = bt_engine.run_backtest(gs, cfg) if cap is None else _run_capped(bt_engine, gs, cfg, cap)
        except Exception as e:
            print(f"  {symbol} {tf} cap={cap}: FAILED {type(e).__name__}: {e}")
            continue
        d = time.time() - t0
        metrics = ""
        if res is not None:
            metrics = (f" ret={res.total_return:.3f} sharpe={res.sharpe_ratio:.2f} "
                       f"dd={res.max_drawdown:.3f} trades={res.num_trades}")
        print(f"  {symbol} {tf:4} cap={str(cap or 'full'):>9}  {d:7.2f}s  "
              f"{rows/max(d,1e-9):>9.0f} bars/s  (data load {t_data:.2f}s){metrics}")


def _run_capped(engine, gs, cfg, cap):
    """Same cerebro, but only the last `cap` bars -- measures a sub-range."""
    import backtrader as bt
    import polars as pl
    df = pl.read_parquet(cfg["path"])
    if cap and df.height > cap:
        df = df.tail(cap)
    tmp = Path("/home/alca/projects/PubBTQuant/.daedalus/bench_slice.parquet")
    df.rename({"datetime": "datetime"}).write_parquet(tmp)
    cfg = dict(cfg, path=str(tmp))
    return engine.run_backtest(gs, cfg)


if __name__ == "__main__":
    main()
