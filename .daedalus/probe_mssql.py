"""Probe the MSSQL data on the ThinkPad: tables, timeframes, coverage.

Never prints the connection string (it carries the password).
"""
import sys

# The C extension must come from dontcommit: importing it a second time
# under backtrader.feeds.mssql.fast_mssql dies with
# 'generic_type: type "ODBCManager" is already registered!'.
from backtrader.dontcommit import connection_string, fast_mssql

import os
import re

# The connection string baked into the fork says Server=localhost, which is
# right on the ThinkPad (where the DB lives) and wrong from this box. Swap
# in the real host instead of editing a file that carries a password.
# The string in dontcommit already points at the ThinkPad
# (SERVER=192.168.0.242,1433) and connects as-is. Rewriting the host was
# what broke the first two attempts — do not touch it.
def _conn_str() -> str:
    return connection_string


def q(sql):
    return fast_mssql.fetch_data_from_db(_conn_str(), sql)


def main():
    import backtrader.dontcommit as _dc
    print("module     :", _dc.__file__)
    print("raw masked :", re.sub(r"(?i)(Password|PWD)=[^;]*", r"\1=***", connection_string))
    # Server identity, password masked.
    m = re.search(r"(?i)(Server|Data Source)=([^;]*)", _conn_str())
    print("server      :", m.group(2) if m else "?")
    m = re.search(r"(?i)Database=([^;]*)", _conn_str())
    print("database    :", m.group(1) if m else "?")

    tables = [r[0] for r in q(
        "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE='BASE TABLE'")]
    print(f"tables      : {len(tables)}")
    klines = sorted(t for t in tables if t.endswith("_klines"))
    print(f"_klines     : {len(klines)}")
    for t in klines[:60]:
        print("   ", t)

    # Timeframe coverage for EVERY klines table — that is the sweep grid.
    import datetime as _dt

    def ts(v):
        try:
            return _dt.datetime.fromtimestamp(int(v) / 1e6).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return str(v)

    inventory = {}
    for t in klines:
        rows = q(f"SELECT Timeframe, COUNT(*), MIN(TimestampStart), MAX(TimestampStart) "
                 f"FROM [{t}] GROUP BY Timeframe")
        inventory[t] = [(r[0], int(r[1]), ts(r[2]), ts(r[3])) for r in rows]

    all_tfs = sorted({tf for v in inventory.values() for tf, *_ in v})
    print("\ntimeframe values present:", all_tfs)
    print(f"{'table':28} " + " ".join(f"{tf:>14}" for tf in all_tfs))
    for t, v in inventory.items():
        cells = []
        for tf in all_tfs:
            hit = next((x for x in v if x[0] == tf), None)
            cells.append(f"{hit[1]:>14,}" if hit else f"{'-':>14}")
        print(f"{t:28} " + " ".join(cells))

    import json as _json
    with open(".daedalus/mssql_inventory.json", "w") as fh:
        _json.dump({k: v for k, v in inventory.items()}, fh, indent=1)
    print("\nwrote .daedalus/mssql_inventory.json")


if __name__ == "__main__":
    main()
