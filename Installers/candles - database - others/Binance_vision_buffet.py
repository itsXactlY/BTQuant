# https://pypi.org/project/binance-historical-data/
# pip install binance-historical-data
#
# Top-N liquid Binance spot */USDT pairs, 1m klines from listing to yesterday (Vision has no "today").
# Only pairs TRADING right now; delisted pairs are never touched.
# Works in batches: dump -> Candle_Database_import.py -> delete CSVs, so disk holds one batch at a time.
# Re-runs resume per symbol from the last candle already in MSSQL.

import os
import sys
import json
import shutil
import datetime
import subprocess
import urllib.request

from binance_historical_data import BinanceDataDumper
from binance_historical_data import data_dumper as bd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import Candle_Database_import as cdi  # noqa: E402

TOP_N = int(os.environ.get("BTQ_TOP_PAIRS", 250))
BATCH = int(os.environ.get("BTQ_BATCH_PAIRS", 25))
DATA_DIR = os.path.expanduser(os.environ.get("BTQ_CANDLE_DIR", "~/.btq-candles"))
DUMP_DIR = os.path.join(DATA_DIR, "candles")

# Stablecoin / fiat bases: pegged, no signal, but high volume would eat slots
EXCLUDED_BASES = {
    "USDC", "FDUSD", "TUSD", "USDP", "PAX", "DAI", "BUSD", "USD1", "RLUSD", "USDE", "PYUSD",
    "BFUSD", "XUSD", "UST", "USTC", "EUR", "AEUR", "EURI", "GBP", "TRY", "RUB", "JPY", "BRL",
    "AUD", "BIDR", "IDRT", "BKRW", "UAH", "ZAR", "NGN", "PLN", "ARS", "MXN", "COP", "CZK",
}

# Skip the SSL country lookup (EU box)
bd.BinanceDataDumper._get_user_country_from_ip = lambda self: "EU"


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def select_pairs() -> list[str]:
    info = get_json("https://api.binance.com/api/v3/exchangeInfo?permissions=SPOT")
    trading = {
        s["symbol"] for s in info["symbols"]
        if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"
        and s.get("isSpotTradingAllowed") and s["baseAsset"] not in EXCLUDED_BASES
    }
    tickers = get_json("https://api.binance.com/api/v3/ticker/24hr")
    ranked = sorted(
        ((float(t["quoteVolume"]), t["symbol"]) for t in tickers if t["symbol"] in trading),
        reverse=True,
    )
    return [s for _, s in ranked[:TOP_N]]


def resume_start(symbol: str):
    """First day of the month holding the last imported candle; None = from listing."""
    table = f"{cdi.safe_ident(symbol)}_klines"
    if not cdi.table_exists(table):
        return None
    last_us = cdi.get_latest_timestamp(table)
    if last_us <= 0:
        return None
    last = datetime.datetime.fromtimestamp(last_us / 1e6, datetime.timezone.utc).date()
    return last.replace(day=1)


def dump_batch(symbols: list[str]):
    dumper = BinanceDataDumper(
        path_dir_where_to_dump=DUMP_DIR,
        asset_class="spot",
        data_type="klines",
        data_frequency="1m",
    )
    for sym in symbols:
        start = resume_start(sym)
        print(f"  ⬇️  {sym} from {start or 'listing'}", flush=True)
        dumper.dump_data(tickers=[sym], date_start=start, is_to_update_existing=False)
    dumper.delete_outdated_daily_results()


def import_batch():
    # Separate process: the importer owns its spawn pool, index drop/rebuild and exit code
    subprocess.run([sys.executable, os.path.join(HERE, "Candle_Database_import.py")],
                   cwd=DATA_DIR, check=True)


def cap_server_memory():
    # SQL Server on Linux defaults to ~80% RAM; import workers need the rest
    total_mb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // 2**20
    cur = cdi.fm_query("SELECT CAST(value_in_use AS BIGINT) FROM sys.configurations WHERE name = 'max server memory (MB)';")
    if cur and int(cur[0][0]) >= 2147483647:
        cap = max(2048, total_mb // 3)
        cdi.fm_exec("EXEC sp_configure 'show advanced options', 1; RECONFIGURE;")
        cdi.fm_exec(f"EXEC sp_configure 'max server memory (MB)', {cap}; RECONFIGURE;")
        print(f"🧠 MSSQL max server memory capped at {cap} MB", flush=True)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    cdi.ensure_database()
    cap_server_memory()
    pairs = select_pairs()
    print(f"📋 {len(pairs)} liquid TRADING */USDT pairs, batches of {BATCH} -> {DATA_DIR}", flush=True)

    for i in range(0, len(pairs), BATCH):
        batch = pairs[i:i + BATCH]
        print(f"\n=== Batch {i // BATCH + 1}/{(len(pairs) + BATCH - 1) // BATCH}: {', '.join(batch)}", flush=True)
        shutil.rmtree(DUMP_DIR, ignore_errors=True)
        dump_batch(batch)
        import_batch()  # raises on any failed table -> CSVs stay for inspection, rerun resumes
        shutil.rmtree(DUMP_DIR, ignore_errors=True)

    print("\n✅ All batches imported.", flush=True)


if __name__ == "__main__":
    main()
