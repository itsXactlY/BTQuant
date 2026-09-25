# Top-N liquid Binance spot */USDT pairs, 1m klines from listing to yesterday (Vision has no "today").
# Only pairs TRADING right now; delisted pairs are never touched.
# Downloads run ahead of the import (own thread, one folder per batch) and pause only when free
# disk drops under BTQ_MIN_FREE_GB; the importer takes each batch as soon as it is complete and
# deletes it after. The pair list is frozen in pairs.json until the run finishes, so a restart
# keeps the same batches. Re-runs resume per symbol from the last candle already in MSSQL.

import os
import sys
import json
import time
import queue
import shutil
import threading
import datetime
import subprocess
import hashlib
import zipfile
import io
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import Candle_Database_import as cdi  # noqa: E402

TOP_N = int(os.environ.get("BTQ_TOP_PAIRS", 250))
BATCH = int(os.environ.get("BTQ_BATCH_PAIRS", 25))
DATA_DIR = os.path.expanduser(os.environ.get("BTQ_CANDLE_DIR", "~/.btq-candles"))
MIN_FREE_GB = float(os.environ.get("BTQ_MIN_FREE_GB", 30))
PAIRS_FILE = os.path.join(DATA_DIR, "pairs.json")

# Stablecoin / fiat bases: pegged, no signal, but high volume would eat slots
EXCLUDED_BASES = {
    "USDC", "FDUSD", "TUSD", "USDP", "PAX", "DAI", "BUSD", "USD1", "RLUSD", "USDE", "PYUSD",
    "BFUSD", "XUSD", "UST", "USTC", "EUR", "AEUR", "EURI", "GBP", "TRY", "RUB", "JPY", "BRL",
    "AUD", "BIDR", "IDRT", "BKRW", "UAH", "ZAR", "NGN", "PLN", "ARS", "MXN", "COP", "CZK",
}

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FILES = "https://data.binance.vision/"
DL_THREADS = int(os.environ.get("BTQ_DL_THREADS", 8))


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


def batch_dir(idx: int) -> str:
    return os.path.join(DATA_DIR, f"b{idx + 1:03d}")


def free_gb(path: str) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 2**30


# ── Vision download ──────────────────────────────────────────────────────────
# Own code instead of binance_historical_data: that library read only the first
# 1000 S3 keys (so the current month's dailies of old pairs never showed up),
# dropped failed downloads silently, and fell back to "listed this month" when
# a listing failed. Here every listing pages, every file is sha256-checked
# against Binance's CHECKSUM, and anything still failing after retries fails
# the batch, which is then neither marked downloaded nor imported.

def fetch(url: str, tries: int = 6) -> bytes:
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(min(60, 2 ** i))
    raise RuntimeError("unreachable")


def list_keys(prefix: str) -> list[str]:
    keys, marker = [], ""
    while True:
        url = f"{S3}?delimiter=/&prefix={urllib.parse.quote(prefix)}"
        if marker:
            url += f"&marker={urllib.parse.quote(marker)}"
        root = ET.fromstring(fetch(url))
        ns = {"s": root.tag.split("}")[0].strip("{")}
        page = [k.text for k in root.findall(".//s:Contents/s:Key", ns)]
        keys += page
        if root.findtext("s:IsTruncated", default="false", namespaces=ns) != "true" or not page:
            return keys
        marker = page[-1]


def download(key: str, dest_dir: str):
    csv = os.path.join(dest_dir, os.path.basename(key)[:-4] + ".csv")
    if os.path.exists(csv) and os.path.getsize(csv) > 0:
        return  # already extracted by an earlier run of this batch
    blob = fetch(FILES + key)
    want = fetch(FILES + key + ".CHECKSUM").decode().split()[0]
    if hashlib.sha256(blob).hexdigest() != want:
        raise ValueError(f"checksum mismatch: {key}")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        z.extractall(dest_dir)


def plan_symbol(sym: str, dump_dir: str) -> list[tuple[str, str]]:
    """(key, dest_dir) for every Vision zip from the resume point up to yesterday."""
    start = resume_start(sym)  # first day of the last imported month, None = from listing
    mon = f"data/spot/monthly/klines/{sym}/1m/"
    all_monthly = sorted(k for k in list_keys(mon) if k.endswith(".zip"))
    monthly = [k for k in all_monthly if not start or k[-11:-4] >= start.strftime("%Y-%m")]
    yesterday = datetime.datetime.now(datetime.timezone.utc).date() - datetime.timedelta(days=1)
    if not all_monthly and not start:
        # listed this month: no monthly file yet, the daily folder is short
        daily = [k for k in list_keys(f"data/spot/daily/klines/{sym}/1m/") if k.endswith(".zip")]
    else:
        # dailies cover what monthly files do not yet: month after the last
        # published monthly (or the resume day) up to yesterday
        first_day = start or datetime.date(2017, 1, 1)
        if all_monthly:
            after = (datetime.date.fromisoformat(all_monthly[-1][-11:-4] + "-01") + datetime.timedelta(days=32)).replace(day=1)
            first_day = max(first_day, after)
        daily, m = [], first_day.replace(day=1)
        while m <= yesterday:
            pre = f"data/spot/daily/klines/{sym}/1m/{sym}-1m-{m:%Y-%m}"
            daily += [k for k in list_keys(pre) if k.endswith(".zip") and k[-14:-4] >= f"{first_day:%Y-%m-%d}"]
            m = (m + datetime.timedelta(days=32)).replace(day=1)
    out = os.path.join(dump_dir, "spot", "{}", "klines", sym, "1m")
    return [(k, out.format("monthly")) for k in monthly] + [(k, out.format("daily")) for k in sorted(daily)]


def dump_batch(symbols: list[str], dump_dir: str):
    jobs = []
    for sym in symbols:
        plan = plan_symbol(sym, dump_dir)
        print(f"  ⬇️  {sym}: {len(plan)} files", flush=True)
        jobs += plan
    with ThreadPoolExecutor(DL_THREADS) as ex:
        for f in [ex.submit(download, k, d) for k, d in jobs]:
            f.result()  # first failure after retries aborts the batch


def producer(batches: list[list[str]], ready: queue.Queue):
    try:
        for idx, batch in enumerate(batches):
            d = batch_dir(idx)
            if not os.path.exists(os.path.join(d, ".downloaded")):
                while free_gb(DATA_DIR) < MIN_FREE_GB:
                    print(f"  ⏸️  download paused: {free_gb(DATA_DIR):.0f} GB free < {MIN_FREE_GB:.0f}", flush=True)
                    time.sleep(60)
                print(f"\n=== Download {idx + 1}/{len(batches)}: {', '.join(batch)}", flush=True)
                dump_batch(batch, os.path.join(d, "candles"))
                open(os.path.join(d, ".downloaded"), "w").close()
            ready.put(idx)
    except BaseException as e:  # surface in the main thread, never hang it
        ready.put(e)
    ready.put(None)


def import_batch(d: str):
    # Separate process: the importer owns its spawn pool, index drop/rebuild and exit code
    subprocess.run([sys.executable, os.path.join(HERE, "Candle_Database_import.py")],
                   cwd=d, check=True)


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

    if os.path.exists(PAIRS_FILE):
        pairs = json.load(open(PAIRS_FILE))
        print(f"📋 resuming frozen list: {len(pairs)} pairs ({PAIRS_FILE})", flush=True)
    else:
        pairs = select_pairs()
        json.dump(pairs, open(PAIRS_FILE, "w"))
        print(f"📋 {len(pairs)} liquid TRADING */USDT pairs, batches of {BATCH} -> {DATA_DIR}", flush=True)
    batches = [pairs[i:i + BATCH] for i in range(0, len(pairs), BATCH)]

    # CSVs from the single-folder layout become batch 1 instead of being fetched again
    legacy = os.path.join(DATA_DIR, "candles")
    if os.path.isdir(legacy) and not os.path.exists(batch_dir(0)):
        os.makedirs(batch_dir(0))
        os.rename(legacy, os.path.join(batch_dir(0), "candles"))

    ready: queue.Queue = queue.Queue()
    threading.Thread(target=producer, args=(batches, ready), daemon=True).start()
    while (item := ready.get()) is not None:
        if isinstance(item, BaseException):
            raise item
        print(f"\n=== Import {item + 1}/{len(batches)}", flush=True)
        import_batch(batch_dir(item))  # raises on a failed table -> CSVs stay, rerun resumes
        shutil.rmtree(batch_dir(item), ignore_errors=True)

    os.remove(PAIRS_FILE)
    print("\n✅ All batches imported.", flush=True)


if __name__ == "__main__":
    main()
