# -*- coding: utf-8 -*-
# ! BTQuant's own ``bruteforce´´ MSSQL importer feat. our inhouse C++ adapter
# - Per-symbol tables (like from first prototypes)
# - Polars for fast CSV load
# - Multiprocessing (spawn)
# - Early file skip (improved: checks start and end)
# - TABLOCK inserts with explicit transactions
# - Drop/rebuild indexes centrally
# - Chronological sorting by file start timestamp to avoid skips
# - Simple tqdm progress bar per table
# - Increased workers + optional per-file timing for optimization
# - Retry logic for all DB operations
# - Configurable batch size
# - Exclude certain symbols by keywords
# - Handles timestamps in seconds, milliseconds, or microseconds

import os
import gc
import time
import glob
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl
import fast_mssql as fm
from tqdm import tqdm  # For progress bars

# ---------- CONFIG ----------
try:
    from backtrader.dontcommit import database, connection_string
    DB_NAME = database
    DB_CONN = connection_string
except Exception:
    DB_NAME = "BinanceData"
    DB_CONN = "DRIVER={ODBC Driver 18 for SQL Server};SERVER=localhost;DATABASE=BinanceData;UID=SA;PWD=YourStrong!Passw0rd;TrustServerCertificate=yes;"

BASE_DIRS = [ # Directories to scan for CSV files
    "candles/spot/monthly/klines/",
    "candles/spot/daily/klines/",
]
EXCLUDED_KEYWORDS = {"BULL", "BEAR", "UP", "DOWN"}

BATCH_SIZE = 200_000  # Try lowering to 100_000 if small files are slow
MAX_WORKERS = min(8, os.cpu_count() or 1)  # Increased to max of 8 for 16-core CPU; test up to 12 with NVME drives

RAW_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
    "ignore"
]

MIN_TIMESTAMP = 1483228800000000  # 2017-01-01 in microseconds (fallback floor)

# ---------- UTILS ----------
def safe_ident(name: str) -> str:
    # Allow only alnum and underscore to avoid SQL injection in identifiers
    s = "".join(ch for ch in name if ch.isalnum() or ch == "_")
    return s


def bracket(name: str) -> str:
    return f"[{name}]"


# ---------- TIMESTAMP HELPERS ----------
def ensure_microseconds_int(ts: int) -> int:
    ts = int(ts)
    if ts >= 1_000_000_000_000_000:  # µs
        return ts
    if ts >= 1_000_000_000_000:      # ms
        return ts * 1_000
    if ts >= 1_000_000_000:          # s
        return ts * 1_000_000
    return ts


def ensure_microseconds_expr(col: pl.Expr) -> pl.Expr:
    return (
        pl.when(col >= 1_000_000_000_000_000).then(col)   # µs
        .when(col >= 1_000_000_000_000).then(col * 1_000) # ms -> µs
        .when(col >= 1_000_000_000).then(col * 1_000_000) # s -> µs
        .otherwise(col)
    ).cast(pl.Int64)


def file_last_close_us(file_path: str):
    try:
        with open(file_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size == 0:
                return None
            block = 4096
            data = b""
            while size > 0:
                read_size = block if size >= block else size
                size -= read_size
                f.seek(size)
                data = f.read(read_size) + data
                if b"\n" in data:
                    break
            lines = data.splitlines()
            for line in reversed(lines):
                if line.strip():
                    parts = line.decode("utf-8", "ignore").split(",")
                    if len(parts) >= 7:
                        return ensure_microseconds_int(int(parts[6]))
    except Exception:
        return None
    return None


def file_first_open_us(file_path: str):
    try:
        with open(file_path, "rb") as f:
            f.seek(0)
            line = f.readline().decode("utf-8", "ignore").strip()
            if not line:
                return None
            parts = line.split(",")
            if len(parts) >= 1:
                return ensure_microseconds_int(int(parts[0]))
    except Exception as e:
        print(f"Peek failed for {file_path}: {e}")
        return None
    return None


def should_skip_file(file_path: str, latest_ts: int) -> bool:
    if latest_ts <= 0:
        return False
    last_us = file_last_close_us(file_path)
    first_us = file_first_open_us(file_path)
    if last_us is None or first_us is None:
        return False  # Don't skip if we can't peek
    # Skip only if ENTIRE file is older than latest_ts
    return last_us <= latest_ts


# ---------- FAST_MSSQL WRAPPERS (RETRY) ----------
def fm_exec(sql: str, tries: int = 10, delay: float = 1.5):
    last = None
    for i in range(tries):
        try:
            fm.execute_non_query(DB_CONN, sql)
            return
        except Exception as e:
            last = e
            try:
                fm.remove_connection(DB_CONN)
            except Exception:
                pass
            time.sleep(delay * (i + 1))
    if last:
        raise last


def fm_query(sql: str, tries: int = 10, delay: float = 1.5):
    last = None
    for i in range(tries):
        try:
            return fm.fetch_data_from_db(DB_CONN, sql)
        except Exception as e:
            last = e
            try:
                fm.remove_connection(DB_CONN)
            except Exception:
                pass
            time.sleep(delay * (i + 1))
    if last:
        raise last


def fm_bulk_tx(insert_sql: str, rows, tries: int = 10, delay: float = 1.5):
    last = None
    for i in range(tries):
        try:
            fm.execute_non_query(DB_CONN, "BEGIN TRANSACTION;")
            fm.bulk_insert(DB_CONN, insert_sql, rows)
            fm.execute_non_query(DB_CONN, "COMMIT;")
            return
        except Exception as e:
            last = e
            try:
                fm.execute_non_query(DB_CONN, "ROLLBACK;")
            except Exception:
                pass
            try:
                fm.remove_connection(DB_CONN)
            except Exception:
                pass
            time.sleep(delay * (i + 1))
    if last:
        raise last


# ---------- DB META ----------
def enable_bulk_db_settings():
    try:
        fm_exec(f"ALTER DATABASE {bracket(DB_NAME)} SET RECOVERY BULK_LOGGED;")
    except Exception as e:
        print(f"[DB] BULK_LOGGED failed: {e}")
    try:
        fm_exec(f"ALTER DATABASE {bracket(DB_NAME)} SET DELAYED_DURABILITY = ALLOWED;")
    except Exception as e:
        print(f"[DB] DELAYED_DURABILITY failed: {e}")


def table_exists(table_name: str) -> bool:
    tname = table_name.replace("'", "''")
    res = fm_query(f"""
        SELECT CASE WHEN EXISTS (
            SELECT 1 FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA='dbo' AND TABLE_NAME='{tname}'
        ) THEN 1 ELSE 0 END
    """)
    if not res:
        return False
    try:
        return int(res[0][0]) == 1
    except Exception:
        return False


def create_table_if_not_exists(table_name: str):
    t = bracket(table_name)
    fm_exec(f"""
    IF NOT EXISTS (SELECT 1 FROM sys.objects WHERE object_id = OBJECT_ID(N'{t}') AND type in (N'U'))
    BEGIN
        CREATE TABLE {t} (
            CandleID INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
            Timeframe VARCHAR(10) NOT NULL,
            TimestampStart BIGINT NOT NULL,
            [Open] DECIMAL(28, 8) NOT NULL,
            [High] DECIMAL(28, 8) NOT NULL,
            [Low] DECIMAL(28, 8) NOT NULL,
            [Close] DECIMAL(28, 8) NOT NULL,
            Volume DECIMAL(28, 8) NOT NULL,
            TimestampEnd BIGINT NOT NULL,
            QuoteVolume DECIMAL(28, 8) NOT NULL,
            Trades INT NOT NULL,
            TakerBaseVolume DECIMAL(28, 8) NOT NULL,
            TakerQuoteVolume DECIMAL(28, 8) NOT NULL
        );
    END
    """)


def get_latest_timestamp(table_name: str) -> int:
    t = bracket(table_name)
    res = fm_query(f"SELECT COALESCE(MAX(TimestampEnd), 0) FROM {t};")
    if res and res[0]:
        try:
            return int(res[0][0])
        except Exception:
            return 0
    return 0


def drop_indexes_for_tables(table_names):
    for t in table_names:
        bt = bracket(t)
        idx1 = f"IX_{t}_TimestampStart"
        idx2 = f"IX_{t}_Timeframe_Timestamp"
        sql = f"""
        IF EXISTS (SELECT 1 FROM sys.indexes WHERE name = '{idx1}' AND object_id = OBJECT_ID(N'{bt}'))
            DROP INDEX [{idx1}] ON {bt};
        IF EXISTS (SELECT 1 FROM sys.indexes WHERE name = '{idx2}' AND object_id = OBJECT_ID(N'{bt}'))
            DROP INDEX [{idx2}] ON {bt};
        """
        try:
            fm_exec(sql)
        except Exception as e:
            print(f"[{t}] Drop indexes failed: {e}")


def create_indexes_for_tables(table_names):
    for t in table_names:
        bt = bracket(t)
        idx1 = f"IX_{t}_TimestampStart"
        idx2 = f"IX_{t}_Timeframe_Timestamp"
        sql = f"""
        IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = '{idx1}' AND object_id = OBJECT_ID(N'{bt}'))
            CREATE NONCLUSTERED INDEX [{idx1}] ON {bt} (TimestampStart);
        IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = '{idx2}' AND object_id = OBJECT_ID(N'{bt}'))
            CREATE NONCLUSTERED INDEX [{idx2}] ON {bt} (Timeframe, TimestampStart);
        """
        try:
            fm_exec(sql)
        except Exception as e:
            print(f"[{t}] Create indexes failed: {e}")


# ---------- DISCOVER TASKS ----------
def discover_tasks():
    tasks = {}
    for base in BASE_DIRS:
        if not os.path.exists(base):
            continue
        for sym_folder in os.listdir(base):
            if any(k in sym_folder for k in EXCLUDED_KEYWORDS):
                continue
            full_sym_path = os.path.join(base, sym_folder)
            if not os.path.isdir(full_sym_path):
                continue

            pair_name = sym_folder.replace("_", "")
            table_name = f"{safe_ident(pair_name)}_klines"

            if not table_exists(table_name):
                create_table_if_not_exists(table_name)
                latest_ts = 0
            else:
                latest_ts = get_latest_timestamp(table_name)

            files_info = []
            for timeframe in os.listdir(full_sym_path):
                tf_path = os.path.join(full_sym_path, timeframe)
                if not os.path.isdir(tf_path):
                    continue
                csvs = sorted(glob.glob(os.path.join(tf_path, "*.csv")))
                for fp in csvs:
                    files_info.append((fp, timeframe))

            if files_info:
                tasks[table_name] = {
                    "latest_ts": latest_ts,
                    "files_info": files_info
                }
    return tasks


# ---------- CSV -> DF ----------
def read_and_prepare_df(file_path: str, timeframe: str, latest_ts: int) -> pl.DataFrame:
    dtype_map = {
        "open_time": pl.Int64,
        "close_time": pl.Int64,
        "number_of_trades": pl.Int64,
        "open": pl.Utf8,
        "high": pl.Utf8,
        "low": pl.Utf8,
        "close": pl.Utf8,
        "volume": pl.Utf8,
        "quote_asset_volume": pl.Utf8,
        "taker_buy_base_asset_volume": pl.Utf8,
        "taker_buy_quote_asset_volume": pl.Utf8,
        "ignore": pl.Utf8,
    }

    try:
        df = pl.read_csv(
            file_path,
            has_header=False,
            new_columns=RAW_COLS,
            schema_overrides=dtype_map
        )
    except TypeError:
        df = pl.read_csv(
            file_path,
            has_header=False,
            new_columns=RAW_COLS,
            dtypes=dtype_map
        )

    df = df.with_columns([
        ensure_microseconds_expr(pl.col("open_time")).alias("ts_start_us"),
        ensure_microseconds_expr(pl.col("close_time")).alias("ts_end_us"),
    ])

    if latest_ts:
        df = df.filter(pl.col("ts_end_us") > pl.lit(latest_ts))

    if df.height == 0:
        return df

    df = df.select([
        pl.lit(timeframe).alias("Timeframe"),
        pl.col("ts_start_us").alias("TimestampStart"),
        pl.col("open").alias("Open"),
        pl.col("high").alias("High"),
        pl.col("low").alias("Low"),
        pl.col("close").alias("Close"),
        pl.col("volume").alias("Volume"),
        pl.col("ts_end_us").alias("TimestampEnd"),
        pl.col("quote_asset_volume").alias("QuoteVolume"),
        pl.col("number_of_trades").alias("Trades"),
        pl.col("taker_buy_base_asset_volume").alias("TakerBaseVolume"),
        pl.col("taker_buy_quote_asset_volume").alias("TakerQuoteVolume"),
    ])

    return df


# ---------- INSERTS ----------
def insert_dataframe_fast(table_name: str, df: pl.DataFrame) -> int:
    if df.height == 0:
        return 0

    # Ensure all are strings for C++ bulk_insert
    df_str = df.with_columns([
        pl.col("TimestampStart").cast(pl.Utf8),
        pl.col("TimestampEnd").cast(pl.Utf8),
        pl.col("Trades").cast(pl.Utf8),
        pl.col("Open").cast(pl.Utf8),
        pl.col("High").cast(pl.Utf8),
        pl.col("Low").cast(pl.Utf8),
        pl.col("Close").cast(pl.Utf8),
        pl.col("Volume").cast(pl.Utf8),
        pl.col("QuoteVolume").cast(pl.Utf8),
        pl.col("TakerBaseVolume").cast(pl.Utf8),
        pl.col("TakerQuoteVolume").cast(pl.Utf8),
        pl.col("Timeframe").cast(pl.Utf8),
    ])

    insert_sql = f"""INSERT INTO {bracket('dbo')}.{bracket(table_name)} WITH (TABLOCK) (
        Timeframe, TimestampStart, [Open], [High], [Low], [Close],
        Volume, TimestampEnd, QuoteVolume, Trades, TakerBaseVolume, TakerQuoteVolume
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

    total = 0
    for start in range(0, df_str.height, BATCH_SIZE):
        batch = df_str.slice(start, BATCH_SIZE)
        rows = [list(map(lambda x: "" if x is None else str(x), r)) for r in batch.iter_rows()]
        fm_bulk_tx(insert_sql, rows)
        total += len(rows)
    return total


# ---------- WORKER ----------
def process_table_task(args):
    table_name, files_info, latest_ts = args
    inserted_total = 0
    t0 = time.time()

    # Sort files by actual start timestamp (chronological order)
    def get_start_ts(info):
        fp, tf = info
        ts = file_first_open_us(fp)
        return ts if ts is not None else MIN_TIMESTAMP  # Fallback to min timestamp if peek fails

    files_info.sort(key=get_start_ts)

    # Simple progress bar for files in this table
    pbar = tqdm(total=len(files_info), desc=f"Processing {table_name}", unit="file", leave=False)

    # OPTIONAL: Enable per-file timing breakdown (uncomment to profile)
    # enable_timing = True
    enable_timing = False

    try:
        for fp, tf in files_info:
            file_t0 = time.time() if enable_timing else 0

            try:
                if should_skip_file(fp, latest_ts):
                    pbar.set_postfix({"status": f"skipped {os.path.basename(fp)}"})
                    pbar.update(1)
                    continue

                read_t0 = time.time() if enable_timing else 0
                df = read_and_prepare_df(fp, tf, latest_ts)
                read_time = time.time() - read_t0 if enable_timing else 0

                if df.height == 0:
                    pbar.set_postfix({"status": f"no new rows in {os.path.basename(fp)}"})
                    pbar.update(1)
                    continue

                insert_t0 = time.time() if enable_timing else 0
                ins = insert_dataframe_fast(table_name, df)
                insert_time = time.time() - insert_t0 if enable_timing else 0

                inserted_total += ins
                if ins > 0:
                    max_ts = df.select(pl.max("TimestampEnd")).item()
                    latest_ts = max(latest_ts, int(max_ts))

                postfix = {"inserted": ins, "file": os.path.basename(fp)}
                if enable_timing:
                    total_file_time = time.time() - file_t0
                    postfix.update({
                        "read_s": f"{read_time:.1f}",
                        "insert_s": f"{insert_time:.1f}",
                        "total_s": f"{total_file_time:.1f}"
                    })
                pbar.set_postfix(postfix)
                pbar.update(1)
                del df
                gc.collect()
            except Exception as e:
                print(f"[{table_name}] ERROR file {fp}: {e}")
                traceback.print_exc()
                pbar.update(1)  # Still progress even on error
    except Exception as e:
        print(f"[{table_name}] FATAL: {e}")
        traceback.print_exc()
    pbar.close()
    dt = time.time() - t0
    print(f"[{table_name}] DONE: inserted {inserted_total} rows in {dt:.1f}s")
    return table_name, inserted_total


# ---------- MAIN ----------
def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    print("Discovering tasks...")
    tasks_map = discover_tasks()
    if not tasks_map:
        print("No tasks found.")
        return

    table_names = list(tasks_map.keys())
    enable_bulk_db_settings()

    print(f"Dropping indexes on {len(table_names)} tables...")
    drop_indexes_for_tables(table_names)

    all_tasks = []
    for table_name, payload in tasks_map.items():
        files_info = payload["files_info"]
        latest_ts = payload["latest_ts"]
        # TIP: To force full re-import for testing, uncomment: latest_ts = 0
        all_tasks.append((table_name, files_info, latest_ts))

    print(f"Processing {len(all_tasks)} tables with {MAX_WORKERS} workers, batch={BATCH_SIZE}...")
    start = time.time()
    total_inserted = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_table_task, t) for t in all_tasks]
        for i, future in enumerate(as_completed(futures), 1):
            try:
                table_name, inserted = future.result()
                total_inserted += inserted
                if i % 10 == 0 or inserted > 0:
                    print(f"Completed {i}/{len(all_tasks)}: {table_name} (+{inserted})")
            except Exception as e:
                print(f"Worker error: {e}")
                traceback.print_exc()

    print(f"Recreating indexes on {len(table_names)} tables...")
    create_indexes_for_tables(table_names)

    print(f"=== ALL DONE in {time.time() - start:.1f}s. Total inserted: {total_inserted} ===")


if __name__ == "__main__":
    main()