# -*- coding: utf-8 -*-
# ! BTQuant's own ``bruteforce´´ MSSQL importer feat. our inhouse C++ adapter
# - Per-symbol tables (like from first prototypes)
# - Polars for fast CSV load with LAZY EVALUATION
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

'''
An whole "Binance Vision Spot" candles dump has be imported in ~10 Hours - working on better ways to do it faster.
This is a brute-force approach, but it works reliably and can be monitored.
🔧 Recreating indexes on 549 tables...
🎉 ALL DONE in 36270.3s. Total: 915,415,800 rows (25239 rows/sec)
╭─alca@alca in repo: BTQuant on  testing-unstable [!] via  v3.13.7 (.btq) took 10h4m41s
╰─λ
'''

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

BATCH_SIZE = 500_000  # Increased for better bulk performance
MAX_WORKERS = min(4, os.cpu_count() or 1)  # Slightly increased but not crazy

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
def fm_exec(sql: str, tries: int = 5, delay: float = 1.0):  # Reduced retries for speed
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

def fm_query(sql: str, tries: int = 5, delay: float = 1.0):
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

def fm_bulk_tx(insert_sql: str, rows, tries: int = 5, delay: float = 1.0):
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

            # Create entry once and accumulate across bases
            if table_name not in tasks:
                # Compute latest_ts once (create table if needed)
                if not table_exists(table_name):
                    create_table_if_not_exists(table_name)
                    latest_ts = 0
                else:
                    latest_ts = get_latest_timestamp(table_name)

                tasks[table_name] = {
                    "latest_ts": latest_ts,
                    "files_info": []
                }

            # Gather CSVs for ALL timeframes (monthly + daily bases merged)
            for timeframe in os.listdir(full_sym_path):
                tf_path = os.path.join(full_sym_path, timeframe)
                if not os.path.isdir(tf_path):
                    continue

                # Include nested dirs if any (some dumps nest per-year, for just in case - unlikely, but why not)
                csvs = sorted(glob.glob(os.path.join(tf_path, "**", "*.csv"), recursive=True))
                
                # Filter out files we can skip BEFORE adding to tasks
                for fp in csvs:
                    if not should_skip_file(fp, tasks[table_name]["latest_ts"]):
                        tasks[table_name]["files_info"].append((fp, timeframe))

    # Optional: drop symbols that ended up with zero files
    tasks = {k: v for k, v in tasks.items() if v["files_info"]}
    return tasks

# ---------- OPTIMIZED CSV -> DF ----------
def clean_numeric_value(val):
    """Clean and validate numeric values for SQL Server DECIMAL columns"""
    if val is None or val == "" or str(val).strip() == "":
        return "0"
    
    val_str = str(val).strip().upper()
    
    # Handle common non-numeric values
    if val_str in ("NULL", "N/A", "NAN", "NONE", "-", ""):
        return "0"
    
    # Remove any non-numeric characters except decimal point and minus sign
    cleaned = ""
    for char in val_str:
        if char.isdigit() or char in ".-":
            cleaned += char
    
    # Validate it's a proper number
    try:
        float(cleaned)
        return cleaned if cleaned else "0"
    except (ValueError, TypeError):
        return "0"


# ---------- OPTIMIZED CSV -> DF ----------
def read_and_prepare_all_lazy(files_info, latest_ts: int) -> pl.DataFrame:
    if not files_info:
        return pl.DataFrame()
    
    # Read everything as strings first, then clean and convert
    dtype_map = {
        "open_time": pl.Int64,
        "close_time": pl.Int64,
        "number_of_trades": pl.Int64,
        # Read all price/volume data as strings for cleaning
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
    
    print(f"    Loading {len(files_info)} files with lazy evaluation...")
    
    lazy_frames = []
    batch_size = 50
    
    for i in range(0, len(files_info), batch_size):
        batch_files = files_info[i:i + batch_size]
        
        for fp, timeframe in batch_files:
            try:
                lazy_df = (
                    pl.scan_csv(
                        fp,
                        has_header=False,
                        new_columns=RAW_COLS,
                        schema_overrides=dtype_map,
                        low_memory=False,
                        ignore_errors=True
                    )
                    .with_columns([
                        ensure_microseconds_expr(pl.col("open_time")).alias("ts_start_us"),
                        ensure_microseconds_expr(pl.col("close_time")).alias("ts_end_us"),
                        pl.lit(timeframe).alias("Timeframe"),
                        # Initial cleaning: remove unwanted chars, replace empty with "0"
                        pl.col("open").str.replace_all(r"[^\d.\-]", "").str.replace("", "0").alias("open_clean"),
                        pl.col("high").str.replace_all(r"[^\d.\-]", "").str.replace("", "0").alias("high_clean"),
                        pl.col("low").str.replace_all(r"[^\d.\-]", "").str.replace("", "0").alias("low_clean"),
                        pl.col("close").str.replace_all(r"[^\d.\-]", "").str.replace("", "0").alias("close_clean"),
                        pl.col("volume").str.replace_all(r"[^\d.\-]", "").str.replace("", "0").alias("volume_clean"),
                        pl.col("quote_asset_volume").str.replace_all(r"[^\d.\-]", "").str.replace("", "0").alias("quote_volume_clean"),
                        pl.col("taker_buy_base_asset_volume").str.replace_all(r"[^\d.\-]", "").str.replace("", "0").alias("taker_base_clean"),
                        pl.col("taker_buy_quote_asset_volume").str.replace_all(r"[^\d.\-]", "").str.replace("", "0").alias("taker_quote_clean"),
                    ])
                    .with_columns([
                        # Handle empty/null values properly
                        pl.when(pl.col("open_clean").is_null() | (pl.col("open_clean") == "")).then(pl.lit("0")).otherwise(pl.col("open_clean")).alias("open_final"),
                        pl.when(pl.col("high_clean").is_null() | (pl.col("high_clean") == "")).then(pl.lit("0")).otherwise(pl.col("high_clean")).alias("high_final"),
                        pl.when(pl.col("low_clean").is_null() | (pl.col("low_clean") == "")).then(pl.lit("0")).otherwise(pl.col("low_clean")).alias("low_final"),
                        pl.when(pl.col("close_clean").is_null() | (pl.col("close_clean") == "")).then(pl.lit("0")).otherwise(pl.col("close_clean")).alias("close_final"),
                        pl.when(pl.col("volume_clean").is_null() | (pl.col("volume_clean") == "")).then(pl.lit("0")).otherwise(pl.col("volume_clean")).alias("volume_final"),
                        pl.when(pl.col("quote_volume_clean").is_null() | (pl.col("quote_volume_clean") == "")).then(pl.lit("0")).otherwise(pl.col("quote_volume_clean")).alias("quote_volume_final"),
                        pl.when(pl.col("taker_base_clean").is_null() | (pl.col("taker_base_clean") == "")).then(pl.lit("0")).otherwise(pl.col("taker_base_clean")).alias("taker_base_final"),
                        pl.when(pl.col("taker_quote_clean").is_null() | (pl.col("taker_quote_clean") == "")).then(pl.lit("0")).otherwise(pl.col("taker_quote_clean")).alias("taker_quote_final"),
                    ])
                )
                
                # Apply filter as early as possible
                if latest_ts > 0:
                    lazy_df = lazy_df.filter(pl.col("ts_end_us") > latest_ts)
                
                lazy_frames.append(lazy_df)
                
            except Exception as e:
                print(f"    ⚠️  Skipping {fp}: {e}")
    
    if not lazy_frames:
        return pl.DataFrame()
    
    print(f"    Executing optimized query plan...")
    
    try:
        result = (
            pl.concat(lazy_frames, how="vertical_relaxed")
            .sort(["ts_start_us", "ts_end_us"])
            .unique(subset=["Timeframe", "ts_start_us"], keep="last", maintain_order=True)
            .select([
                pl.col("Timeframe"),
                pl.col("ts_start_us").alias("TimestampStart"),
                pl.col("open_final").alias("Open"),
                pl.col("high_final").alias("High"),
                pl.col("low_final").alias("Low"),
                pl.col("close_final").alias("Close"),
                pl.col("volume_final").alias("Volume"),
                pl.col("ts_end_us").alias("TimestampEnd"),
                pl.col("quote_volume_final").alias("QuoteVolume"),
                pl.col("number_of_trades").alias("Trades"),
                pl.col("taker_base_final").alias("TakerBaseVolume"),
                pl.col("taker_quote_final").alias("TakerQuoteVolume"),
            ])
            .collect(streaming=True)
        )
        
        print(f"    ✅ Processed {len(result)} rows")
        return result
        
    except Exception as e:
        print(f"    ❌ Lazy execution failed: {e}")
        traceback.print_exc()
        return pl.DataFrame()

# ---------- SIMPLIFIED INSERT WITH ROBUST VALIDATION ----------
def insert_dataframe_fast(table_name: str, df: pl.DataFrame) -> int:
    if df.height == 0:
        return 0

    print(f"    Inserting {df.height} rows in batches of {BATCH_SIZE}...")

    # Convert to strings with Python-based cleaning
    def clean_numeric_python(val):
        if val is None:
            return "0"
        
        val_str = str(val).strip()
        if val_str == "" or val_str.upper() in ("NULL", "N/A", "NAN", "NONE"):
            return "0"
        
        # Keep only digits, decimal point, and minus sign
        cleaned = ''.join(c for c in val_str if c.isdigit() or c in '.-')
        
        try:
            # Validate it's a proper number
            float(cleaned)
            return cleaned if cleaned else "0"
        except (ValueError, TypeError):
            return "0"

    def clean_integer_python(val):
        try:
            return str(int(float(clean_numeric_python(val))))
        except (ValueError, TypeError):
            return "0"

    # Convert DataFrame to list of rows for processing
    # aLca :: quick rant above the lines below:
    # As much i hate pandas, heres the necessary evil for easy dict conversion
    '''
    If you have ns-precision temporal values you should be aware that Python natively only supports up to μs-precision; 
    ns-precision values will be truncated to microseconds on conversion to Python. 
    If this matters to your use-case you should export to a different format (such as Arrow or NumPy).
    '''
    rows_data = df.to_pandas().to_dict('records')  # aLca :: Convert to pandas via poars, then dict for easier processing over bringing numPy into the game (less dependencies and headaches)
    
    insert_sql = f"""INSERT INTO {bracket('dbo')}.{bracket(table_name)} WITH (TABLOCK) (
        Timeframe, TimestampStart, [Open], [High], [Low], [Close],
        Volume, TimestampEnd, QuoteVolume, Trades, TakerBaseVolume, TakerQuoteVolume
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

    total = 0
    num_batches = (len(rows_data) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(rows_data))
        batch_rows = rows_data[start_idx:end_idx]
        
        clean_batch = []
        for row in batch_rows:
            try:
                clean_row = [
                    str(row.get("Timeframe", "")),                           # String
                    clean_integer_python(row.get("TimestampStart", 0)),      # BIGINT
                    clean_numeric_python(row.get("Open", 0)),                # DECIMAL
                    clean_numeric_python(row.get("High", 0)),                # DECIMAL
                    clean_numeric_python(row.get("Low", 0)),                 # DECIMAL
                    clean_numeric_python(row.get("Close", 0)),               # DECIMAL
                    clean_numeric_python(row.get("Volume", 0)),              # DECIMAL
                    clean_integer_python(row.get("TimestampEnd", 0)),        # BIGINT
                    clean_numeric_python(row.get("QuoteVolume", 0)),         # DECIMAL
                    clean_integer_python(row.get("Trades", 0)),              # INT
                    clean_numeric_python(row.get("TakerBaseVolume", 0)),     # DECIMAL
                    clean_numeric_python(row.get("TakerQuoteVolume", 0)),    # DECIMAL
                ]
                clean_batch.append(clean_row)
                
            except Exception as e:
                print(f"    ⚠️  Skipping malformed row: {e}")
                continue
        
        if not clean_batch:
            print(f"    ⚠️  Batch {batch_idx + 1} empty after cleaning")
            continue
            
        try:
            fm_bulk_tx(insert_sql, clean_batch)
            total += len(clean_batch)
            
            if batch_idx % 5 == 0 and num_batches > 5:
                print(f"      Batch {batch_idx + 1}/{num_batches} complete ({len(clean_batch)} rows)")
                
        except Exception as e:
            print(f"    ❌ Batch {batch_idx + 1} failed: {e}")
            if clean_batch:
                print(f"    Sample row: {clean_batch[0]}")
            continue
    
    return total

# ---------- OPTIMIZED WORKER ----------
def process_table_task(args):
    table_name, files_info, latest_ts = args
    t0 = time.time()

    print(f"[{table_name}] Processing {len(files_info)} files...")

    try:
        # Optimized lazy loading
        df = read_and_prepare_all_lazy(files_info, latest_ts)

        if df.height == 0:
            print(f"[{table_name}] ✅ No new rows (skipped)")
            return table_name, 0

        # Explicit garbage collection before insert
        gc.collect()
        
        inserted_total = insert_dataframe_fast(table_name, df)

        # Clean up
        del df
        gc.collect()

        dt = time.time() - t0
        rate = inserted_total / dt if dt > 0 else 0
        print(f"[{table_name}] ✅ DONE: {inserted_total:,} rows in {dt:.1f}s ({rate:.0f} rows/sec)")
        return table_name, inserted_total

    except Exception as e:
        dt = time.time() - t0
        print(f"[{table_name}] ❌ FAILED after {dt:.1f}s: {e}")
        traceback.print_exc()
        return table_name, 0

# ---------- MAIN ----------
def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    print("🔍 Discovering import tasks...")
    tasks_map = discover_tasks()
    if not tasks_map:
        print("❌ No tasks found.")
        return

    table_names = list(tasks_map.keys())
    total_files = sum(len(payload["files_info"]) for payload in tasks_map.values())
    print(f"📊 Found {len(table_names)} tables, {total_files:,} total files")
    
    enable_bulk_db_settings()

    print(f"🗑️  Dropping indexes on {len(table_names)} tables...")
    drop_indexes_for_tables(table_names)

    # Build optimized task list (files already pre-filtered)
    all_tasks = []
    for table_name, payload in tasks_map.items():
        files_info = payload["files_info"]
        latest_ts = payload["latest_ts"]
        # TIP: To force full re-import for testing, uncomment: latest_ts = 0
        all_tasks.append((table_name, files_info, latest_ts))

    print(f"🚀 Processing {len(all_tasks)} tables with {MAX_WORKERS} workers (batch={BATCH_SIZE:,})...")
    start = time.time()
    total_inserted = 0
    completed_tables = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_table_task, t) for t in all_tasks]
        
        for future in as_completed(futures):
            try:
                table_name, inserted = future.result()
                total_inserted += inserted
                completed_tables += 1
                
                elapsed = time.time() - start
                rate = total_inserted / elapsed if elapsed > 0 else 0
                
                print(f"📈 Progress: {completed_tables}/{len(all_tasks)} tables, "
                      f"{total_inserted:,} total rows, {rate:.0f} rows/sec avg")
                      
            except Exception as e:
                print(f"❌ Worker error: {e}")
                traceback.print_exc()

    print(f"🔧 Recreating indexes on {len(table_names)} tables...")
    create_indexes_for_tables(table_names)

    elapsed = time.time() - start
    rate = total_inserted / elapsed if elapsed > 0 else 0
    print(f"🎉 ALL DONE in {elapsed:.1f}s. Total: {total_inserted:,} rows ({rate:.0f} rows/sec)")

if __name__ == "__main__":
    main()