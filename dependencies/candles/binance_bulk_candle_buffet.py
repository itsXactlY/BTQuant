# https://pypi.org/project/binance-historical-data/
# pip install binance-historical-data

import gc
import os
import csv
import shutil
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from binance_historical_data import BinanceDataDumper
from backtrader.dontcommit import database, connection_string

# === SETUP BINANCE DATA DUMP ===
data_dumper = BinanceDataDumper(
    path_dir_where_to_dump="candles/",
    asset_class="spot",
    data_type="klines",
    data_frequency="1m"
)

# --- Patch to skip SSL country lookup ---
from binance_historical_data import data_dumper as bd
bd.BinanceDataDumper._get_user_country_from_ip = lambda self: "EU"

data_dumper.dump_data(
    tickers=None,
    date_start=datetime.date(2016, 1, 1),
    date_end=datetime.date(2026, 1, 1),
    is_to_update_existing=False,
    tickers_to_exclude=["TUSD", "TRY", "RUB", "PAX", "JPY", "GBG", "FDUSD", "EUR", "ETH", "DOWNUSDT", "BUSD", "BRL", "BNB", "BKWR", "BIDR", "AUD", "BULL", "BEAR", "UP", "DOWN"]
)

data_dumper.delete_outdated_daily_results()

# === DELETE BULL/BEAR DIRS ===
def delete_folders_with_keywords(root_folder, keywords):
    for root, dirs, _ in os.walk(root_folder, topdown=False):
        for d in dirs:
            if any(k in d for k in keywords):
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)

delete_folders_with_keywords("candles/", ["BULL", "BEAR", "UP", "DOWN"])

# === CONFIGURATION ===
BATCH_SIZE = 100000
MAX_WORKERS = 8
BASE_DIRS = ["candles/spot/monthly/klines/", "candles/spot/daily/klines/"]

# === TIMESTAMP CONVERSION UTILITIES ===
def ensure_microseconds(timestamp_value):
    timestamp = int(timestamp_value)

    if timestamp >= 1000000000000000:
        return timestamp
    elif timestamp >= 1000000000000:
        return timestamp * 1000
    elif timestamp >= 1000000000:
        return timestamp * 1000000
    else:
        print(f"Warning: Unexpected timestamp format: {timestamp}")
        return timestamp

def validate_timestamp_format(timestamp_str, source_info=""):
    """
    Validate and convert timestamp to microseconds with logging
    """
    try:
        original = int(timestamp_str)
        converted = ensure_microseconds(original)
        
        # if original != converted:
        #     print(f"Timestamp conversion: {original} -> {converted} {source_info}")
        
        return converted
    except ValueError as e:
        print(f"Error converting timestamp '{timestamp_str}' {source_info}: {e}")
        raise

# === DATABASE UTILS ===
import pyodbc
def get_db_connection():
    return pyodbc.connect(connection_string)

def get_master_db_connection():
    master_conn_str = connection_string.replace(f'DATABASE={database}', 'DATABASE=master')
    return pyodbc.connect(master_conn_str)

def database_exists():
    try:
        with get_master_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM sys.databases WHERE name = ?", (database,))
                return cursor.fetchone()[0] > 0
    except Exception as e:
        print(f"Error checking if database exists: {e}")
        return False

def create_database():
    try:
        with get_master_db_connection() as conn:
            conn.autocommit = True
            with conn.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE [{database}]")
        print(f"Database '{database}' created successfully.")
    except Exception as e:
        print(f"Error creating database: {e}")
        raise

def table_exists(cursor, table_name):
    cursor.execute("SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = ?", (table_name,))
    return cursor.fetchone()[0] > 0

def create_table(cursor, table_name):
    cursor.execute(f"""
    CREATE TABLE [{table_name}] (
        CandleID INT PRIMARY KEY IDENTITY(1,1),
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
        TakerQuoteVolume DECIMAL(28, 8) NOT NULL,
        
        -- Add indexes for better query performance
        INDEX IX_{table_name}_TimestampStart (TimestampStart),
        INDEX IX_{table_name}_Timeframe_Timestamp (Timeframe, TimestampStart)
    )
    """)
    print(f"Created table {table_name} with microsecond timestamp support and indexes")

def get_latest_timestamp(cursor, table_name):
    cursor.execute(f"SELECT MAX(TimestampEnd) FROM [{table_name}]")
    result = cursor.fetchone()[0]
    return result if result else 0

# === CSV PROCESSING WITH TIMESTAMP CONVERSION ===

def process_files_for_table(args):
    table_name, files_info, latest_timestamp = args
    total_inserted_rows = 0
    conversion_count = 0

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                sql = f"""INSERT INTO [{table_name}] (
                    Timeframe, TimestampStart, [Open], [High], [Low], [Close],
                    Volume, TimestampEnd, QuoteVolume, Trades,
                    TakerBaseVolume, TakerQuoteVolume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

                for file_path, timeframe in files_info:
                    inserted_rows = 0
                    file_conversion_count = 0

                    with open(file_path, 'r') as file:
                        csv_reader = csv.reader(file)
                        batch = []

                        for row_num, row in enumerate(csv_reader, 1):
                            try:
                                ts_start = validate_timestamp_format(row[0], f"{file_path}:row{row_num}:start")
                                ts_end   = validate_timestamp_format(row[6], f"{file_path}:row{row_num}:end")

                                if int(row[0]) != ts_start:
                                    file_conversion_count += 1

                                # Only skip if really older than latest_timestamp
                                if ts_end <= latest_timestamp:
                                    continue

                                processed_row = [
                                    timeframe,
                                    ts_start,
                                    row[1], row[2], row[3], row[4],
                                    row[5],
                                    ts_end,
                                    row[7], row[8], row[9], row[10]
                                ]

                                batch.append(processed_row)

                                if len(batch) >= BATCH_SIZE:
                                    cursor.executemany(sql, batch)
                                    conn.commit()
                                    inserted_rows += len(batch)
                                    batch.clear()

                            except Exception as e:
                                print(f"[{table_name}] Error processing row {row_num} in {file_path}: {e}")
                                continue

                        if batch:
                            cursor.executemany(sql, batch)
                            conn.commit()
                            inserted_rows += len(batch)
                    # ✅ free memory after each CSV
                    del csv_reader
                    del batch
                    gc.collect()
                    
                    total_inserted_rows += inserted_rows
                    conversion_count += file_conversion_count
                    print(f"[{table_name}] Processed {file_path} - Inserted: {inserted_rows}, Converted: {file_conversion_count}")

    except Exception as e:
        print(f"[{table_name}] Fatal error: {e}")
        import traceback
        traceback.print_exc()

    print(f"[{table_name}] SUMMARY: Total inserted: {total_inserted_rows}, Timestamp conversions: {conversion_count}")


def main():
    # Prepare tables and latest timestamps
    table_timestamps = {}
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            for data_directory in BASE_DIRS:
                if not os.path.exists(data_directory):
                    continue

                for folder in os.listdir(data_directory):
                    full_path = os.path.join(data_directory, folder)
                    if not os.path.isdir(full_path):
                        continue

                    pair_name = folder.replace("_", "")
                    table_name = f"{pair_name}_klines"

                    if not table_exists(cursor, table_name):
                        create_table(cursor, table_name)
                        conn.commit()
                        table_timestamps[table_name] = 0
                    else:
                        table_timestamps[table_name] = get_latest_timestamp(cursor, table_name)

    # Build task list
    tasks = []
    for data_directory in BASE_DIRS:
        if not os.path.exists(data_directory):
            continue

        for folder in os.listdir(data_directory):
            pair_name = folder.replace("_", "")
            table_name = f"{pair_name}_klines"
            latest_ts = table_timestamps.get(table_name, 0)
            full_folder_path = os.path.join(data_directory, folder)

            if not os.path.isdir(full_folder_path):
                continue

            files_info = []
            for timeframe in os.listdir(full_folder_path):
                tf_path = os.path.join(full_folder_path, timeframe)
                if not os.path.isdir(tf_path):
                    continue

                for file in os.listdir(tf_path):
                    if file.endswith(".csv"):
                        files_info.append((os.path.join(tf_path, file), timeframe))

            if files_info:
                tasks.append((table_name, files_info, latest_ts))

    print(f"Processing {len(tasks)} tables with up to {MAX_WORKERS} threads...")

    # Threaded execution with exception visibility
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_files_for_table, t) for t in tasks]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Thread error: {e}")

    print("=== ALL DATA IMPORTED ===")

if __name__ == "__main__":
    main()