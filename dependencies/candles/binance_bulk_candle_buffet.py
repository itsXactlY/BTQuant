# https://pypi.org/project/binance-historical-data/
# pip install binance-historical-data

import os
import csv
import shutil
import datetime
import time
from concurrent.futures import ThreadPoolExecutor
from binance_historical_data import BinanceDataDumper
from backtrader.dontcommit import database, connection_string, fast_mssql

# === SETUP BINANCE DATA DUMP ===
# data_dumper = BinanceDataDumper(
#     path_dir_where_to_dump="candles/",
#     asset_class="spot",
#     data_type="klines",
#     data_frequency="1m"
# )

# # --- Patch to skip SSL country lookup ---
# from binance_historical_data import data_dumper as bd
# bd.BinanceDataDumper._get_user_country_from_ip = lambda self: "US"

# data_dumper.dump_data(
#     tickers=None,
#     date_start=datetime.date(2016, 1, 1),
#     date_end=datetime.date(2026, 1, 1),
#     is_to_update_existing=False,
#     tickers_to_exclude=[
#         "TUSD", "TRY", "RUB", "PAX", "JPY", "GBG", "FDUSD", "EUR", "ETH",
#         "DOWNUSDT", "BUSD", "BRL", "BNB", "BKWR", "BIDR", "AUD", "BULL",
#         "BEAR", "UP", "DOWN"
#     ]
# )

# data_dumper.delete_outdated_daily_results()

# # === DELETE BULL/BEAR DIRS ===
# def delete_folders_with_keywords(root_folder, keywords):
#     for root, dirs, _ in os.walk(root_folder, topdown=False):
#         for d in dirs:
#             if any(k in d for k in keywords):
#                 shutil.rmtree(os.path.join(root, d), ignore_errors=True)

# delete_folders_with_keywords("candles/", ["BULL", "BEAR", "UP", "DOWN"])

# === CONFIGURATION ===
BATCH_SIZE = 100_000
MAX_WORKERS = 8
BASE_DIRS = ["candles/spot/monthly/klines/", "candles/spot/daily/klines/"]

# === TIMESTAMP CONVERSION UTILITIES ===
def ensure_microseconds(timestamp_value):
    timestamp = int(timestamp_value)
    if timestamp >= 1_000_000_000_000_000:  # already µs
        return timestamp
    elif timestamp >= 1_000_000_000_000:   # ms → µs
        return timestamp * 1000
    elif timestamp >= 1_000_000_000:       # s → µs
        return timestamp * 1_000_000
    else:
        print(f"Warning: Unexpected timestamp format: {timestamp}")
        return timestamp

def validate_timestamp_format(timestamp_str, source_info=""):
    try:
        original = int(timestamp_str)
        converted = ensure_microseconds(original)
        return converted
    except ValueError as e:
        print(f"Error converting timestamp '{timestamp_str}' {source_info}: {e}")
        raise

# === DATABASE UTILS (via fast_mssql) ===
def database_exists():
    try:
        query = f"SELECT COUNT(*) FROM sys.databases WHERE name = '{database}'"
        result = fast_mssql.fetch_data_from_db(connection_string, query)
        return int(result[0][0]) > 0
    except Exception as e:
        print(f"Error checking if database exists: {e}")
        return False

def create_database():
    try:
        query = f"CREATE DATABASE [{database}]"
        fast_mssql.execute_non_query(connection_string.replace(f"DATABASE={database}", "DATABASE=master"), query)
        print(f"Database '{database}' created successfully.")
    except Exception as e:
        print(f"Error creating database: {e}")
        raise

def table_exists(table_name):
    query = f"SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{table_name}'"
    result = fast_mssql.fetch_data_from_db(connection_string, query)
    return int(result[0][0]) > 0

def create_table(table_name):
    create_sql = f"""
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
        INDEX IX_{table_name}_TimestampStart (TimestampStart),
        INDEX IX_{table_name}_Timeframe_Timestamp (Timeframe, TimestampStart)
    )
    """
    fast_mssql.execute_non_query(connection_string, create_sql)
    print(f"Created table {table_name} with microsecond timestamp support and indexes")

def get_latest_timestamp(table_name):
    query = f"SELECT MAX(TimestampEnd) FROM [{table_name}]"
    result = fast_mssql.fetch_data_from_db(connection_string, query)
    return int(result[0][0]) if result and result[0][0] is not None else 0

# === CSV PROCESSING WITH TIMESTAMP CONVERSION ===
def process_files_for_table(args):
    table_name, files_info, latest_timestamp = args
    total_inserted_rows = 0
    conversion_count = 0

    try:
        insert_sql = f"""INSERT INTO [{table_name}] (
            Timeframe, TimestampStart, [Open], [High], [Low], [Close],
            Volume, TimestampEnd, QuoteVolume, Trades,
            TakerBaseVolume, TakerQuoteVolume
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

        for file_path, timeframe in files_info:
            inserted_rows = 0
            file_conversion_count = 0
            batch = []

            with open(file_path, 'r') as file:
                csv_reader = csv.reader(file)

                for row_num, row in enumerate(csv_reader, 1):
                    try:
                        timestamp_start_original = row[0]
                        timestamp_end_original = row[6]

                        timestamp_start = validate_timestamp_format(
                            timestamp_start_original, 
                            f"in {file_path}:row{row_num}:start"
                        )
                        timestamp_end = validate_timestamp_format(
                            timestamp_end_original,
                            f"in {file_path}:row{row_num}:end"
                        )

                        if int(timestamp_start_original) != timestamp_start:
                            file_conversion_count += 1

                        if timestamp_end > latest_timestamp:
                            processed_row = [
                                timeframe,
                                str(timestamp_start),  # must be strings for fast_mssql v1
                                row[1],  # Open
                                row[2],  # High
                                row[3],  # Low
                                row[4],  # Close
                                row[5],  # Volume
                                str(timestamp_end),
                                row[7],  # Quote volume
                                row[8],  # Trades
                                row[9],  # Taker base volume
                                row[10]  # Taker quote volume
                            ]
                            batch.append(processed_row)

                            if len(batch) >= BATCH_SIZE:
                                fast_mssql.bulk_insert(connection_string, insert_sql, batch)
                                inserted_rows += len(batch)
                                batch.clear()
                    
                    except Exception as e:
                        print(f"[{table_name}] Error processing row {row_num} in {file_path}: {e}")
                        print(f"Row data: {row}")
                        continue

                if batch:
                    fast_mssql.bulk_insert(connection_string, insert_sql, batch)
                    inserted_rows += len(batch)

            total_inserted_rows += inserted_rows
            conversion_count += file_conversion_count

            print(f"[{table_name}] Processed {file_path}")
            print(f"  - Inserted: {inserted_rows} rows")
            print(f"  - Timestamp conversions: {file_conversion_count}")

    except Exception as e:
        print(f"[{table_name}] Error processing files: {e}")
        import traceback
        traceback.print_exc()

    print(f"[{table_name}] SUMMARY:")
    print(f"  - Total rows inserted: {total_inserted_rows}")
    print(f"  - Total timestamp conversions: {conversion_count}")
    print(f"  - All timestamps stored as microseconds")

# === MAIN LOGIC ===
def main():
    print("=== CRYPTO DATA IMPORT WITH MICROSECOND TIMESTAMP STANDARDIZATION ===")
    
    if not database_exists():
        print(f"Database '{database}' not found. Creating...")
        create_database()
        time.sleep(2)
        if not database_exists():
            print(f"Failed to create database '{database}'.")
            return
        print(f"Database '{database}' created successfully.")
    else:
        print(f"Database '{database}' already exists.")

    table_timestamps = {}
    for data_directory in BASE_DIRS:
        if not os.path.exists(data_directory):
            continue

        for folder in os.listdir(data_directory):
            full_folder_path = os.path.join(data_directory, folder)
            if not os.path.isdir(full_folder_path):
                continue

            pair_name = folder.replace('_', '')
            table_name = f'{pair_name}_klines'

            if table_name not in table_timestamps:
                if not table_exists(table_name):
                    create_table(table_name)
                    table_timestamps[table_name] = 0
                else:
                    latest_ts = get_latest_timestamp(table_name)
                    table_timestamps[table_name] = latest_ts
                    print(f"{table_name} latest timestamp: {latest_ts} (µs)")

    # Build file processing map
    table_files_map = {}
    for data_directory in BASE_DIRS:
        if not os.path.exists(data_directory):
            continue

        for folder in os.listdir(data_directory):
            pair_name = folder.replace('_', '')
            table_name = f'{pair_name}_klines'
            latest_ts = table_timestamps.get(table_name, 0)
            full_folder_path = os.path.join(data_directory, folder)

            if not os.path.isdir(full_folder_path):
                continue

            for timeframe in os.listdir(full_folder_path):
                tf_path = os.path.join(full_folder_path, timeframe)
                if not os.path.isdir(tf_path):
                    continue

                for file in os.listdir(tf_path):
                    if file.endswith(".csv"):
                        file_path = os.path.join(tf_path, file)
                        table_files_map.setdefault(table_name, []).append((file_path, timeframe))

    # Process all tables
    tasks = []
    for table_name, files_info in table_files_map.items():
        tasks.append((table_name, files_info, table_timestamps.get(table_name, 0)))

    print(f"Processing {len(tasks)} tables using up to {MAX_WORKERS} threads...")
    print("All new data will be stored with microsecond timestamps")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(process_files_for_table, tasks)

    print("=== IMPORT COMPLETED ===")
    print("All timestamps have been standardized to microseconds")

if __name__ == "__main__":
    main()
