# https://pypi.org/project/binance-historical-data/
# pip install binance-historical-data

import os
import csv
import shutil
import datetime
import time
from concurrent.futures import ThreadPoolExecutor
from binance_historical_data import BinanceDataDumper
from backtrader.dontcommit import database, connection_string

# === SETUP BINANCE DATA DUMP ===
data_dumper = BinanceDataDumper(
    path_dir_where_to_dump="candles/",
    asset_class="spot",
    data_type="klines",
    data_frequency="1m"
)

data_dumper.dump_data(
    tickers=None,
    date_start=datetime.date(2016, 1, 1),
    date_end=datetime.date(2026, 1, 1),
    is_to_update_existing=True,
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
        TakerQuoteVolume DECIMAL(28, 8) NOT NULL
    )
    """)


def get_latest_timestamp(cursor, table_name):
    cursor.execute(f"SELECT MAX(TimestampEnd) FROM [{table_name}]")
    result = cursor.fetchone()[0]
    return result if result else 0


# === CSV PROCESSING ===
def process_files_for_table(args):
    table_name, files_info, latest_timestamp = args
    total_inserted_rows = 0

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
                    with open(file_path, 'r') as file:
                        csv_reader = csv.reader(file)
                        batch = []
                        for row in csv_reader:
                            if int(row[0]) > latest_timestamp:
                                batch.append([timeframe] + row[:11])
                                if len(batch) >= BATCH_SIZE:
                                    cursor.executemany(sql, batch)
                                    conn.commit()
                                    inserted_rows += len(batch)
                                    batch.clear()
                        if batch:
                            cursor.executemany(sql, batch)
                            conn.commit()
                            inserted_rows += len(batch)

                    total_inserted_rows += inserted_rows
                    print(f"[{table_name}] Processed {file_path} - Inserted {inserted_rows} rows.")

    except Exception as e:
        print(f"[{table_name}] Error processing files: {e}")

    print(f"[{table_name}] Finished processing all files. Total rows inserted: {total_inserted_rows}")


# === MAIN LOGIC ===
def main():
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
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
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
                            if not table_exists(cursor, table_name):
                                create_table(cursor, table_name)
                                conn.commit()
                                print(f"Created table {table_name}")
                                table_timestamps[table_name] = 0
                            else:
                                latest_ts = get_latest_timestamp(cursor, table_name)
                                table_timestamps[table_name] = latest_ts
                                print(f"{table_name} latest ts: {latest_ts}")

    except Exception as e:
        print(f"Database preparation error: {e}")
        return

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

    tasks = []
    for table_name, files_info in table_files_map.items():
        tasks.append((table_name, files_info, table_timestamps.get(table_name, 0)))

    print(f"Processing {len(tasks)} tables using up to {MAX_WORKERS} threads...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(process_files_for_table, tasks)

    print("All tables processed.")


if __name__ == "__main__":
    main()
