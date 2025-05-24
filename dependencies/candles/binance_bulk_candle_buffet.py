# https://pypi.org/project/binance-historical-data/
# pip install binance-historical-data

import os
import csv
import shutil
import datetime
import pyodbc
from concurrent.futures import ThreadPoolExecutor
from binance_historical_data import BinanceDataDumper
from backtrader.dontcommit import driver, server, database, username, password

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

# # === DATABASE CONNECTION ===
def get_db_connection():
    return pyodbc.connect(
        f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password};TrustServerCertificate=yes;'
    )

def table_exists(cursor, table_name):
    cursor.execute(f"""
    SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = ?
    """, (table_name,))
    return cursor.fetchone()[0] > 0

def get_latest_timestamp(cursor, table_name):
    cursor.execute(f"SELECT MAX(TimestampEnd) FROM [{table_name}]")
    result = cursor.fetchone()[0]
    return result if result else 0

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

# # === CSV PROCESSING ===
BATCH_SIZE = 250000

def process_csv_file(file_info):
    table_name, file_path, timeframe_folder, latest_timestamp = file_info
    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        sql = f"""INSERT INTO [{table_name}] (
            Timeframe, TimestampStart, [Open], [High], [Low], [Close],
            Volume, TimestampEnd, QuoteVolume, Trades,
            TakerBaseVolume, TakerQuoteVolume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            batch = []
            inserted = 0
            for row in csv_reader:
                if int(row[0]) > latest_timestamp:
                    batch.append([timeframe_folder] + row[:11])
                    if len(batch) == BATCH_SIZE:
                        cursor.executemany(sql, batch)
                        connection.commit()
                        inserted += len(batch)
                        batch.clear()
            if batch:
                cursor.executemany(sql, batch)
                connection.commit()
                inserted += len(batch)

        print(f"[{table_name}] Finished {file_path} - Inserted {inserted} rows.")
        cursor.close()
        connection.close()
    except Exception as e:
        print(f"[{table_name}] ERROR {file_path}: {e}")

# === MAIN ===
def main():
    base_dirs = ["candles/spot/monthly/klines/", "candles/spot/daily/klines/"]
    file_info_list = []

    conn = get_db_connection()
    cursor = conn.cursor()

    for data_directory in base_dirs:
        if not os.path.exists(data_directory):
            continue

        folders = [f for f in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, f))]
        for folder in folders:
            pair_name = folder.replace('_', '')
            table_name = f'{pair_name}_klines'

            if not table_exists(cursor, table_name):
                create_table(cursor, table_name)
                print(f"Created table {table_name}")
                latest_ts = 0
            else:
                latest_ts = get_latest_timestamp(cursor, table_name)
                print(f"{table_name} latest ts: {latest_ts}")

            timeframe_folders = os.listdir(os.path.join(data_directory, folder))
            for tf in timeframe_folders:
                tf_path = os.path.join(data_directory, folder, tf)
                if not os.path.isdir(tf_path):
                    continue
                for file in os.listdir(tf_path):
                    if file.endswith(".csv"):
                        file_path = os.path.join(tf_path, file)
                        file_info_list.append((table_name, file_path, tf, latest_ts))

    cursor.close()
    conn.close()

    # MULTITHREADED EXECUTION
    print(f"Starting to process {len(file_info_list)} files with multithreading...")
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        executor.map(process_csv_file, file_info_list)

    print("All files processed.")

if __name__ == "__main__":
    main()
