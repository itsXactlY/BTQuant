import os
import pyodbc
import time
import csv
from backtrader.dontcommit import driver, server, database, username, password

def table_exists(cursor, table_name):
    cursor.execute(f"""
    SELECT COUNT(*) 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_NAME = N'{table_name}'
    """)
    return cursor.fetchone()[0] > 0

def get_latest_timestamp(cursor, table_name):
    cursor.execute(f"SELECT MAX(TimestampEnd) FROM [{table_name}]")
    result = cursor.fetchone()[0]
    return result if result else 0

BATCH_SIZE = 250000

try:
    connection = pyodbc.connect(f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password};TrustServerCertificate=yes;')
    cursor = connection.cursor()
    print("Connected to the BacktraderData database successfully")

    data_directory = 'candles/spot/monthly/klines/' # adjust for ur path
    folders = [f for f in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, f))]
    print(f"Found {len(folders)} folders")

    for folder in folders:
        pair_name = folder.replace('_', '')
        # table_name = 'BTCUSDT_1s_klines' # example for 1s feed(s)
        table_name = f'{pair_name}_klines'
        print(f"Checking table {table_name}")

        if not table_exists(cursor, table_name):
            print(f"Table {table_name} does not exist. Creating it.")
            create_table_sql = f"""
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
            """
            cursor.execute(create_table_sql)
            print(f"Table {table_name} created successfully")
            latest_timestamp = 0
        else:
            print(f"Table {table_name} already exists. Appending data.")
            latest_timestamp = get_latest_timestamp(cursor, table_name)

        print(f"Processing table {table_name}")

        timeframe_folders = [f for f in os.listdir(os.path.join(data_directory, folder)) if os.path.isdir(os.path.join(data_directory, folder, f))]
        print(f"Processing {folder} with {len(timeframe_folders)} timeframes")

        for timeframe_folder in timeframe_folders:
            csv_files = sorted([f for f in os.listdir(os.path.join(data_directory, folder, timeframe_folder)) if f.endswith('.csv')])
            print(f"Found {len(csv_files)} CSV files for {timeframe_folder}")

        for csv_file in csv_files:
            file_path = os.path.join(data_directory, folder, timeframe_folder, csv_file)
            print(f"Processing {file_path}")

            start_time = time.time()
            rows_inserted = 0

            with open(file_path, 'r') as file:
                csv_reader = csv.reader(file)

                sql = f"""INSERT INTO [{table_name}] (Timeframe,
                TimestampStart, 
                [Open], 
                [High], 
                [Low], 
                [Close], 
                Volume, 
                TimestampEnd, 
                QuoteVolume, 
                Trades, 
                TakerBaseVolume, 
                TakerQuoteVolume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

                batch = []
                for row in csv_reader:
                    if int(row[0]) > latest_timestamp:
                        batch.append([timeframe_folder] + row[:11])
                        if len(batch) == BATCH_SIZE:
                            cursor.executemany(sql, batch)
                            rows_inserted += len(batch)
                            elapsed_time = time.time() - start_time
                            print(f"Inserted {rows_inserted} rows. Elapsed time: {elapsed_time:.2f} seconds")
                            connection.commit()
                            batch = []

                if batch:
                    cursor.executemany(sql, batch)
                    rows_inserted += len(batch)
                    connection.commit()

            print(f"Committed changes for {table_name}")
            print(f"Total rows inserted: {rows_inserted}")
            print(f"Total time for {csv_file}: {time.time() - start_time:.2f} seconds")

except Exception as e:
    print(f"An error occurred: {str(e)}")
finally:
    if connection:
        connection.close()
        print("Connection closed")

print("Script execution completed")