# https://pypi.org/project/binance-historical-data/
# pip install binance-historical-data
from binance_historical_data import BinanceDataDumper
import datetime

data_dumper = BinanceDataDumper(
    path_dir_where_to_dump="candles/",
    asset_class="spot",  # spot, um, cm
    data_type="klines",  # aggTrades, klines, trades
    data_frequency="1s"
)

data_dumper.dump_data(
    tickers=["BTCUSDT"], # Or tickers=None for all tickers excluding blacklist
    date_start=datetime.date(year=2024, month=1, day=1),
    date_end=datetime.date(year=2025, month=1, day=1),
    is_to_update_existing=True,
    tickers_to_exclude=["TUSD", "TRY", "RUB", "PAX", "JPY", "GBG", "FDUSD", "EUR", "ETH", "DOWNUSDT", "BUSD", "BRL", "BNB", "BKWR", "BIDR", "AUD", "BULL", "BEAR", "UP", "DOWN"])


data_dumper.delete_outdated_daily_results()


import os
import shutil

# Keywords to look for in folder names
keywords = ["BULL", "BEAR", "UP", "DOWN"]

def delete_folders_with_keywords(root_folder, keywords):
    for root, dirs, files in os.walk(root_folder, topdown=False):
        for dir_name in dirs:
            if any(keyword in dir_name for keyword in keywords):
                dir_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(dir_path)
                    print(f"Deleted folder: {dir_path}")
                except Exception as e:
                    print(f"Error deleting folder {dir_path}: {e}")

# Path to the root folder where the script will start
root_folder_path = "candles/"

# Run the function
delete_folders_with_keywords(root_folder_path, keywords)
