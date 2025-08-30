# https://pypi.org/project/binance-historical-data/
# pip install binance-historical-data

import gc
import os
import shutil
import datetime
from binance_historical_data import BinanceDataDumper

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
gc.collect()
