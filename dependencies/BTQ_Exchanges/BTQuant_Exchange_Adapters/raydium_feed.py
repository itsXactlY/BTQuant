from collections import deque
import time
import pandas as pd
from backtrader.feed import DataBase
from backtrader.utils import date2num
import threading

class RaydiumData(DataBase):
    params = (
        ('poll_interval', 5),  # seconds between API requests
        ('debug', False)
    )
    
    _ST_LIVE, _ST_HISTORBACK, _ST_OVER = range(3)

    def __init__(self, store, start_date=None):
        self.store = store
        self.token_info = store.token_info
        self._data = deque()

    def start(self):
        super().start()
        self._state = self._ST_LIVE
        self.put_notification(self.LIVE)
        
        threading.Thread(target=self._poll_api, daemon=True).start()

    def _poll_api(self):
        while True:
            try:
                self.store.fetch_latest_price()
                price = self.token_info.price
                timestamp = time.time()
                kline = [timestamp, price, price, price, price, 1]  # Dummy OHLCV
                self._data.append(kline)
                if self.p.debug:
                    print(f"Polled data: {kline}")
            except Exception as e:
                print(f"Error polling Raydium API: {e}")
            time.sleep(self.p.poll_interval)

    def _load(self):
        if self._state != self._ST_LIVE:
            return False

        try:
            kline = self._data.popleft()
        except IndexError:
            return None

        timestamp, open_, high, low, close, volume = kline

        self.lines.datetime[0] = date2num(pd.Timestamp(timestamp, unit='s'))
        self.lines.open[0] = open_
        self.lines.high[0] = high
        self.lines.low[0] = low
        self.lines.close[0] = close
        self.lines.volume[0] = volume
        return True
