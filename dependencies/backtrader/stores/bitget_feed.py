from collections import deque
import time
import json
import pandas as pd
import numpy as np
from queue import Empty
from backtrader.dataseries import TimeFrame
from backtrader.feed import DataBase
from backtrader.utils import date2num
import threading
# from fastquant.strategys.base import function_trapper

def identify_gaps(df, expected_interval):
    df['timestamp'] = pd.to_datetime(df.index)
    df['time_diff'] = df['timestamp'].diff()
    gaps = df[df['time_diff'] > expected_interval]
    return gaps

class BitgetData(DataBase):
    params = (
        ('drop_newest', False),
        ('update_interval_seconds', 1),
        ('debug', False)
    )

    _ST_LIVE, _ST_HISTORBACK, _ST_OVER = range(3)

    def __init__(self, store, start_date=None):
        super().__init__()
        self.start_date = start_date
        self._store = store
        self._data = deque()
        self.interval = self._store.get_interval(TimeFrame.Seconds, compression=1)
        if self.interval is None:
            raise ValueError("Unsupported timeframe/compression")
        self.ws_url = store.ws_url
        self._state = self._ST_HISTORBACK if start_date else self._ST_LIVE

    # @function_trapper
    def handle_websocket_message(self, message):
        try:
            data = json.loads(message)
            # if self.p.debug:
            #     print(f"Received websocket message: {data}")
            if "action" in data and data["action"] == "snapshot":
                if self.p.debug:
                    print('HACK :: dropping snapshot... dont preprocess old candles...')
                return
            if 'arg' in data and 'channel' in data['arg'] and data['arg']['channel'].startswith('candle'):
                for k in data.get('data', []):
                    kline = self._parser_to_kline(k[0], k)
                    self._data.append(kline)
                    # if self.p.debug:
                    #     print('Received fresh data:', kline)
        except Exception as e:
            if self.p.debug:
                print(f"Error handling WebSocket message: {e}")
            return

    # @function_trapper
    def _load(self):
        if self._state == self._ST_OVER:
            return False
        elif self._state == self._ST_LIVE:
            return self._load_kline()
        elif self._state == self._ST_HISTORBACK:
            if self._load_kline():
                return True
            else:
                self._start_live()

    # @function_trapper
    def _load_kline(self):
        try:
            kline = self._data.popleft()
            if self.p.debug:
                print(f"Processing kline: {kline}")
        except IndexError:
            return None

        timestamp, open_, high, low, close, volume = kline

        # Quick 'n dirty check for missing, None, or NaN values
        values = [timestamp, open_, high, low, close, volume]
        if any(v is None for v in values):
            if self.p.debug:
                print(f"Skipping kline due to None value: {kline}")
            return self._load_kline()
        if any(np.isnan(v) if isinstance(v, (int, float)) else False for v in values):  # Check for NaN
            if self.p.debug:
                print(f"Skipping kline due to NaN value: {kline}")
            return self._load_kline()
        if volume == 0:
            if self.p.debug:
                print(f"Skipping kline due to zero volume: {kline}")
            return self._load_kline()

        # All values are valid, proceed to load the kline
        self.lines.datetime[0] = date2num(timestamp)
        self.lines.open[0] = open_
        self.lines.high[0] = high
        self.lines.low[0] = low
        self.lines.close[0] = close
        self.lines.volume[0] = volume
        return True

    # @function_trapper
    def _parser_dataframe(self, data):
        df = data.copy()
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        return df

    # @function_trapper
    def _parser_to_kline(self, timestamp, kline):
        dt = pd.to_datetime(pd.to_numeric(timestamp), unit='ms', utc=True)
        return [
            dt,
            float(kline[1]),
            float(kline[2]),
            float(kline[3]),
            float(kline[4]),
            float(kline[5])
        ]

    # @function_trapper
    def _start_live(self):
        print("Starting live data...")
        self._store.start_socket()
        self._state = self._ST_LIVE
        self.put_notification(self.LIVE)
        print("Starting live data and purging historical data...")
        threading.Thread(target=self._process_websocket_messages, daemon=True).start()

    # @function_trapper
    def haslivedata(self):
        return self._state == self._ST_LIVE and len(self._data) > 0

    # @function_trapper
    def islive(self):
        return True

    # @function_trapper
    def start(self):
        DataBase.start(self)

        print("Starting WebSocket connection...")
        print("WebSocket connection started.")

        if self.start_date:
            self._state = self._ST_HISTORBACK
            self.put_notification(self.DELAYED)
        self._start_live()

    # @function_trapper
    def _process_websocket_messages(self):
        while True:
            try:
                message = self._store.message_queue.get(timeout=1)
                self.handle_websocket_message(message)
            except Empty:
                time.sleep(1)
