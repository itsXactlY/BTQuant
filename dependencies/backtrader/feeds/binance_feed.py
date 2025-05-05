from collections import deque
import queue
import time
import json
import pandas as pd
import numpy as np
from backtrader.dataseries import TimeFrame
from backtrader.feed import DataBase
from backtrader.utils import date2num
import threading

def identify_gaps(df, expected_interval):
    df['timestamp'] = pd.to_datetime(df.index)
    df['time_diff'] = df['timestamp'].diff()
    gaps = df[df['time_diff'] > expected_interval]
    return gaps

class BinanceData(DataBase):
    params = (
        ('drop_newest', False),
        ('update_interval_seconds', 1),
        ('debug', False),
        ('max_buffer', 1)
    )

    _ST_LIVE, _ST_HISTORBACK, _ST_OVER = range(3)

    def __init__(self, store, start_date=None):
        super().__init__()
        self.start_date = start_date
        self._store = store
        self._data = deque(maxlen=self.p.max_buffer)
        self.interval = self._store.get_interval(TimeFrame.Seconds, compression=1)
        if self.interval is None:
            raise ValueError("Unsupported timeframe/compression")
        self.ws_url = store.ws_url
        self._state = self._ST_HISTORBACK if start_date else self._ST_LIVE

    # 
    def handle_websocket_message(self, message):
        try:
            data = json.loads(message)
            
            kline = self._parser_to_kline(data['k']['t'], [
                data['k']['t'],
                data['k']['o'],
                data['k']['h'],
                data['k']['l'],
                data['k']['c'],
                data['k']['v'],
            ])
            
            print('DEBUG :: POPLEFT handle_websocket_message')
            self._data.popleft()
            self._data.append(kline)
            
            if self.p.debug:
                print('received fresh data:', kline)
        except Exception as e:
            print(f"Error handling WebSocket message: {e}")

    # 
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

    # 
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

        self.lines.datetime[0] = date2num(timestamp)
        self.lines.open[0] = open_
        self.lines.high[0] = high
        self.lines.low[0] = low
        self.lines.close[0] = close
        self.lines.volume[0] = volume

        del kline
        return True

    # 
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

    # 
    def _start_live(self):
        print("Starting live data...")
        self._store.start_socket()
        self._store.start_memory_monitor()
        self._state = self._ST_LIVE
        self.put_notification(self.LIVE)
        print("Starting live data and purging historical data...")

    # 
    def haslivedata(self):
        return self._state == self._ST_LIVE and len(self._data) > 0

    # 
    def islive(self):
        return True

    # 
    def start(self):
        DataBase.start(self)

        if self.start_date:
            self._state = self._ST_HISTORBACK
            self.put_notification(self.DELAYED)

            klines = self._store.fetch_ohlcv(
                self._store.symbol,
                self.interval,
                since=int(self.start_date.timestamp() * 1000))
            
            if klines:
                if self.p.drop_newest and klines:
                    klines.pop()
                
                df = pd.DataFrame(klines)
                if df.shape[1] > 6:
                    df.drop(df.columns[6:], axis=1, inplace=True)
                df = self._parser_dataframe(df)
                self._data.extend(df.values.tolist())
        
        self.data_queue = self._store.start_socket()
        
        if not hasattr(self, '_processing_thread') or not self._processing_thread.is_alive():
            self._processing_thread = threading.Thread(
                target=self._process_queue, 
                daemon=True,
                name="BinanceDataProcessor"
            )
            self._processing_thread.start()

        if self._state == self._ST_HISTORBACK:
            self._start_live()
        else:
            self._state = self._ST_LIVE
            self.put_notification(self.LIVE)

    def _process_queue(self):
        """Process data from queue with memory management"""
        cleanup_counter = 0
        while self._state != self._ST_OVER:
            try:
                kline_data = self.data_queue.get(timeout=1)
                timestamp = pd.to_datetime(kline_data[0], unit='ms')
                processed_data = [
                    timestamp,
                    kline_data[1],  # open
                    kline_data[2],  # high
                    kline_data[3],  # low
                    kline_data[4],  # close
                    kline_data[5]   # volume
                ]

                while len(self._data) >= self.p.max_buffer:
                    self._data.popleft()
                    
                self._data.append(processed_data)
                del kline_data
                del processed_data

                cleanup_counter += 1
                if cleanup_counter >= 100:
                    import gc
                    gc.collect()
                    cleanup_counter = 0

            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                print(f"Error in _process_queue: {e}")

    # 
    def _process_websocket_messages(self):
        while not self._state == self._ST_OVER:
            try:
                message = self._store.message_queue.get(timeout=1)
                self.handle_websocket_message(message)
                del message
            except queue.Empty:
                time.sleep(0.1)

