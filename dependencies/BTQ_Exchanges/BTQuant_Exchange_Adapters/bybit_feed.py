from collections import deque
import json
import time
import pandas as pd
from queue import Empty
from backtrader.dataseries import TimeFrame
from backtrader.feed import DataBase
from backtrader.utils import date2num
import threading

class ByBitData(DataBase):
    params = (
        ('drop_newest', False),
        ('update_interval_seconds', 1),
        ('debug', True)
    )

    _ST_LIVE, _ST_HISTORBACK, _ST_OVER = range(3)

    def __init__(self, store, start_date=None):
        self.start_date = start_date
        self._store = store
        self._data = deque()
        self.interval = self._store.get_interval(TimeFrame.Seconds, compression=1)
        if self.interval is None:
            raise ValueError("Unsupported timeframe/compression")
        self.ws_url = store.ws_url

    def handle_websocket_message(self, message):
        try:
            data = json.loads(message)  # Parse the JSON message
            if self.p.debug:
                print(f"Received websocket message: {data}")
            if 'data' in data and isinstance(data['data'], list):
                for kline in data['data']:
                    kline_data = self._parser_to_kline(
                        kline['timestamp'],
                        [
                            kline['timestamp'],
                            kline['open'],
                            kline['high'],
                            kline['low'],
                            kline['close'],
                            kline['volume']
                        ]
                    )
                    self._data.append(kline_data.values.tolist()[0])
                    if self.p.debug:
                        print('Received fresh data:', kline_data)
        except Exception as e:
            print(f"Error handling WebSocket message: {e}")
            print(f"Message content: {message}")
            import traceback
            traceback.print_exc()

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

    def _load_kline(self):
        try:
            kline = self._data.popleft()
            if self.p.debug:
                print(f"Processing kline: {kline}")
        except IndexError:
            return None

        timestamp, open_, high, low, close, volume = kline

        # Skip processing if the volume is zero
        if volume == 0:
            if self.p.debug:
                print(f"Skipping kline with zero volume: {kline}")
            return self._load_kline()  # Load the next kline instead

        self.lines.datetime[0] = date2num(pd.Timestamp(timestamp, unit='ms'))
        self.lines.open[0] = open_
        self.lines.high[0] = high
        self.lines.low[0] = low
        self.lines.close[0] = close
        self.lines.volume[0] = volume
        return True

    def _parser_dataframe(self, data):
        if isinstance(data, list):
            # aLca :: hack for the first 6 columns are what we need
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'ignore'])
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]  # dropping the extra column
        elif isinstance(data, pd.DataFrame):
            df = data
            if len(df.columns) > 6:
                df = df.iloc[:, :6]  # Keep only the first 6 columns
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df

    def _parser_to_kline(self, timestamp, kline):
        df = pd.DataFrame([kline], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return self._parser_dataframe(df)

    def _start_live(self):
        print("Starting live data...")
        self._store.start_socket()
        self._state = self._ST_LIVE
        self.put_notification(self.LIVE)

    def haslivedata(self):
        return self._state == self._ST_LIVE and len(self._data) > 0

    def islive(self):
        return True

    def start(self):
        DataBase.start(self)

        print("Starting WebSocket connection...")

        if self.start_date:
            self._state = self._ST_HISTORBACK
            self.put_notification(self.DELAYED)

            # historical data
            klines = self._store.fetch_ohlcv(
                self._store.symbol,
                self._store.get_interval(TimeFrame.Seconds, 1),
                since=int(self.start_date.timestamp() * 1000))

            if self.p.debug:
                print(f"Fetched historical klines: {klines}")

            if klines:
                if self.p.drop_newest and klines:
                    klines.pop()

                df = self._parser_dataframe(klines)
                self._data.extend(df.values.tolist())
            else:
                print("No historical data fetched")
        else:
            self._start_live()
        threading.Thread(target=self._process_websocket_messages, daemon=True).start()

    def _process_websocket_messages(self):
        while True:
            try:
                message = self._store.message_queue.get(timeout=1)
                self.handle_websocket_message(message)
                time.sleep(1)
            except Empty:
                time.sleep(1)