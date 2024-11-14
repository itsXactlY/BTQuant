from collections import deque
import json
import time
import pandas as pd
from queue import Empty
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
        ('debug', False)
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
            kline = self._parser_to_kline(data['k']['t'], [
                data['k']['t'],
                data['k']['o'],
                data['k']['h'],
                data['k']['l'],
                data['k']['c'],
                data['k']['v'],
            ])
            self._data.extend(kline.values.tolist())
            if self.p.debug:
                print('recieved fresh data:', kline)
        except Exception as e:
            print(f"Error handling WebSocket message: {e}")

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
            # if self.p.debug:
            # print(f"Skipping kline with zero volume: {kline}")
            return self._load_kline()  # check the next kline instead

        self.lines.datetime[0] = date2num(pd.Timestamp(timestamp))
        self.lines.open[0] = open_
        self.lines.high[0] = high
        self.lines.low[0] = low
        self.lines.close[0] = close
        self.lines.volume[0] = volume
        return True


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
        df = pd.DataFrame([[timestamp, kline[1], kline[2], kline[3], kline[4], kline[5]]])
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
        print("WebSocket connection started.")

        if self.start_date:
            self._state = self._ST_HISTORBACK
            self.put_notification(self.DELAYED)

            # Fetch historical data
            klines = self._store.fetch_ohlcv(
                self._store.symbol,
                self._store.get_interval(TimeFrame.Seconds, 1),
                since=int(self.start_date.timestamp() * 1000))

            if self.p.debug:
                print(f"Fetched historical klines: {klines}")

            if klines:
                # Check if klines have the required number of elements before popping
                if self.p.drop_newest and klines:
                    if klines:
                        klines.pop()
                
                df = pd.DataFrame(klines)
                # Identify gaps
                gaps = identify_gaps(df, pd.Timedelta(self.start_date.timestamp() * 1000))
                if not gaps.empty:
                    print(f"Gaps found in the data:")
                    print(gaps)

                if df.shape[1] > 6:
                    df.drop(df.columns[6:], axis=1, inplace=True)
                df = self._parser_dataframe(df)
                self._data.extend(df.values.tolist())
            else:
                print("No historical data fetched")
        else:
            self._start_live()

        # Process the messages from the WebSocket
        threading.Thread(target=self._process_websocket_messages, daemon=True).start()

    def _process_websocket_messages(self):
        while True:
            try:
                message = self._store.message_queue.get(timeout=1)
                self.handle_websocket_message(message)
            except Empty:
                time.sleep(1)