from collections import deque
import time
import pandas as pd
from queue import Queue, Empty
from backtrader.dataseries import TimeFrame
from backtrader.feed import DataBase
from backtrader.utils import date2num
import threading
from collections import deque
import time
from web3 import Web3

class PancakeSwapData(DataBase):
    params = (
        ('drop_newest', False),
        ('update_interval_seconds', 1),
        ('debug', False)
    )

    _ST_LIVE, _ST_HISTORBACK, _ST_OVER = range(3)

    def __init__(self, store, token_address, start_date=None):
        self.start_date = start_date
        self._store = store
        self._data = deque()
        self.token_address = Web3.to_checksum_address(token_address)
        self.interval = self._store.get_interval(TimeFrame.Seconds, compression=1)
        if self.interval is None:
            raise ValueError("Unsupported timeframe/compression")
        self.ws_url = store.ws_url
        self.message_queue = store.message_queue

    def handle_websocket_message(self, price_data):
        try:
            if self.p.debug:
                print(f"Received price data: {price_data}")
            kline = self._parser_to_kline(price_data['timestamp'], [
                price_data['timestamp'],
                price_data['open'],
                price_data['high'],
                price_data['low'],
                price_data['close'],
                price_data['volume'],
            ])
            self._data.extend(kline.values.tolist())
            if self.p.debug:
                print('received fresh data:', kline)
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
        # if volume == 0:
        #     if self.p.debug:
        #         print(f"Skipping kline with zero volume") #: {kline}"
        #     return self._load_kline()  # Load the next kline instead

        self.lines.datetime[0] = date2num(pd.Timestamp(timestamp, unit='s'))
        self.lines.open[0] = open_
        self.lines.high[0] = high
        self.lines.low[0] = low
        self.lines.close[0] = close
        self.lines.volume[0] = volume
        return True

    def _parser_dataframe(self, data):
        df = data.copy()
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
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
        print("Starting live data...")  # Debugging
        self._store.start_socket(self.token_address)
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

            # Fetch historical data (you may need to implement this method in the store)
            klines = self._store.fetch_ohlcv(
                self.token_address,
                self._store.get_interval(TimeFrame.Seconds, 1),
                since=int(self.start_date.timestamp()))

            if self.p.debug:
                print(f"Fetched historical klines: {klines}")

            if klines:
                if self.p.drop_newest and klines:
                    klines.pop()

                df = pd.DataFrame(klines)
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
                message = self.message_queue.get(timeout=1)
                self.handle_websocket_message(message)
            except Empty:
                time.sleep(1)




# Factory and Pair ABIs
FACTORY_ABI = [{
    'inputs': [{'internalType': 'address', 'name': '', 'type': 'address'},
               {'internalType': 'address', 'name': '', 'type': 'address'}],
    'name': 'getPair',
    'outputs': [{'internalType': 'address', 'name': '', 'type': 'address'}],
    'stateMutability': 'view',
    'type': 'function'
}]
PAIR_ABI = [
    {'inputs': [], 'name': 'getReserves', 'outputs': [
        {'internalType': 'uint112', 'name': '_reserve0', 'type': 'uint112'},
        {'internalType': 'uint112', 'name': '_reserve1', 'type': 'uint112'},
        {'internalType': 'uint32', 'name': '_blockTimestampLast', 'type': 'uint32'}
    ], 'stateMutability': 'view', 'type': 'function'},
    {'inputs': [], 'name': 'token0', 'outputs': [{'internalType': 'address', 'name': '', 'type': 'address'}],
     'stateMutability': 'view', 'type': 'function'},
    {'inputs': [], 'name': 'token1', 'outputs': [{'internalType': 'address', 'name': '', 'type': 'address'}],
     'stateMutability': 'view', 'type': 'function'},
    {'anonymous': False, 'inputs': [
        {'indexed': False, 'internalType': 'uint112', 'name': 'reserve0', 'type': 'uint112'},
        {'indexed': False, 'internalType': 'uint112', 'name': 'reserve1', 'type': 'uint112'}
    ], 'name': 'Sync', 'type': 'event'}
]