from datetime import datetime
import threading
import pytz
import requests
from queue import Queue
from backtrader.dataseries import TimeFrame
from .binance_feed import BinanceData
import websocket
from fastquant.strategys.base import function_trapper


class BinanceStore(object):
    _GRANULARITIES = {
        (TimeFrame.Seconds, 1): '1s',
        (TimeFrame.Minutes, 1): '1m',
        (TimeFrame.Minutes, 3): '3m',
        (TimeFrame.Minutes, 5): '5m',
        (TimeFrame.Minutes, 15): '15m',
        (TimeFrame.Minutes, 30): '30m',
        (TimeFrame.Minutes, 60): '1h',
        (TimeFrame.Minutes, 120): '2h',
        (TimeFrame.Minutes, 240): '4h',
        (TimeFrame.Minutes, 360): '6h',
        (TimeFrame.Minutes, 480): '8h',
        (TimeFrame.Minutes, 720): '12h',
        (TimeFrame.Days, 1): '1d',
        (TimeFrame.Days, 3): '3d',
        (TimeFrame.Weeks, 1): '1w',
        (TimeFrame.Months, 1): '1M',
    }

    def __init__(self, coin_refer, coin_target):
        self.coin_refer = coin_refer
        self.coin_target = coin_target
        self.symbol = f"{coin_refer}{coin_target}".upper()
        self.ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@kline_{self.get_interval(TimeFrame.Seconds, 1)}"
        print(f"WebSocket URL: {self.ws_url}")

        self.websocket = None
        self.websocket_thread = None
        self.message_queue = Queue()

    @function_trapper
    def getdata(self, start_date=None):
        if not hasattr(self, '_data'):
            self._data = BinanceData(store=self, start_date=start_date)
        return self._data

    @function_trapper
    def get_interval(self, timeframe, compression):
        return self._GRANULARITIES.get((timeframe, compression))

    @function_trapper
    def on_message(self, ws, message):
        self.message_queue.put(message)
        # print("Raw message received:", repr(message))  # check exactly what is received (ping/pong debug...)

    @function_trapper
    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")

    @function_trapper
    def on_close(self, ws, close_status_code, close_msg):
        print(f"WebSocket connection closed: {close_status_code} - {close_msg}")

    @function_trapper
    def on_open(self, ws):
        print("WebSocket connection opened")

    @function_trapper
    def start_socket(self):
        def run_socket():
            import time
            while True:
                try:
                    print("Starting WebSocket connection...")
                    self.websocket = websocket.WebSocketApp(
                        self.ws_url,
                        on_message=self.on_message,
                        on_error=self.on_error,
                        on_close=self.on_close,
                        on_open=self.on_open
                    )

                    self.websocket.run_forever(ping_interval=10, ping_timeout=3)
                except Exception as e:
                    print(f"WebSocket encountered an exception: {e}")
                print("WebSocket disconnected. Reconnecting in 3 seconds...")
                time.sleep(3)
        self.websocket_thread = threading.Thread(target=run_socket, daemon=True)
        self.websocket_thread.start()

    @function_trapper
    def stop_socket(self):
        if self.websocket:
            self.websocket.close()
            print("WebSocket connection closed.")

    @function_trapper
    def fetch_ohlcv(self, symbol, interval, since=None, until=None):
        import time

        print('STORE::FETCH SINCE:', since)
        start_timestamp = since
        data = []
        max_retries = 5

        while True:
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': 1000  # Maximum limit per request
            }

            if start_timestamp:
                params['startTime'] = start_timestamp

            if until:
                params['endTime'] = until

            url = "https://api.binance.com/api/v3/klines"

            retries = 0
            new_data = None
            while retries < max_retries:
                try:
                    response = requests.get(url, params=params, timeout=10)
                    new_data = response.json()
                    break  # exit retry loop if successful
                except (requests.exceptions.RequestException, requests.exceptions.JSONDecodeError) as e:
                    wait_time = 2 ** retries  # exponential backoff
                    print(f"Error fetching data (attempt {retries + 1}/{max_retries}): {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    retries += 1

            if retries == max_retries:
                print("Max retries reached. Exiting fetch.")
                break

            if not new_data or len(new_data) == 0:
                break

            data.extend(new_data)

            # Update the start timestamp for the next request
            start_timestamp = new_data[-1][0] + 1

        if data:
            start_time = datetime.fromtimestamp(data[0][0] / 1000, tz=pytz.UTC)
            end_time = datetime.fromtimestamp(data[-1][0] / 1000, tz=pytz.UTC)
            print(f"Fetched data from {start_time} to {end_time}")

        return data