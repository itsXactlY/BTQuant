import threading
import requests
from queue import Queue
from backtrader.dataseries import TimeFrame
from .binance_feed import BinanceData
import websocket

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

    def getdata(self, start_date=None):
        if not hasattr(self, '_data'):
            self._data = BinanceData(store=self, start_date=start_date)
        return self._data

    def get_interval(self, timeframe, compression):
        return self._GRANULARITIES.get((timeframe, compression))

    def on_message(self, ws, message):
        self.message_queue.put(message)

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print(f"WebSocket connection closed: {close_status_code} - {close_msg}")

    def on_open(self, ws):
        print("WebSocket connection opened")

    def start_socket(self):
        def run_socket():
            print("Starting WebSocket connection...")
            self.websocket = websocket.WebSocketApp(self.ws_url,
                                                    on_message=self.on_message,
                                                    on_error=self.on_error,
                                                    on_close=self.on_close,
                                                    on_open=self.on_open)
            self.websocket.run_forever(reconnect=5)

        self.websocket_thread = threading.Thread(target=run_socket, daemon=True)
        self.websocket_thread.start()

    def stop_socket(self):
        if self.websocket:
            self.websocket.close()
            print("WebSocket connection closed.")

    def fetch_ohlcv(self, symbol, interval, since=None):
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}"
        if since:
            url += f"&startTime={since}"
        response = requests.get(url)
        data = response.json()
        return data