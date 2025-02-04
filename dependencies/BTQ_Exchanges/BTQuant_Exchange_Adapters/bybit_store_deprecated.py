import threading
import requests
from queue import Queue
from backtrader.dataseries import TimeFrame
from .bybit_feed_deprecated import ByBitData
import json
import websocket

class BybitStore(object):
    _GRANULARITIES = {
        (TimeFrame.Seconds, 1): '1',
        (TimeFrame.Minutes, 1): '1',
        (TimeFrame.Minutes, 3): '3',
        (TimeFrame.Minutes, 5): '5',
        (TimeFrame.Minutes, 15): '15',
        (TimeFrame.Minutes, 30): '30',
        (TimeFrame.Minutes, 60): '60',
        (TimeFrame.Minutes, 120): '120',
        (TimeFrame.Minutes, 240): '240',
        (TimeFrame.Minutes, 360): '360',
        (TimeFrame.Minutes, 480): '480',
        (TimeFrame.Minutes, 720): '720',
        (TimeFrame.Days, 1): 'D',
        (TimeFrame.Days, 3): '3D',
        (TimeFrame.Weeks, 1): 'W',
        (TimeFrame.Months, 1): 'M',
    }

    def __init__(self, coin_refer, coin_target):
        self.coin_refer = coin_refer
        self.coin_target = coin_target
        self.symbol = f"{coin_refer}{coin_target}".upper()
        self.ws_url = "wss://stream.bybit.com/v5/public/spot"
        print(f"WebSocket URL: {self.ws_url}")

        self.websocket = None
        self.websocket_thread = None
        self.message_queue = Queue()

    def getdata(self, start_date=None):
        if not hasattr(self, '_data'):
            self._data = ByBitData(store=self, start_date=start_date)
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
        subscription_message = {
            "op": "subscribe",
            "args": [f"kline.{self.get_interval(TimeFrame.Seconds, 1)}.{self.symbol}"]
        }
        ws.send(json.dumps(subscription_message))

    def start_socket(self):
        def run_socket():
            print("Starting WebSocket connection...")
            # websocket.enableTrace(True)
            self.websocket = websocket.WebSocketApp(
                self.ws_url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            self.websocket.run_forever(reconnect=5)

        self.websocket_thread = threading.Thread(target=run_socket, daemon=True)
        self.websocket_thread.start()

    def stop_socket(self):
        if self.websocket:
            self.websocket.close()
            print("WebSocket connection closed.")

    def fetch_ohlcv(self, symbol, interval, since=None):
        url = f"https://api.bybit.com/v5/market/kline?symbol={symbol}&interval={interval}"
        if since:
            url += f"&from={since}"
        response = requests.get(url)
        try:
            data = response.json()
            if 'ret_code' in data and data['ret_code'] != 0:
                raise Exception(f"Error fetching data: {data['ret_msg']}")
            return data['result']
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f"Error fetching OHLCV data: {e}")
            return []