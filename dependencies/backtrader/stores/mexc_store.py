from datetime import datetime
import threading
import pytz
import requests
import time
import json
from queue import Queue
from backtrader.dataseries import TimeFrame
from backtrader.feeds.mexc_feed import MexcData
import websocket



class MexcStore(object):
    _GRANULARITIES = {
        (TimeFrame.Seconds, 1): 'Min1',
        (TimeFrame.Minutes, 1): 'Min1',
        (TimeFrame.Minutes, 3): 'Min3',
        (TimeFrame.Minutes, 5): 'Min5',
        (TimeFrame.Minutes, 15): 'Min15',
        (TimeFrame.Minutes, 30): 'Min30',
        (TimeFrame.Minutes, 60): 'Hour1',
        (TimeFrame.Minutes, 120): 'Hour2',
        (TimeFrame.Minutes, 240): 'Hour4',
        (TimeFrame.Minutes, 360): 'Hour6',
        (TimeFrame.Minutes, 480): 'Hour8',
        (TimeFrame.Minutes, 720): 'Hour12',
        (TimeFrame.Days, 1): 'Day1',
        (TimeFrame.Days, 3): 'Day3',
        (TimeFrame.Weeks, 1): 'Week1',
        (TimeFrame.Months, 1): 'Month1',
    }

    def __init__(self, coin_refer, coin_target):
        self.coin_refer = coin_refer
        self.coin_target = coin_target
        self.symbol = f"{coin_refer}{coin_target}".upper()
        self.ws_url = "wss://wbs.mexc.com/ws"
        print(f"WebSocket URL: {self.ws_url}")

        self.websocket = None
        self.websocket_thread = None
        self.message_queue = Queue()

    # 
    def getdata(self, start_date=None):
        if not hasattr(self, '_data'):
            self._data = MexcData(store=self, start_date=start_date)
        return self._data

    # 
    def get_interval(self, timeframe, compression):
        return self._GRANULARITIES.get((timeframe, compression))

    # 
    def on_message(self, ws, message):
        try:
            if isinstance(message, str):
                self.message_queue.put(message)
                # print("Raw message received:", repr(message))  # check exactly what is received (ping/pong debug...)
            else:
                self.message_queue.put(json.dumps(message))
        except Exception as e:
            print(f"Error processing WebSocket message: {e}")

    # 
    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")

    # 
    def on_close(self, ws, close_status_code, close_msg):
        print(f"WebSocket connection closed: {close_status_code} - {close_msg}")

    # 
    def on_open(self, ws):
        print("WebSocket connection opened")

        payload = {
            "method": "SUBSCRIPTION",
            "params": [f"spot@public.kline.v3.api@{self.symbol}@{self.get_interval(TimeFrame.Minutes, 1)}"],
            "id": 1
        }
        ws.send(json.dumps(payload))

    # 
    def start_socket(self):
        def run_socket():
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

    # 
    def stop_socket(self):
        if self.websocket:
            self.websocket.close()
            print("WebSocket connection closed.")

    # 
    def fetch_ohlcv(self, symbol, interval, since=None, until=None):
        print('STORE::FETCH SINCE:', since)
        print(f"Fetching historical data for {symbol} with interval {interval}")

        if until is None:
            until = int(time.time() * 1000)
        if since is None:
            since = until - (2 * 60 * 60 * 1000)
        
        params = {
            'symbol': symbol,
            'interval': '1m',
            'limit': 1000,
            'startTime': since,
            'endTime': until
        }
        
        url = "https://api.mexc.com/api/v3/klines"
        print(f"Request URL: {url}")
        print(f"Request parameters: {params}")
        
        max_retries = 5
        retries = 0
        data = []
        
        while retries < max_retries:
            try:
                response = requests.get(url, params=params, timeout=10)
                if response.status_code != 200:
                    print(f"API error: {response.status_code} - {response.text}")
                    retries += 1
                    time.sleep(2 ** retries)  # exponential backoff
                    continue
                    
                data = response.json()
                print(f"Response received: {len(data)} records")
                break  # exit retry loop if successful
            except (requests.exceptions.RequestException, requests.exceptions.JSONDecodeError) as e:
                wait_time = 2 ** retries  # exponential backoff
                print(f"Error fetching data (attempt {retries + 1}/{max_retries}): {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1

        if retries == max_retries:
            print("Max retries reached. Returning empty dataset.")
            return []

        if data:
            start_time = datetime.fromtimestamp(data[0][0] / 1000, tz=pytz.UTC)
            end_time = datetime.fromtimestamp(data[-1][0] / 1000, tz=pytz.UTC)
            print(f"Fetched {len(data)} candles from {start_time} to {end_time}")
        return data
