from datetime import datetime
import threading
import time
from queue import Queue
from backtrader.dataseries import TimeFrame

import websocket
import json
from .bitget_feed import BitgetData

class BitgetStore(object):
    _GRANULARITIES = {
        (TimeFrame.Seconds, 1): '1s',
        (TimeFrame.Minutes, 1): '1min',
        (TimeFrame.Minutes, 3): '3min',
        (TimeFrame.Minutes, 5): '5min',
        (TimeFrame.Minutes, 15): '15min',
        (TimeFrame.Minutes, 30): '30min',
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
        self.ws_url = "wss://ws.bitget.com/spot/v1/stream"
        print(f"WebSocket URL: {self.ws_url}")

        self.websocket = None
        self.websocket_thread = None
        self.message_queue = Queue()

    
    def getdata(self, start_date=None):
        if not hasattr(self, '_data'):
            self._data = BitgetData(store=self, start_date=start_date)
        return self._data

    
    def get_interval(self, timeframe, compression):
        return self._GRANULARITIES.get((timeframe, compression))

    
    def on_message(self, ws, message):
        try:
            if isinstance(message, str):
                self.message_queue.put(message)
                # print("Raw message received:", repr(message))  # check exactly what is received (ping/pong debug...)
            else:
                self.message_queue.put(json.dumps(message))
        except Exception as e:
            print(f"Error processing WebSocket message: {e}")

    
    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")

    
    def on_close(self, ws, close_status_code, close_msg):
        print(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        self.keep_pinging = False

    
    def on_open(self, ws):
        print(f"{datetime.now()} WebSocket connection opened")
        print("No Warmup for BITGET - HACK :: Caching Live Candles till start - Warming up...")
        granularity = self.get_interval(TimeFrame.Seconds, 1)
        payload = {
            "op": "subscribe",
            "args": [
                {
                    "instType": "SP",
                    "channel": "candle" + granularity,
                    "instId": self.symbol
                }
            ]
        }
        ws.send(json.dumps(payload))
        print(f"Subscribed to {self.symbol} candlestick data with granularity {granularity}")
        self.start_ping(ws)

    
    def start_ping(self, ws):
        self.keep_pinging = True 

        def ping_loop():
            while self.keep_pinging:
                if ws.sock and ws.sock.connected:
                    try:
                        ws.send("ping")
                    except Exception as e:
                        print("Error sending ping:", e)
                        break
                else:
                    break
                time.sleep(30)
        threading.Thread(target=ping_loop, daemon=True).start()

    
    def start_socket(self):
        def run_socket():
            while True:
                try:
                    print("Starting WebSocket connection...")
                    websocket.enableTrace(False)
                    self.websocket = websocket.WebSocketApp(
                        self.ws_url,
                        on_message=self.on_message,
                        on_error=self.on_error,
                        on_close=self.on_close,
                        on_open=self.on_open
                    )
                    self.websocket.run_forever()
                except Exception as e:
                    print(f"WebSocket encountered an exception: {e}")
                print("WebSocket disconnected. Reconnecting in 3 seconds...")
                time.sleep(3)
        self.websocket_thread = threading.Thread(target=run_socket, daemon=True)
        self.websocket_thread.start()

    
    def stop_socket(self):
        if self.websocket:
            self.websocket.close()
            print("WebSocket connection closed.")


    
    def fetch_ohlcv(self, symbol, interval, since=None, until=None):
        '''BITGET IS A BIT HACKY - NOT IN A GOOD WAY...'''
        return