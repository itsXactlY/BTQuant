import threading
import time
from backtrader.dataseries import TimeFrame
from queue import Queue, Empty
import websocket
import threading
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
        self.websocket = None
        self.websocket_thread = None
        self.message_queue = Queue(maxsize=100000)
        self._ping_thread = None
        
        self._ws_options = {
            'ping_interval': None,  # We handle pings manually
            'ping_timeout': None,   # Avoid automatic ping logic
        }
        
        # performance stats
        self._message_count = 0
        self._connection_count = 0
        self._last_stats_time = time.time()
    
    def getdata(self, start_date=None):
        if not hasattr(self, '_data'):
            self._data = MexcData(store=self, start_date=start_date)
        return self._data

    def get_interval(self, timeframe, compression):
        return self._GRANULARITIES.get((timeframe, compression))

    def on_open(self, ws):
        print("WebSocket connection opened.")
        payload = {
            "method": "SUBSCRIPTION",
            "params": [f"spot@public.deals.v3.api@{self.symbol}"],
            "id": 1
        }
        ws.send(json.dumps(payload))

        if self._ping_thread is None:
            def ping_loop():
                while True:
                    time.sleep(30)
                    try:
                        ws.send(json.dumps({"method": "PING", "id": 2}))
                    except Exception:
                        break
            self._ping_thread = threading.Thread(target=ping_loop, daemon=True)
            self._ping_thread.start()

    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print(f"WebSocket connection closed: {close_status_code} - {close_msg}")
    
    def on_message(self, ws, message):
        """Optimized message handler to reduce processing overhead"""
        try:
            self._message_count += 1
            self.message_queue.put_nowait(message)
        except:
            # Queue full, log but dont block
            if self._message_count % 1000 == 0:  # Log every 1000th overflow
                print(f"Warning: Message queue full, dropping messages")
        
        # stats reporting - as for now
        # now = time.time()
        # if now - self._last_stats_time > 15:
        #     elapsed = now - self._last_stats_time
        #     rate = self._message_count / elapsed if elapsed > 0 else 0
        #     print(f"WebSocket stats: Received {self._message_count} messages ({rate:.1f}/sec)")
        #     self._message_count = 0
        #     self._last_stats_time = now
    
    def start_socket(self):
        """Start WebSocket with optimized connection settings"""
        def run_socket():
            while True:
                try:
                    self._connection_count += 1
                    print(f"Starting WebSocket connection (attempt {self._connection_count})...")
                    
                    self.websocket = websocket.WebSocketApp(
                        self.ws_url,
                        on_open=self.on_open,
                        on_message=self.on_message,
                        on_error=self.on_error,
                        on_close=self.on_close
                    )
                    
                    if self._connection_count > 1:
                        while not self.message_queue.empty():
                            try:
                                self.message_queue.get_nowait()
                            except Empty:
                                break
                    
                    self.websocket.run_forever(**self._ws_options)
                    
                    print("WebSocket disconnected. Reconnecting in 3 seconds...")
                    time.sleep(3)
                except Exception as e:
                    print(f"Error in websocket thread: {e}")
                    time.sleep(5)
        
        self.websocket_thread = threading.Thread(target=run_socket, daemon=True)
        self.websocket_thread.start()