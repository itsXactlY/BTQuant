from datetime import datetime
import threading
import pytz
import requests
import queue
from backtrader.dataseries import TimeFrame
from backtrader.feeds.binance_feed import BinanceData
import websocket
import json
import time


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
        self._data = None
        self.broker = None
        self._running = False
        self.websocket = None
        self.websocket_thread = None
        self._ST_OVER = 'OVER'
        self._state = None
        
        self.q_streaming = queue.Queue()
        self.q_store = queue.Queue()
        self.q_notifications = queue.Queue()

    def getdata(self, start_date=None):
        if self._data is None:
            self._data = BinanceData(store=self, start_date=start_date)
        return self._data

    def get_interval(self, timeframe, compression):
        return self._GRANULARITIES.get((timeframe, compression))
        
    def put_notification(self, msg, *args, **kwargs):
        self.q_notifications.put((msg, args, kwargs))
        
    def get_notifications(self):
        """Return pending notifications"""
        notifications = []
        while True:
            try:
                notifications.append(self.q_notifications.get_nowait())
            except queue.Empty:
                break
        return notifications

    def start_socket(self, timeframe=TimeFrame.Seconds, compression=1):
        """Start websocket connection and return queue for data consumption"""
        if self._running:
            return self.q_store
            
        interval = self.get_interval(timeframe, compression)
        if not interval:
            self.put_notification(f"Unsupported timeframe-compression combination: {timeframe}-{compression}")
            return None
            
        self.ws_url = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@kline_{interval}"
        print(f"WebSocket URL: {self.ws_url}")
        
        self._running = True
        t = threading.Thread(target=self._t_streaming_listener)
        t.daemon = True
        t.start()
        self.websocket_thread = t
        
        # processing thread that moves data from streaming queue to store queue
        t = threading.Thread(target=self._t_data_processor)
        t.daemon = True
        t.start()
        self._processing_thread = t
        
        return self.q_store

    def _t_streaming_listener(self):
        """Thread that handles the websocket connection and pushes data to streaming queue"""
        retry_count = 0
        max_retries = 5
        
        while self._running and retry_count < max_retries:
            try:
                websocket.enableTrace(False)
                ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close
                )
                self.websocket = ws
                ws.run_forever(ping_interval=10, ping_timeout=5, reconnect=5)
                
                # If we get here, the websocket was closed
                if not self._running:
                    break
                    
                retry_count += 1
                wait_time = 2 ** retry_count  # exponential backoff
                print(f"WebSocket disconnected, retrying in {wait_time} seconds ({retry_count}/{max_retries})")
                time.sleep(wait_time)
                
            except Exception as e:
                self.put_notification(f"WebSocket error: {e}")
                retry_count += 1
                time.sleep(2 ** retry_count)
        
        if retry_count >= max_retries:
            self.put_notification("Max retries reached for WebSocket connection")

    def _t_data_processor(self):
        """Thread that processes data from streaming queue to store queue"""
        while self._running:
            try:
                message = self.q_streaming.get(timeout=1.0)
                if message is None:  # sentinel value for shutdown
                    break
                    
                # Process the message
                try:
                    data = json.loads(message)
                    if 'k' in data:
                        kline_data = (
                            data['k']['t'],  # timestamp
                            float(data['k']['o']),  # open
                            float(data['k']['h']),  # high
                            float(data['k']['l']),  # low
                            float(data['k']['c']),  # close
                            float(data['k']['v'])   # volume
                        )

                        try:
                            self.q_store.put(kline_data, block=False)
                        except queue.Full:
                            _ = self.q_store.get(block=False)
                            self.q_store.put(kline_data, block=False)

                    del data
                    
                except json.JSONDecodeError:
                    self.put_notification(f"Invalid JSON received: {message[:100]}...")
                except Exception as e:
                    self.put_notification(f"Error processing message: {e}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.put_notification(f"Error in data processor: {e}")
                
        print("Data processor thread exiting")

    def _on_open(self, ws):
        print("WebSocket connection opened")
        self.put_notification("WebSocket connection opened")

    def _on_message(self, ws, message):
        try:
            self.q_streaming.put(message, block=False)
        except queue.Full:
            _ = self.q_streaming.get(block=False)
            self.q_streaming.put(message, block=False)

    def _on_error(self, ws, error):
        print(f"WebSocket error: {error}")
        self.put_notification(f"WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        print(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        self.put_notification(f"WebSocket connection closed: {close_status_code} - {close_msg}")

    def stop(self):
        """Clean shutdown of data feed"""
        print(f"Stopping BinanceStore for {self.symbol}")
        self._running = False
        self._state = self._ST_OVER
        
        try:
            self.q_streaming.put(None, block=False)
        except queue.Full:
            _ = self.q_streaming.get(block=False)
            self.q_streaming.put(None, block=False)
            
        if self.websocket:
            self.websocket.close()

        if hasattr(self, '_processing_thread') and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=2)

        if self._data:
            self._data.clear()

        self._clear_queue(self.q_streaming)
        self._clear_queue(self.q_store)
        
        print(f"Data feed for {self.symbol} shut down cleanly")
        
    def _clear_queue(self, q):
        """Clear all items from a queue"""
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    def fetch_ohlcv(self, symbol, interval, since=None, until=None):
        """Fetch historical OHLCV data from Binance API"""
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

    def start_memory_monitor(self, interval_seconds=60):
        """Start a thread to monitor memory usage at regular intervals"""
        def _monitor_memory():
            print('start _monitoring')
            while getattr(self, '_running', True):
                try:
                    import psutil
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Memory usage: {memory_mb:.2f} MB")
                except ImportError:
                    print("psutil not installed. Cannot monitor memory.")
                    break
                except Exception as e:
                    print(f"Error in memory monitor: {e}")
                
                time.sleep(interval_seconds)
        
        monitor_thread = threading.Thread(target=_monitor_memory, daemon=True)
        monitor_thread.start()
        return monitor_thread