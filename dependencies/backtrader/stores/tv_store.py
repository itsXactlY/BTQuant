import threading
import queue
import time
from typing import Optional
from dataclasses import dataclass
from backtrader.dataseries import TimeFrame
from backtrader.feeds.tv_feed import TradingViewData


@dataclass
class Candle:
    start_time: int = 0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    initialized: bool = False
    
    def reset(self, price: float, volume: float, timestamp: int):
        self.start_time = timestamp
        self.open = price
        self.high = price
        self.low = price
        self.close = price
        self.volume = volume
        self.initialized = True
    
    def update(self, price: float, volume: float):
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume = volume  # TradingView sends cumulative volume
    
    def to_tuple(self):
        return (
            self.start_time * 1000,  # milliseconds
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume
        )


class TradingViewStore:
    _GRANULARITIES = {
        (TimeFrame.Seconds, 1): 1,
        (TimeFrame.Minutes, 1): 60,
        (TimeFrame.Minutes, 5): 300,
        (TimeFrame.Minutes, 15): 900,
    }

    def __init__(self, symbol: str):
        self.symbol = symbol
        self._data: Optional[TradingViewData] = None
        self._running = False
        self._ticker = None

        self.q_store = queue.Queue(maxsize=1000)
        self.q_notifications = queue.Queue()

        self._current_candle = Candle()
        self._candle_interval = 1  # seconds
        self._lock = threading.Lock()

        self._timer_thread: Optional[threading.Thread] = None

    def getdata(self, start_date=None) -> TradingViewData:
        if self._data is None:
            self._data = TradingViewData(store=self, start_date=start_date)
        return self._data

    def get_interval(self, timeframe: TimeFrame, compression: int) -> Optional[int]:
        return self._GRANULARITIES.get((timeframe, compression))
    
    def put_notification(self, msg: str, *args, **kwargs):
        self.q_notifications.put((msg, args, kwargs))
    
    def get_notifications(self):
        notifications = []
        while True:
            try:
                notifications.append(self.q_notifications.get_nowait())
            except queue.Empty:
                break
        return notifications

    def start_socket(self, timeframe=TimeFrame.Seconds, compression=1):
        if self._running:
            return self.q_store
        
        interval_seconds = self.get_interval(timeframe, compression)
        if not interval_seconds:
            self.put_notification(f"Unsupported timeframe: {timeframe}-{compression}")
            return None
        
        self._candle_interval = interval_seconds
        self._running = True
        
        print(f"Starting TradingView ticker for {self.symbol}")
        print(f"Candle interval: {interval_seconds}s")
        
        try:
            from .ticker import ticker
        except ImportError:
            print("ERROR: ticker library not found. Install from the provided file.")
            return None

        self._ticker = ticker(self.symbol, save=False, verbose=False)
        self._ticker.cb = self._handle_tick
        self._ticker.start()

        self._timer_thread = threading.Thread(
            target=self._candle_timer,
            daemon=True,
            name="TVStore-CandleTimer"
        )
        self._timer_thread.start()
        
        return self.q_store

    def _handle_tick(self, ticker_name: str, data: dict):
        if not self._running:
            return
        
        price = data.get('price')
        volume = data.get('volume')
        timestamp = data.get('time')
        
        if price is None or timestamp is None:
            return
        
        with self._lock:
            if not self._current_candle.initialized:
                # start first candle
                self._current_candle.reset(price, volume, timestamp)
            else:
                # check if we need to close current candle
                if timestamp - self._current_candle.start_time >= self._candle_interval:
                    # emit completed candle
                    self._emit_candle()
                    # start new candle
                    self._current_candle.reset(price, volume, timestamp)
                else:
                    # update current candle
                    self._current_candle.update(price, volume)

    def _candle_timer(self):
        print("Candle timer started")
        
        while self._running:
            time.sleep(self._candle_interval)
            
            with self._lock:
                if self._current_candle.initialized:
                    current_time = int(time.time())
                    candle_age = current_time - self._current_candle.start_time

                    if candle_age >= self._candle_interval:
                        self._emit_candle()
                        price = self._current_candle.close
                        self._current_candle.reset(price, 0, current_time)
        
        print("Candle timer stopped")

    def _emit_candle(self):
        """Emit current candle to queue (must be called with lock held)."""
        if not self._current_candle.initialized:
            return
        
        kline = self._current_candle.to_tuple()
        
        try:
            self.q_store.put(kline, block=False)
            # print(f"Emitted candle: {self._current_candle.start_time} "
            #       f"O:{self._current_candle.open:.12f} "
            #       f"H:{self._current_candle.high:.12f} "
            #       f"L:{self._current_candle.low:.12f} "
            #       f"C:{self._current_candle.close:.12f} "
            #       f"V:{self._current_candle.volume:.12f}")
        except queue.Full:
            try:
                self.q_store.get(block=False)
                self.q_store.put(kline, block=False)
            except queue.Empty:
                pass

    def stop(self):
        print(f"Stopping TradingViewStore for {self.symbol}")
        self._running = False
        
        if self._ticker:
            self._ticker.stop()
        
        if self._timer_thread and self._timer_thread.is_alive():
            self._timer_thread.join(timeout=2)
        
        if self._data:
            self._data.stop()

        try:
            while True:
                self.q_store.get_nowait()
        except queue.Empty:
            pass
        
        print("Store stopped")

    def start_memory_monitor(self, interval_seconds=60):
        def _monitor():
            while self._running:
                try:
                    import psutil
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Memory: {memory_mb:.2f} MB")
                except:
                    pass
                time.sleep(interval_seconds)
        
        t = threading.Thread(target=_monitor, daemon=True)
        t.start()
        return t