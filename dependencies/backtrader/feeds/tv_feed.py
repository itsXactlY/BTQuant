from collections import deque
from enum import Enum, auto
from typing import Optional, Tuple, List
import queue
import pandas as pd
import numpy as np
from backtrader.dataseries import TimeFrame
from backtrader.feed import DataBase
from backtrader.utils import date2num
import threading
import gc

class DataState(Enum):
    LIVE = auto()
    HISTORBACK = auto()
    OVER = auto()

class KlineData:
    __slots__ = ('timestamp', 'open', 'high', 'low', 'close', 'volume')
    
    def __init__(self, timestamp: pd.Timestamp, open_: float, high: float, 
                 low: float, close: float, volume: float):
        self.timestamp = timestamp
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
    
    def is_valid(self) -> bool:
        values = [self.timestamp, self.open, self.high, self.low, self.close, self.volume]
        if any(v is None for v in values):
            return False
        if any(np.isnan(v) if isinstance(v, (int, float)) else False for v in values):
            return False
        if self.volume == 0:
            return False
        return True
    
    def to_tuple(self) -> Tuple:
        return (self.timestamp, self.open, self.high, self.low, self.close, self.volume)

class ThreadSafeDataBuffer:
    def __init__(self, max_size: int = 1000):
        self._buffer: deque = deque(maxlen=max_size)
        self._lock: threading.Lock = threading.Lock()
        self._data_available: threading.Event = threading.Event()
        self._max_size: int = max_size
    
    def append(self, item: KlineData) -> None:
        with self._lock:
            while len(self._buffer) >= self._max_size:
                self._buffer.popleft()
            
            self._buffer.append(item)
            self._data_available.set()
    
    def pop_left(self, timeout: float = 1.0) -> Optional[KlineData]:
        if not self._data_available.wait(timeout=timeout):
            return None
        
        with self._lock:
            if len(self._buffer) == 0:
                self._data_available.clear()
                return None
            
            try:
                item = self._buffer.popleft()
                if len(self._buffer) == 0:
                    self._data_available.clear()
                return item
            except IndexError:
                self._data_available.clear()
                return None
    
    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._data_available.clear()
    
    @property
    def size(self) -> int:
        with self._lock:
            return len(self._buffer)
    
    def has_data(self) -> bool:
        with self._lock:
            return len(self._buffer) > 0

class DataProcessor:
    def __init__(self, data_queue: queue.Queue, buffer: ThreadSafeDataBuffer, 
                 state_manager: 'StateManager', debug: bool = False):
        self._data_queue = data_queue
        self._buffer = buffer
        self._state_manager = state_manager
        self._debug = debug
        self._thread: Optional[threading.Thread] = None
        self._cleanup_counter = 0
        self._gc_interval = 100
    
    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._process_loop,
                daemon=True,
                name="TradingViewDataProcessor"
            )
            self._thread.start()
    
    def _process_loop(self) -> None:
        while not self._state_manager.is_over:
            try:
                raw_data = self._data_queue.get(timeout=1)
                kline = self._parse_raw_data(raw_data)
                
                if kline and kline.is_valid():
                    self._buffer.append(kline)
                    if self._debug:
                        print(f"Processed and buffered kline: {kline.timestamp}")
                elif self._debug:
                    print(f"Skipped invalid kline: {raw_data}")
                
                self._perform_periodic_cleanup()
                
            except queue.Empty:
                continue
            except Exception as e:
                self._handle_processing_error(e)
    
    def _parse_raw_data(self, raw_data: List) -> Optional[KlineData]:
        try:
            timestamp = pd.to_datetime(raw_data[0], unit='ms', utc=True)
            return KlineData(
                timestamp=timestamp,
                open_=raw_data[1],
                high=raw_data[2],
                low=raw_data[3],
                close=raw_data[4],
                volume=raw_data[5]
            )
        except (IndexError, ValueError, TypeError) as e:
            if self._debug:
                print(f"Error parsing raw data: {e}")
            return None
    
    def _perform_periodic_cleanup(self) -> None:
        self._cleanup_counter += 1
        if self._cleanup_counter >= self._gc_interval:
            gc.collect()
            self._cleanup_counter = 0
    
    def _handle_processing_error(self, error: Exception) -> None:
        print(f"Error in data processor: {error}")
        if self._debug:
            import traceback
            traceback.print_exc()

class StateManager:
    def __init__(self, initial_state: DataState = DataState.LIVE):
        self._state = initial_state
        self._lock = threading.Lock()
    
    @property
    def state(self) -> DataState:
        with self._lock:
            return self._state
    
    @state.setter
    def state(self, new_state: DataState) -> None:
        with self._lock:
            self._state = new_state
    
    @property
    def is_live(self) -> bool:
        return self.state == DataState.LIVE
    
    @property
    def is_over(self) -> bool:
        return self.state == DataState.OVER
    
    def set_live(self) -> None:
        self.state = DataState.LIVE
    
    def set_over(self) -> None:
        self.state = DataState.OVER

class TradingViewData(DataBase):
    """
    TradingView data feed for Backtrader with thread-safe operations.
    
    This class provides a live data feed from TradingView WebSocket with:
    - Thread-safe data buffering
    - Event-driven data availability signaling
    - Automatic data validation
    - Memory-efficient processing
    """
    
    params = (
        ('drop_newest', False),
        ('update_interval_seconds', 1),
        ('debug', True),
        ('max_buffer', 1000),
        ('load_timeout', 0.05),
    )

    def __init__(self, store, start_date: Optional[pd.Timestamp] = None):
        super().__init__()
        self._store = store
        self._start_date = start_date
        
        self.symbol = store.symbol
        self.params.symbol = store.symbol  # Also add to params
        
        self._state_manager = StateManager(DataState.LIVE)
        self._buffer = ThreadSafeDataBuffer(max_size=self.p.max_buffer)
        self._processor: Optional[DataProcessor] = None

        self._interval = self._validate_interval()
    
    def _validate_interval(self) -> str:
        interval = self._store.get_interval(TimeFrame.Seconds, compression=1)
        if interval is None:
            raise ValueError("Unsupported timeframe/compression")
        return interval
    
    # ==================== Backtrader Interface Methods ====================
    
    def start(self) -> None:
        super().start()

        data_queue = self._store.start_socket()
        self._processor = DataProcessor(
            data_queue=data_queue,
            buffer=self._buffer,
            state_manager=self._state_manager,
            debug=self.p.debug
        )
        self._processor.start()

        self._state_manager.set_live()
        self.put_notification(self.LIVE)

        self._store.start_memory_monitor()
        
        print(f"TradingView data feed started for {self._store.symbol}")
    
    def stop(self) -> None:
        self._state_manager.set_over()
        self._buffer.clear()
        super().stop()
    
    def islive(self) -> bool:
        return True
    
    def haslivedata(self) -> bool:
        return self._state_manager.is_live and self._buffer.has_data()
    
    def _load(self) -> Optional[bool]:
        if self._state_manager.is_over:
            return False
        
        if self._state_manager.is_live:
            return self._load_kline()
        
        return False
    
    # ==================== Data Loading Methods ====================
    
    def _load_kline(self) -> Optional[bool]:
        kline = self._buffer.pop_left(timeout=self.p.load_timeout)
        
        if kline is None:
            return None
        
        if self._state_manager.is_over:
            return False

        if not kline.is_valid():
            if self.p.debug:
                print(f"Skipping invalid kline: {kline.to_tuple()}")
            return self._load_kline()

        self._populate_lines(kline)
        
        if self.p.debug:
            print(f"Loaded kline: {kline.timestamp} - Close: {kline.close}")
        
        return True
    
    def _populate_lines(self, kline: KlineData) -> None:
        self.lines.datetime[0] = date2num(kline.timestamp)
        self.lines.open[0] = kline.open
        self.lines.high[0] = kline.high
        self.lines.low[0] = kline.low
        self.lines.close[0] = kline.close
        self.lines.volume[0] = kline.volume
    
    # ==================== Properties ====================
    
    @property
    def buffer_size(self) -> int:
        """Get current buffer size."""
        return self._buffer.size
    
    @property
    def state(self) -> DataState:
        """Get current feed state."""
        return self._state_manager.state