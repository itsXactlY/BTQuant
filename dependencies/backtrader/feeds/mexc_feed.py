from collections import deque
import random
import time
import json
import threading
import queue
from backtrader.dataseries import TimeFrame
from backtrader.feed import DataBase
from backtrader.utils import date2num
from datetime import datetime, timezone

class TickSampler:
    """
    Efficiently samples incoming ticks and converts them to time-based OHLCV candles.
    Instead of processing every tick, this sampler strategically selects ticks
    to generate accurate candles with minimal processing overhead.
    """
    
    def __init__(self, timeframe_seconds=1):
        self.timeframe_ms = timeframe_seconds * 1000
        self.current_candle = None
        self.current_period = None
        self.tick_buffer = []
        self.max_buffer_size = 100
        self.batch_process_size = 10
        self.processed_periods = set()

        # for performance tracking - as for now
        self.total_ticks_received = 0
        self.ticks_processed = 0
        self.candles_generated = 0
        self.last_stat_time = time.time()
    
    def add_tick(self, timestamp_ms, price, volume):
        """Add a tick to the buffer for efficient batch processing"""
        self.total_ticks_received += 1
        
        if len(self.tick_buffer) >= self.max_buffer_size:
            self.process_tick_buffer()
        
        self.tick_buffer.append((timestamp_ms, price, volume))
        
        if (len(self.tick_buffer) >= self.batch_process_size or 
            (self.tick_buffer and 
            self.tick_buffer[-1][0] - self.tick_buffer[0][0] > self.timeframe_ms)):
            return self.process_tick_buffer()
        
        return None
    
    def process_tick_buffer(self):
        if not self.tick_buffer:
            return None
        
        self.ticks_processed += len(self.tick_buffer)
        period_ticks = {}
        
        for timestamp_ms, price, volume in self.tick_buffer:
            period_start = timestamp_ms - (timestamp_ms % self.timeframe_ms)
            if period_start not in period_ticks:
                period_ticks[period_start] = {
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': volume,
                    'ticks': 1,
                    'timestamp': period_start
                }
            else:
                data = period_ticks[period_start]
                data['high'] = max(data['high'], price)
                data['low'] = min(data['low'], price)
                data['close'] = price
                data['volume'] += volume
                data['ticks'] += 1
        
        self.tick_buffer = []
        
        if self.tick_buffer:
            current_time = self.tick_buffer[-1][0]
        else:
            current_time = int(time.time() * 1000)
        
        current_period = current_time - (current_time % self.timeframe_ms)
        
        completed_candles = []
        
        for period in sorted(period_ticks.keys()):
            candle_data = period_ticks[period]
            
            if period == current_period:
                self.current_period = period
                self.current_candle = candle_data
            elif period < current_period:
                if period not in self.processed_periods:
                    completed_candles.append(candle_data)
                    self.candles_generated += 1
                    self.processed_periods.add(period)
        
        if len(self.processed_periods) > 1000:
            oldest_period = min(self.processed_periods)
            self.processed_periods.remove(oldest_period)
        
        return completed_candles[-1] if completed_candles else None

class MexcData(DataBase):
    """
    Backtrader Data Feed for MEXC that uses strategic tick sampling
    to compress X-second candles without processing every single incoming trade tick.
    """
    params = (
        ('timeframe', TimeFrame.Seconds),
        ('compression', 1),
        ('max_bar_buffer', 300),
        ('debug', False),
    )

    _ST_LIVE, _ST_HISTORBACK, _ST_OVER = range(3)

    def __init__(self, store, start_date=None):
        super().__init__()
        self.start_date = start_date
        self._store = store
        self.ws_url = store.ws_url
        
        self._timeframe = self.p.timeframe
        self._compression = self.p.compression

        self.tick_sampler = TickSampler(timeframe_seconds=self.p.compression)
        
        self._bar_buffer = deque(maxlen=self.p.max_bar_buffer)

        self._state = self._ST_LIVE
        self._running = True
        self._first_data_received = False
        
        self._processing_thread = None

    def start(self):
        """Start the data feed and WebSocket connection"""
        DataBase.start(self)
        self._store.start_socket()
        self._processing_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._processing_thread.start()

    def _process_loop(self):
        """Efficient processing loop that samples ticks instead of processing each one"""
        while self._running:
            try:
                messages = []
                batch_start = time.time()
                
                while len(messages) < 100 and time.time() - batch_start < 0.1:
                    try:
                        msg = self._store.message_queue.get(timeout=0.001)
                        messages.append(msg)
                    except queue.Empty:
                        break
                
                if not messages:
                    time.sleep(0.1)
                    continue
                
                for message in messages:
                    try:
                        data = json.loads(message)
                        deals = None
                        
                        if 'd' in data:
                            d = data['d']
                            if 'deals' in d:
                                deals = d['deals']
                            elif 'data' in d:
                                deals = d['data']
                        
                        if not deals:
                            continue
                        
                        for trade in deals:
                            if 'p' in trade:
                                timestamp_ms = int(trade['t'])
                                price = float(trade['p'])
                                qty = float(trade['v'])
                            else:
                                timestamp_ms = int(trade['tradeTime'])
                                price = float(trade['price'])
                                qty = float(trade['quantity'])
                            
                            candle = self.tick_sampler.add_tick(timestamp_ms, price, qty)
                            
                            if candle:
                                bar_dt = datetime.fromtimestamp(candle['timestamp'] / 1000, tz=timezone.utc)
                                self._bar_buffer.append([
                                    bar_dt,
                                    candle['open'],
                                    candle['high'],
                                    candle['low'],
                                    candle['close'],
                                    candle['volume']
                                ])
                                
                                if not self._first_data_received:
                                    self._first_data_received = True
                                    self._state = self._ST_LIVE
                                    self.put_notification(self.LIVE)
                                    if self.p.debug:
                                        print(f"First data received, state set to LIVE")
                        
                    except Exception as e:
                        if self.p.debug:
                            print(f"Message processing error: {e}")
                
                if self.p.debug and random.random() < 0.01:  # only 1% of loops
                    print(f"Bar buffer size: {len(self._bar_buffer)}")
                
            except Exception as e:
                if self.p.debug:
                    print(f"Processing loop error: {e}")
                time.sleep(0.1)  # Sleep on error
    
    def stop(self):
        """Stop the data feed cleanly"""
        self._running = False
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=1.0)
        return super().stop()

    def _load(self):
        """Load next bar from buffer into lines"""
        try:
            bar = self._bar_buffer.popleft()
        except IndexError:
            return None

        dt, o, h, l, c, v = bar
        self.lines.datetime[0] = date2num(dt)
        self.lines.open[0] = o
        self.lines.high[0] = h
        self.lines.low[0] = l
        self.lines.close[0] = c
        self.lines.volume[0] = v
        return True

    def islive(self):
        """This data feed is always considered live"""
        return True

    def haslivedata(self):
        """Check if there's data available in the buffer"""
        return bool(self._bar_buffer)
