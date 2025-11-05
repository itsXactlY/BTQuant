from collections import deque
import queue
import time
import json
import threading

import pandas as pd
import numpy as np
from backtrader.dataseries import TimeFrame
from backtrader.feed import DataBase
from backtrader.utils import date2num


def identify_gaps(df, expected_interval):
    """
    Simple helper to detect gaps in historical data.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df.index)
    df["time_diff"] = df["timestamp"].diff()
    gaps = df[df["time_diff"] > expected_interval]
    return gaps


class BinanceData(DataBase):
    """
    Backtrader DataFeed that consumes:
    - Historical data from BinanceStore.fetch_ohlcv
    - Live klines via BinanceStore.q_store

    The main CPU fix is:
    - When no new live bar is available, _load() sleeps for
      update_interval_seconds before returning None, avoiding a tight
      busy-loop in Cerebro.
    """

    params = (
        ("drop_newest", False),
        # Sleep duration used to throttle the live-polling loop when no new bar
        # is available. 0.05–0.5 is a good range; default 0.25s.
        ("update_interval_seconds", 0.25),
        # Debug logging is extremely verbose; keep off by default.
        ("debug", False),
        # Maximum number of bars in the in-memory buffer.
        ("max_buffer", 1),
    )

    _ST_LIVE, _ST_HISTORBACK, _ST_OVER = range(3)

    def __init__(self, store, start_date=None):
        super().__init__()
        self.start_date = start_date
        self._store = store
        self._data = deque(maxlen=self.p.max_buffer)

        self.interval = self._store.get_interval(TimeFrame.Seconds, compression=1)
        if self.interval is None:
            raise ValueError("Unsupported timeframe/compression")

        self.ws_url = store.ws_url
        self._state = self._ST_HISTORBACK if start_date else self._ST_LIVE

        self.data_queue = None
        self._processing_thread = None

    # ------------------------------------------------------------------
    # WebSocket message handler (unused in current pipeline but kept
    # for compatibility / future use)
    # ------------------------------------------------------------------
    def handle_websocket_message(self, message):
        """
        Optional direct WebSocket handler; currently not used because we
        go through q_store + _process_queue. Left here for compatibility.
        """
        try:
            data = json.loads(message)
            k = data["k"]

            kline = self._parser_to_kline(
                k["t"],
                [
                    k["t"],
                    k["o"],
                    k["h"],
                    k["l"],
                    k["c"],
                    k["v"],
                ],
            )

            if self.p.debug:
                print("DEBUG :: POPLEFT handle_websocket_message")

            if self._data:
                # Drop oldest if buffer is full; deque(maxlen=...) already
                # enforces this, but we keep the explicit popleft() for clarity.
                self._data.popleft()
            self._data.append(kline)

            if self.p.debug:
                print("received fresh data:", kline)
        except Exception as e:
            print(f"Error handling WebSocket message: {e}")

    # ------------------------------------------------------------------
    # Core Backtrader hook: _load()
    # ------------------------------------------------------------------
    def _load(self):
        """
        Backtrader calls this in a loop.

        Return:
            - True  -> new bar delivered
            - False -> data stream is over
            - None  -> no data yet (live) -> Cerebro will poll again

        The key CPU optimization is the sleep in LIVE mode when no bar
        is available.
        """
        if self._state == self._ST_OVER:
            return False

        if self._state == self._ST_LIVE:
            res = self._load_kline()
            if res is None:
                # No data currently in buffer: throttle the poll loop
                time.sleep(self.p.update_interval_seconds)
            return res

        if self._state == self._ST_HISTORBACK:
            res = self._load_kline()
            if res:
                return True

            # No more historical data – transition to live
            self._start_live()
            return None

        # Fallback (should not normally hit)
        return None

    def _load_kline(self):
        """
        Pull the next valid kline from the internal deque.

        Iterative (no recursion) to avoid deep call stacks when many
        invalid/filtered klines appear.
        """
        while True:
            try:
                kline = self._data.popleft()
                if self.p.debug:
                    print(f"Processing kline: {kline}")
            except IndexError:
                # Buffer empty – caller decides how to handle
                return None

            timestamp, open_, high, low, close, volume = kline

            # Basic sanity checks
            values = [timestamp, open_, high, low, close, volume]

            # Skip None values
            if any(v is None for v in values):
                if self.p.debug:
                    print(f"Skipping kline due to None value: {kline}")
                continue

            # Skip NaNs
            if any(np.isnan(v) if isinstance(v, (int, float)) else False for v in values):
                if self.p.debug:
                    print(f"Skipping kline due to NaN value: {kline}")
                continue

            # Optionally skip zero-volume bars (common on illiquid/1s streams)
            if volume == 0:
                if self.p.debug:
                    print(f"Skipping kline due to zero volume: {kline}")
                continue

            # Map into Backtrader's line buffers
            self.lines.datetime[0] = date2num(timestamp)
            self.lines.open[0] = float(open_)
            self.lines.high[0] = float(high)
            self.lines.low[0] = float(low)
            self.lines.close[0] = float(close)
            self.lines.volume[0] = float(volume)

            del kline
            return True

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def _parser_dataframe(self, data):
        df = pd.DataFrame(data)
        # Ensure we only keep the first 6 columns if Binance changes payload
        if df.shape[1] > 6:
            df = df.iloc[:, :6]
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        return df

    def _parser_to_kline(self, timestamp, kline):
        dt = pd.to_datetime(pd.to_numeric(timestamp), unit="ms", utc=True)
        return [
            dt,
            float(kline[1]),
            float(kline[2]),
            float(kline[3]),
            float(kline[4]),
            float(kline[5]),
        ]

    # ------------------------------------------------------------------
    # Live start / state helpers
    # ------------------------------------------------------------------
    def _start_live(self):
        print("Starting live data...")
        # start_socket is idempotent: if already running, it simply
        # returns the existing queue.
        self.data_queue = self._store.start_socket()
        self._store.start_memory_monitor()
        self._state = self._ST_LIVE
        self.put_notification(self.LIVE)
        print("Starting live data and purging historical data...")

    def haslivedata(self):
        # Used by Backtrader to know whether live data is expected
        return self._state == self._ST_LIVE and len(self._data) > 0

    def islive(self):
        return True

    # ------------------------------------------------------------------
    # Backtrader lifecycle
    # ------------------------------------------------------------------
    def start(self):
        """
        Called by Backtrader when the feed is started.

        - Loads historical candles (if start_date given)
        - Starts the WebSocket socket + memory monitor
        - Spawns a background thread that moves q_store -> _data
        """
        DataBase.start(self)

        # Historical bootstrap
        if self.start_date:
            self._state = self._ST_HISTORBACK
            self.put_notification(self.DELAYED)

            klines = self._store.fetch_ohlcv(
                self._store.symbol,
                self.interval,
                since=int(self.start_date.timestamp() * 1000),
            )

            if klines:
                if self.p.drop_newest and klines:
                    # Some exchanges send a partially-formed last candle
                    klines.pop()

                df = self._parser_dataframe(klines)
                # df.values.tolist() -> rows: [ts, o, h, l, c, v]
                self._data.extend(df.values.tolist())

        # Live socket: q_store
        self.data_queue = self._store.start_socket()

        # Background thread that moves q_store -> _data
        if not (self._processing_thread and self._processing_thread.is_alive()):
            self._processing_thread = threading.Thread(
                target=self._process_queue,
                daemon=True,
                name="BinanceDataProcessor",
            )
            self._processing_thread.start()

        if self._state == self._ST_HISTORBACK:
            # We will transition to LIVE via _load() when history exhausted.
            self.put_notification(self.DELAYED)
        else:
            self._state = self._ST_LIVE
            self.put_notification(self.LIVE)

    def _process_queue(self):
        """
        Process data from q_store with memory management.

        This runs in a separate daemon thread and fills self._data,
        which is consumed by _load_kline().
        """
        cleanup_counter = 0
        while self._state != self._ST_OVER:
            try:
                kline_data = self.data_queue.get(timeout=1.0)
                # Sentinel used by stop() to break out quickly
                if kline_data is None:
                    break

                timestamp = pd.to_datetime(kline_data[0], unit="ms", utc=True)
                processed_data = [
                    timestamp,
                    kline_data[1],  # open
                    kline_data[2],  # high
                    kline_data[3],  # low
                    kline_data[4],  # close
                    kline_data[5],  # volume
                ]

                # deque(maxlen=...) already keeps buffer bounded; no need for
                # manual while-loop here.
                self._data.append(processed_data)

                del kline_data
                del processed_data

                cleanup_counter += 1
                # GC is relatively expensive; do it rarely (e.g. every 1000 bars)
                if cleanup_counter >= 1000:
                    import gc

                    gc.collect()
                    cleanup_counter = 0

            except queue.Empty:
                # Nothing to process right now; short sleep to avoid spin
                time.sleep(self.p.update_interval_seconds)
            except Exception as e:
                print(f"Error in _process_queue: {e}")

    def _process_websocket_messages(self):
        """
        Legacy helper for direct message_queue -> _data processing.
        Not used in the current pipeline, but kept for compatibility.
        """
        while self._state != self._ST_OVER:
            try:
                message = self._store.message_queue.get(timeout=1)
                self.handle_websocket_message(message)
                del message
            except queue.Empty:
                time.sleep(0.1)

    def stop(self):
        """
        Ensure clean shutdown of background processing thread and
        align with Backtrader's DataBase.stop().
        """
        # Mark feed as over so _process_queue can exit its outer loop
        self._state = self._ST_OVER

        # Unblock _process_queue if it is waiting on the queue
        if isinstance(self.data_queue, queue.Queue):
            try:
                self.data_queue.put_nowait(None)
            except queue.Full:
                # If full, drop one and reinsert the sentinel
                try:
                    _ = self.data_queue.get_nowait()
                    self.data_queue.put_nowait(None)
                except queue.Empty:
                    # If it's empty when we get here, nothing to do
                    pass

        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=2.0)

        DataBase.stop(self)
