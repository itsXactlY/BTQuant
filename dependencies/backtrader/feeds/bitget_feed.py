from collections import deque
import time
import json
import queue
import threading
from typing import Optional, Deque, List, Any

import pandas as pd
import numpy as np
from backtrader.dataseries import TimeFrame
from backtrader.feed import DataBase
from backtrader.utils import date2num


def identify_gaps(df, expected_interval):
    df = df.copy()
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.to_datetime(df.index)
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["time_diff"] = df["timestamp"].diff()
    return df[df["time_diff"] > expected_interval]


class BitgetData(DataBase):
    """
    Historical → warmup (DELAYED) → LIVE, low-CPU, queue-driven.

    Live flow:
      BitgetStore.message_queue (raw json)  →  _process_websocket_messages()
      → handle_websocket_message() → append normalized klines to self._data
      → _load_kline() maps to Backtrader lines
    """

    params = (
        ("drop_newest", False),
        ("update_interval_seconds", 0.25),  # throttle when no new bar
        ("debug", False),
        ("max_buffer", None),               # None = unbounded (good for warmup)
    )

    _ST_LIVE, _ST_HISTORBACK, _ST_OVER = range(3)

    def __init__(self, store, start_date=None):
        super().__init__()
        self.start_date = start_date
        self._store = store

        maxlen = self.p.max_buffer if self.p.max_buffer and self.p.max_buffer > 0 else None
        self._data: Deque[List[Any]] = deque(maxlen=maxlen)

        self.interval_ws = self._store.get_interval(TimeFrame.Seconds, 1)
        if self.interval_ws is None:
            raise ValueError("Unsupported timeframe/compression")

        self._state = self._ST_HISTORBACK if start_date else self._ST_LIVE

        self._ws_thread: Optional[threading.Thread] = None
        self._msg_queue = None

    # ---------------- Backtrader hooks ----------------

    def start(self):
        DataBase.start(self)

        # Historical bootstrap (warmup)
        if self.start_date:
            self._state = self._ST_HISTORBACK
            self.put_notification(self.DELAYED)

            interval_rest = self._store.get_interval_rest(TimeFrame.Seconds, 1)
            rows = self._store.fetch_ohlcv(
                interval=interval_rest,
                since_ms=int(self.start_date.timestamp() * 1000),
            )
            if rows:
                if self.p.drop_newest and rows:
                    rows.pop()  # drop possibly forming last candle
                df = self._parser_dataframe(rows)
                self._data.extend(df.values.tolist())
                if self.p.debug:
                    print(f"[BitgetData] Loaded {len(df)} historical bars")

        # Start socket (idempotent) + background consumer thread
        self._msg_queue = self._store.start_socket()
        if not (self._ws_thread and self._ws_thread.is_alive()):
            self._ws_thread = threading.Thread(
                target=self._process_websocket_messages, daemon=True, name="BitgetDataWSConsumer"
            )
            self._ws_thread.start()

        # If no history requested → LIVE immediately
        if not self.start_date:
            self._state = self._ST_LIVE
            self.put_notification(self.LIVE)

    def stop(self):
        # Mark over; unblocks consumer loop
        self._state = self._ST_OVER
        if isinstance(self._msg_queue, queue.Queue):
            try:
                self._msg_queue.put_nowait(None)
            except queue.Full:
                pass
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=2.0)
        DataBase.stop(self)

    def _load(self):
        if self._state == self._ST_OVER:
            return False

        if self._state == self._ST_HISTORBACK:
            res = self._load_kline()
            if res:
                return True
            # no more history → LIVE
            self._state = self._ST_LIVE
            self.put_notification(self.LIVE)
            return None

        if self._state == self._ST_LIVE:
            res = self._load_kline()
            if res is None:
                time.sleep(self.p.update_interval_seconds)  # CPU friendly
                return None
            return res

        return None

    # ---------------- live queue consumer ----------------

    def _process_websocket_messages(self):
        """
        Consume raw WS messages and append normalized klines into self._data.
        """
        q: queue.Queue = self._msg_queue
        sleep_s = self.p.update_interval_seconds
        while self._state != self._ST_OVER:
            try:
                message = q.get(timeout=1.0)
                if message is None:
                    break
                self.handle_websocket_message(message)
            except queue.Empty:
                time.sleep(sleep_s)
            except Exception as e:
                if self.p.debug:
                    print(f"[BitgetData] WS consumer error: {e}")
                time.sleep(sleep_s)

    # ---------------- parsing ----------------

    def handle_websocket_message(self, message: str):
        """
        Bitget WS message → extract candle arrays from 'arg.channel' == 'candle*'
        and append normalized rows to self._data.
        """
        try:
            data = json.loads(message)

            # Drop snapshots of ancient backfill if you don't want them
            if data.get("action") == "snapshot":
                if self.p.debug:
                    print("[BitgetData] drop snapshot")
                return

            arg = data.get("arg", {})
            channel = arg.get("channel", "")
            if not channel.startswith("candle"):
                return

            for k in data.get("data", []):
                # Expected k: [ts, open, high, low, close, volume, ...]
                ts = k[0]
                kline = self._parser_to_kline(ts, k)
                self._data.append(kline)

        except Exception as e:
            if self.p.debug:
                print(f"[BitgetData] parse error: {e}")

    def _parser_dataframe(self, rows):
        df = pd.DataFrame(rows)
        # Keep first 6 fields [ts, o, h, l, c, v]
        if df.shape[1] > 6:
            df = df.iloc[:, :6]
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        return df

    def _parser_to_kline(self, ts, kline):
        dt = pd.to_datetime(pd.to_numeric(ts), unit="ms", utc=True)
        return [
            dt,
            float(kline[1]),
            float(kline[2]),
            float(kline[3]),
            float(kline[4]),
            float(kline[5]),
        ]

    # ---------------- map into BT lines ----------------

    def _load_kline(self):
        try:
            kline = self._data.popleft()
        except IndexError:
            return None

        if self.p.debug:
            print(f"[BitgetData] Processing kline: {kline}")

        ts, o, h, _l, c, v = kline
        vals = [ts, o, h, _l, c, v]

        if any(vv is None for vv in vals):
            return self._load_kline()
        if any(np.isnan(vv) if isinstance(vv, (int, float, np.floating)) else False for vv in vals):
            return self._load_kline()
        if v == 0:
            return self._load_kline()

        self.lines.datetime[0] = date2num(ts)
        self.lines.open[0] = float(o)
        self.lines.high[0] = float(h)
        self.lines.low[0] = float(_l)
        self.lines.close[0] = float(c)
        self.lines.volume[0] = float(v)
        return True

    # ---------------- BT helpers ----------------

    def _start_live(self):
        # Not used; we transition in _load()
        pass

    def haslivedata(self):
        return self._state == self._ST_LIVE and len(self._data) > 0

    def islive(self):
        return True
