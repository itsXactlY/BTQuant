import json
import time
import queue
import threading
import requests
from typing import Optional, Tuple, Dict
from backtrader.dataseries import TimeFrame
import websocket


class BitgetStore:
    """
    Minimal Bitget store:
      - Manages WebSocket connection
      - Emits raw JSON messages into self.message_queue
      - Provides fetch_ohlcv() for historical bootstrap
    """

    _INTERVALS_WS: Dict[Tuple[TimeFrame, int], str] = {
        (TimeFrame.Seconds, 1): "1S",
        (TimeFrame.Minutes, 1): "1M",
        (TimeFrame.Minutes, 3): "3M",
        (TimeFrame.Minutes, 5): "5M",
        (TimeFrame.Minutes, 15): "15M",
        (TimeFrame.Minutes, 30): "30M",
        (TimeFrame.Minutes, 60): "1H",
        (TimeFrame.Minutes, 120): "2H",
        (TimeFrame.Minutes, 240): "4H",
        (TimeFrame.Days, 1): "1D",
        (TimeFrame.Weeks, 1): "1W",
    }

    _INTERVALS_REST: Dict[Tuple[TimeFrame, int], str] = {
        (TimeFrame.Seconds, 1): "1m",
        (TimeFrame.Minutes, 1): "1m",
        (TimeFrame.Minutes, 3): "3m",
        (TimeFrame.Minutes, 5): "5m",
        (TimeFrame.Minutes, 15): "15m",
        (TimeFrame.Minutes, 30): "30m",
        (TimeFrame.Minutes, 60): "1h",
        (TimeFrame.Minutes, 120): "2h",
        (TimeFrame.Minutes, 240): "4h",
        (TimeFrame.Days, 1): "1d",
        (TimeFrame.Weeks, 1): "1w",
    }

    def __init__(
        self,
        symbol: str,
        product: str = "spot",  # "mix" (perps) or "spot"
        region: str = "global",  # url variants if you use them
        debug: bool = False,
    ):
        """
        Args:
            symbol: Bitget instrument id, e.g. 'BTCUSDT_UMCBL' for USDT-M perp
            product: 'mix' for perps / futures, 'spot' for spot
        """
        self.symbol = symbol
        self.product = product
        self.region = region
        self.debug = debug

        # WS endpoints (adjust if you use regionals)
        self.ws_url = (
            "wss://ws.bitget.com/mix/v1/stream"
            if product == "mix"
            else "wss://ws.bitget.com/spot/v1/stream"
        )

        # REST base (public market candles)
        self.rest_base = (
            "https://api.bitget.com/api/mix/v1/market"
            if product == "mix"
            else "https://api.bitget.com/api/spot/v1/market"
        )

        self.message_queue: queue.Queue[str] = queue.Queue(maxsize=5000)
        self._running = False
        self._ws: Optional[websocket.WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._sub_payload = None

    # ------------- helpers -------------

    def get_interval(self, timeframe, compression) -> Optional[str]:
        return self._INTERVALS_WS.get((timeframe, compression))

    def get_interval_rest(self, timeframe, compression) -> Optional[str]:
        return self._INTERVALS_REST.get((timeframe, compression))

    def put_notification(self, msg: str):
        # Placeholder hook if you want to surface store notifications
        if self.debug:
            print(f"[BitgetStore] {msg}")

    # ------------- REST: historical -------------

    def fetch_ohlcv(self, interval: str, since_ms: Optional[int] = None, until_ms: Optional[int] = None):
        """
        Fetch historical candles. Returns list of rows:
        [timestamp(ms), open, high, low, close, volume]
        """
        url = f"{self.rest_base}/candles"
        params = {"symbol": self.symbol, "granularity": interval, "limit": 1000}
        if since_ms:
            params["startTime"] = since_ms
        if until_ms:
            params["endTime"] = until_ms

        out = []
        max_retries = 5
        while True:
            retries = 0
            payload = None
            while retries < max_retries:
                try:
                    r = requests.get(url, params=params, timeout=10)
                    r.raise_for_status()
                    payload = r.json()
                    break
                except requests.RequestException as e:
                    wait = 2 ** retries
                    if self.debug:
                        print(f"[BitgetStore] REST error ({retries+1}/{max_retries}): {e} → sleep {wait}s")
                    time.sleep(wait)
                    retries += 1

            if payload is None:
                break

            # Bitget returns newest→oldest in some endpoints; normalize to oldest→newest
            rows = payload
            if rows and rows[0][0] > rows[-1][0]:
                rows = list(reversed(rows))

            if not rows:
                break

            out.extend(rows)

            # paginate by moving start forward
            last_ts = rows[-1][0]
            if since_ms and last_ts <= since_ms:
                break
            params["startTime"] = last_ts + 1

            # stop if we reached until
            if until_ms and last_ts >= until_ms:
                break

            # safety: Bitget sometimes returns identical tail
            if len(rows) < 1000:
                break

        return out

    # ------------- WS lifecycle -------------

    def start_socket(self, timeframe=TimeFrame.Seconds, compression=1):
        """
        Start WebSocket and subscribe to candle channel.
        """
        if self._running:
            return self.message_queue

        code = self.get_interval(timeframe, compression)
        if not code:
            self.put_notification(f"Unsupported TF {timeframe}x{compression}")
            return self.message_queue

        channel = f"candle{code}"
        self._sub_payload = {
            "op": "subscribe",
            "args": [{"channel": channel, "instId": self.symbol}],
        }

        self._running = True
        t = threading.Thread(target=self._ws_thread_fn, daemon=True, name="BitgetWS")
        t.start()
        self._ws_thread = t
        return self.message_queue

    def _ws_thread_fn(self):
        retry = 0
        while self._running and retry < 10:
            try:
                websocket.enableTrace(False)
                self._ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever(ping_interval=10, ping_timeout=5)
                if not self._running:
                    break
                retry += 1
                wait = min(60, 2 ** retry)
                if self.debug:
                    print(f"[BitgetStore] WS disconnected, retry in {wait}s")
                time.sleep(wait)
            except Exception as e:
                self.put_notification(f"WS fatal error: {e}")
                retry += 1
                time.sleep(min(60, 2 ** retry))

    def stop(self):
        self._running = False
        try:
            # unblock any consumer waiting on the queue
            self.message_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=2)

    # ------------- WS callbacks -------------

    def _on_open(self, ws):
        if self.debug:
            print("[BitgetStore] WS open")
        try:
            if self._sub_payload:
                ws.send(json.dumps(self._sub_payload))
        except Exception as e:
            self.put_notification(f"Subscribe send error: {e}")

    def _on_message(self, ws, message: str):
        # Pass raw JSON to the consumer (BitgetData)
        try:
            self.message_queue.put_nowait(message)
        except queue.Full:
            # drop oldest to keep fresh
            try:
                _ = self.message_queue.get_nowait()
                self.message_queue.put_nowait(message)
            except queue.Empty:
                pass

    def _on_error(self, ws, error):
        self.put_notification(f"WS error: {error}")

    def _on_close(self, ws, code, reason):
        if self.debug:
            print(f"[BitgetStore] WS close {code} {reason}")
