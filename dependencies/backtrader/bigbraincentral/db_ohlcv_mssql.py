"""
Read-only OHLCV feed for BTQuant/Backtrader using SQL Server (fast_mssql).

- ZERO writes, ZERO schema changes
- Nur SELECTs aus bestehenden Tabellen
- Die Datenbank wird extern gefüttert (ccapi C++, andere Collector)

Unterstützte Modi:
- mode="global":   eine gemeinsame OHLCV-Tabelle (Default: "ohlcv")
                   Spalten: timestamp, exchange, symbol, timeframe,
                            open, high, low, close, volume
- mode="per_pair": eine Tabelle pro (exchange, symbol),
                   Name aus table_pattern.format(exchange=..., symbol=...)
                   z.B. "{symbol}_klines" -> "BTCUSDT_klines"
                        "{exchange}_{symbol}_klines" -> "binance_BTCUSDT_klines"
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Deque, List, Any

import time
from collections import deque
from backtrader.feed import DataBase
from backtrader.utils import date2num
from backtrader.dataseries import TimeFrame

import fast_mssql


# ----------------------------------------------------------------------
# 1. Konfiguration für DB-Zugriff (nur Lesen)
# ----------------------------------------------------------------------
@dataclass
class MSSQLFeedConfig:
    server: str = "localhost"
    database: str = ""
    username: str = "SA"
    password: str = ""
    driver: str = "{ODBC Driver 18 for SQL Server}"
    trust_server_certificate: bool = True

    def connection_string(self) -> str:
        trust = "yes" if self.trust_server_certificate else "no"
        return (
            f"DRIVER={self.driver};"
            f"SERVER={self.server};"
            f"DATABASE={self.database};"
            f"UID={self.username};"
            f"PWD={self.password};"
            f"TrustServerCertificate={trust};"
        )


class ReadOnlyOHLCV:
    """
    Minimaler read-only Wrapper um fast_mssql für OHLCV-Zugriffe.
    Macht KEINE DDL, KEINE Inserts/Updates – nur SELECT.
    """

    def __init__(
        self,
        config: MSSQLFeedConfig,
        mode: str = "global",
        global_table: str = "ohlcv",
        schema: str = "dbo",
        table_pattern: str = "{symbol}_klines",
    ) -> None:
        """
        :param mode: "global" oder "per_pair"
        :param global_table: Tabellenname für mode="global"
        :param schema: Schema-Name, meist "dbo"
        :param table_pattern: Muster für per_pair-Tabellen
                              Platzhalter: {exchange}, {symbol}
        """
        if mode not in ("global", "per_pair"):
            raise ValueError("mode must be 'global' or 'per_pair'")

        self.config = config
        self.mode = mode
        self.global_table = global_table
        self.schema = schema
        self.table_pattern = table_pattern

        self._conn_str = config.connection_string()

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        # Groß-/Kleinschreibung erhalten (BTCUSDT), aber Trenner entfernen
        return symbol.replace("/", "").replace("-", "").replace("_", "")

    def _table_name(self, exchange: str, symbol: str) -> str:
        if self.mode == "global":
            return f"{self.schema}.{self.global_table}"
        # per_pair
        sym = self._normalize_symbol(symbol)
        ex = exchange.lower()
        tbl = self.table_pattern.format(exchange=ex, symbol=sym)
        return f"{self.schema}.{tbl}"

    def get_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """
        OHLCV-Zeilen als list[dict] lesen.

        Erwartete Spalten:
          - timestamp (DATETIME/DATETIME2)
          - open, high, low, close, volume
          - bei mode == "global": exchange, symbol, timeframe
        """
        if end is None:
            end = datetime.utcnow()

        start_str = start.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end.strftime("%Y-%m-%d %H:%M:%S")

        table = self._table_name(exchange, symbol)
        top_clause = f"TOP {limit}" if limit else ""

        if self.mode == "global":
            query = f"""
                SELECT {top_clause}
                       timestamp, [open], high, low, [close], volume
                FROM {table}
                WHERE exchange = '{exchange}'
                  AND symbol = '{symbol}'
                  AND timeframe = '{timeframe}'
                  AND timestamp >= '{start_str}' AND timestamp < '{end_str}'
                ORDER BY timestamp ASC;
            """
        else:
            # per-pair-Tabellen brauchen i.d.R. keinen exchange/symbol-Filter
            query = f"""
                SELECT {top_clause}
                       timestamp, [open], high, low, [close], volume
                FROM {table}
                WHERE timeframe = '{timeframe}'
                  AND timestamp >= '{start_str}' AND timestamp < '{end_str}'
                ORDER BY timestamp ASC;
            """

        rows = fast_mssql.fetch_data_from_db(self._conn_str, query)
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        return [dict(zip(cols, r)) for r in rows]


# ----------------------------------------------------------------------
# 2. Backtrader-Feed, der aus ReadOnlyOHLCV liest
# ----------------------------------------------------------------------
class DatabaseOHLCVData(DataBase):
    """
    Backtrader-Feed, der OHLCV aus SQL Server liest.

    - rein lesend
    - History per Bulk-Select
    - Live via DB-Polling (kein Socket, kein Async)
    """

    params = (
        ("db_config", None),        # MSSQLFeedConfig
        ("exchange", None),        # "binance", "bitget", "mexc", ...
        ("symbol", None),          # "btcusdt", "ethusdt", ...
        ("timeframe", TimeFrame.Seconds),
        ("compression", 1),
        ("fromdate", None),        # datetime
        ("todate", None),          # optional
        ("live", False),
        ("poll_interval", 0.25),
        ("mode", "global"),        # "global" | "per_pair"
        ("global_table", "ohlcv"),
        ("schema", "dbo"),
        ("table_pattern", "{symbol}_klines"),
        ("debug", False),
    )

    _ST_HIST, _ST_LIVE, _ST_OVER = range(3)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.p.db_config is None or not isinstance(self.p.db_config, MSSQLFeedConfig):
            raise ValueError("db_config (MSSQLFeedConfig) muss gesetzt sein")

        if not self.p.exchange:
            raise ValueError("exchange muss gesetzt sein")
        if not self.p.symbol:
            raise ValueError("symbol muss gesetzt sein")

        self._reader = ReadOnlyOHLCV(
            config=self.p.db_config,
            mode=self.p.mode,
            global_table=self.p.global_table,
            schema=self.p.schema,
            table_pattern=self.p.table_pattern,
        )

        self._buffer: Deque[list] = deque()
        self._state = self._ST_HIST if self.p.fromdate else (self._ST_LIVE if self.p.live else self._ST_OVER)
        self._tf_str = self._timeframe_to_str(self.p.timeframe, self.p.compression)
        self._last_ts: Optional[datetime] = None

    # ---- helpers --------------------------------------------------
    @staticmethod
    def _timeframe_to_str(tf: TimeFrame, compression: int) -> str:
        if tf == TimeFrame.Seconds:
            return f"{compression}s"
        if tf == TimeFrame.Minutes:
            return f"{compression}m"
        if tf == TimeFrame.Hours:
            return f"{compression}h"
        if tf == TimeFrame.Days:
            return f"{compression}d"
        raise ValueError(f"Unsupported TimeFrame/compression: {tf}, {compression}")

    @staticmethod
    def _ensure_datetime(ts: Any) -> datetime:
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts / 1000.0)
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts)
            except Exception:
                if "." in ts:
                    base = ts.split(".")[0]
                    return datetime.strptime(base, "%Y-%m-%d %H:%M:%S")
                raise
        raise TypeError(f"Unsupported timestamp type: {type(ts)}")

    def _load_history(self):
        if not self.p.fromdate:
            return

        end = self.p.todate or datetime.utcnow()

        rows = self._reader.get_ohlcv(
            exchange=self.p.exchange,
            symbol=self.p.symbol,
            timeframe=self._tf_str,
            start=self.p.fromdate,
            end=end,
        )

        for r in rows:
            dt = self._ensure_datetime(r["timestamp"])
            dtnum = date2num(dt)
            bar = [
                dtnum,
                float(r["open"]),
                float(r["high"]),
                float(r["low"]),
                float(r["close"]),
                float(r["volume"]),
            ]
            self._buffer.append(bar)
            if self._last_ts is None or dt > self._last_ts:
                self._last_ts = dt

    def _poll_new(self):
        if self._last_ts is None:
            # fallback: fromdate oder "jetzt - 1 Tag"
            start = self.p.fromdate or (datetime.utcnow() - timedelta(days=1))
        else:
            start = self._last_ts

        rows = self._reader.get_ohlcv(
            exchange=self.p.exchange,
            symbol=self.p.symbol,
            timeframe=self._tf_str,
            start=start,
            end=None,
        )

        if not rows:
            return

        new = []
        for r in rows:
            dt = self._ensure_datetime(r["timestamp"])
            if self._last_ts is None or dt > self._last_ts:
                dtnum = date2num(dt)
                new.append([
                    dtnum,
                    float(r["open"]),
                    float(r["high"]),
                    float(r["low"]),
                    float(r["close"]),
                    float(r["volume"]),
                ])
                self._last_ts = dt

        for bar in new:
            self._buffer.append(bar)

    # ---- Backtrader Hooks -----------------------------------------
    def start(self):
        DataBase.start(self)

        if self._state == self._ST_HIST:
            self.put_notification(self.DELAYED)
            self._load_history()

        if self.p.live:
            self.put_notification(self.LIVE)

    def stop(self):
        DataBase.stop(self)

    def _load(self):
        if self._state == self._ST_OVER:
            return False

        # 1) Bar im Buffer?
        if self._buffer:
            dtnum, o, h, l, c, v = self._buffer.popleft()
            self.lines.datetime[0] = dtnum
            self.lines.open[0] = o
            self.lines.high[0] = h
            self.lines.low[0] = l
            self.lines.close[0] = c
            self.lines.volume[0] = v
            return True

        # 2) History leer -> evtl. in LIVE wechseln
        if self._state == self._ST_HIST:
            if self.p.live:
                self._state = self._ST_LIVE
                return None
            else:
                self._state = self._ST_OVER
                return False

        # 3) LIVE: DB pollen
        if self._state == self._ST_LIVE:
            self._poll_new()
            if self._buffer:
                return None
            time.sleep(self.p.poll_interval)
            return None

        return False


# Convenience-Subklassen je Exchange
class BinanceDBData(DatabaseOHLCVData):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("exchange", "binance")
        super().__init__(*args, **kwargs)


class BitgetDBData(DatabaseOHLCVData):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("exchange", "bitget")
        super().__init__(*args, **kwargs)


class MexcDBData(DatabaseOHLCVData):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("exchange", "mexc")
        super().__init__(*args, **kwargs)
