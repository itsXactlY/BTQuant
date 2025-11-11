from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List
import time
from collections import deque

from backtrader.feed import DataBase
from backtrader.utils import date2num
from backtrader.dataseries import TimeFrame

import fast_mssql

@dataclass
class MSSQLFeedConfig:
    server: str = "localhost"
    database: str = "BTQ_MarketData"
    username: str = "SA"
    password: str = "q?}33YIToo:H%xue$Kr*"
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
    def __init__(
        self,
        config: MSSQLFeedConfig,
        mode: str = "global",
        global_table: str = "ohlcv",
        schema: str = "dbo",
        table_pattern: str = "{symbol}_klines",
    ) -> None:
        if mode not in ("global", "per_pair"):
            raise ValueError("mode must be 'global' or 'per_pair'")
        self.config = config
        self.mode = mode
        self.global_table = global_table
        self.schema = schema
        self.table_pattern = table_pattern
        self._conn_str = config.connection_string()

    @staticmethod
    def _sym_compact(symbol: str) -> str:  # BTC-USDT -> BTCUSDT
        return symbol.replace("/", "").replace("-", "").replace("_", "")

    @staticmethod
    def _sym_underscore(symbol: str) -> str:  # BTC-USDT -> BTC_USDT
        return symbol.replace("/", "_").replace("-", "_")

    @staticmethod
    def _qv(s: str) -> str:
        return s.replace("'", "''")

    @staticmethod
    def _ident(s: str) -> str:
        import re

        if not re.fullmatch(r"[A-Za-z0-9_]+", s):
            raise ValueError(f"invalid identifier: {s}")
        return s

    def _table_exists(self, schema: str, table: str) -> bool:
        q = (
            "SELECT 1 FROM sys.tables t "
            "JOIN sys.schemas s ON s.schema_id=t.schema_id "
            f"WHERE s.name='{self._qv(schema)}' AND t.name='{self._qv(table)}'"
        )
        rows = fast_mssql.fetch_data_from_db(self._conn_str, q)
        return bool(rows)

    def _table_name(self, exchange: str, symbol: str) -> str:
        if self.mode == "global":
            return f"{self._ident(self.schema)}.{self._ident(self.global_table)}"
        ex = exchange.lower()
        cand1 = self.table_pattern.format(exchange=ex, symbol=self._sym_underscore(symbol))
        cand2 = self.table_pattern.format(exchange=ex, symbol=self._sym_compact(symbol))
        if self._table_exists(self.schema, cand1):
            return f"{self._ident(self.schema)}.{self._ident(cand1)}"
        if self._table_exists(self.schema, cand2):
            return f"{self._ident(self.schema)}.{self._ident(cand2)}"
        # fallback to first candidate
        return f"{self._ident(self.schema)}.{self._ident(cand1)}"

    def get_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
        strict_gt: bool = False,
    ) -> List[dict]:
        if end is None:
            end = datetime.now(timezone.utc)
        fqtn = self._table_name(exchange, symbol)
        comp = ">" if strict_gt else ">="
        top_clause = f"TOP {int(limit)} " if limit else ""
        start_str = start.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end.strftime("%Y-%m-%d %H:%M:%S")

        if self.mode == "global":
            ex = self._qv(exchange)
            sym = self._qv(symbol)
            tf = self._qv(timeframe)
            query = (
                f"SELECT {top_clause}"
                f"timestamp, [open], high, low, [close], volume "
                f"FROM {fqtn} "
                f"WHERE exchange='{ex}' AND symbol='{sym}' AND timeframe='{tf}' "
                f"AND timestamp {comp} '{start_str}' AND timestamp < '{end_str}' "
                f"ORDER BY timestamp ASC;"
            )
        else:
            tf = self._qv(timeframe)
            query = (
                f"SELECT {top_clause}"
                f"timestamp, [open], high, low, [close], volume "
                f"FROM {fqtn} "
                f"WHERE timeframe='{tf}' "
                f"AND timestamp {comp} '{start_str}' AND timestamp < '{end_str}' "
                f"ORDER BY timestamp ASC;"
            )

        rows = fast_mssql.fetch_data_from_db(self._conn_str, query)
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        return [dict(zip(cols, r)) for r in rows]

class ReadOnlyTradesAgg(ReadOnlyOHLCV):
    @staticmethod
    def _qv(s: str) -> str:
        return s.replace("'", "''")

    def _retry_fetch(self, sql: str, retries: int = 4, base_delay: float = 0.05):
        for i in range(retries):
            try:
                return fast_mssql.fetch_data_from_db(self._conn_str, sql)
            except RuntimeError as e:
                msg = str(e).lower()
                if "deadlock" in msg or "1205" in msg:
                    time.sleep(base_delay * (2 ** i))
                    continue
                raise

    def get_ticks_by_id(
        self, exchange: str, symbol: str, last_id: int, limit: int = 1000
    ) -> list[dict]:
        # build alias set to match different stored symbol forms
        aliases = {
            symbol,
            symbol.replace("-", "/"),
            symbol.replace("/", "-"),
            symbol.replace("/", ""),
            symbol.replace("-", ""),
            symbol.replace("/", "_"),
        }
        syms = "', '".join(self._qv(s) for s in sorted(aliases))
        ex = self._qv(exchange.lower())
        sql = f"""
            SELECT TOP ({int(limit)}) t.id, t.[timestamp], t.price, t.quantity
            FROM dbo.trades AS t WITH (READPAST, ROWLOCK)
            WHERE t.exchange='{ex}'
              AND t.symbol IN ('{syms}')
              AND t.id > {int(last_id)}
            ORDER BY t.id ASC
            OPTION (MAXDOP 1, OPTIMIZE FOR UNKNOWN, RECOMPILE);
        """
        rows = self._retry_fetch(sql)
        out = []
        for rid, ts, price, qty in rows:
            out.append(
                {
                    "id": int(rid),
                    "timestamp": ts,
                    "open": float(price),
                    "high": float(price),
                    "low": float(price),
                    "close": float(price),
                    "volume": float(qty),
                }
            )
        return out

class DatabaseOHLCVData(DataBase):
    params = (
        ("db_config", None),  # MSSQLFeedConfig
        ("exchange", None),  # "binance", "okx", ...
        ("symbol", None),  # "BTC-USDT", ...
        ("timeframe", TimeFrame.Seconds),
        ("compression", 1),

        ("fromdate", None),  # datetime
        ("todate", None),  # optional

        ("live", True),
        ("poll_interval", 0.10),  # legacy fixed sleep; adaptive poll is _cur_poll
        ("mode", "global"),  # "global" | "per_pair"
        ("global_table", "ohlcv"),
        ("schema", "dbo"),
        ("table_pattern", "{symbol}_klines"),
        ("source", "auto"),  # "auto" | "klines" | "trades"

        # Tick mode / performance knobs
        ("ticks", True),              # raw per-trade mode (no fixed N-second aggregation)
        ("tick_batch_limit", 1000),   # batch size per poll to keep CPU low
        ("min_poll", 0.05),           # fastest when busy
        ("max_poll", 0.50),           # slowest when idle

        ("debug", False),
    )

    _ST_HIST, _ST_LIVE, _ST_OVER = range(3)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._klines = ReadOnlyOHLCV(
            self.p.db_config, self.p.mode, self.p.global_table, self.p.schema, self.p.table_pattern
        )
        self._trades = ReadOnlyTradesAgg(
            self.p.db_config, self.p.mode, self.p.global_table, self.p.schema, self.p.table_pattern
        )

        self._buffer = deque()
        self._state = self._ST_HIST if self.p.fromdate else (self._ST_LIVE if self.p.live else self._ST_OVER)
        self._tf_str = self._timeframe_to_str(self.p.timeframe, self.p.compression)  # e.g. "1m", "1s"
        self._last_ts = None
        self._last_id = 0  # monotonic per symbol
        self._cur_poll = self.p.max_poll

    @property
    def symbol(self):
        return self.p.symbol

    @property
    def exchange(self):
        return self.p.exchange

    @staticmethod
    def _timeframe_to_str(tf, compression):
        if tf == TimeFrame.Seconds:
            return f"{compression}s"
        if tf == TimeFrame.Minutes:
            return f"{compression}m"
        if tf == TimeFrame.Hours:
            return f"{compression}h"
        if tf == TimeFrame.Days:
            return f"{compression}d"
        raise ValueError("Unsupported timeframe")

    @staticmethod
    def _ensure_datetime(ts):
        """Coerce DB value (datetime, str, int ms) to timezone-aware UTC datetime."""
        if isinstance(ts, datetime):
            return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
        if isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except Exception:
                base = ts.split(".")[0] if "." in ts else ts
                dt = datetime.strptime(base, "%Y-%m-%d %H:%M:%S")
                return dt.replace(tzinfo=timezone.utc)
        raise TypeError(f"Unsupported timestamp type: {type(ts)}")

    def _is_seconds(self) -> tuple[bool, int]:
        tf = self._tf_str
        if tf.endswith("s"):
            return True, int(tf[:-1] or "1")
        return False, 0

    def _push_row(self, r: dict):
        """Append one OHLCV row into BT buffer lines and update last timestamp."""
        dt = self._ensure_datetime(r["timestamp"])
        self._buffer.append(
            [
                date2num(dt),
                float(r["open"]),
                float(r["high"]),
                float(r["low"]),
                float(r["close"]),
                float(r["volume"]),
            ]
        )
        if self._last_ts is None or dt > self._last_ts:
            self._last_ts = dt

    def _select_reader(self):
        if getattr(self.p, "ticks", False):
            return "ticks", None
        if self.p.source == "klines":
            return "klines", None
        sec, n = self._is_seconds()
        return ("trades", n) if sec else ("klines", None)

    @staticmethod
    def _qv(s: str) -> str:
        return s.replace("'", "''")

    @staticmethod
    def _ts_ms(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def _bootstrap_last_id(self):
        dt_anchor = (self.p.fromdate or datetime.now(timezone.utc))
        ex = self._qv(self.p.exchange.lower())
        aliases = {
            self.p.symbol,
            self.p.symbol.replace("-", "/"),
            self.p.symbol.replace("/", "-"),
            self.p.symbol.replace("/", ""),
            self.p.symbol.replace("-", ""),
            self.p.symbol.replace("/", "_"),
        }
        syms = "', '".join(self._qv(s) for s in sorted(aliases))
        ts = self._ts_ms(dt_anchor)
        sql = f"""
            SELECT TOP (1) id
            FROM dbo.trades WITH (READPAST)
            WHERE exchange='{ex}' AND symbol IN ('{syms}') AND [timestamp] <= '{ts}'
            ORDER BY id DESC;
        """
        rows = fast_mssql.fetch_data_from_db(self._trades._conn_str, sql)
        self._last_id = int(rows[0][0]) if rows else 0

    def start(self):
        DataBase.start(self)
        if self.p.fromdate and self.p.fromdate.tzinfo is None:
            self.p.fromdate = self.p.fromdate.replace(tzinfo=timezone.utc)
        self._bootstrap_last_id()
        self._load_history()
        self._cur_poll = self.p.max_poll

    def _load_history(self):
        if not self.p.fromdate:
            return
        rows = self._fetch(self.p.fromdate, self.p.todate or datetime.now(timezone.utc), limit=None, strict_gt=False)
        for r in rows:
            self._push_row(r)

    def _fetch(self, start, end, limit, strict_gt):
        kind, bucket = self._select_reader()
        if kind == "ticks":
            lim = limit or self.p.tick_batch_limit
            rows = self._trades.get_ticks_by_id(self.p.exchange, self.p.symbol, self._last_id, lim)
            if rows:
                self._last_id = rows[-1]["id"]
            return rows
        if kind == "klines":
            return self._klines.get_ohlcv(
                self.p.exchange, self.p.symbol, self._tf_str, start=start or datetime.now(timezone.utc),
                end=end or datetime.now(timezone.utc), limit=limit, strict_gt=strict_gt
            )
        return []

    def _poll_new(self):
        rows = self._fetch(None, None, self.p.tick_batch_limit, True)
        for r in rows:
            self._push_row(r)
        if rows:
            self._cur_poll = max(self.p.min_poll, self._cur_poll * 0.5)
        else:
            self._cur_poll = min(self.p.max_poll, self._cur_poll * 1.25)
        time.sleep(self._cur_poll)

    def _load(self):
        if self._state == self._ST_OVER:
            return False

        if self._buffer:
            dtnum, o, h, _l, c, v = self._buffer.popleft()
            self.lines.datetime[0] = dtnum
            self.lines.open[0] = o
            self.lines.high[0] = h
            self.lines.low[0] = _l
            self.lines.close[0] = c
            self.lines.volume[0] = v
            return True

        if self._state == self._ST_HIST:
            if self.p.live:
                self._state = self._ST_LIVE
                self.put_notification(self.LIVE)
                return None
            self._state = self._ST_OVER
            return False

        if self._state == self._ST_LIVE:
            self._poll_new()
            if self._buffer:
                return None
            if self.p.debug:
                print("Poll Interval:", round(self._cur_poll, 3))
            time.sleep(max(0.0, self.p.poll_interval - self._cur_poll))
            return None

        return False

class BinanceDBData(DatabaseOHLCVData):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("exchange", "binance")
        super().__init__(*args, **kwargs)

class OkxDBData(DatabaseOHLCVData):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("exchange", "okx")
        super().__init__(*args, **kwargs)