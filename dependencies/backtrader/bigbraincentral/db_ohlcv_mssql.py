from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, List

import time
from collections import deque
from backtrader.feed import DataBase
from backtrader.utils import date2num
from backtrader.dataseries import TimeFrame

import fast_mssql

@dataclass
class MSSQLFeedConfig: # TODO :: use dontcommit
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
    def __init__(self, config: MSSQLFeedConfig, mode: str = "global",
                 global_table: str = "ohlcv", schema: str = "dbo",
                 table_pattern: str = "{symbol}_klines") -> None:
        if mode not in ("global", "per_pair"):
            raise ValueError("mode must be 'global' or 'per_pair'")
        self.config = config
        self.mode = mode
        self.global_table = global_table
        self.schema = schema
        self.table_pattern = table_pattern
        self._conn_str = config.connection_string()

    @staticmethod
    def _sym_compact(symbol: str) -> str:     # BTC-USDT -> BTCUSDT
        return symbol.replace("/", "").replace("-", "").replace("_", "")

    @staticmethod
    def _sym_underscore(symbol: str) -> str:  # BTC-USDT -> BTC_USDT
        return symbol.replace("/", "_").replace("-", "_")

    def _q(self, s: str) -> str:
        return s.replace("'", "''")

    def _ident(self, s: str) -> str:
        import re
        if not re.fullmatch(r"[A-Za-z0-9_]+", s):
            raise ValueError(f"invalid identifier: {s}")
        return s

    def _table_exists(self, schema: str, table: str) -> bool:
        q = (
            "SELECT 1 FROM sys.tables t "
            "JOIN sys.schemas s ON s.schema_id=t.schema_id "
            f"WHERE s.name='{self._q(schema)}' AND t.name='{self._q(table)}'"
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
        return f"{self._ident(self.schema)}.{self._ident(cand1)}"

    def get_ohlcv(
        self, exchange: str, symbol: str, timeframe: str,
        start: datetime, end: Optional[datetime] = None,
        limit: Optional[int] = None, strict_gt: bool = False
    ) -> List[dict]:
        if end is None:
            end = datetime.utcnow()

        fqtn = self._table_name(exchange, symbol)
        comp = ">" if strict_gt else ">="
        top_clause = f"TOP {int(limit)} " if limit else ""
        start_str = start.strftime("%Y-%m-%d %H:%M:%S")
        end_str   = end.strftime("%Y-%m-%d %H:%M:%S")

        if self.mode == "global":
            ex = self._q(exchange)
            sym = self._q(symbol)
            tf  = self._q(timeframe)
            query = (
                f"SELECT {top_clause}"
                f"timestamp, [open], high, low, [close], volume "
                f"FROM {fqtn} "
                f"WHERE exchange='{ex}' AND symbol='{sym}' AND timeframe='{tf}' "
                f"AND timestamp {comp} '{start_str}' AND timestamp < '{end_str}' "
                f"ORDER BY timestamp ASC;"
            )
        else:
            tf  = self._q(timeframe)
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
    def get_ohlcv_from_trades(
        self,
        exchange: str,
        symbol: str,
        bucket_s: int,              # 1 for 1s, 15 for 15s, 60 for 1m...
        start: datetime,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
        strict_gt: bool = False,
    ) -> List[dict]:
        if end is None:
            end = datetime.utcnow()

        ex = self._q(exchange)
        sym = self._q(symbol)
        start_str = start.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # ms ok
        end_str   = end.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        query = f"""
        WITH b AS (
          SELECT
            DATEADD(SECOND,
                    (DATEDIFF(SECOND, '1970-01-01', [timestamp]) / {bucket_s}) * {bucket_s},
                    '1970-01-01') AS ts,
            [timestamp], price, quantity
          FROM dbo.trades
          WHERE exchange = '{ex}'
            AND symbol = '{sym}'
            AND [timestamp] {'>' if strict_gt else '>='} '{start_str}'
            AND [timestamp] < '{end_str}'
        ),
        r AS (
          SELECT
            ts, [timestamp], price, quantity,
            ROW_NUMBER() OVER (PARTITION BY ts ORDER BY [timestamp])        AS rn_asc,
            ROW_NUMBER() OVER (PARTITION BY ts ORDER BY [timestamp] DESC)   AS rn_desc
          FROM b
        )
        SELECT
          ts AS timestamp,
          MAX(CASE WHEN rn_asc  = 1 THEN price END) AS [open],
          MAX(price)                                AS high,
          MIN(price)                                AS low,
          MAX(CASE WHEN rn_desc = 1 THEN price END) AS [close],
          SUM(quantity)                             AS volume
        FROM r
        GROUP BY ts
        ORDER BY ts ASC
        {f'OFFSET 0 ROWS FETCH NEXT {int(limit)} ROWS ONLY' if limit else ''};
        """

        rows = fast_mssql.fetch_data_from_db(self._conn_str, query)
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        print(rows, cols, [dict(zip(cols, r)) for r in rows])
        return [dict(zip(cols, r)) for r in rows]
        
    def _retry_fetch(self, query: str, retries: int = 4, base_delay: float = 0.05):
        for i in range(retries):
            try:
                return fast_mssql.fetch_data_from_db(self._conn_str, query)
            except RuntimeError as e:
                msg = str(e).lower()
                if "deadlock" in msg or "1205" in msg:
                    time.sleep(base_delay * (2 ** i))
                    continue
                raise

    def get_ticks(self, exchange: str, symbol: str,
                start: datetime, end: Optional[datetime] = None,
                limit: Optional[int] = 5000, strict_gt: bool = False) -> List[dict]:
        if end is None:
            end = datetime.now(timezone.utc)
        aliases = {
            symbol, symbol.replace('-', '/'), symbol.replace('/', '-'),
            symbol.replace('-', ''), symbol.replace('/', ''), symbol.replace('/', '_')
        }
        ex = self._q(exchange.lower())
        syms = "', '".join(self._q(s) for s in aliases)
        comp = '>' if strict_gt else '>='
        top = f"TOP {int(limit)} " if limit else ""
        startstr = start.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        endstr   = end.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        query = f"""
            SELECT {top} t.[timestamp], t.price, t.quantity
            FROM dbo.trades AS t WITH (READPAST, ROWLOCK, INDEX(IX_trades_ex_sym_ts))
            WHERE t.exchange='{ex}' AND t.symbol IN ('{syms}')
            AND t.[timestamp] {comp} '{startstr}' AND t.[timestamp] < '{endstr}'
            ORDER BY t.[timestamp] ASC;
        """
        rows = self._retry_fetch(query)
        return [{'timestamp': ts,
                'open': price, 'high': price, 'low': price, 'close': price,
                'volume': qty} for ts, price, qty in rows]


class DatabaseOHLCVData(DataBase):
    params = (
        ("db_config", None),        # MSSQLFeedConfig
        ("exchange", None),         # "binance", "bitget", "mexc", ...
        ("symbol", None),           # "btcusdt", "ethusdt", ...
        ("timeframe",               TimeFrame.Seconds),
        ("compression", 1),
        ("fromdate", None),         # datetime
        ("todate", None),           # optional
        ("live", True),
        ("poll_interval", 1),
        ("mode", "global"),         # "global" | "per_pair"
        ("global_table", "ohlcv"),
        ("schema", "dbo"),
        ("table_pattern", "{symbol}_klines"),
        ("source", "auto"),    # "auto" | "klines" | "trades"
        ('ticks', False),
        ("debug", True),
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
        self._state  = self._ST_HIST if self.p.fromdate else (self._ST_LIVE if self.p.live else self._ST_OVER)
        self._tf_str = self._timeframe_to_str(self.p.timeframe, self.p.compression)  # e.g. "1m", "15s"
        self._last_ts = None

    @property
    def symbol(self):
        return self.p.symbol

    @property
    def exchange(self):
        return self.p.exchange

    @staticmethod
    def _timeframe_to_str(tf, compression):
        if tf == TimeFrame.Seconds: return f"{compression}s"
        if tf == TimeFrame.Minutes: return f"{compression}m"
        if tf == TimeFrame.Hours:   return f"{compression}h"
        if tf == TimeFrame.Days:    return f"{compression}d"
        raise ValueError("Unsupported timeframe")

    @staticmethod
    def _ensure_datetime(ts):
        """Coerce DB value (datetime, str, int ms) to timezone-aware UTC datetime."""
        if isinstance(ts, datetime):
            # assume already UTC/naive; normalize to aware UTC
            return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
        if isinstance(ts, str):
            # try ISO first, then a common fallback without fractional seconds
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

    # DatabaseOHLCVData
    def _select_reader(self):
        if self.p.ticks:
            return "ticks", None
        if self.p.source == "klines":
            return "klines", None
        sec, n = self._is_seconds()
        return ("trades", n) if sec else ("klines", None)

    def _fetch(self, start, end, limit, strict_gt):
        kind, bucket = self._select_reader()
        if kind == "ticks":
            return self._trades.get_ticks(self.p.exchange, self.p.symbol,
                                        start=start, end=end, limit=limit, strict_gt=strict_gt)
        if kind == "klines":
            return self._klines.get_ohlcv(self.p.exchange, self.p.symbol, self._tf_str,
                                        start=start, end=end, limit=limit, strict_gt=strict_gt)
        return self._trades.get_ohlcv_from_trades(self.p.exchange, self.p.symbol,
                                                bucket_s=bucket or 1,
                                                start=start, end=end, limit=limit, strict_gt=strict_gt)

    def _load_history(self):
        if not self.p.fromdate:
            return

        end = self.p.todate or datetime.now(timezone.utc)
        rows = self._fetch(self.p.fromdate, end, limit=None, strict_gt=False)

        for r in rows:
            dt = self._ensure_datetime(r["timestamp"])
            self._buffer.append([
                date2num(dt),
                float(r["open"]), float(r["high"]),
                float(r["low"]),  float(r["close"]),
                float(r["volume"]),
            ])
            self._last_ts = dt if self._last_ts is None or dt > self._last_ts else self._last_ts

    def _poll_new(self):
        # next start strictly after last delivered bar to avoid duplicates
        if self._last_ts is None:
            start = self.p.fromdate or (datetime.now(timezone.utc) - timedelta(days=1))
            strict_gt = False
        else:
            start = self._last_ts + timedelta(microseconds=1)
            strict_gt = True

        rows = self._fetch(start=start, end=None, limit=1000, strict_gt=strict_gt)
        if not rows:
            return

        for r in rows:
            dt = self._ensure_datetime(r["timestamp"])
            self._buffer.append([
                date2num(dt),
                float(r["open"]), float(r["high"]),
                float(r["low"]),  float(r["close"]),
                float(r["volume"]),
            ])
            if self._last_ts is None or dt > self._last_ts:
                self._last_ts = dt

    def start(self):
        DataBase.start(self)
        self._load_history()
        if self.p.live:
            t0 = time.time()
            # bootstrap up to 2s for first tick
            while not getattr(self, "_buffer", None) and time.time() - t0 < 2.0:
                self._poll_new()
                if not self._buffer:
                    time.sleep(self.p.poll_interval)
            self.put_notification(self.LIVE)
        elif self._state == self._ST_LIVE:
            self.put_notification(self.LIVE)

    def _load(self):
        if self._state == self._ST_OVER:
            return False

        # 1) Drain any buffered bars
        if self._buffer:
            dtnum, o, h, l, c, v = self._buffer.popleft()
            self.lines.datetime[0] = dtnum
            self.lines.open[0] = o
            self.lines.high[0] = h
            self.lines.low[0] = l
            self.lines.close[0] = c
            self.lines.volume[0] = v
            return True

        # 2) If we just finished history and want live, flip state
        if self._state == self._ST_HIST:
            if self.p.live:
                self._state = self._ST_LIVE
                self.put_notification(self.LIVE)
                return None
            self._state = self._ST_OVER
            return False

        # 3) Live polling
        if self._state == self._ST_LIVE:
            self._poll_new()
            if self._buffer:
                # tell BT “data will be available on the next cycle”
                return None
            print('Poll Interval: ', self.p.poll_interval)
            time.sleep(self.p.poll_interval)
            return None

        return False

# Convenience-subclasses per Exchange
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
