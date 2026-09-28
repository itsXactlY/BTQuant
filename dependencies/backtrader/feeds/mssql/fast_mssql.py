"""
fast_mssql — compatibility shim for BTQuant's C++ fast_mssql module.
Provides the same API using pyodbc (when available).
BTQuant uses this via: from backtrader.dontcommit import fast_mssql
"""
from typing import Any

_HAS_PYODBC = False
try:
    import pyodbc as _pyodbc
    _HAS_PYODBC = True
except ImportError:
    pass

_CONNECTIONS: dict = {}


def fetch_data_from_db(conn_str: str, query: str) -> list[dict]:
    if not _HAS_PYODBC:
        raise RuntimeError("pyodbc not installed — install with: pip install pyodbc")
    if conn_str not in _CONNECTIONS:
        _CONNECTIONS[conn_str] = _pyodbc.connect(conn_str, timeout=5, autocommit=True)
    conn = _CONNECTIONS[conn_str]
    cursor = conn.cursor()
    cursor.execute(query)
    columns = [c[0] for c in cursor.description]
    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    cursor.close()
    return rows


def execute_non_query(conn_str: str, query: str) -> int:
    if not _HAS_PYODBC:
        raise RuntimeError("pyodbc not installed")
    if conn_str not in _CONNECTIONS:
        _CONNECTIONS[conn_str] = _pyodbc.connect(conn_str, timeout=5, autocommit=True)
    cursor = _CONNECTIONS[conn_str].cursor()
    cursor.execute(query)
    rc = cursor.rowcount
    cursor.close()
    return rc


def bulk_insert(conn_str: str, query: str, rows: list[list[Any]]) -> int:
    if not _HAS_PYODBC:
        raise RuntimeError("pyodbc not installed")
    if conn_str not in _CONNECTIONS:
        _CONNECTIONS[conn_str] = _pyodbc.connect(conn_str, timeout=10, autocommit=False)
    conn = _CONNECTIONS[conn_str]
    cursor = conn.cursor()
    cursor.executemany(query, rows)
    conn.commit()
    rc = cursor.rowcount
    cursor.close()
    return rc


def close_all_connections() -> None:
    for conn in _CONNECTIONS.values():
        try:
            conn.close()
        except Exception:
            pass
    _CONNECTIONS.clear()


def get_pool_size() -> int:
    return len(_CONNECTIONS)


def remove_connection(conn_str: str) -> None:
    conn = _CONNECTIONS.pop(conn_str, None)
    if conn:
        try:
            conn.close()
        except Exception:
            pass