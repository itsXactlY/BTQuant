"""
fast_mssql — compatibility shim for BTQuant's C++ fast_mssql module.
Provides the same API using pyodbc so BTQuant works without the compiled C++ extension.
BTQuant uses this via: from backtrader.dontcommit import fast_mssql
Expected API: fetch_data_from_db, execute_non_query, bulk_insert, close_all_connections, etc.
"""
import pyodbc
from typing import Any


def fetch_data_from_db(conn_str: str, query: str) -> list[dict]:
    """Execute SELECT query, return list of dicts."""
    conn = pyodbc.connect(conn_str, timeout=5, autocommit=True)
    cursor = conn.cursor()
    cursor.execute(query)
    columns = [c[0] for c in cursor.description]
    rows = []
    for row in cursor.fetchall():
        rows.append(dict(zip(columns, row)))
    cursor.close()
    conn.close()
    return rows


def execute_non_query(conn_str: str, query: str) -> int:
    """Execute INSERT/UPDATE/DELETE, return row count."""
    conn = pyodbc.connect(conn_str, timeout=5, autocommit=True)
    cursor = conn.cursor()
    cursor.execute(query)
    rc = cursor.rowcount
    cursor.close()
    conn.close()
    return rc


def bulk_insert(conn_str: str, query: str, rows: list[list[Any]]) -> int:
    """Bulk insert using executemany."""
    conn = pyodbc.connect(conn_str, timeout=10, autocommit=False)
    cursor = conn.cursor()
    cursor.executemany(query, rows)
    conn.commit()
    rc = cursor.rowcount
    cursor.close()
    conn.close()
    return rc


def close_all_connections() -> None:
    pass


def get_pool_size() -> int:
    return 0


def remove_connection(conn_str: str) -> None:
    pass