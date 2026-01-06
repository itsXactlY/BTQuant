"""
Utilities - Helper functions for the market data collector

This module provides utility functions used throughout the
market data collector, equivalent to utilities.h in C++.
"""

import time
from datetime import datetime


def get_current_timestamp() -> str:
    """Get current timestamp formatted for logging (equivalent to C++ function)"""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S") + f".{now.microsecond:06d}"


def create_connection_string(server: str, database: str, user: str, password: str) -> str:
    """
    Create MS SQL Server connection string

    Args:
        server: SQL Server hostname/IP
        database: Database name
        user: Username
        password: Password

    Returns:
        ODBC connection string
    """
    return (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={user};"
        f"PWD={password};"
        "TrustServerCertificate=yes;"
        "MARS_Connection=yes;"
        "Connection Timeout=30;"
        "Command Timeout=60;"
    )


def now_micros() -> int:
    """Get current time in microseconds since epoch"""
    return int(time.time() * 1_000_000)


def safe_parse_double(label: str, s: str, ok: list) -> float:
    """
    Safely parse a string to float with error handling

    Args:
        label: Label for error messages
        s: String to parse
        ok: List with single boolean element to indicate success

    Returns:
        Parsed float value (0.0 on error)
    """
    ok[0] = False
    if not s:
        return 0.0

    try:
        # Remove any whitespace
        s = s.strip()
        if not s:
            return 0.0

        val = float(s)
        ok[0] = True
        return val
    except ValueError as e:
        print(f"safe_parse_double({label}): exception for '{s}': {e}")
        return 0.0


def split(string: str, delimiter: str) -> list:
    """
    Split string by delimiter (equivalent to C++ split function)

    Args:
        string: String to split
        delimiter: Delimiter character

    Returns:
        List of split parts
    """
    if not delimiter:
        return [string]

    parts = []
    current = ""
    for char in string:
        if char == delimiter:
            parts.append(current)
            current = ""
        else:
            current += char

    if current:  # Add remaining part
        parts.append(current)

    return parts