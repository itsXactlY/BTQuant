"""
btq_secrets — single source of truth for BTQuant SQL Server credentials.

WHY THIS EXISTS
---------------
The DB password used to be hardcoded in a dozen places across this repo.
It was git-tracked with a live GitHub remote, so the value was baked into
history and every hardcoded copy was a dead placeholder anyway. This module
replaces all of them with one indirection: values live in a local, mode-600
secrets file OUTSIDE the repo, and this loader reads it at import time.

This file is intentionally VALUE-FREE. Never write a credential here — the
repo is tracked and published, so anything committed here leaks.

RESOLUTION ORDER
----------------
1. ``$BTQ_SECRETS``                      explicit override (path to a .py file)
2. ``<sys.prefix>/etc/btquant/secrets.py``  tpad layout (inside a venv)
3. ``~/.btq/etc/btquant/secrets.py``        home fallback (no venv)

Later candidates win, so a home-dir file overrides a venv one. The override
is authoritative and never falls through.

The secrets file is plain Python, so it can hold arbitrary values without
quoting concerns. Only public (non-underscore) names are exported.
"""

from __future__ import annotations

import os
import sys

#: Names we know how to use. Anything else in the file is still exported.
DEFAULTS = {
    "server": "localhost",
    "candle_database": "BinanceData",
    "optuna_database": "OptunaBT",
    "marketdata_database": "BTQ_MarketData",
    "username": "SA",
    "password": "",
    "driver": "{ODBC Driver 18 for SQL Server}",
    "trust_server_certificate": "yes",
}

_CANDIDATES = (
    os.path.join(sys.prefix, "etc", "btquant", "secrets.py"),
    os.path.expanduser("~/.btq/etc/btquant/secrets.py"),
)


def secrets_path() -> str | None:
    """Return the secrets file that will be used, or None if there is none."""
    override = os.environ.get("BTQ_SECRETS")
    if override:
        return override
    for path in reversed(_CANDIDATES):
        if os.path.isfile(path):
            return path
    return None


def load() -> dict:
    """Return the merged secrets dict (file values win over DEFAULTS)."""
    merged = dict(DEFAULTS)
    path = secrets_path()
    if not path:
        return merged
    namespace: dict = {}
    try:
        with open(path) as handle:
            exec(compile(handle.read(), path, "exec"), namespace)
    except OSError as exc:
        raise RuntimeError(f"btq_secrets: cannot read {path}: {exc}") from exc
    merged.update({k: v for k, v in namespace.items() if not k.startswith("_")})
    merged["_secrets_file"] = path
    return merged


def get(name: str, default=None):
    """Read a single secret. Convenience for one-off call sites."""
    return load().get(name, default)


def connection_string(
    database: str | None = None,
    driver: str | None = None,
    trust_server_certificate: str | None = None,
) -> str:
    """Build an ODBC connection string from the resolved secrets."""
    cfg = load()
    return (
        f"DRIVER={driver or cfg['driver']};"
        f"SERVER={cfg['server']};"
        f"DATABASE={database or cfg['marketdata_database']};"
        f"UID={cfg['username']};"
        f"PWD={cfg['password']};"
        f"TrustServerCertificate="
        f"{trust_server_certificate or cfg['trust_server_certificate']};"
    )


def sqlcmd_env() -> dict:
    """Env for sqlcmd, so the password never appears in argv (visible via ps).

    Use with ``subprocess.run(..., env=btq_secrets.sqlcmd_env())``.
    """
    cfg = load()
    env = dict(os.environ)
    env["SQLCMDPASSWORD"] = cfg["password"]
    return env


def require() -> dict:
    """Like load(), but fails loudly when no password is available.

    Use in code paths that cannot do anything useful without credentials,
    so the operator sees one clear error instead of an auth failure later.
    """
    cfg = load()
    if not cfg.get("password"):
        raise RuntimeError(
            "btq_secrets: no password loaded. Create the secrets file with:\n"
            "  mkdir -p ~/.btq/etc/btquant && chmod 700 ~/.btq/etc/btquant\n"
            "  # then write server/database/username/password into\n"
            "  # ~/.btq/etc/btquant/secrets.py and chmod 600 that file\n"
            "Or point $BTQ_SECRETS at your own file. "
            f"(searched: {', '.join(_CANDIDATES)})"
        )
    return cfg
