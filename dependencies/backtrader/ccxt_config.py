from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if path.is_file():
        with path.open() as f:
            return json.load(f)
    return {}


def load_ccxt_config(
    exchange: str,
    account: str = "main",
    *,
    default_type: str = "spot",
    require_keys: bool = True,
) -> Dict[str, Any]:
    """Load a CCXT config dict for `exchange` and `account`.
    Search order:
      1) <venv>/ccxt/{exchange}_{account}.json
      2) <venv>/ccxt/{exchange}.json
      3) Environment variables (override / fill):
         - BTQ_{EXCHANGE}_{ACCOUNT}_API_KEY
         - BTQ_{EXCHANGE}_{ACCOUNT}_API_SECRET
         - BTQ_{EXCHANGE}_{ACCOUNT}_PASSWORD
         - BTQ_{EXCHANGE}_{ACCOUNT}_UID
         (fallback: BTQ_{EXCHANGE}_API_KEY, ...)

    Returns:
      dict suitable for ccxt.<exchange>(config)

    Example venv layout (venv = .btq or .priv etc.):

      /home/you/projects/BTQuant/.btq/
        └─ ccxt/
           ├─ binance_main.json
           └─ bybit_main.json
    """
    base = Path(sys.prefix)          # venv root (.btq, .priv, .fuqtoy, etc.)
    cfg_dir = base / "ccxt"

    cfg: Dict[str, Any] = {}
    primary = cfg_dir / f"{exchange}_{account}.json"
    fallback = cfg_dir / f"{exchange}.json"
    cfg.update(_load_json_if_exists(primary) or _load_json_if_exists(fallback))
    exu = exchange.upper()
    acu = account.upper()
    env_key = (
        os.getenv(f"BTQ_{exu}_{acu}_API_KEY")
        or os.getenv(f"BTQ_{exu}_API_KEY")
    )
    env_secret = (
        os.getenv(f"BTQ_{exu}_{acu}_API_SECRET")
        or os.getenv(f"BTQ_{exu}_API_SECRET")
    )
    env_password = (
        os.getenv(f"BTQ_{exu}_{acu}_PASSWORD")
        or os.getenv(f"BTQ_{exu}_PASSWORD")
    )
    env_uid = (
        os.getenv(f"BTQ_{exu}_{acu}_UID")
        or os.getenv(f"BTQ_{exu}_UID")
    )
    env_default_type = (
        os.getenv(f"BTQ_{exu}_{acu}_DEFAULT_TYPE")
        or os.getenv(f"BTQ_{exu}_DEFAULT_TYPE")
    )
    if env_key:
        cfg["apiKey"] = env_key
    if env_secret:
        cfg["secret"] = env_secret
    if env_password:
        cfg["password"] = env_password
    if env_uid:
        cfg["uid"] = env_uid

    cfg.setdefault("enableRateLimit", True)
    options = cfg.setdefault("options", {})
    options.setdefault("adjustForTimeDifference", True)
    options.setdefault("defaultType", env_default_type or default_type)

    if require_keys and ("apiKey" not in cfg or "secret" not in cfg):
        raise RuntimeError(
            f"Missing API key/secret for {exchange} account {account}.\n"
            f"Either create:\n  {primary}\n"
            f"or set env vars:\n"
            f"  BTQ_{exu}_{acu}_API_KEY / BTQ_{exu}_{acu}_API_SECRET"
        )

    return cfg
