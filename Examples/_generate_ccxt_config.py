#!/usr/bin/env python3
from __future__ import annotations

'''
Use like: _generate_ccxt_config.py binance --account main --default-type spot
'''

import argparse
import json
import sys
from pathlib import Path

try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init(autoreset=True)
    _COLOR = True
except ImportError:
    _COLOR = False
    class _Dummy:
        RESET_ALL = ""
    class _DummyFore:
        RED = GREEN = YELLOW = CYAN = MAGENTA = ""
    Fore = _DummyFore()
    Style = _Dummy()


def _ok(msg: str) -> None:
    if _COLOR:
        print(f"{Fore.GREEN}[OK]{Style.RESET_ALL} {msg}")
    else:
        print(f"[OK] {msg}")


def _warn(msg: str) -> None:
    if _COLOR:
        print(f"{Fore.YELLOW}[!]{Style.RESET_ALL} {msg}")
    else:
        print(f"[!] {msg}")


def _info(msg: str) -> None:
    if _COLOR:
        print(f"{Fore.CYAN}[*]{Style.RESET_ALL} {msg}")
    else:
        print(f"[*] {msg}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a CCXT config JSON template inside the current venv."
    )
    parser.add_argument(
        "exchange",
        help="CCXT exchange id, e.g. binance, bybit, kraken",
    )
    parser.add_argument(
        "-a",
        "--account",
        default="main",
        help="Logical account name (default: main)",
    )
    parser.add_argument(
        "-t",
        "--default-type",
        default="future",
        choices=["spot", "margin", "future"],
        help="CCXT defaultType (spot/margin/future). Default: future",
    )
    parser.add_argument(
        "-f",
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON if it already exists",
    )

    args = parser.parse_args()

    venv_root = Path(sys.prefix)
    cfg_dir = venv_root / "ccxt"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{args.exchange}_{args.account}.json"
    if cfg_path.exists() and not args.overwrite:
        _warn(f"{cfg_path} already exists.")
        _info("Use --overwrite if you really want to replace it.")
        return 1
    template = {
        "apiKey": "PUT_YOUR_API_KEY_HERE",
        "secret": "PUT_YOUR_API_SECRET_HERE",
        "enableRateLimit": True,
        "options": {
            "defaultType": args.default_type,
            "adjustForTimeDifference": True,
        },
    }
    with cfg_path.open("w") as f:
        json.dump(template, f, indent=2)

    exu = args.exchange.upper()
    acu = args.account.upper()

    _ok(f"Wrote CCXT config template for '{args.exchange}' / '{args.account}':")
    print(f"     {cfg_path}")
    print()
    _info("Now edit that file and put in your real apiKey/secret.")
    _info("Or leave them as-is and use env vars instead, e.g.:")
    print(f"  export BTQ_{exu}_{acu}_API_KEY='your_key_here'")
    print(f"  export BTQ_{exu}_{acu}_API_SECRET='your_secret_here'")
    print()
    _info("This works together with load_ccxt_config(exchange, account) from ccxt_config.py.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
