"""MCP tools ⟷ exchange connectors status & API keys check."""

import os
import json
from pathlib import Path

BTQUANT_ROOT = Path(__file__).resolve().parent.parent.parent

from registry import register_tool


@register_tool(
    "exchange_status",
    "List all exchange connectors and their configuration status.",
    {"type": "object", "properties": {}, "required": []},
)
def exchange_status(_args) -> dict:
    result = {}

    # ── CCXT config files ──────────────────────────────────────────────
    ccxt_glob = list(BTQUANT_ROOT.rglob("ccxt/*.json"))
    result["ccxt_config_files"] = [str(p.relative_to(BTQUANT_ROOT)) for p in ccxt_glob]

    # ── Check API key env vars ─────────────────────────────────────────
    api_vars = {}
    for var in sorted(os.environ):
        if var.startswith("BTQ_") or var.startswith("LIVE_API"):
            api_vars[var] = "***" + os.environ[var][-4:] if os.environ[var] else "(empty)"
    result["env_api_keys"] = api_vars

    # ── CCXT store files ───────────────────────────────────────────────
    stores_dir = BTQUANT_ROOT / "dependencies" / "backtrader" / "stores"
    result["available_stores"] = sorted([
        f.stem for f in stores_dir.glob("*_store.py") if f.stem != "__init__"
    ])

    # ── Broker files ───────────────────────────────────────────────────
    brokers_dir = BTQUANT_ROOT / "dependencies" / "backtrader" / "brokers"
    result["available_brokers"] = sorted([
        f.stem for f in brokers_dir.glob("*_broker.py") if f.stem != "__init__"
    ] + [f.stem for f in brokers_dir.glob("*broker.py") if f.stem != "__init__"])

    # ── Feed files ─────────────────────────────────────────────────────
    feeds_dir = BTQUANT_ROOT / "dependencies" / "backtrader" / "feeds"
    result["available_feeds"] = sorted([
        f.stem for f in feeds_dir.glob("*_feed.py") if f.stem != "__init__"
    ])

    # ── dontcommit.py secrets (all keys present?) ──────────────────────
    dontcommit_path = BTQUANT_ROOT / "dependencies" / "backtrader" / "dontcommit.py"
    if dontcommit_path.exists():
        content = dontcommit_path.read_text()
        # Check for common placeholders
        placeholders = []
        for line in content.split("\n"):
            if line.strip().startswith("#"):
                continue
            if "=" in line and ('"' in line or "'" in line):
                var = line.split("=")[0].strip()
                val = line.split("=")[1].strip().strip('"').strip("'")
                if not val or val == "":
                    placeholders.append(var)
        result["dontcommit_placeholders"] = placeholders
    else:
        result["dontcommit_placeholders"] = []

    return result


@register_tool(
    "exchange_not_connected",
    "List exchanges and DEXes that are NOT yet connected to BTQuant.",
    {"type": "object", "properties": {}, "required": []},
)
def exchange_not_connected(_args) -> dict:
    return {
        "missing_perp_dexes": [
            {"name": "Hyperliquid", "reason": "ECDSA wallet auth needed, EIP-712 signing, no CCXT support. Largest perp DEX."},
            {"name": "dYdX", "reason": "Requires Starkware/zk integration or dYdX Chain REST."},
            {"name": "Vertex", "reason": "Arbitrum perp DEX, no connector exists."},
            {"name": "GMX / GMX v2", "reason": "AMM-based perp, no connector."},
            {"name": "SynFutures", "reason": "Orderbook perp on Blast, no connector."},
            {"name": "Drift", "reason": "Solana perp DEX, no connector."},
            {"name": "Jupiter Perp", "reason": "Solana aggregator, no connector."},
            {"name": "Gains Network", "reason": "Synthetic perp, no connector."},
        ],
        "missing_cex_futures": [
            {"name": "Binance Futures", "reason": "CCAPI collects only spot data."},
            {"name": "Bybit Futures", "reason": "CCAPI collects only spot data."},
            {"name": "OKX Futures", "reason": "CCAPI collects only spot data."},
        ],
        "note": "All 10+ current connectors are spot-only. See invariant:btquant-no-perp-connector in memory."
    }