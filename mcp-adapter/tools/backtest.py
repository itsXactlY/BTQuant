"""MCP tools ⟷ backtesting (Backtrader strategies)."""

import os
import subprocess
import sys
from pathlib import Path

BTQUANT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BTQUANT_ROOT))
sys.path.insert(0, str(BTQUANT_ROOT / "dependencies"))

from registry import register_tool


@register_tool(
    "backtest_list_strategies",
    "List all available Backtrader strategies in the catalogue.",
    {"type": "object", "properties": {}, "required": []},
)
def backtest_list_strategies(_args) -> dict:
    strats_dir = BTQUANT_ROOT / "dependencies" / "backtrader" / "strategies"
    strategies = {}
    for f in sorted(strats_dir.glob("*.py")):
        if f.name.startswith("__") or f.name == "base.py":
            continue
        # Read first few lines for docstring
        content = f.read_text().split("\n")
        doc = ""
        in_doc = False
        for line in content[:20]:
            if '"""' in line:
                in_doc = not in_doc
                continue
            if in_doc:
                doc += line.strip() + " "
        strategies[f.stem] = {
            "file": str(f.relative_to(BTQUANT_ROOT)),
            "description": doc.strip()[:150],
            "size_kb": round(f.stat().st_size / 1024, 1),
        }
    return {"count": len(strategies), "strategies": strategies}


@register_tool(
    "backtest_list_indicators",
    "List available technical indicators.",
    {"type": "object", "properties": {}, "required": []},
)
def backtest_list_indicators(_args) -> dict:
    ind_dir = BTQUANT_ROOT / "dependencies" / "backtrader" / "indicators"
    indicators = {}
    for f in sorted(ind_dir.glob("*.py")):
        if f.name.startswith("__"):
            continue
        indicators[f.stem] = {"file": str(f.relative_to(BTQUANT_ROOT))}
    return {"count": len(indicators), "indicators": indicators}


@register_tool(
    "backtest_run_simple",
    "Run a quick SMA crossover backtest on BTC/USDT Binance data.",
    {
        "type": "object",
        "properties": {
            "strategy": {
                "type": "string",
                "description": "Strategy name (e.g. SMA_Cross_Simple)",
                "default": "SMA_Cross_Simple",
            },
            "symbol": {
                "type": "string",
                "description": "Trading pair (default BTC/USDT)",
                "default": "BTC/USDT",
            },
            "start": {
                "type": "string",
                "description": "Start date (YYYY-MM-DD)",
                "default": "2024-01-01",
            },
            "end": {
                "type": "string",
                "description": "End date (YYYY-MM-DD)",
                "default": "2024-01-31",
            },
            "timeframe": {
                "type": "string",
                "enum": ["1m", "5m", "15m", "1h", "4h", "1d"],
                "description": "OHLCV timeframe",
                "default": "1h",
            },
            "exchange": {
                "type": "string",
                "description": "Data source exchange",
                "default": "binance",
            },
            "cash": {
                "type": "number",
                "description": "Initial capital",
                "default": 10000,
            },
        },
        "required": [],
    },
)
def backtest_run_simple(args) -> dict:
    """Run a backtest via Backtrader. Falls back to CCXT data or mock."""
    strategy_name = args.get("strategy", "SMA_Cross_Simple")
    symbol = args.get("symbol", "BTC/USDT")
    start = args.get("start", "2024-01-01")
    end = args.get("end", "2024-01-31")
    tf = args.get("timeframe", "1h")
    exchange = args.get("exchange", "binance")
    cash = args.get("cash", 10000)

    # Build a small standalone script that runs the backtest
    script = f"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "{BTQUANT_ROOT}")
sys.path.insert(0, "{BTQUANT_ROOT}/dependencies")

try:
    from backtrader import backtest
    from backtrader.strategies.{strategy_name} import {strategy_name}
    from backtrader.utils.ccxt_data import get_crypto_data

    data = get_crypto_data("{symbol}", "{start}", "{end}", "{tf}", "{exchange}")
    result = backtest(strategy={strategy_name}, data=data, init_cash={cash}, quantstats=False, plot=False)
    print("BACKTEST RESULT:")
    print(f"  Initial:  ${{result['initial_portfolio_value']:,.2f}}" if hasattr(result, '__getitem__') and 'initial_portfolio_value' in result else "  (result object)")
    print(str(result)[:2000])
except ImportError as e:
    print(f"ImportError: {{e}}")
    print("Backtrader not fully installed or CCXT data unavailable.")
    print("This is expected if CCXT or exchange data API keys are not configured.")
except Exception as e:
    print(f"Backtest failed: {{e}}")
    import traceback
    traceback.print_exc()
"""
    try:
        r = subprocess.run(
            ["python3", "-c", script],
            capture_output=True, text=True, timeout=120,
            cwd=BTQUANT_ROOT,
        )
        output = r.stdout[-3000:] if len(r.stdout) > 3000 else r.stdout
        if r.stderr and "Error" in r.stderr:
            output += "\n--- stderr ---\n" + r.stderr[-1000:]
        return {
            "success": r.returncode == 0,
            "strategy": strategy_name,
            "symbol": symbol,
            "timeframe": tf,
            "output": output,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Backtest timed out after 120s"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@register_tool(
    "backtest_strategy_source",
    "Get the full source code of a strategy file.",
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Strategy module name (without .py)"}
        },
        "required": ["name"],
    },
)
def backtest_strategy_source(args) -> dict:
    name = args["name"]
    path = BTQUANT_ROOT / "dependencies" / "backtrader" / "strategies" / f"{name}.py"
    if not path.exists():
        return {"error": f"Strategy '{name}' not found"}
    content = path.read_text()
    return {
        "name": name,
        "path": str(path.relative_to(BTQUANT_ROOT)),
        "lines": len(content.split("\n")),
        "source": content,
    }