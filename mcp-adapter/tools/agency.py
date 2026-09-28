"""MCP tools ⟷ Autonomous Agency management."""

import json
import os
import subprocess
from pathlib import Path

BTQUANT_ROOT = Path(__file__).resolve().parent.parent.parent
import sys  # noqa: E402

from registry import register_tool


AGENCY_SCRIPT = BTQUANT_ROOT / "run_agency.py"


@register_tool(
    "agency_status",
    "Check Autonomous Agency status: config, log, database health.",
    {"type": "object", "properties": {}, "required": []},
)
def agency_status(_args) -> dict:
    status = {
        "script_exists": AGENCY_SCRIPT.exists(),
        "script_size_kb": round(AGENCY_SCRIPT.stat().st_size / 1024, 1) if AGENCY_SCRIPT.exists() else 0,
    }

    # Config
    try:
        sys.path.insert(0, str(BTQUANT_ROOT))
        from autonomous_agency.config import AgencyConfig
        cfg = AgencyConfig()
        status["config"] = {
            "ai_model": cfg.ai_model_name,
            "live_exchange": cfg.live_exchange,
            "live_trading_enabled": cfg.enable_live_trading,
            "min_sharpe": cfg.min_sharpe_ratio,
            "max_drawdown": cfg.max_drawdown_limit,
            "cycle_interval_h": cfg.cycle_interval_hours,
            "strategies_dir": cfg.strategies_dir,
        }
    except Exception as e:
        status["config_error"] = str(e)

    # Process
    try:
        r = subprocess.run(["pgrep", "-f", "run_agency.py"], capture_output=True, text=True, timeout=5)
        pids = r.stdout.strip().split()
        status["running_pids"] = pids if pids[0] else []
    except Exception:
        status["running_pids"] = []

    # Log file
    log_path = BTQUANT_ROOT / "autonomous_agency.log"
    if log_path.exists():
        status["log_size_kb"] = round(log_path.stat().st_size / 1024, 1)
        status["log_tail"] = _tail(log_path, 20)

    # DB file
    db_path = BTQUANT_ROOT / "autonomous_agency.db"
    if db_path.exists():
        status["db_size_kb"] = round(db_path.stat().st_size / 1024, 1)

    return status


@register_tool(
    "agency_start",
    "Start the Autonomous Agency (non-blocking, background process).",
    {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["full", "backtest", "deploy"],
                "description": "Agency run mode (default: full)",
                "default": "backtest",
            },
        },
        "required": [],
    },
)
def agency_start(args) -> dict:
    mode = args.get("mode", "backtest")
    cmd = ["python3", str(AGENCY_SCRIPT), "--mode", mode]

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=BTQUANT_ROOT)
    return {
        "started": r.returncode == 0,
        "mode": mode,
        "pid": _find_agency_pid(),
        "output": (r.stdout[-2000:] if len(r.stdout) > 2000 else r.stdout),
        "error": r.stderr[-500:] if r.stderr else None,
    }


@register_tool(
    "agency_stop",
    "Stop running Autonomous Agency processes.",
    {"type": "object", "properties": {}, "required": []},
)
def agency_stop(_args) -> dict:
    try:
        r = subprocess.run(["pkill", "-f", "run_agency.py"], capture_output=True, text=True, timeout=10)
        return {"killed": r.returncode == 0, "output": r.stdout.strip() or "(no process found)"}
    except Exception as e:
        return {"error": str(e)}


@register_tool(
    "agency_logs",
    "Read agency logs (last N lines).",
    {
        "type": "object",
        "properties": {
            "lines": {"type": "integer", "description": "Number of lines (default: 50)", "default": 50},
        },
    },
)
def agency_logs(args) -> dict:
    log_path = BTQUANT_ROOT / "autonomous_agency.log"
    if not log_path.exists():
        return {"error": "agency log not found"}
    return {"log": _tail(log_path, args.get("lines", 50))}


@register_tool(
    "agency_strategies",
    "List generated strategy files.",
    {"type": "object", "properties": {}, "required": []},
)
def agency_strategies(_args) -> dict:
    strat_dir = BTQUANT_ROOT / "autonomous_agency" / "strategies"
    results_dir = BTQUANT_ROOT / "autonomous_agency" / "results"
    files = {}
    for d, label in [(strat_dir, "strategies"), (results_dir, "results")]:
        if d.exists():
            items = []
            for f in sorted(d.iterdir()):
                if f.is_file() and not f.name.startswith("."):
                    items.append({"name": f.name, "size_kb": round(f.stat().st_size / 1024, 1)})
            files[label] = items
    return files


def _tail(path: Path, n: int) -> str:
    lines = path.read_text().splitlines()
    return "\n".join(lines[-n:])


def _find_agency_pid() -> str:
    try:
        r = subprocess.run(["pgrep", "-f", "run_agency.py"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip()
    except Exception:
        return ""