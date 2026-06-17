"""MCP tools ⟷ data collection, HotSpine, MsSQL, and mock data."""

import os
import subprocess
from pathlib import Path

BTQUANT_ROOT = Path(__file__).resolve().parent.parent.parent

from registry import register_tool


@register_tool(
    "data_hotspine_status",
    "Check HotSpine shared memory status.",
    {"type": "object", "properties": {}, "required": []},
)
def data_hotspine_status(_args) -> dict:
    result = {}
    try:
        r = subprocess.run(
            ["ls", "-la", "/dev/shm/"],
            capture_output=True, text=True, timeout=5,
        )
        shm_files = [l for l in r.stdout.split("\n") if "BTQ" in l]
        result["shared_memory_files"] = shm_files
        result["btq_found"] = len(shm_files) > 0
    except Exception as e:
        result["error"] = str(e)

    # Check library
    lib_path = BTQUANT_ROOT / "hotspine" / "libhotspine_reader.so"
    result["hotspine_lib_exists"] = lib_path.exists()
    if lib_path.exists():
        result["hotspine_lib_size_kb"] = round(lib_path.stat().st_size / 1024, 1)

    return result


@register_tool(
    "data_hotspine_run_test",
    "Run a HotSpine basic connectivity test.",
    {
        "type": "object",
        "properties": {
            "test": {
                "type": "string",
                "enum": ["basic", "comprehensive", "core", "integration"],
                "description": "Which test to run",
                "default": "basic",
            },
        },
    },
)
def data_hotspine_run_test(args) -> dict:
    test_map = {
        "basic": "test_hotspine_basic.py",
        "comprehensive": "test_hotspine_comprehensive.py",
        "core": "test_hotspine_core.py",
        "integration": "test_hotspine_integration.py",
    }
    test_file = test_map.get(args.get("test", "basic"))
    test_path = BTQUANT_ROOT / "hotspine" / test_file
    if not test_path.exists():
        return {"error": f"{test_file} not found in hotspine/"}
    try:
        r = subprocess.run(
            ["python3", str(test_path)],
            capture_output=True, text=True, timeout=60,
            cwd=BTQUANT_ROOT / "hotspine",
        )
        output = r.stdout[-3000:] if len(r.stdout) > 3000 else r.stdout
        return {
            "success": r.returncode == 0,
            "test": test_file,
            "output": output,
            "error": r.stderr[-500:] if r.stderr else None,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@register_tool(
    "data_mssql_status",
    "Check MsSQL BigBrainCentral database connectivity.",
    {"type": "object", "properties": {}, "required": []},
)
def data_mssql_status(_args) -> dict:
    # Check config file
    config_paths = [
        BTQUANT_ROOT / "dependencies" / "datacollector" / "config.json",
        BTQUANT_ROOT / "dependencies" / "ccapi" / "example" / "build" / "src" / "market_data_collector" / "config.json",
        BTQUANT_ROOT / "dependencies" / "ccapi" / "example" / "src" / "market_data_collector" / "config.json",
    ]
    result = {"configs_found": []}
    for p in config_paths:
        if p.exists():
            import json
            cfg = json.loads(p.read_text())
            result["configs_found"].append({
                "path": str(p.relative_to(BTQUANT_ROOT)),
                "server": cfg.get("db", {}).get("server", "?"),
                "database": cfg.get("db", {}).get("database", "?"),
            })

    # Try connection via pyodbc
    try:
        import pyodbc
        conn_str = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=127.0.0.1;DATABASE=BTQ_MarketData;UID=SA;PWD=q?}33YIToo:H%xue$Kr*"
        conn = pyodbc.connect(conn_str, timeout=5)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sys.tables")
        table_count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        result["connection"] = "ok"
        result["table_count"] = table_count
    except Exception as e:
        result["connection"] = "failed"
        result["connection_error"] = str(e)

    # Try sqlcmd fallback
    try:
        r = subprocess.run(
            ["sqlcmd", "-S", "127.0.0.1", "-U", "SA", "-P", "q?}33YIToo:H%xue$Kr*",
             "-Q", "SELECT COUNT(*) FROM sys.tables"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            result["sqlcmd_found"] = True
            result["sqlcmd_output"] = r.stdout[-500:]
        else:
            result["sqlcmd_found"] = False
    except FileNotFoundError:
        result["sqlcmd_installed"] = False
    except Exception as e:
        result["sqlcmd_error"] = str(e)

    return result


@register_tool(
    "data_mock_producer",
    "Run the mock data producer (generates test market data).",
    {
        "type": "object",
        "properties": {
            "timeout": {"type": "integer", "description": "Runtime seconds (default 10)"},
        },
    },
)
def data_mock_producer(args) -> dict:
    script = BTQUANT_ROOT / "mock_data_producer.py"
    if not script.exists():
        return {"error": "mock_data_producer.py not found"}
    timeout = args.get("timeout", 10)
    try:
        r = subprocess.run(
            ["python3", str(script)],
            capture_output=True, text=True, timeout=timeout, cwd=BTQUANT_ROOT,
        )
        output = r.stdout[-2000:] if len(r.stdout) > 2000 else r.stdout
        return {
            "success": r.returncode == 0,
            "output": output,
            "error": r.stderr[-500:] if r.stderr else None,
        }
    except subprocess.TimeoutExpired:
        return {"success": True, "output": f"(ran for {timeout}s — mock producer keeps running)"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@register_tool(
    "data_ccapi_process",
    "Check if CCAPI market data collector process is running.",
    {"type": "object", "properties": {}, "required": []},
)
def data_ccapi_process(_args) -> dict:
    try:
        r = subprocess.run(
            ["pgrep", "-f", "market_data_collector"],
            capture_output=True, text=True, timeout=5,
        )
        pids = r.stdout.strip().split() if r.stdout.strip() else []
        return {"running": len(pids) > 0, "pids": pids}
    except Exception as e:
        return {"error": str(e)}


@register_tool(
    "data_detector_process",
    "Check if manipulation detector process is running.",
    {"type": "object", "properties": {}, "required": []},
)
def data_detector_process(_args) -> dict:
    try:
        r = subprocess.run(
            ["pgrep", "-f", "manipulation_monitor"],
            capture_output=True, text=True, timeout=5,
        )
        pids = r.stdout.strip().split() if r.stdout.strip() else []
        return {"running": len(pids) > 0, "pids": pids}
    except Exception as e:
        return {"error": str(e)}


@register_tool(
    "data_dashboard_process",
    "Check if QuantStats dashboard is running.",
    {"type": "object", "properties": {}, "required": []},
)
def data_dashboard_process(_args) -> dict:
    try:
        r = subprocess.run(
            ["pgrep", "-f", "quantstats_dashboard"],
            capture_output=True, text=True, timeout=5,
        )
        pids = r.stdout.strip().split() if r.stdout.strip() else []
        return {"running": len(pids) > 0, "pids": pids}
    except Exception as e:
        return {"error": str(e)}