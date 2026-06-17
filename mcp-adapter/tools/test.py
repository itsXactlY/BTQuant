"""MCP tools ⟷ test execution (pytest, hotspine tests)."""

import os
import subprocess
from pathlib import Path

BTQUANT_ROOT = Path(__file__).resolve().parent.parent.parent

from registry import register_tool


TEST_SUITES = {
    "pytest_all": {
        "description": "Full pytest suite",
        "cwd": str(BTQUANT_ROOT),
        "cmd": ["python", "-m", "pytest", "-x", "-v", "--timeout=120"],
    },
    "ms_sql_hotswap": {
        "description": "MsSQL database connection tests",
        "cwd": str(BTQUANT_ROOT),
        "cmd": ["python", "-m", "pytest", "-x", "-v", "test_ms_sql_hotswap.py"],
    },
    "neural_pipeline": {
        "description": "Neural trading pipeline tests",
        "cwd": str(BTQUANT_ROOT),
        "cmd": ["python", "-m", "pytest", "-x", "-v",
                "neural-memory-neural-trading-pipeline/tests/"],
    },
    "hotspine_basic": {
        "description": "HotSpine basic integration test",
        "cwd": str(BTQUANT_ROOT / "hotspine"),
        "cmd": ["python", "test_hotspine_basic.py"],
    },
    "hotspine_comprehensive": {
        "description": "HotSpine comprehensive test",
        "cwd": str(BTQUANT_ROOT / "hotspine"),
        "cmd": ["python", "test_hotspine_comprehensive.py"],
    },
    "hotspine_core": {
        "description": "HotSpine core reader test",
        "cwd": str(BTQUANT_ROOT / "hotspine"),
        "cmd": ["python", "test_hotspine_core.py"],
    },
    "hotspine_integration": {
        "description": "HotSpine full system integration",
        "cwd": str(BTQUANT_ROOT / "hotspine"),
        "cmd": ["python", "test_hotspine_integration.py"],
    },
    "hotspine_sql_arch": {
        "description": "HotSpine SQL architecture validation",
        "cwd": str(BTQUANT_ROOT / "hotspine"),
        "cmd": ["python", "test_hotspine_sql_architecture.py"],
    },
    "hotspine_system": {
        "description": "HotSpine system-level test",
        "cwd": str(BTQUANT_ROOT / "hotspine"),
        "cmd": ["python", "test_hotspine_system.py"],
    },
    "config": {
        "description": "Configuration validation test",
        "cwd": str(BTQUANT_ROOT),
        "cmd": ["python", "test_configuration.py"],
    },
}


@register_tool(
    "test_list",
    "List all available test suites.",
    {"type": "object", "properties": {}, "required": []},
)
def test_list(_args) -> dict:
    return {name: meta["description"] for name, meta in TEST_SUITES.items()}


@register_tool(
    "test_run",
    "Run a test suite and return results.",
    {
        "type": "object",
        "properties": {
            "suite": {
                "type": "string",
                "enum": list(TEST_SUITES.keys()),
                "description": "Which test suite to run",
            },
            "extra_args": {
                "type": "string",
                "description": "Additional pytest arguments (e.g. '-k pattern')",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 300)",
            },
        },
        "required": ["suite"],
    },
)
def test_run(args) -> dict:
    suite = args["suite"]
    meta = TEST_SUITES.get(suite)
    if not meta:
        return {"error": f"Unknown suite: {suite}"}

    cmd = list(meta["cmd"])
    extra = args.get("extra_args", "")
    if extra:
        cmd.extend(extra.split())

    timeout = args.get("timeout", 300)

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=meta["cwd"], timeout=timeout)
        log = r.stdout[-3000:] if len(r.stdout) > 3000 else r.stdout
        if r.stderr:
            log += "\n--- stderr ---\n" + r.stderr[-1000:]
        return {
            "success": r.returncode == 0,
            "suite": suite,
            "exit_code": r.returncode,
            "output": log,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "suite": suite, "error": f"Timed out after {timeout}s"}
    except Exception as e:
        return {"success": False, "suite": suite, "error": str(e)}


@register_tool(
    "test_lint",
    "Run flake8 linter on BTQuant Python files.",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File or directory to lint (default: .)"},
        },
    },
)
def test_lint(args) -> dict:
    path = args.get("path", ".")
    cmd = ["flake8", path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=BTQUANT_ROOT, timeout=60)
        lines = r.stdout.split("\n")
        # Filter out intentional E501 in AI prompt files if desired
        return {
            "success": r.returncode == 0,
            "total_issues": len([l for l in lines if l.strip()]),
            "output": r.stdout[-3000:] if len(r.stdout) > 3000 else r.stdout,
        }
    except Exception as e:
        return {"error": str(e)}