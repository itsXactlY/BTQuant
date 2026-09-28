"""MCP tools ⟷ Neural Trading Pipeline."""

import os
import subprocess
from pathlib import Path

BTQUANT_ROOT = Path(__file__).resolve().parent.parent.parent
PIPELINE_DIR = BTQUANT_ROOT / "neural-memory-neural-trading-pipeline"

from registry import register_tool


@register_tool(
    "neural_status",
    "Check Neural Pipeline status: files, config, trained models.",
    {"type": "object", "properties": {}, "required": []},
)
def neural_status(_args) -> dict:
    status = {
        "pipeline_exists": PIPELINE_DIR.exists(),
        "scripts": [],
        "config": None,
        "models": [],
    }

    if not PIPELINE_DIR.exists():
        return status

    # Scripts
    scripts_dir = PIPELINE_DIR / "scripts"
    if scripts_dir.exists():
        status["scripts"] = sorted([f.name for f in scripts_dir.glob("*.py")])

    # Config
    config_path = PIPELINE_DIR / "config" / "config.yaml"
    if config_path.exists():
        status["config"] = config_path.read_text()[:1000]

    # Model files
    for pattern in ["*.pt", "*.pth", "*.h5", "*.keras"]:
        for f in PIPELINE_DIR.rglob(pattern):
            status["models"].append({
                "name": str(f.relative_to(PIPELINE_DIR)),
                "size_kb": round(f.stat().st_size / 1024, 1),
            })

    return status


@register_tool(
    "neural_run_feature_selection",
    "Run feature selection script from the neural pipeline.",
    {
        "type": "object",
        "properties": {
            "timeout": {"type": "integer", "description": "Timeout seconds (default 300)"},
        },
    },
)
def neural_run_feature_selection(args) -> dict:
    script = PIPELINE_DIR / "scripts" / "run_feature_selection.py"
    if not script.exists():
        return {"error": "run_feature_selection.py not found"}
    timeout = args.get("timeout", 300)
    try:
        r = subprocess.run(
            ["python3", str(script)],
            capture_output=True, text=True, timeout=timeout, cwd=PIPELINE_DIR,
        )
        output = r.stdout[-3000:] if len(r.stdout) > 3000 else r.stdout
        return {"success": r.returncode == 0, "output": output, "error": r.stderr[-500:] if r.stderr else None}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Timed out after {timeout}s"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@register_tool(
    "neural_run_walk_forward",
    "Run walk-forward backtest from the neural pipeline.",
    {
        "type": "object",
        "properties": {
            "timeout": {"type": "integer", "description": "Timeout seconds (default 600)"},
        },
    },
)
def neural_run_walk_forward(args) -> dict:
    script = PIPELINE_DIR / "scripts" / "run_walk_forward.py"
    if not script.exists():
        return {"error": "run_walk_forward.py not found"}
    timeout = args.get("timeout", 600)
    try:
        r = subprocess.run(
            ["python3", str(script)],
            capture_output=True, text=True, timeout=timeout, cwd=PIPELINE_DIR,
        )
        output = r.stdout[-3000:] if len(r.stdout) > 3000 else r.stdout
        return {"success": r.returncode == 0, "output": output, "error": r.stderr[-500:] if r.stderr else None}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Timed out after {timeout}s"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@register_tool(
    "neural_pipeline_config",
    "Get neural pipeline configuration.",
    {"type": "object", "properties": {}, "required": []},
)
def neural_pipeline_config(_args) -> dict:
    import yaml
    config_path = PIPELINE_DIR / "config" / "config.yaml"
    if not config_path.exists():
        return {"error": "config.yaml not found"}
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        return cfg
    except Exception as e:
        return {"error": str(e), "raw": config_path.read_text()[:2000]}