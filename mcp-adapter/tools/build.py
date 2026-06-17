"""MCP tools ⟷ C++ build (manipulation detectors, CCAPI, render engine)."""

import os
import subprocess
from pathlib import Path

BTQUANT_ROOT = Path(__file__).resolve().parent.parent.parent

from registry import register_tool


BUILD_TARGETS = {
    "detectors": {
        "name": "BTQuant Manipulation Detectors",
        "cmake_dir": BTQUANT_ROOT / "tests" / "new",
        "build_dir": BTQUANT_ROOT / "tests" / "new" / "build",
        "targets": ["manipulation_monitor", "simple_monitor", "multi_exchange_monitor"],
    },
    "ccapi": {
        "name": "CCAPI Market Data Collector",
        "cmake_dir": BTQUANT_ROOT / "dependencies" / "ccapi" / "example",
        "build_dir": BTQUANT_ROOT / "dependencies" / "ccapi" / "example" / "build",
        "targets": ["market_data_collector"],
    },
    "render_engine": {
        "name": "BTQ Render Engine (Vulkan)",
        "cmake_dir": BTQUANT_ROOT / "dependencies" / "BTQ_Render_Engine",
        "build_dir": BTQUANT_ROOT / "dependencies" / "BTQ_Render_Engine" / "build",
        "targets": ["footprint_panel", "watchlist_panel", "microstructure_renderer", "correlation_heatmap"],
    },
}


@register_tool(
    "build_list_targets",
    "List all available C++ build targets across BTQuant's 3 CMake projects.",
    {"type": "object", "properties": {}, "required": []},
)
def build_list_targets(_args) -> dict:
    result = {}
    for key, meta in BUILD_TARGETS.items():
        build_dir = meta["build_dir"]
        result[key] = {
            "name": meta["name"],
            "cmake_dir": str(meta["cmake_dir"]),
            "build_dir": str(build_dir),
            "exists": build_dir.exists(),
            "targets": meta["targets"],
        }
    return result


@register_tool(
    "build_run",
    "Configure and build a C++ CMake project. Set target='all' to build everything.",
    {
        "type": "object",
        "properties": {
            "project": {
                "type": "string",
                "enum": list(BUILD_TARGETS.keys()),
                "description": "Which CMake project to build",
            },
            "target": {
                "type": "string",
                "description": "Specific make target (default: all targets in project)",
            },
            "build_type": {
                "type": "string",
                "enum": ["Release", "Debug"],
                "description": "CMake build type (default: Release)",
            },
            "clean": {
                "type": "boolean",
                "description": "Remove build dir before building",
            },
            "jobs": {
                "type": "integer",
                "description": "Parallel jobs (default: nproc)",
            },
        },
        "required": ["project"],
    },
)
def build_run(args) -> dict:
    project = args["project"]
    meta = BUILD_TARGETS.get(project)
    if not meta:
        return {"error": f"Unknown project: {project}. Use build_list_targets to see available projects."}

    build_type = args.get("build_type", "Release")
    clean = args.get("clean", False)
    jobs = args.get("jobs", os.cpu_count() or 4)
    specific_target = args.get("target", "")

    cmake_dir = meta["cmake_dir"]
    build_dir = meta["build_dir"]

    if clean and build_dir.exists():
        import shutil
        shutil.rmtree(build_dir)

    build_dir.mkdir(parents=True, exist_ok=True)

    cmds = [
        f"cmake {cmake_dir} -DCMAKE_BUILD_TYPE={build_type}",
        f"make -j{jobs} {' '.join([specific_target] if specific_target else meta['targets'])}",
    ]

    log = []
    for cmd in cmds:
        log.append(f"$ {cmd}")
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=build_dir, timeout=300)
        log.append(r.stdout[-2000:] if len(r.stdout) > 2000 else r.stdout)
        if r.returncode != 0:
            log.append(f"FAILED (exit={r.returncode}):")
            log.append(r.stderr[-1000:])
            return {"success": False, "project": project, "log": "\n".join(log[-20:]), "exit_code": r.returncode}

    return {"success": True, "project": project, "log": "\n".join(log[-20:]), "exit_code": 0}


@register_tool(
    "build_clean_all",
    "Remove all C++ build directories (clean slate).",
    {"type": "object", "properties": {}, "required": []},
)
def build_clean_all(_args) -> dict:
    import shutil
    cleaned = []
    for key, meta in BUILD_TARGETS.items():
        bd = meta["build_dir"]
        if bd.exists():
            shutil.rmtree(bd)
            cleaned.append(key)
    return {"cleaned": cleaned}


@register_tool(
    "build_check_compiler",
    "Check C++ compiler version and availability.",
    {"type": "object", "properties": {}, "required": []},
)
def build_check_compiler(_args) -> dict:
    result = {}
    for binary in ["g++", "gcc", "clang++", "cmake", "make"]:
        try:
            r = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=5)
            version = r.stdout.split("\n")[0] if r.stdout else r.stderr.split("\n")[0]
            result[binary] = version[:120]
        except FileNotFoundError:
            result[binary] = "NOT FOUND"
        except Exception as e:
            result[binary] = str(e)
    return result