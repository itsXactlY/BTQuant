"""MCP tools ⟷ project info & structure."""

import json
import os
from pathlib import Path

BTQUANT_ROOT = Path(__file__).resolve().parent.parent.parent

from registry import register_tool


@register_tool(
    "project_info",
    "BTQuant project overview: root, branch, sizes, Python/C++ counts, last commit.",
    {"type": "object", "properties": {}, "required": []},
)
def project_info(_args) -> dict:
    info = {
        "root": str(BTQUANT_ROOT),
        "branch": _git("rev-parse --abbrev-ref HEAD") or "unknown",
        "last_commit": _git("log --oneline -1") or "unknown",
        "size_gb": _du_sk(),
    }
    info["file_counts"] = _count_files()
    info["python_version"] = os.popen("python3 --version 2>/dev/null").read().strip() or "unknown"
    info["directories"] = sorted([
        d.name for d in BTQUANT_ROOT.iterdir() if d.is_dir() and not d.name.startswith(".")
    ])
    return info


@register_tool(
    "project_git_status",
    "Current git status: branch, dirty files, commit log.",
    {
        "type": "object",
        "properties": {"count": {"type": "integer", "description": "Commits to show (default 5)"}},
    },
)
def project_git_status(args) -> dict:
    count = args.get("count", 5)
    return {
        "branch": _git("rev-parse --abbrev-ref HEAD"),
        "status": _git("status --short") or "(clean)",
        "log": _git(f"log --oneline -{count}") or "(no commits)",
    }


@register_tool(
    "project_cmakelists",
    "List all CMakeLists.txt files (C++ build targets).",
    {"type": "object", "properties": {}, "required": []},
)
def project_cmakelists(_args) -> dict:
    cmakes = list(BTQUANT_ROOT.rglob("CMakeLists.txt"))
    return {
        "count": len(cmakes),
        "files": sorted([str(p.relative_to(BTQUANT_ROOT)) for p in cmakes]),
    }


@register_tool(
    "project_file_tree",
    "List files and dirs at a given depth inside BTQuant.",
    {
        "type": "object",
        "properties": {
            "subdir": {"type": "string", "description": "Relative subdirectory (default: '.')"},
            "depth": {"type": "integer", "description": "Max depth (default: 2)"},
        },
        "required": [],
    },
)
def project_file_tree(args) -> dict:
    subdir = args.get("subdir", ".")
    depth = args.get("depth", 2)
    target = BTQUANT_ROOT / subdir
    if not target.exists():
        return {"error": f"Path not found: {subdir}"}
    tree = _build_tree(target, depth, indent=0)
    return {"root": str(target.relative_to(BTQUANT_ROOT)), "tree": tree}


# ── helpers ───────────────────────────────────────────────────────────────
def _git(cmd: str) -> str:
    try:
        r = os.popen(f"cd {BTQUANT_ROOT} && git {cmd} 2>/dev/null").read()
        return r.strip()
    except Exception:
        return ""


def _du_sk() -> float:
    try:
        import subprocess
        r = subprocess.run(["du", "-sk", BTQUANT_ROOT], capture_output=True, text=True)
        kb = int(r.stdout.split()[0])
        return round(kb / 1_048_576, 2)  # GB
    except Exception:
        return 0.0


def _count_files() -> dict:
    counts = {}
    for ext, label in [(".py", "python"), (".cpp", "c++"), (".hpp", "c++_header"),
                        (".h", "c_header"), (".json", "json"), (".md", "markdown"),
                        (".yaml", "yaml"), (".yml", "yaml"), (".sql", "sql")]:
        counts[label] = len(list(BTQUANT_ROOT.rglob(f"*{ext}")))
    return counts


def _build_tree(path: Path, max_depth: int, indent: int) -> list:
    if max_depth <= 0 or not path.is_dir():
        return []
    items = []
    for child in sorted(path.iterdir()):
        name = child.name
        if name.startswith("."):
            continue
        entry = {"name": name, "type": "dir" if child.is_dir() else "file"}
        if child.is_dir() and indent < max_depth - 1:
            entry["children"] = _build_tree(child, max_depth - 1, indent + 1)
        items.append(entry)
    return items