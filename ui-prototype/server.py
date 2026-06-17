#!/usr/bin/env python3
"""BTQuant Terminal Backend — FastAPI WebSocket server.

Connects the terminal UI to the real BTQuant stack.
Commands execute against actual BTQuant components.

Usage:
    cd /home/alca/projects/PubBTQuant/ui-prototype
    python server.py [--port 8888]
    
Then open http://localhost:8888
"""

import asyncio
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from collections import deque

# ── BTQuant paths ──
BTQUANT_ROOT = Path("/home/alca/projects/PubBTQuant")
BACKTRADER_DIR = BTQUANT_ROOT / "dependencies" / "backtrader"
STRATEGIES_DIR = BACKTRADER_DIR / "strategies"
INDICATORS_DIR = BACKTRADER_DIR / "indicators"
FEEDS_DIR = BACKTRADER_DIR / "feeds"
BROKERS_DIR = BACKTRADER_DIR / "brokers"
STORES_DIR = BACKTRADER_DIR / "stores"
ANALYZERS_DIR = BACKTRADER_DIR / "analyzers"
HOTSPINE_DIR = BACKTRADER_DIR / "hotspine"
AGENCY_DIR = BTQUANT_ROOT / "autonomous_agency"
NEURAL_DIR = BTQUANT_ROOT / "neural-memory-neural-trading-pipeline"
MCP_DIR = BTQUANT_ROOT / "mcp-adapter"
CCAPI_DIR = BTQUANT_ROOT / "dependencies" / "ccapi"
RENDER_DIR = BTQUANT_ROOT / "dependencies" / "BTQ_Render_Engine"
TESTS_DIR = BTQUANT_ROOT / "tests"
MSSQL_DIR = BACKTRADER_DIR / "bigbraincentral"

sys.path.insert(0, str(BACKTRADER_DIR.parent))
sys.path.insert(0, str(BTQUANT_ROOT))

# ── Try imports ──
HAS_BT = False
HAS_PSUTIL = False

try:
    import backtrader
    HAS_BT = True
except ImportError:
    pass

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    pass

# ── FastAPI ──
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="BTQuant Terminal")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── State ──
start_time = time.time()
command_log: deque = deque(maxlen=200)
active_ws: Optional[WebSocket] = None

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def shell(cmd: str, timeout: int = 10) -> Dict:
    """Run a shell command and return result."""
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return {"ok": True, "stdout": r.stdout.strip(), "stderr": r.stderr.strip(), "code": r.returncode}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def get_btquant_processes() -> List[Dict]:
    """Get running BTQuant-related processes."""
    procs = []
    if HAS_PSUTIL:
        for p in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
            try:
                info = p.info
                cmdline = ' '.join(info.get('cmdline') or [])
                name = info.get('name', '')
                btq_keywords = ['btq', 'backtrader', 'hotspine', 'detector', 'mcp',
                               'ccapi', 'agency', 'run_agency', 'render', 'vulkan',
                               'uvicorn', 'fastapi', 'server.py']
                if any(kw in (cmdline + name).lower() for kw in btq_keywords):
                    mem = info.get('memory_info')
                    procs.append({
                        "pid": info['pid'],
                        "name": name,
                        "cmdline": cmdline[:120],
                        "cpu": info.get('cpu_percent', 0),
                        "mem_mb": round(mem.rss / 1024 / 1024, 1) if mem else 0,
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    else:
        r = shell("ps aux | grep -E 'btq|backtrader|hotspine|detector|mcp|ccapi|agency|render|vulkan|uvicorn|server.py' | grep -v grep")
        if r['ok'] and r['stdout']:
            for line in r['stdout'].split('\n'):
                parts = line.split(None, 10)
                if len(parts) >= 11:
                    procs.append({
                        "pid": int(parts[1]),
                        "name": parts[10].split()[0] if parts[10] else parts[0],
                        "cmdline": parts[10][:120],
                        "cpu": float(parts[2]),
                        "mem_mb": round(float(parts[5]) / 1024, 1),
                    })
    return procs

def list_py_files(directory: Path, exclude: list = None) -> List[str]:
    """List .py files in a directory."""
    exclude = exclude or ['__init__.py', '__pycache__', 'base.py', 'doncommit.py', 'version.py', 'metabase.py']
    if not directory.exists():
        return []
    return sorted([
        f.stem for f in directory.glob("*.py")
        if f.stem not in exclude and not f.stem.startswith('_')
    ])

def scan_strategy_classes(filepath: Path) -> List[Dict]:
    """Scan a strategy file for class definitions."""
    try:
        content = filepath.read_text()
        import re
        classes = re.findall(r'class\s+(\w+)\s*\(', content)
        params = re.findall(r'params\s*=\s*\((.*?)\)', content, re.DOTALL)
        return [
            {"name": c, "params": p.strip()[:80] if p else ""}
            for c, p in zip(classes, params)
        ]
    except Exception:
        return []

def check_port(port: int) -> bool:
    """Check if a port is in use."""
    r = shell(f"ss -tlnp | grep :{port} | head -1")
    return bool(r.get('ok') and r.get('stdout'))

def check_shm() -> Dict:
    """Check HotSpine shared memory."""
    r = shell("ls -la /dev/shm/BTQ* 2>/dev/null | head -5")
    exists = r['ok'] and 'BTQ' in r.get('stdout', '')
    r2 = shell("ls -la /dev/shm/ | grep BTQ | wc -l")
    count = 0
    if r2['ok']:
        try:
            count = int(r2['stdout'].strip())
        except ValueError:
            pass
    return {"exists": exists, "files": count, "details": r.get('stdout', '')}

# ══════════════════════════════════════════════════════════════
# COMMAND HANDLERS
# ══════════════════════════════════════════════════════════════

def L(text: str, cls: str = "") -> Dict:
    """Helper to create a line dict."""
    return {"text": text, "cls": cls}

async def cmd_status(args: List[str]) -> Dict:
    """Real stack status — check all subsystems."""
    lines = []
    inspector = {"title": "Stack Status", "sections": []}

    # HotSpine
    hs = check_shm()
    hs_cls = "green" if hs['exists'] else "red"
    hs_text = f"● HOTSPINE   {'OK — ' + str(hs['files']) + ' SHM files' if hs['exists'] else 'NOT FOUND — no /dev/shm/BTQ*'}"
    lines.append(L(f"  {hs_text}", hs_cls))

    # Processes
    procs = get_btquant_processes()
    proc_cls = "green" if procs else "amber"
    lines.append(L(f"  ● PROCESSES {len(procs)} running", proc_cls))

    # MCP
    mcp_up = check_port(8910)
    mcp_cls = "green" if mcp_up else "red"
    lines.append(L(f"  ● MCP        {':8910 LISTENING' if mcp_up else 'NOT LISTENING on :8910'}", mcp_cls))

    # MSSQL
    mssql_r = shell("ss -tlnp | grep 1433 | head -1")
    mssql_up = mssql_r['ok'] and '1433' in mssql_r.get('stdout', '')
    mssql_cls = "green" if mssql_up else "amber"
    lines.append(L(f"  ● DB         {'MSSQL :1433 LISTENING' if mssql_up else 'MSSQL :1433 not detected'}", mssql_cls))

    # BTQuant Python
    lines.append(L(f"  ● BACKTRADER {'imported (bt module)' if HAS_BT else 'NOT importable (check sys.path)'}", "green" if HAS_BT else "red"))

    # PSUTIL
    lines.append(L(f"  ● PSUTIL     {'OK' if HAS_PSUTIL else 'NOT installed — limited process monitoring'}", "green" if HAS_PSUTIL else "amber"))

    # Agency
    agency_py = AGENCY_DIR / "orchestrator.py"
    lines.append(L(f"  ● AGENCY     {'orchestrator.py EXISTS' if agency_py.exists() else 'NOT FOUND'}", "green" if agency_py.exists() else "red"))

    # Neural
    neural_cfg = NEURAL_DIR / "config" / "config.yaml"
    lines.append(L(f"  ● NEURAL     {'config.yaml EXISTS' if neural_cfg.exists() else 'NOT FOUND'}", "green" if neural_cfg.exists() else "red"))

    lines.append(L(""))
    lines.append(L(f"  Backend: {'backtrader' if HAS_BT else 'NOT CONNECTED'} │ Root: {BTQUANT_ROOT}", "dim"))

    inspector["sections"].append({
        "title": "Dependencies",
        "kv": [
            ["backtrader", "OK" if HAS_BT else "MISSING"],
            ["psutil", "OK" if HAS_PSUTIL else "MISSING"],
            ["HotSpine SHM", "OK" if hs['exists'] else "MISSING"],
            ["MCP :8910", "UP" if mcp_up else "DOWN"],
            ["MSSQL :1433", "UP" if mssql_up else "DOWN"],
        ]
    })
    inspector["sections"].append({
        "title": "Processes",
        "kv": [[f"PID {p['pid']}", f"{p['name']} — {p['cmdline'][:60]}"] for p in procs[:8]]
    })

    return {"type": "output", "lines": lines, "inspector": inspector}

async def cmd_strategies(args: List[str]) -> Dict:
    """List real strategy files from the strategies directory."""
    lines = []
    files = list_py_files(STRATEGIES_DIR, ['__init__.py', 'base.py'])
    
    lines.append(L(f"  STRATEGY FILES — {len(files)} found in strategies/", "hi"))
    lines.append(L(""))
    
    for f in files:
        fp = STRATEGIES_DIR / f"{f}.py"
        size = fp.stat().st_size if fp.exists() else 0
        lines.append(L(f"  {f:<40} {size:>6} bytes", "bright"))
    
    lines.append(L(""))
    lines.append(L(f"  Base class: strategies/base.py (BaseStrategy → bt.Strategy)", "dim"))
    
    if HAS_BT:
        try:
            sys.path.insert(0, str(STRATEGIES_DIR.parent))
            import importlib
            # Just list the files, don't import (avoids side effects)
            lines.append(L(f"  bt module available for execution", "green"))
        except Exception as e:
            lines.append(L(f"  bt import warning: {e}", "amber"))
    
    inspector = {"title": "Strategies", "sections": [
        {"title": "Directory", "kv": [
            ["Path", str(STRATEGIES_DIR)],
            ["Files", str(len(files))],
            ["Base", "BaseStrategy (1401 lines)"],
        ]},
        {"title": "File List", "kv": [[f, f"exists"] for f in files[:15]]}
    ]}
    
    return {"type": "output", "lines": lines, "inspector": inspector}

async def cmd_indicators(args: List[str]) -> Dict:
    """List real indicator files."""
    files = list_py_files(INDICATORS_DIR)
    lines = []
    lines.append(L(f"  INDICATOR FILES — {len(files)} found in indicators/", "hi"))
    lines.append(L(""))
    
    # Group by first letter
    current_letter = ""
    for f in files:
        letter = f[0].upper()
        if letter != current_letter:
            current_letter = letter
            lines.append(L(f"  ── {letter} ──", "cyan"))
        lines.append(L(f"    {f}", "bright"))
    
    inspector = {"title": "Indicators", "sections": [
        {"title": "Directory", "kv": [["Path", str(INDICATORS_DIR)], ["Count", str(len(files))]]}
    ]}
    return {"type": "output", "lines": lines, "inspector": inspector}

async def cmd_feeds(args: List[str]) -> Dict:
    """List real feed files."""
    files = list_py_files(FEEDS_DIR)
    lines = []
    lines.append(L(f"  FEED FILES — {len(files)} found in feeds/", "hi"))
    lines.append(L(""))
    for f in files:
        lines.append(L(f"  {f}", "bright"))
    inspector = {"title": "Feeds", "sections": [
        {"title": "Directory", "kv": [["Path", str(FEEDS_DIR)], ["Count", str(len(files))]]}
    ]}
    return {"type": "output", "lines": lines, "inspector": inspector}

async def cmd_brokers(args: List[str]) -> Dict:
    files = list_py_files(BROKERS_DIR)
    lines = []
    lines.append(L(f"  BROKER FILES — {len(files)} found in brokers/", "hi"))
    lines.append(L(""))
    for f in files:
        lines.append(L(f"  {f}", "bright"))
    inspector = {"title": "Brokers", "sections": [
        {"title": "Directory", "kv": [["Path", str(BROKERS_DIR)], ["Count", str(len(files))]]}
    ]}
    return {"type": "output", "lines": lines, "inspector": inspector}

async def cmd_stores(args: List[str]) -> Dict:
    files = list_py_files(STORES_DIR)
    lines = []
    lines.append(L(f"  STORE FILES — {len(files)} found in stores/", "hi"))
    lines.append(L(""))
    for f in files:
        lines.append(L(f"  {f}", "bright"))
    inspector = {"title": "Stores", "sections": [
        {"title": "Directory", "kv": [["Path", str(STORES_DIR)], ["Count", str(len(files))]]}
    ]}
    return {"type": "output", "lines": lines, "inspector": inspector}

async def cmd_hotspine(args: List[str]) -> Dict:
    """Check real HotSpine status."""
    lines = []
    hs = check_shm()
    
    lines.append(L("  HOTSPINE — Shared Memory Subsystem", "hi"))
    lines.append(L(""))
    lines.append(L(f"  SHM path:     /dev/shm/BTQ*", "bright"))
    lines.append(L(f"  SHM exists:   {'YES' if hs['exists'] else 'NO'}", "green" if hs['exists'] else "red"))
    lines.append(L(f"  SHM files:    {hs['files']}", "bright"))
    
    if hs['details']:
        lines.append(L(""))
        lines.append(L("  SHM contents:", "dim"))
        for line in hs['details'].split('\n')[:5]:
            lines.append(L(f"    {line}", "dim"))
    
    # Check library
    lib = HOTSPINE_DIR / "libhotspine_reader.so"
    lines.append(L(f"  Library:      {'libhotspine_reader.so EXISTS' if lib.exists() else 'libhotspine_reader.so NOT FOUND'}", 
                   "green" if lib.exists() else "red"))
    
    # Check reader.py
    reader = BACKTRADER_DIR / "hotspine" / "reader.py"
    lines.append(L(f"  Reader:       {'hotspine/reader.py EXISTS' if reader.exists() else 'NOT FOUND'}",
                   "green" if reader.exists() else "red"))
    
    # Check config
    config = BACKTRADER_DIR / "hotspine" / "config.py"
    lines.append(L(f"  Config:       {'hotspine/config.py EXISTS' if config.exists() else 'NOT FOUND'}",
                   "green" if config.exists() else "red"))
    
    inspector = {"title": "HotSpine", "sections": [
        {"title": "SHM", "kv": [
            ["Path", "/dev/shm/BTQ*"],
            ["Exists", str(hs['exists'])],
            ["Files", str(hs['files'])],
        ]},
        {"title": "Components", "kv": [
            ["Library", "libhotspine_reader.so" + (" ✓" if lib.exists() else " ✗")],
            ["Reader", "hotspine/reader.py" + (" ✓" if reader.exists() else " ✗")],
            ["Config", "hotspine/config.py" + (" ✓" if config.exists() else " ✗")],
        ]}
    ]}
    return {"type": "output", "lines": lines, "inspector": inspector}

async def cmd_ccapi(args: List[str]) -> Dict:
    lines = []
    lines.append(L("  CCAPI — Exchange Connectors", "hi"))
    lines.append(L(""))
    
    src = CCAPI_DIR / "example" / "src"
    if src.exists():
        modules = sorted([d.name for d in src.iterdir() if d.is_dir()])
        lines.append(L(f"  Source: {src}", "dim"))
        lines.append(L(f"  Modules: {len(modules)}", "bright"))
        lines.append(L(""))
        for m in modules:
            lines.append(L(f"  {m}", "bright"))
    else:
        lines.append(L("  CCAPI source NOT FOUND", "red"))
    
    lines.append(L(""))
    lines.append(L("  Connectors: Binance OKX Bybit Coinbase Kraken (SPOT ONLY)", "amber"))
    
    inspector = {"title": "CCAPI", "sections": [
        {"title": "Source", "kv": [["Path", str(src)], ["Exists", str(src.exists())]]}
    ]}
    return {"type": "output", "lines": lines, "inspector": inspector}

async def cmd_render(args: List[str]) -> Dict:
    lines = []
    lines.append(L("  BTQ RENDER ENGINE — Vulkan", "hi"))
    lines.append(L(""))
    
    src = RENDER_DIR / "src"
    if src.exists():
        files = sorted([f.name for f in src.glob("*.cpp")])
        lines.append(L(f"  Source: {src}", "dim"))
        lines.append(L(f"  C++ files: {len(files)}", "bright"))
        lines.append(L(""))
        for f in files:
            lines.append(L(f"  {f}", "bright"))
    else:
        lines.append(L("  Render engine source NOT FOUND", "red"))
    
    inspector = {"title": "Render Engine", "sections": [
        {"title": "Source", "kv": [["Path", str(src)], ["Exists", str(src.exists())]]}
    ]}
    return {"type": "output", "lines": lines, "inspector": inspector}

async def cmd_db(args: List[str]) -> Dict:
    lines = []
    lines.append(L("  BIGBRAINCENTRAL — MSSQL", "hi"))
    lines.append(L(""))
    
    # Check if MSSQL is listening
    mssql_r = shell("ss -tlnp | grep 1433 | head -1")
    mssql_up = mssql_r['ok'] and '1433' in mssql_r.get('stdout', '')
    lines.append(L(f"  Port 1433:    {'LISTENING' if mssql_up else 'NOT LISTENING'}", "green" if mssql_up else "red"))
    
    # Check storage_mssql.py
    storage = MSSQL_DIR / "storage_mssql.py"
    lines.append(L(f"  storage_mssql: {'EXISTS' if storage.exists() else 'NOT FOUND'}", 
                   "green" if storage.exists() else "red"))
    
    # Check fast_mssql
    fast = BTQUANT_ROOT / "dependencies" / "MsSQL" / "fast_mssql.cpp"
    lines.append(L(f"  fast_mssql.cpp: {'EXISTS' if fast.exists() else 'NOT FOUND'}",
                   "green" if fast.exists() else "red"))
    
    # Check config
    lines.append(L(f"  Database:     BTQ_MarketData", "bright"))
    lines.append(L(f"  Host:         127.0.0.1:1433", "bright"))
    lines.append(L(f"  Python:       3.14 fast_mssql shim", "dim"))
    lines.append(L(""))
    lines.append(L("  WARNING: Python 3.14 native fast_mssql limitation", "amber"))
    
    inspector = {"title": "BigBrainCentral", "sections": [
        {"title": "Connection", "kv": [
            ["Host", "127.0.0.1:1433"],
            ["Database", "BTQ_MarketData"],
            ["Status", "LISTENING" if mssql_up else "DOWN"],
        ]},
        {"title": "Files", "kv": [
            ["storage_mssql.py", str(storage.exists())],
            ["fast_mssql.cpp", str(fast.exists())],
        ]}
    ]}
    return {"type": "output", "lines": lines, "inspector": inspector}

async def cmd_agency(args: List[str]) -> Dict:
    lines = []
    lines.append(L("  AUTONOMOUS AGENCY", "hi"))
    lines.append(L(""))
    
    # Check files
    orchestrator = AGENCY_DIR / "orchestrator.py"
    hypothesis = AGENCY_DIR / "hypothesis_generator.py"
    strategy_factory = AGENCY_DIR / "strategy_factory.py"
    backtester = AGENCY_DIR / "backtester.py"
    evaluator = AGENCY_DIR / "evaluator.py"
    evolution = AGENCY_DIR / "evolution_engine.py"
    deployer = AGENCY_DIR / "live_deployment.py"
    config = AGENCY_DIR / "config.py"
    
    components = [
        ("Orchestrator", orchestrator),
        ("Hypothesis Gen", hypothesis),
        ("Strategy Factory", strategy_factory),
        ("Backtester", backtester),
        ("Evaluator", evaluator),
        ("Evolution Engine", evolution),
        ("Live Deployer", deployer),
        ("Config", config),
    ]
    
    for name, path in components:
        status = "●" if path.exists() else "○"
        cls = "green" if path.exists() else "red"
        lines.append(L(f"  {status} {name:<20} {path.name}", cls))
    
    # Check run_agency.py
    run_agency = BTQUANT_ROOT / "run_agency.py"
    lines.append(L(""))
    lines.append(L(f"  Entry point:  {'run_agency.py EXISTS' if run_agency.exists() else 'NOT FOUND'}",
                   "green" if run_agency.exists() else "red"))
    
    # Check generated strategies
    gen_strats = AGENCY_DIR / "strategies"
    if gen_strats.exists():
        count = len(list(gen_strats.glob("*.py")))
        lines.append(L(f"  Generated:    {count} strategies in agency/strategies/", "bright"))
    
    inspector = {"title": "Agency", "sections": [
        {"title": "Components", "kv": [[n, "✓" if p.exists() else "✗"] for n, p in components]},
        {"title": "Entry", "kv": [
            ["run_agency.py", str(run_agency.exists())],
            ["strategies dir", str(gen_strats.exists()) if gen_strats else "N/A"],
        ]}
    ]}
    return {"type": "output", "lines": lines, "inspector": inspector}

async def cmd_neural(args: List[str]) -> Dict:
    lines = []
    lines.append(L("  NEURAL TRADING PIPELINE — V3", "hi"))
    lines.append(L(""))
    
    subs = ["data", "models", "training", "backtesting", "rl", "scripts", "config", "tests"]
    for sub in subs:
        p = NEURAL_DIR / sub
        exists = p.exists()
        count = len(list(p.glob("*.py"))) if exists else 0
        status = "●" if exists else "○"
        cls = "green" if exists else "dim"
        lines.append(L(f"  {status} {sub:<14} {count} files", cls))
    
    # Config
    cfg = NEURAL_DIR / "config" / "config.yaml"
    lines.append(L(""))
    lines.append(L(f"  config.yaml:  {'EXISTS' if cfg.exists() else 'NOT FOUND'}",
                   "green" if cfg.exists() else "red"))
    
    inspector = {"title": "Neural Pipeline", "sections": [
        {"title": "Subpackages", "kv": [[s, str((NEURAL_DIR / s).exists())] for s in subs]}
    ]}
    return {"type": "output", "lines": lines, "inspector": inspector}

async def cmd_mcp(args: List[str]) -> Dict:
    lines = []
    lines.append(L("  MCP ADAPTER", "hi"))
    lines.append(L(""))
    
    server = MCP_DIR / "server.py"
    lines.append(L(f"  server.py:    {'EXISTS' if server.exists() else 'NOT FOUND'}",
                   "green" if server.exists() else "red"))
    
    # List tool modules
    tools_dir = MCP_DIR / "tools"
    if tools_dir.exists():
        tool_files = sorted(tools_dir.glob("*.py"))
        lines.append(L(f"  Tool modules: {len(tool_files)}", "bright"))
        lines.append(L(""))
        for tf in tool_files:
            if tf.name == '__init__.py':
                continue
            lines.append(L(f"    {tf.stem}", "bright"))
    
    # Check port
    mcp_up = check_port(8910)
    lines.append(L(""))
    lines.append(L(f"  Port 8910:    {'LISTENING' if mcp_up else 'NOT LISTENING'}", 
                   "green" if mcp_up else "red"))
    
    inspector = {"title": "MCP Adapter", "sections": [
        {"title": "Server", "kv": [
            ["File", str(server)],
            ["Port", "8910"],
            ["Status", "UP" if mcp_up else "DOWN"],
        ]}
    ]}
    return {"type": "output", "lines": lines, "inspector": inspector}

async def cmd_processes(args: List[str]) -> Dict:
    procs = get_btquant_processes()
    lines = []
    lines.append(L(f"  BTQUANT PROCESSES — {len(procs)} found", "hi"))
    lines.append(L(""))
    
    if procs:
        lines.append(L(f"  {'PID':>7}  {'CPU%':>5}  {'MEM':>7}  COMMAND", "dim"))
        lines.append(L(f"  {'─'*7}  {'─'*5}  {'─'*7}  {'─'*40}", "dim"))
        for p in procs:
            lines.append(L(f"  {p['pid']:>7}  {p['cpu']:>5.1f}  {p['mem_mb']:>6.1f}M  {p['cmdline'][:60]}", "bright"))
    else:
        lines.append(L("  No BTQuant processes detected", "amber"))
    
    inspector = {"title": "Processes", "sections": [
        {"title": "Running", "kv": [[f"PID {p['pid']}", p['name']] for p in procs[:10]]}
    ]}
    return {"type": "output", "lines": lines, "inspector": inspector}

async def cmd_backtest(args: List[str]) -> Dict:
    """Show backtest info and available strategies."""
    lines = []
    lines.append(L("  BACKTEST RUNNER", "hi"))
    lines.append(L(""))
    
    if HAS_BT:
        lines.append(L("  backtrader:   imported ✓", "green"))
    else:
        lines.append(L("  backtrader:   NOT importable", "red"))
    
    # List available strategies
    strats = list_py_files(STRATEGIES_DIR, ['__init__.py', 'base.py'])
    lines.append(L(f"  Strategies:   {len(strats)} available", "bright"))
    lines.append(L(""))
    
    for s in strats[:20]:
        lines.append(L(f"    {s}", "bright"))
    
    lines.append(L(""))
    lines.append(L("  Usage: python -c \"import bt; c=bt.Cerebro(); ...\"", "dim"))
    
    inspector = {"title": "Backtest", "sections": [
        {"title": "Engine", "kv": [
            ["backtrader", "OK" if HAS_BT else "MISSING"],
            ["Strategies", str(len(strats))],
        ]},
        {"title": "Usage", "kv": [
            ["CLI", "btq backtest / btq optimize"],
            ["Python", "import bt; c = bt.Cerebro()"],
        ]}
    ]}
    return {"type": "output", "lines": lines, "inspector": inspector}

async def cmd_detectors(args: List[str]) -> Dict:
    lines = []
    lines.append(L("  C++ MANIPULATION DETECTORS", "hi"))
    lines.append(L(""))
    
    det_dir = TESTS_DIR / "new" / "src" / "detectors"
    if det_dir.exists():
        dets = sorted([f.stem for f in det_dir.glob("*.cpp")])
        lines.append(L(f"  Source: {det_dir}", "dim"))
        lines.append(L(f"  Detectors: {len(dets)}", "bright"))
        lines.append(L(""))
        for d in dets:
            lines.append(L(f"  ● {d}", "green"))
    else:
        lines.append(L("  Detector source NOT FOUND", "red"))
    
    inspector = {"title": "Detectors", "sections": [
        {"title": "Directory", "kv": [["Path", str(det_dir)], ["Exists", str(det_dir.exists())]]}
    ]}
    return {"type": "output", "lines": lines, "inspector": inspector}

async def cmd_limits(args: List[str]) -> Dict:
    lines = []
    lines.append(L("  KNOWN LIMITATIONS", "hi"))
    lines.append(L(""))
    limits = [
        ("CRITICAL", "No perpetual/futures support — spot only"),
        ("CRITICAL", "No Hyperliquid connector"),
        ("CRITICAL", "CCAPI spot-only — all 5 exchange connectors are spot"),
        ("HIGH", "Detectors spot-only"),
        ("HIGH", "Live Deployer sandbox by default"),
        ("MEDIUM", "Python 3.14 native fast_mssql limitation"),
        ("MEDIUM", "HotSpine access may require root/group permissions"),
        ("LOW", "mypy missing — no static type checking"),
        ("LOW", "Expected flake8 E501 warnings"),
    ]
    for sev, text in limits:
        color = "red" if sev == "CRITICAL" else "amber" if sev == "HIGH" else "dim"
        lines.append(L(f"  [{sev:<8}] {text}", color))
    return {"type": "output", "lines": lines, "inspector": {"title": "Limitations", "sections": []}}

async def cmd_config(args: List[str]) -> Dict:
    lines = []
    lines.append(L("  BTQUANT CONFIGURATION", "hi"))
    lines.append(L(""))
    lines.append(L(f"  Root:           {BTQUANT_ROOT}", "bright"))
    lines.append(L(f"  Backtrader:     {BACKTRADER_DIR}", "bright"))
    lines.append(L(f"  Agency:         {AGENCY_DIR}", "bright"))
    lines.append(L(f"  Neural:         {NEURAL_DIR}", "bright"))
    lines.append(L(f"  MCP:            {MCP_DIR}", "bright"))
    lines.append(L(f"  backtrader OK:  {HAS_BT}", "green" if HAS_BT else "red"))
    lines.append(L(f"  psutil OK:      {HAS_PSUTIL}", "green" if HAS_PSUTIL else "amber"))
    lines.append(L(f"  Python:         {sys.version.split()[0]}", "bright"))
    lines.append(L(f"  PID:            {os.getpid()}", "dim"))
    return {"type": "output", "lines": lines, "inspector": {"title": "Config", "sections": []}}

async def cmd_whoami(args: List[str]) -> Dict:
    lines = []
    lines.append(L(f"  User:     {os.getenv('USER', 'unknown')}", "bright"))
    lines.append(L(f"  Home:     {Path.home()}", "bright"))
    lines.append(L(f"  Project:  {BTQUANT_ROOT}", "hi"))
    lines.append(L(f"  Server:   BTQuant Terminal Backend", "cyan"))
    lines.append(L(f"  PID:      {os.getpid()}", "dim"))
    return {"type": "output", "lines": lines, "inspector": {"title": "Operator", "sections": []}}

async def cmd_uptime(args: List[str]) -> Dict:
    ms = time.time() - start_time
    s = int(ms)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    lines = []
    lines.append(L(f"  Server uptime: {h}h {m}m {s}s", "hi"))
    lines.append(L(f"  Started: {datetime.fromtimestamp(start_time).strftime('%H:%M:%S')}", "dim"))
    return {"type": "output", "lines": lines, "inspector": {"title": "Uptime", "sections": []}}

async def cmd_help(args: List[str]) -> Dict:
    lines = []
    lines.append(L("  BTQUANT TERMINAL — COMMANDS", "hi"))
    lines.append(L(""))
    cmds = [
        ("status", "Real stack status — check all subsystems"),
        ("strategies", "List strategy files from strategies/"),
        ("indicators", "List indicator files from indicators/"),
        ("feeds", "List feed files from feeds/"),
        ("brokers", "List broker files from brokers/"),
        ("stores", "List store files from stores/"),
        ("detectors", "Check C++ detector source"),
        ("hotspine", "Check HotSpine SHM status"),
        ("ccapi", "Check CCAPI exchange connectors"),
        ("render", "Check BTQ Render Engine"),
        ("db", "Check BigBrainCentral MSSQL"),
        ("backtest", "Show backtest runner info"),
        ("agency", "Check autonomous agency status"),
        ("neural", "Check neural pipeline status"),
        ("mcp", "Check MCP adapter status"),
        ("processes", "List running BTQuant processes"),
        ("limits", "Show known limitations"),
        ("config", "Show BTQuant configuration"),
        ("whoami", "Show operator info"),
        ("uptime", "Show server uptime"),
        ("help", "This help"),
        ("clear", "Clear terminal (client-side)"),
    ]
    for name, desc in cmds:
        lines.append(L(f"  {name:<14} {desc}", "bright"))
    lines.append(L(""))
    lines.append(L("  Keyboard: F1=help  Ctrl+K=palette  ↑↓=history  Tab=complete", "dim"))
    return {"type": "output", "lines": lines, "inspector": {"title": "Help", "sections": []}}

async def cmd_pulse(args: List[str]) -> Dict:
    lines = []
    lines.append(L("  PULSE — MACRO INTEGRATION", "hi"))
    lines.append(L(""))
    lines.append(L("  PULSE maps macro trends to detector and feed behavior.", "dim"))
    lines.append(L("  Alerts via Discord webhook on cron schedule.", "dim"))
    return {"type": "output", "lines": lines, "inspector": {"title": "PULSE", "sections": []}}

async def cmd_test(args: List[str]) -> Dict:
    lines = []
    lines.append(L("  BTQUANT TEST SUITES", "hi"))
    lines.append(L(""))
    test_dirs = [
        ("tests/new", "C++ detector tests"),
        ("tests", "General tests"),
        ("hotspine", "HotSpine tests"),
    ]
    for d, desc in test_dirs:
        p = BTQUANT_ROOT / d
        if p.exists():
            count = len(list(p.glob("test_*.py")))
            lines.append(L(f"  ● {d:<20} {count} test files — {desc}", "green"))
        else:
            lines.append(L(f"  ○ {d:<20} NOT FOUND — {desc}", "dim"))
    lines.append(L(""))
    lines.append(L("  Run: python -m pytest tests/ -v", "dim"))
    return {"type": "output", "lines": lines, "inspector": {"title": "Tests", "sections": []}}

async def cmd_logs(args: List[str]) -> Dict:
    lines = []
    lines.append(L("  BTQUANT LOG FILES", "hi"))
    lines.append(L(""))
    
    log_locations = [
        ("autonomous_agency.log", AGENCY_DIR / ".." / "autonomous_agency.log"),
        ("agency/logs/", AGENCY_DIR / "logs"),
    ]
    for name, path in log_locations:
        if path.exists():
            if path.is_file():
                size = path.stat().st_size
                lines.append(L(f"  ● {name:<30} {size} bytes", "green"))
            else:
                count = len(list(path.glob("*.log")))
                lines.append(L(f"  ● {name:<30} {count} log files", "green"))
        else:
            lines.append(L(f"  ○ {name:<30} NOT FOUND", "dim"))
    return {"type": "output", "lines": lines, "inspector": {"title": "Logs", "sections": []}}

async def cmd_signals(args: List[str]) -> Dict:
    lines = []
    lines.append(L("  SIGNAL MATRIX", "hi"))
    lines.append(L(""))
    lines.append(L("  No active signals — connect to live feeds to generate signals.", "dim"))
    lines.append(L("  Run a backtest: import bt; c = bt.Cerebro(); ...", "dim"))
    return {"type": "output", "lines": lines, "inspector": {"title": "Signals", "sections": []}}

# ══════════════════════════════════════════════════════════════
# COMMAND ROUTER
# ══════════════════════════════════════════════════════════════

HANDLERS = {
    "status": cmd_status,
    "strategies": cmd_strategies, "strat": cmd_strategies,
    "indicators": cmd_indicators, "ind": cmd_indicators,
    "feeds": cmd_feeds,
    "brokers": cmd_brokers,
    "stores": cmd_stores,
    "hotspine": cmd_hotspine, "hs": cmd_hotspine,
    "ccapi": cmd_ccapi,
    "render": cmd_render,
    "db": cmd_db, "mssql": cmd_db,
    "backtest": cmd_backtest, "bt": cmd_backtest,
    "agency": cmd_agency,
    "neural": cmd_neural,
    "mcp": cmd_mcp,
    "processes": cmd_processes, "ps": cmd_processes,
    "detectors": cmd_detectors, "det": cmd_detectors,
    "limits": cmd_limits,
    "config": cmd_config,
    "whoami": cmd_whoami,
    "uptime": cmd_uptime,
    "help": cmd_help,
    "pulse": cmd_pulse,
    "test": cmd_test,
    "logs": cmd_logs,
    "signals": cmd_signals,
}

async def execute_command(cmd: str) -> Dict:
    """Route command to handler."""
    parts = cmd.strip().split()
    if not parts:
        return {"type": "output", "lines": [L("")]}

    name = parts[0].lower()
    args = parts[1:]

    handler = HANDLERS.get(name)
    if handler:
        try:
            result = await handler(args)
            command_log.append({"cmd": cmd, "time": time.time(), "ok": True})
            return result
        except Exception as e:
            command_log.append({"cmd": cmd, "time": time.time(), "ok": False, "error": str(e)})
            return {
                "type": "error",
                "lines": [
                    L(f"  Error in '{name}': {e}", "red"),
                    L(f"  {traceback.format_exc().split(chr(10))[-2]}", "dim"),
                ],
            }
    else:
        return {
            "type": "output",
            "lines": [
                L(f"  Unknown command: {name}", "red"),
                L('  Type "help" for available commands', "dim"),
            ],
        }

# ══════════════════════════════════════════════════════════════
# WEBSOCKET
# ══════════════════════════════════════════════════════════════

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global active_ws
    await ws.accept()
    active_ws = ws
    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                cmd = msg.get("command", "")
                result = await execute_command(cmd)
                result["id"] = msg.get("id", "")
                await ws.send_json(result)
            except json.JSONDecodeError:
                # Treat raw text as command
                result = await execute_command(data)
                await ws.send_json(result)
    except WebSocketDisconnect:
        active_ws = None
    except Exception:
        active_ws = None

# ══════════════════════════════════════════════════════════════
# HTTP ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/")
async def index():
    return FileResponse(BTQUANT_ROOT / "ui-prototype" / "quantower.html")

@app.get("/api/status")
async def api_status():
    result = await execute_command("status")
    return JSONResponse(result)

@app.get("/api/health")
async def api_health():
    return {"status": "ok", "uptime_s": int(time.time() - start_time), "backend": HAS_BT}

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BTQuant Terminal Backend")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"BTQuant Terminal Backend starting on http://{args.host}:{args.port}")
    print(f"  Root:     {BTQUANT_ROOT}")
    print(f"  backtrader: {'OK' if HAS_BT else 'NOT FOUND'}")
    print(f"  psutil:     {'OK' if HAS_PSUTIL else 'NOT FOUND'}")
    print()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
