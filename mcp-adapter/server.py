"""
BTQuant MCP Adapter — Main Server
===================================
Exposes BTQuant's full framework as MCP tools for AI agents.

Protocol: MCP over HTTP+SSE (standard Model Context Protocol)
Transport: uvicorn on 0.0.0.0:8910

Usage:
    python server.py                  # HTTP+SSE mode
    python server.py --stdio          # stdio mode (for Hermes plugin)
    python server.py --list-tools     # Print tool manifest and exit
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# ── ensure BTQuant backtrader fork is importable ──────────────────────────
BTQUANT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BTQUANT_ROOT))
sys.path.insert(0, str(BTQUANT_ROOT / "dependencies"))

# ── tool registry ─────────────────────────────────────────────────────────
from registry import TOOLS, register_tool


# ── import tool modules (they self-register via @register_tool) ───────────
def _load_tools():
    mods = [
        "tools.project",
        "tools.build",
        "tools.test",
        "tools.backtest",
        "tools.agency",
        "tools.neural",
        "tools.data",
        "tools.exchange",
    ]
    for mod_name in mods:
        # Try both absolute (mcp_adapter.tools.*) and relative (tools.*) import paths
        loaded = False
        for prefix in ("mcp_adapter.", ""):
            try:
                __import__(f"{prefix}{mod_name}", fromlist=["_"])
                loaded = True
                break
            except ImportError:
                continue
        if not loaded:
            logging.warning("tool module %s could not be loaded", mod_name)


# ── MCP protocol handlers ─────────────────────────────────────────────────
def handle_mcp_request(body: dict) -> dict:
    """Route a single JSON-RPC 2.0 request to the right handler."""
    req_id = body.get("id", 0)
    method = body.get("method", "")
    params = body.get("params", {})

    if method == "initialize":
        return _jsonrpc(req_id, {
            "protocolVersion": "2025-03-26",
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False},
                "prompts": {},
            },
            "serverInfo": {"name": "btquant-mcp", "version": "1.0.0"},
        })

    if method == "ping":
        return _jsonrpc(req_id, {})

    if method == "tools/list":
        tool_list = []
        for name, meta in TOOLS.items():
            tool_list.append({
                "name": name,
                "description": meta["description"],
                "inputSchema": meta["input_schema"],
            })
        return _jsonrpc(req_id, {"tools": tool_list})

    if method == "tools/call":
        name = params.get("name", "")
        args = params.get("arguments", {})
        meta = TOOLS.get(name)
        if not meta:
            return _jsonrpc_error(req_id, -32601, f"Unknown tool: {name}")
        try:
            result = meta["handler"](args)
            return _jsonrpc(req_id, {"content": [{"type": "text", "text": json.dumps(result, default=str)}]})
        except Exception as exc:
            logging.exception("tool %s failed", name)
            return _jsonrpc_error(req_id, -32000, f"Tool {name} failed: {exc}")

    return _jsonrpc_error(req_id, -32601, f"Unknown method: {method}")


def _jsonrpc(id_val, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": id_val, "result": result}


def _jsonrpc_error(id_val, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id_val, "error": {"code": code, "message": message}}


# ── HTTP transport (uvicorn / SSE) ────────────────────────────────────────
def build_app():
    """Return a Starlette ASGI app for HTTP+SSE MCP transport."""
    try:
        from sse_starlette.sse import EventSourceResponse
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse
        from starlette.routing import Route
    except ImportError as exc:
        print("Missing deps: pip install starlette sse-starlette uvicorn", file=sys.stderr)
        raise SystemExit(1) from exc

    async def mcp_http(request: Request):
        body = await request.json()
        result = handle_mcp_request(body)
        return JSONResponse(result)

    async def mcp_sse(request: Request):
        """SSE endpoint: client sends POST to /mcp, gets streamed responses."""
        async def event_gen():
            yield {"event": "endpoint", "data": json.dumps({"url": "/mcp"})}
            # keep-alive every 30s
            import asyncio
            while True:
                await asyncio.sleep(30)
                yield {"event": "keepalive", "data": ""}

        return EventSourceResponse(event_gen())

    async def list_tools_http(request: Request):
        """GET /tools — list all registered tools (debug helper)."""
        return JSONResponse({
            name: {"description": meta["description"], "inputSchema": meta["input_schema"]}
            for name, meta in TOOLS.items()
        })

    async def health_http(request: Request):
        return JSONResponse({
            "status": "ok",
            "tools": len(TOOLS),
            "btquant_root": str(BTQUANT_ROOT),
        })

    app = Starlette(
        debug=False,
        routes=[
            Route("/mcp", mcp_http, methods=["POST"]),
            Route("/sse", mcp_sse, methods=["GET"]),
            Route("/tools", list_tools_http, methods=["GET"]),
            Route("/health", health_http, methods=["GET"]),
        ],
    )
    return app


# ── stdio transport (for Hermes plugin: transport: stdio) ─────────────────
def run_stdio():
    """Read JSON-RPC 2.0 lines from stdin, write responses to stdout."""
    import sys
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            body = json.loads(line)
        except json.JSONDecodeError:
            continue
        result = handle_mcp_request(body)
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()


# ── immediate: load tools at import time (before uvicorn takes over) ─────
_load_tools()

# ── entry point ───────────────────────────────────────────────────────────
def main():
    _load_tools()

    parser = argparse.ArgumentParser(description="BTQuant MCP Adapter")
    parser.add_argument("--stdio", action="store_true", help="Run in stdio mode (for Hermes plugin)")
    parser.add_argument("--port", type=int, default=8910, help="HTTP port (default: 8910)")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind address")
    parser.add_argument("--list-tools", action="store_true", help="Print tool manifest and exit")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(name)s %(message)s")

    if args.list_tools:
        manifest = {}
        for name, meta in TOOLS.items():
            manifest[name] = {"description": meta["description"], "inputSchema": meta["input_schema"]}
        print(json.dumps(manifest, indent=2))
        return

    if args.stdio:
        logging.info("Starting BTQuant MCP in stdio mode (%d tools)", len(TOOLS))
        run_stdio()
    else:
        logging.info("Starting BTQuant MCP on http://%s:%d (%d tools)", args.host, args.port, len(TOOLS))
        import uvicorn
        app = build_app()
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()