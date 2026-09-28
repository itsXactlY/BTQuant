#!/bin/bash
# Start the BTQuant MCP Adapter server
# Usage: ./scripts/run.sh [--port PORT] [--stdio]

set -e
cd "$(dirname "$0")/.."

if [ "$1" = "--stdio" ]; then
    exec python3 server.py --stdio
elif [ "$1" = "--list-tools" ]; then
    exec python3 server.py --list-tools
else
    PORT="${2:-8910}"
    echo "Starting BTQuant MCP on 0.0.0.0:${PORT} ..."
    exec python3 server.py --port "${PORT}"
fi