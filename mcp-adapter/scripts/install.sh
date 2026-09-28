#!/bin/bash
# Install BTQuant MCP Adapter dependencies
set -e
cd "$(dirname "$0")/.."
pip install -r requirements.txt
echo "Done. Run: python server.py --port 8910"