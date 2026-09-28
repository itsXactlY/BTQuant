# BTQuant MCP Adapter

MCP (Model Context Protocol) server that exposes BTQuant's full framework as tools for AI agents.

## Structure

```
mcp-adapter/
├── server.py          # Main server: HTTP+SSE and stdio transports
├── requirements.txt   # Python dependencies
├── __init__.py
├── tools/
│   ├── __init__.py
│   ├── project.py     # Project info, git status, file tree
│   ├── build.py       # C++ CMake build (detectors, ccapi, render engine)
│   ├── test.py        # Test suites (pytest, hotspine, linting)
│   ├── backtest.py    # Backtrader strategies, indicators, backtest execution
│   ├── agency.py      # Autonomous Agency lifecycle (status, start, stop, logs)
│   ├── neural.py      # Neural Trading Pipeline (feature selection, walk-forward)
│   ├── data.py        # HotSpine, MsSQL, CCAPI, detector processes, mock data
│   └── exchange.py    # Exchange connector status, missing perp DEX list
└── scripts/           # (optional helper scripts)
```

## Usage

### HTTP+SSE mode (for remote agents, Cline, etc.)
```bash
cd /home/alca/projects/PubBTQuant/mcp-adapter
pip install -r requirements.txt
python server.py --port 8910
# → http://0.0.0.0:8910/mcp (JSON-RPC POST)
# → http://0.0.0.0:8910/sse (SSE endpoint)
# → http://0.0.0.0:8910/health
# → http://0.0.0.0:8910/tools  (tool manifest)
```

### Stdio mode (for Hermes plugin)
```bash
python server.py --stdio
```

### List tools
```bash
python server.py --list-tools
```

## Hermes Config

Add to `~/.hermes/config.yaml` under `mcp.servers`:

```yaml
  btquant:
    description: "BTQuant HFT framework — build, test, backtest, agency, data"
    enabled: true
    transport: http
    url: http://127.0.0.1:8910/mcp
    # OR stdio mode:
    # transport: stdio
    # command: /usr/bin/python3
    # args: [/home/alca/projects/PubBTQuant/mcp-adapter/server.py, --stdio]
```

## Tools Summary

| Tool | Description |
|------|-------------|
| `project_info` | BTQuant project overview |
| `project_git_status` | Git branch, dirty files, log |
| `project_cmakelists` | CMakeLists.txt locations |
| `project_file_tree` | Directory tree at depth |
| `build_list_targets` | Available C++ build targets |
| `build_run` | Configure + make a CMake project |
| `build_clean_all` | Remove all build dirs |
| `build_check_compiler` | g++/clang/cmake versions |
| `test_list` | Available test suites |
| `test_run` | Run a test suite |
| `test_lint` | Flake8 linting |
| `backtest_list_strategies` | Strategy catalogue |
| `backtest_list_indicators` | Indicator catalogue |
| `backtest_run_simple` | Quick SMA crossover backtest |
| `backtest_strategy_source` | Full strategy source code |
| `agency_status` | Agency config, process, log |
| `agency_start` | Start agency (background) |
| `agency_stop` | Stop agency |
| `agency_logs` | Read agency logs |
| `agency_strategies` | Generated strategy files |
| `neural_status` | Pipeline status |
| `neural_run_feature_selection` | Feature selection |
| `neural_run_walk_forward` | Walk-forward backtest |
| `neural_pipeline_config` | Pipeline config |
| `data_hotspine_status` | Shared memory status |
| `data_hotspine_run_test` | HotSpine connectivity test |
| `data_mssql_status` | MsSQL database status |
| `data_mock_producer` | Mock data generator |
| `data_ccapi_process` | CCAPI collector process |
| `data_detector_process` | Manipulation detector process |
| `data_dashboard_process` | QuantStats dashboard |
| `exchange_status` | Exchange connector config |
| `exchange_not_connected` | Missing perp DEX list |

## Adding New Tools

1. Create `tools/your_module.py`
2. Import `from server import register_tool`
3. Decorate a function with `@register_tool(name, description, input_schema)`
4. Restart server — auto-discovered via `_load_tools()`
