# BTQuant Harmony Operating Map

This directory contains a static, self-contained prototype for the full-stack BTQuant UI.

Open:

- `index.html`

It implements the v3 full-stack operating map concept:

- Stack Harmony Vector across all BTQuant domains
- BTQuant Operating Map graph
- Signal Matrix
- Data lane: CCAPI, HotSpine, detectors, Vulkan render
- Strategy lane: Backtrader, strategy catalogue, indicators, feeds, stores
- Execution lane: brokers, orders, live gate, BigBrain DB
- Intelligence lane: autonomous agency, evolution, neural pipeline, QuantStats
- Control lane: MCP adapter, PULSE macro integration, command audit
- Known Limitations Board

## How to view

From the BTQuant repo root:

```bash
python -m http.server 8123 --directory ui-prototype
```

Then open:

```text
http://127.0.0.1:8123/
```

## Important

This is a static prototype with mock data. It does not send live trading commands, read secrets, or modify BTQuant state.

Production integration should connect to:

- BTQuant MCP adapter on port 8910
- `/mcp` and `/sse`
- HotSpine health adapter
- CCAPI collector status
- detector alerts
- Backtrader strategy state
- BigBrainCentral DB health
- agency/neural pipeline status
- PULSE macro signal stream
