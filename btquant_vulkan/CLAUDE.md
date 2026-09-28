# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**btquant_vulkan** - High-Frequency Trading Terminal
- C++26 Vulkan GPU-only rendering engine
- ImGui for UI windows, ImPlot for mathematical visualization
- Real-time sub-second latency data from `/dev/shm/btquant_hotspine`

## Data Source

**Binary Format** (`/dev/shm/btquant_hotspine`):
- Header: "UQTB" magic (4 bytes) + version + symbol_count at offset 0x00
- Data starts at 0x1000, struct per symbol:
  - `bid_price`, `ask_price` (double)
  - `bid_size`, `ask_size` (double)
  - `timestamp` (uint64, microseconds)
  - `seq` (uint32)
- Symbols: `/dev/shm/btquant_symbols.json` (Binance, Bybit, OKX)

## Architecture

### Core Layers
1. **Vulkan Renderer** - GPU-only rendering, no CPU blits
2. **Data Ingestion** - Memory-mapped file I/O, lock-free ring buffers
3. **Compute Shaders** - GPU-side aggregation for heatmaps, VPVR
4. **UI Layer** - ImGui + ImPlot for all windows

### Key Systems
- `VulkanBackend` - Device selection, swapchain, render passes
- `DataSpine` - Lock-free SPSC queues from mmap'd hotspine
- `RenderGraph` - Frame graph for heatmap/chart/DOM composition
- `WindowManager` - Multi-window docking system

## Build Commands

```bash
# Build with CMake
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Or use the build script
./build.sh
```

## Key Conventions

- All rendering via Vulkan compute/graphics pipelines
- Data ingestion uses lock-free SPSC queues
- GPU buffers for all visualization data
- ImGui for windows, ImPlot for charts/graphs
- C++26 with `<std>`, `std::expected`, `std::print`
