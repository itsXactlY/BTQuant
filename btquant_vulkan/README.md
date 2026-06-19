# btquant_vulkan

Eine High-Frequency Trading Terminal-Anwendung mit Vulkan GPU-only Rendering Engine.

## Features

- **C++23 Vulkan GPU-only rendering engine** - Vollständig GPU-basiertes Rendering ohne CPU-Blitting
- **Real-time market data** - Sub-second latency data from `/dev/shm/btquant_hotspine`
- **Modern UI** - ImGui for UI windows, ImPlot for mathematical visualization
- **Multi-symbol support** - Handles data from Binance, Bybit, OKX exchanges
- **Advanced visualizations** - Order book, DOM, trades feed, TPO charts

## Architektur

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

## Build Requirements

- C++23 compliant compiler
- CMake 3.28+
- Vulkan SDK
- GLFW3 or SDL2
- Git for fetching dependencies

## Build Instructions

```bash
# Clone and build
git clone <repository-url>
cd btquant_vulkan
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Or use the build script if available
./build.sh
```

## Usage

The application connects to market data stored in `/dev/shm/btquant_hotspine`. Make sure this shared memory segment contains market data in the expected binary format:

- Header: "UQTB" magic (4 bytes) + version + symbol_count at offset 0x00
- Data starts at 0x1000, struct per symbol:
  - `bid_price`, `ask_price` (double)
  - `bid_size`, `ask_size` (double)
  - `timestamp` (uint64, microseconds)
  - `seq` (uint32)

Symbol definitions are expected in `/dev/shm/btquant_symbols.json`.

## Project Structure

```
src/
├── core/              # Vulkan context and utilities
├── data/              # Market data structures and ingestion
├── renderer/          # GPU rendering and compute pipelines
├── ui/                # ImGui integration and window management
└── widgets/           # Specialized trading widgets
```

## Key Conventions

- All rendering via Vulkan compute/graphics pipelines
- Data ingestion uses lock-free SPSC queues
- GPU buffers for all visualization data
- ImGui for windows, ImPlot for charts/graphs
- C++23 with modern features

## License

See LICENSE file for licensing information.