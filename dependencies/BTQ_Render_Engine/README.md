# BTQ_Render_Engine Documentation

## Overview

The BTQ_Render_Engine is a high-performance, GPU-accelerated real-time financial data visualization engine built with Vulkan. It provides professional-grade tools for traders and quantitative analysts to monitor, analyze, and visualize market data with sub-millisecond latency.

## Key Features

- **Real-time Data Processing**: Handles 10,000+ market data updates per second with <100µs latency
- **GPU-Accelerated Visualization**: Vulkan-based rendering with compute shaders for complex calculations
- **Comprehensive Analytics**: Technical indicators, pattern recognition, and risk management
- **Professional UI**: Modern ImGui-based interface with customizable layouts
- **HotSpine Integration**: Zero-copy shared memory architecture for ultra-fast data transfer
- **High Performance**: Optimized for low latency and high throughput

## Getting Started Guide

### System Requirements

- **OS**: Linux (tested on Ubuntu 20.04 and later)
- **GPU**: Vulkan-capable GPU with at least 4GB VRAM
- **CPU**: Multi-core processor (8+ cores recommended)
- **Memory**: 16GB RAM minimum, 32GB recommended
- **Storage**: 10GB free disk space

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/BTQ_Render_Engine.git
cd BTQ_Render_Engine

# Create build directory
mkdir build
cd build

# Configure CMake
cmake ..

# Build (using all available cores)
make -j$(nproc)
```

### First Run

```bash
# Run the advanced dashboard
./dashboard_advanced

# Run integration tests
./test_integration

# Run market data processor test
./test_market_data
```

### Configuration

The engine uses a YAML configuration file (`dashboard_config.yaml`) to customize behavior. See the [Configuration Guide](#configuration-guide) for detailed options.

## Architecture Overview

### System Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                     BTQ_Render_Engine                            │
├───────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │   HotSpine Data  │  │ Market Data      │  │  Symbol Manager  │  │
│  │   Bridge         │  │ Processor        │  │                  │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
│              │                  │                  │              │
│              └──────────────────┼──────────────────┘              │
│                                 ▼                                 │
│                      ┌──────────────────┐                         │
│                      │ Data Visualization│                         │
│                      │ Engine           │                         │
│                      └──────────────────┘                         │
│                                 ▼                                 │
│                      ┌──────────────────┐                         │
│                      │ Vulkan Rendering │                         │
│                      │ Pipeline         │                         │
│                      └──────────────────┘                         │
│                                 ▼                                 │
│                      ┌──────────────────┐                         │
│                      │ UI Components    │                         │
│                      │ - Charts         │                         │
│                      │ - Order Book     │                         │
│                      │ - Heatmap        │                         │
│                      │ - Data Grid      │                         │
│                      │ - Log Display    │                         │
│                      └──────────────────┘                         │
│                                 ▼                                 │
│                      ┌──────────────────┐                         │
│                      │ Performance      │                         │
│                      │ Monitor          │                         │
│                      └──────────────────┘                         │
└───────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
HotSpine (Shared Memory) → HotSpineDataBridge → MarketDataProcessor → DataVisualizationEngine → VulkanRenderer → UI
     ↓
SymbolManager ← DashboardConfig
```

## Components Documentation

### 1. HotSpineDataBridge

The HotSpineDataBridge provides zero-copy access to real-time market data from shared memory. It handles both live data from HotSpine and simulated data for testing.

**Key Features**:
- Lock-free shared memory reader
- Thread-safe instrument management
- Automatic reconnection
- Supports both live and simulated data

**File**: [`include/hotspine_data_bridge.hpp`](include/hotspine_data_bridge.hpp)

### 2. MarketDataProcessor

The MarketDataProcessor aggregates and analyzes market data in real-time.

**Key Features**:
- OHLCV aggregation
- VWAP, momentum, and volatility calculations
- Symbol rankings
- Market statistics

**File**: [`include/market_data_processor.hpp`](include/market_data_processor.hpp)

### 3. DataVisualizationEngine

Responsible for preparing data for GPU rendering.

**Key Features**:
- GPU-optimized data structures
- Vulkan pipeline management
- Chart rendering
- Heatmap visualization
- Order book display

**File**: [`include/data_visualization_engine.hpp`](include/data_visualization_engine.hpp)

### 4. SymbolManager

Manages trading symbols and their metadata.

**Key Features**:
- Symbol registration and lookup
- Filtering and search
- Statistics tracking
- Exchange and market type management

**File**: [`include/symbol_manager.hpp`](include/symbol_manager.hpp)

### 5. DashboardConfig

Comprehensive configuration management system.

**Key Features**:
- Configuration loading/saving
- Theme management
- Layout management
- Performance tuning

**File**: [`include/dashboard_config.hpp`](include/dashboard_config.hpp)

### 6. PerformanceMonitor

Monitors system performance and provides detailed metrics.

**Key Features**:
- Frame rate tracking
- Latency measurement
- Memory usage monitoring
- Network status
- System health assessment

**File**: [`include/performance_monitor.hpp`](include/performance_monitor.hpp)

## UI Components

### QuantWorkspaceComponent

Main workspace component that manages instrument charts and analysis tools.

**File**: [`include/components/quant_workspace_component.hpp`](include/components/quant_workspace_component.hpp)

### TechnicalIndicatorsComponent

Displays technical analysis indicators.

**File**: [`include/components/technical_indicators_component.hpp`](include/components/technical_indicators_component.hpp)

### TapeComponent

Real-time trade tape display.

**File**: [`include/components/tape_component.hpp`](include/components/tape_component.hpp)

### HeatmapComponent

Visualizes market data heatmaps.

**File**: [`include/components/heatmap_component.hpp`](include/components/heatmap_component.hpp)

### DataGridComponent

Tabular data display with sorting and filtering.

**File**: [`include/components/data_grid_component.hpp`](include/components/data_grid_component.hpp)

### LogDisplayComponent

System and trading log display.

**File**: [`include/components/log_display_component.hpp`](include/components/log_display_component.hpp)

## API Reference

### VulkanDashboard Class

The main entry point for the rendering engine.

```cpp
class VulkanDashboard {
public:
    VulkanDashboard(uint32_t width, uint32_t height,
                   std::shared_ptr<HotSpineDataBridge> bridge,
                   const VulkanDashboardConfig &config);
    ~VulkanDashboard();
    
    void initialize();
    void shutdown();
    void render_frame();
    void handle_events();
    bool should_close() const;
    void set_active_symbol(const std::string &s);
    std::string get_active_symbol() const;
    VulkanCore *get_vulkan_core();
};
```

### HotSpineDataBridge Class

```cpp
class HotSpineDataBridge {
public:
    HotSpineDataBridge(const std::string &shm_path = "/btquant");
    ~HotSpineDataBridge();
    
    bool start();
    void stop();
    void poll();
    
    std::map<std::string, std::shared_ptr<MarketInstrument>> GetInstruments();
    void setMarketDataProcessor(std::shared_ptr<RenderEngine::MarketDataProcessor> processor);
};
```

### DashboardConfig Class

```cpp
class DashboardConfig {
public:
    DashboardConfig();
    ~DashboardConfig();
    
    bool loadConfiguration(const std::string& config_file = "dashboard_config.yaml");
    bool saveConfiguration() const;
    void resetToDefaults();
    
    const DataSourceConfig& getDataSourceConfig() const;
    const DisplayConfig& getDisplayConfig() const;
    const ThemeConfig& getThemeConfig() const;
    const LayoutConfig& getLayoutConfig() const;
    const PerformanceConfig& getPerformanceConfig() const;
    const UserPreferences& getUserPreferences() const;
    
    void setDataSourceConfig(const DataSourceConfig& config);
    void setDisplayConfig(const DisplayConfig& config);
    void setThemeConfig(const ThemeConfig& config);
    void setLayoutConfig(const LayoutConfig& config);
    void setPerformanceConfig(const PerformanceConfig& config);
    void setUserPreferences(const UserPreferences& preferences);
    
    bool applyTheme(const std::string& theme_name);
    bool saveLayout(const std::string& layout_name);
    bool loadLayout(const std::string& layout_name);
};
```

### TechnicalIndicators Class

```cpp
class TechnicalIndicators {
public:
    static IndicatorResult simple_moving_average(const std::vector<OHLCV> &data, int period);
    static IndicatorResult exponential_moving_average(const std::vector<OHLCV> &data, int period);
    static std::vector<IndicatorResult> bollinger_bands(const std::vector<OHLCV> &data, int period, double std_dev = 2.0);
    static IndicatorResult rsi(const std::vector<OHLCV> &data, int period = 14);
    static std::vector<IndicatorResult> macd(const std::vector<OHLCV> &data, int fast_period = 12, int slow_period = 26, int signal_period = 9);
    static std::vector<IndicatorResult> stochastic(const std::vector<OHLCV> &data, int k_period = 14, int d_period = 3);
};
```

## Configuration Guide

### Configuration File Structure

The engine uses `dashboard_config.yaml` for configuration:

```yaml
# BTQuant Dashboard Configuration
version=1

[data_source]
hotspine_shm_name=/btquant_hotspine
symbols_file=/dev/shm/btquant_symbols.json
auto_reconnect=true
reconnect_interval_ms=5000
max_symbols=1000

[display]
window_width=1920
window_height=1080
fullscreen=false
vsync=true
target_fps=120
msaa_samples=4

[theme]
name=dark
background_color=0.100,0.100,0.100,1.000
text_color=0.900,0.900,0.900,1.000
accent_color=0.200,0.600,1.000,1.000
positive_color=0.000,0.800,0.000,1.000
negative_color=0.800,0.000,0.000,1.000
neutral_color=0.500,0.500,0.500,1.000
grid_line_color=0.300,0.300,0.300,1.000
font_size=14
line_height=1.2

[layout]
grid_columns=10
grid_rows=20
show_grid=true
show_heatmap=true
show_charts=true
show_orderbook=true
show_logs=true
show_performance=true
panel_spacing=8
panel_padding=12

[performance]
monitoring_enabled=true
monitoring_interval_ms=1000
history_size=300
fps_alert_threshold=30
latency_alert_threshold_ms=10
memory_alert_threshold_mb=1024
enable_profiling=false

[user_preferences]
default_exchange=binance
default_symbol=BTCUSDT
auto_save_layout=true
show_tooltips=true
animation_speed=1
update_frequency_hz=30
current_layout=default
```

### Configuration Options

#### Data Source Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `hotspine_shm_name` | Shared memory segment name | `/btquant_hotspine` |
| `symbols_file` | Path to symbols configuration file | `/dev/shm/btquant_symbols.json` |
| `auto_reconnect` | Auto-reconnect on connection loss | `true` |
| `reconnect_interval_ms` | Reconnect interval in milliseconds | `5000` |
| `max_symbols` | Maximum number of symbols to track | `1000` |

#### Display Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `window_width` | Window width in pixels | `1920` |
| `window_height` | Window height in pixels | `1080` |
| `fullscreen` | Enable fullscreen mode | `false` |
| `vsync` | Enable vertical sync | `true` |
| `target_fps` | Target frames per second | `120` |
| `msaa_samples` | MSAA anti-aliasing samples | `4` |

#### Theme Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `name` | Theme name (dark/light/blue/green) | `dark` |
| `background_color` | Background color (RGBA) | `0.1, 0.1, 0.1, 1.0` |
| `text_color` | Text color (RGBA) | `0.9, 0.9, 0.9, 1.0` |
| `accent_color` | Accent color (RGBA) | `0.2, 0.6, 1.0, 1.0` |
| `positive_color` | Positive change color (RGBA) | `0.0, 0.8, 0.0, 1.0` |
| `negative_color` | Negative change color (RGBA) | `0.8, 0.0, 0.0, 1.0` |

#### Performance Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `monitoring_enabled` | Enable performance monitoring | `true` |
| `monitoring_interval_ms` | Monitoring interval in milliseconds | `1000` |
| `history_size` | Metric history size | `300` |
| `fps_alert_threshold` | FPS alert threshold | `30` |
| `latency_alert_threshold_ms` | Latency alert threshold in ms | `10` |
| `memory_alert_threshold_mb` | Memory alert threshold in MB | `1024` |
| `enable_profiling` | Enable detailed profiling | `false` |

## Troubleshooting

### Common Issues

#### 1. Vulkan Initialization Failed

**Symptoms**: Engine fails to start with Vulkan initialization errors.

**Solutions**:
- Check that your GPU supports Vulkan 1.0 or later
- Ensure you have the latest GPU drivers installed
- Verify that Vulkan libraries are installed:
  ```bash
  sudo apt install vulkan-utils libvulkan1
  ```
- Run `vulkaninfo` to check Vulkan capabilities

#### 2. Shared Memory Connection Failed

**Symptoms**: Engine can't connect to HotSpine shared memory.

**Solutions**:
- Verify HotSpine is running
- Check shared memory permissions
- Ensure `/dev/shm/btquant_symbols.json` exists and is readable
- Try running with sudo if permissions are restricted

#### 3. Low Frame Rate

**Symptoms**: Frame rate below target FPS.

**Solutions**:
- Reduce window resolution
- Disable MSAA or reduce sample count
- Close unnecessary applications
- Ensure GPU drivers are up to date
- Check for GPU overheating

#### 4. High Latency

**Symptoms**: Data takes too long to appear on screen.

**Solutions**:
- Ensure HotSpine is running on the same machine
- Check network connectivity if using remote data source
- Verify shared memory permissions
- Reduce update frequency in configuration

#### 5. Symbol Loading Failed

**Symptoms**: Symbols not appearing in the interface.

**Solutions**:
- Check `/dev/shm/btquant_symbols.json` file format
- Verify symbols file contains valid JSON
- Ensure symbols file is readable by the engine
- Check log output for errors

## Examples

### Basic Usage Example

```cpp
#include "vulkan_dashboard_advanced.hpp"
#include <cstdio>
#include <cstdlib>

using namespace BTQuant;

int main(int argc, char *argv[]) {
    try {
        VulkanDashboardConfig config;
        config.enable_msaa = false;
        config.enable_validation_layers = false;
        
        auto hotspine_bridge = std::make_shared<HotSpineDataBridge>(config);
        auto dashboard = std::make_unique<VulkanDashboard>(1280, 720, hotspine_bridge, config);
        
        dashboard->initialize();
        while (!dashboard->should_close()) {
            dashboard->handle_events();
            dashboard->render_frame();
        }
        dashboard->shutdown();
        
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "Dashboard error: %s\n", e.what());
        return 1;
    }
}
```

### Custom Configuration Example

```cpp
#include "dashboard_config.hpp"

using namespace BTQuant::RenderEngine;

int main() {
    DashboardConfig config;
    
    // Load from custom file
    config.loadConfiguration("my_config.yaml");
    
    // Modify configuration
    DisplayConfig display = config.getDisplayConfig();
    display.window_width = 1920;
    display.window_height = 1080;
    display.fullscreen = true;
    config.setDisplayConfig(display);
    
    // Apply theme
    config.applyTheme("light");
    
    // Save changes
    config.saveConfiguration();
    
    return 0;
}
```

### Data Analysis Example

```cpp
#include "market_data_processor.hpp"
#include "hotspine_data_bridge.hpp"

using namespace BTQuant;

int main() {
    auto bridge = std::make_shared<HotSpineDataBridge>();
    auto processor = std::make_shared<RenderEngine::MarketDataProcessor>();
    
    bridge->setMarketDataProcessor(processor);
    bridge->start();
    
    // Wait for data to accumulate
    std::this_thread::sleep_for(std::chrono::seconds(10));
    
    // Get market summary
    auto summary = processor->getMarketSummary();
    std::cout << "Total symbols: " << summary.total_symbols << std::endl;
    std::cout << "Total trades: " << summary.total_trades << std::endl;
    std::cout << "Avg volatility: " << summary.avg_volatility << std::endl;
    
    // Get top volume symbols
    auto rankings = processor->getRankings(RankingCriteria::VOLUME, 10);
    std::cout << "\nTop 10 volume symbols:" << std::endl;
    for (const auto& rank : rankings) {
        std::cout << rank.symbol_id << ": " << rank.volume << " " << rank.price << std::endl;
    }
    
    bridge->stop();
    return 0;
}
```

## Performance Optimization Tips

### 1. GPU Optimization

- **Reduce Overdraw**: Use efficient shader techniques
- **Minimize State Changes**: Batch similar rendering operations
- **Optimize Vertex Data**: Use instanced rendering for repeated elements
- **Texture Compression**: Use GPU-native texture formats

### 2. CPU Optimization

- **Data Processing**: Offload calculations to GPU compute shaders
- **Memory Management**: Use pooling and pre-allocation
- **Threading**: Optimize thread utilization and synchronization
- **Cache Optimization**: Improve data locality

### 3. Network Optimization

- **Compression**: Use efficient data compression
- **Batching**: Combine multiple updates into single messages
- **Protocol Optimization**: Use binary protocols for lower overhead

### 4. System Optimization

- **Process Affinity**: Pin threads to specific CPU cores
- **Memory Bandwidth**: Use high-performance memory
- **Storage**: Use SSD for faster asset loading
- **Power Management**: Disable CPU throttling

## Advanced Features

### Custom Shaders

The engine supports custom Vulkan shaders for advanced visualization. Shaders are stored in the `shaders/` directory and compiled to SPIR-V format.

### Extending Components

New UI components can be created by extending the `UIComponent` base class.

```cpp
struct MyCustomComponent : public UIComponent {
    MyCustomComponent(const glm::vec2 &p, const glm::vec2 &s) 
        : UIComponent(p, s) {}
        
    void update(float dt) override {
        // Update logic
    }
    
    void render_gui() override {
        // ImGui rendering
    }
};
```

### Plugin System

The engine supports a plugin system for extending functionality:

- Custom indicators
- Trading strategies
- Data sources
- Visualization components

## License

The BTQ_Render_Engine is released under the MIT License.

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting a pull request.

## Support

For support and troubleshooting:

1. Check the [Troubleshooting](#troubleshooting) section
2. Look for existing issues on GitHub
3. Create a new issue with detailed information
4. Join the community Discord server

## Acknowledgments

- ImGui - Dear ImGui: Bloat-free Graphical User interface for C++
- ImPlot - Immediate Mode Plotting for ImGui
- Vulkan - Low-overhead, cross-platform 3D graphics and compute API
- GLFW - Multi-platform library for OpenGL, OpenGL ES, and Vulkan development
- GLM - OpenGL Mathematics
