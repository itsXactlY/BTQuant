# Phase 1: High-Performance Candle Rendering Implementation Plan

## Overview
Enhance the existing QuantWorkspaceComponent to support high-performance, fully configurable candle rendering with OHLCV data and technical indicator overlays.

## Current Implementation Analysis

### QuantWorkspaceComponent
Current functionality:
- Renders single instrument chart per symbol
- Uses ImPlot for basic plotting
- Manual candlestick rendering via ImDrawList
- Volume profile visualization
- Neon-cyan and neon-red color scheme

Limitations:
- No indicator overlays
- No timeframe selection
- Basic zoom/pan functionality
- Manual rendering not GPU-accelerated

### MarketDataProcessor
Current functionality:
- OHLCV aggregation for multiple timeframes
- Basic analytics (VWAP, momentum, volatility)
- Candle management (current/in-progress)

Limitations:
- No indicator calculations
- Limited timeframe support
- No data buffering optimization

## Implementation Plan

### 1. Enhanced QuantWorkspaceComponent
**File:** dependencies/BTQ_Render_Engine/src/components/quant_workspace_component.cpp

#### Changes:
```cpp
// Add timeframe selection
void render_timeframe_selector();

// Add indicator overlay selection
void render_indicator_selector();

// Add interactive chart controls
void render_chart_controls();

// Enhanced instrument chart rendering
void render_instrument_chart(const std::string &symbol, const InstrumentStore &inst);

// Add support for multiple timeframes
std::unordered_map<std::string, TimeFrame> symbol_timeframes_;

// Add indicator configuration
struct IndicatorConfig {
    bool show_sma_10 = true;
    bool show_sma_20 = true;
    bool show_sma_50 = false;
    bool show_ema_10 = false;
    bool show_ema_20 = false;
    bool show_ema_50 = false;
    bool show_rsi = true;
    bool show_macd = true;
    bool show_bollinger = false;
    bool show_stochastic = false;
};
std::unordered_map<std::string, IndicatorConfig> indicator_configs_;
```

### 2. Enhanced MarketDataProcessor
**File:** dependencies/BTQ_Render_Engine/src/data/market_data_processor.cpp

#### Changes:
```cpp
// Add indicator calculation methods
std::vector<double> calculate_sma(const std::vector<OHLCVCandle> &candles, int period);
std::vector<double> calculate_ema(const std::vector<OHLCVCandle> &candles, int period);
std::vector<double> calculate_rsi(const std::vector<OHLCVCandle> &candles, int period);
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> calculate_macd(
    const std::vector<OHLCVCandle> &candles, int fast_period, int slow_period, int signal_period);
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> calculate_bollinger_bands(
    const std::vector<OHLCVCandle> &candles, int period, double std_dev);
std::tuple<std::vector<double>, std::vector<double>> calculate_stochastic(
    const std::vector<OHLCVCandle> &candles, int k_period, int d_period);

// Add indicator data storage to SymbolAnalytics
struct SymbolAnalytics {
    // Existing fields...
    
    // Indicator data
    std::unordered_map<TimeFrame, std::vector<double>> sma_10;
    std::unordered_map<TimeFrame, std::vector<double>> sma_20;
    std::unordered_map<TimeFrame, std::vector<double>> sma_50;
    std::unordered_map<TimeFrame, std::vector<double>> ema_10;
    std::unordered_map<TimeFrame, std::vector<double>> ema_20;
    std::unordered_map<TimeFrame, std::vector<double>> ema_50;
    std::unordered_map<TimeFrame, std::vector<double>> rsi_14;
    std::unordered_map<TimeFrame, std::vector<double>> macd_line;
    std::unordered_map<TimeFrame, std::vector<double>> macd_signal;
    std::unordered_map<TimeFrame, std::vector<double>> macd_histogram;
    std::unordered_map<TimeFrame, std::vector<double>> bollinger_mid;
    std::unordered_map<TimeFrame, std::vector<double>> bollinger_upper;
    std::unordered_map<TimeFrame, std::vector<double>> bollinger_lower;
    std::unordered_map<TimeFrame, std::vector<double>> stochastic_k;
    std::unordered_map<TimeFrame, std::vector<double>> stochastic_d;
};

// Update indicator calculations on candle updates
void update_indicators(SymbolAnalytics &symbol_data, TimeFrame timeframe);
```

### 3. Enhanced DataVisualizationEngine
**File:** dependencies/BTQ_Render_Engine/src/visualization/data_visualization_engine.cpp

#### Changes:
```cpp
// Add indicator buffer support
VkBuffer indicator_buffer_;
VkDeviceMemory indicator_memory_;

// Add indicator data structure
struct IndicatorDataGPU {
    uint32_t symbol_id;
    uint32_t timeframe;
    float value;
    uint32_t indicator_type; // 0=SMA10, 1=SMA20, etc.
    ColorRGBA color;
    float padding[2];
};

// Update indicator data
void updateIndicatorData(uint32_t symbol_id, TimeFrame timeframe, 
                         IndicatorType type, const std::vector<double> &values);

// Get indicator buffer
VkBuffer getIndicatorBuffer() const;

// Enhanced buffer initialization
bool initializeBuffers() override;
```

### 4. New ChartManager Component
**File:** dependencies/BTQ_Render_Engine/include/components/chart_manager.hpp

```cpp
#pragma once

#include "market_data_processor.hpp"
#include "hotspine_data_bridge.hpp"
#include <unordered_map>
#include <string>
#include <memory>

namespace BTQuant {

struct ChartInstance {
    std::string symbol;
    RenderEngine::TimeFrame timeframe;
    uint32_t chart_id;
    bool visible = true;
    bool minimized = false;
    ImVec2 position = {0, 0};
    ImVec2 size = {600, 400};
};

class ChartManager {
public:
    ChartManager(std::shared_ptr<HotSpineDataBridge> bridge,
                 std::shared_ptr<RenderEngine::MarketDataProcessor> processor);
    
    uint32_t create_chart(const std::string &symbol, RenderEngine::TimeFrame timeframe);
    void destroy_chart(uint32_t chart_id);
    void toggle_chart_visibility(uint32_t chart_id);
    void toggle_chart_minimization(uint32_t chart_id);
    void update_chart_position(uint32_t chart_id, const ImVec2 &position);
    void update_chart_size(uint32_t chart_id, const ImVec2 &size);
    
    const std::unordered_map<uint32_t, ChartInstance> &get_charts() const;
    std::vector<ChartInstance> get_visible_charts() const;
    std::vector<ChartInstance> get_charts_for_symbol(const std::string &symbol) const;
    
    void update();
    
private:
    std::shared_ptr<HotSpineDataBridge> bridge_;
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
    std::unordered_map<uint32_t, ChartInstance> charts_;
    uint32_t next_chart_id_ = 0;
};

} // namespace BTQuant
```

### 5. New IndicatorRenderer Component
**File:** dependencies/BTQ_Render_Engine/include/components/indicator_renderer.hpp

```cpp
#pragma once

#include "market_data_processor.hpp"
#include "vulkan_base_types.hpp"
#include "data_visualization_engine.hpp"
#include <vector>
#include <string>

namespace BTQuant {

enum class IndicatorType {
    SMA_10, SMA_20, SMA_50,
    EMA_10, EMA_20, EMA_50,
    RSI_14,
    MACD, MACD_SIGNAL, MACD_HISTOGRAM,
    BOLLINGER_MID, BOLLINGER_UPPER, BOLLINGER_LOWER,
    STOCHASTIC_K, STOCHASTIC_D
};

struct IndicatorParams {
    IndicatorType type;
    int period1 = 10;
    int period2 = 20;
    int period3 = 9;
    double std_dev = 2.0;
    ImVec4 color = {0.0f, 0.94f, 1.0f, 1.0f};
    float line_width = 1.0f;
    bool visible = true;
};

class IndicatorRenderer {
public:
    IndicatorRenderer(VulkanCore *vulkan_core,
                     std::shared_ptr<RenderEngine::MarketDataProcessor> processor);
    
    void render_indicators(const std::string &symbol, 
                         RenderEngine::TimeFrame timeframe,
                         const std::vector<IndicatorParams> &indicators);
    
    void initialize_vulkan_resources();
    void cleanup_vulkan_resources();
    
private:
    VulkanCore *vulkan_core_;
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
    RenderEngine::DataVisualizationEngine *visualization_engine_;
    
    std::vector<IndicatorParams> default_indicators_;
    
    void render_sma(const std::string &symbol, RenderEngine::TimeFrame timeframe,
                   const IndicatorParams &params);
    void render_ema(const std::string &symbol, RenderEngine::TimeFrame timeframe,
                   const IndicatorParams &params);
    void render_rsi(const std::string &symbol, RenderEngine::TimeFrame timeframe,
                   const IndicatorParams &params);
    void render_macd(const std::string &symbol, RenderEngine::TimeFrame timeframe,
                    const IndicatorParams &params);
    void render_bollinger(const std::string &symbol, RenderEngine::TimeFrame timeframe,
                        const IndicatorParams &params);
    void render_stochastic(const std::string &symbol, RenderEngine::TimeFrame timeframe,
                         const IndicatorParams &params);
};

} // namespace BTQuant
```

## Testing Strategy
1. **Unit Tests:** Test indicator calculation methods
2. **Integration Tests:** Verify chart creation and data flow
3. **Performance Tests:** Measure rendering performance with various indicator combinations
4. **Stress Tests:** Test with high-frequency data and multiple active charts

## Milestones
1. Complete enhanced QuantWorkspaceComponent - 5 days
2. Implement ChartManager and IndicatorRenderer - 3 days
3. Enhance MarketDataProcessor with indicator calculations - 4 days
4. Enhance DataVisualizationEngine - 2 days
5. Testing and optimization - 3 days

## Expected Results
- Smooth, GPU-accelerated candle rendering at 60+ FPS
- Real-time indicator overlays with customizable parameters
- Interactive timeframe selection
- Efficient memory management with 10+ active charts
- Comprehensive testing coverage
