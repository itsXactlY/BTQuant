#pragma once

#include "market_data_processor.hpp"
#include "vulkan_base_types.hpp"
#include <vector>
#include <string>
#include "imgui.h"
#include <unordered_map>
#include <mutex>

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

// Caching structure for indicator calculations
struct IndicatorCacheEntry {
    uint64_t last_update;
    std::vector<double> values;
};

struct IndicatorCache {
    std::unordered_map<std::string, IndicatorCacheEntry> cache;
    std::mutex mutex;
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
    
    // Cache management
    void clear_cache(const std::string &symbol);
    void clear_all_caches();
    
private:
    VulkanCore *vulkan_core_;
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
    
    std::vector<IndicatorParams> default_indicators_;
    
    // Indicator cache
    std::unordered_map<std::string, IndicatorCache> indicator_caches_;
    
    // Helper to map symbol name to ID
    uint32_t getSymbolId(const std::string &symbol) const;
    
    // Cached indicator calculation methods
    const std::vector<double>& calculate_sma(const std::string &symbol,
                                             RenderEngine::TimeFrame timeframe,
                                             int period);
    const std::vector<double>& calculate_ema(const std::string &symbol,
                                             RenderEngine::TimeFrame timeframe,
                                             int period);
    const std::vector<double>& calculate_rsi(const std::string &symbol,
                                             RenderEngine::TimeFrame timeframe,
                                             int period);
    const std::vector<double>& calculate_macd(const std::string &symbol,
                                              RenderEngine::TimeFrame timeframe,
                                              int period1, int period2, int period3);
    const std::vector<double>& calculate_bollinger(const std::string &symbol,
                                                  RenderEngine::TimeFrame timeframe,
                                                  int period, double std_dev);
    const std::vector<double>& calculate_stochastic(const std::string &symbol,
                                                   RenderEngine::TimeFrame timeframe,
                                                   int period1, int period2);
    
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
