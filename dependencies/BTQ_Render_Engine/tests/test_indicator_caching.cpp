#include "../include/components/chart_panel.hpp"
#include "../include/components/chart_manager.hpp"
#include "../include/hotspine_data_bridge.hpp"
#include "../include/market_data_processor.hpp"

#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

using namespace BTQuant;

// Mock classes for testing
class MockHotSpineDataBridge : public HotSpineDataBridge {
public:
    MockHotSpineDataBridge() : HotSpineDataBridge({}, {}) {}
    
    void update() override {}
    bool isConnected() const override { return true; }
    void subscribeToSymbol(const std::string& symbol) override {}
    void unsubscribeFromSymbol(const std::string& symbol) override {}
    std::vector<MarketData> getRecentData(const std::string& symbol, int count) override {
        return {};
    }
};

class MockMarketDataProcessor : public RenderEngine::MarketDataProcessor {
public:
    MockMarketDataProcessor() : RenderEngine::MarketDataProcessor() {}
    
    void update() override {}
    void processMarketData(const MarketData& data) override {}
    std::vector<OHLCVData> getHistoricalData(uint32_t symbol_id, RenderEngine::TimeFrame tf, int count) override {
        return {};
    }
    double getTimeFrameDuration(RenderEngine::TimeFrame tf) override { return 1000000.0; } // 1 second in microseconds
    std::vector<VolumeProfileLevel> getVolumeProfile(uint32_t symbol_id, RenderEngine::TimeFrame tf) override {
        return {};
    }
};

void test_sma_caching() {
    std::cout << "Testing SMA Caching..." << std::endl;

    // Create mock objects
    auto bridge = std::make_shared<MockHotSpineDataBridge>();
    auto processor = std::make_shared<MockMarketDataProcessor>();
    ChartManager chart_manager;
    
    PanelConfig config;
    config.title = "Test Chart";
    config.position = {0, 0};
    config.size = {800, 600};
    
    ChartPanel chart_panel(config, bridge, processor, &chart_manager);
    
    // Create test data
    std::vector<float> prices = {100.0f, 101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f, 108.0f, 109.0f};
    
    // Calculate SMA twice with same data - should use cache second time
    auto sma1 = chart_panel.calculate_sma(prices, 5);
    auto sma2 = chart_panel.calculate_sma(prices, 5);
    
    // Results should be identical
    assert(sma1.size() == sma2.size());
    for (size_t i = 0; i < sma1.size(); ++i) {
        assert(sma1[i] == sma2[i]);
    }
    
    std::cout << "✓ SMA caching test passed" << std::endl;
    
    // Test with different period - should not use cache
    auto sma3 = chart_panel.calculate_sma(prices, 3);
    assert(sma1.size() == sma3.size());
    
    std::cout << "✓ SMA different period test passed" << std::endl;
    
    // Test with extended data - cache should be invalidated
    std::vector<float> extended_prices = {100.0f, 101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f, 108.0f, 109.0f, 110.0f};
    chart_panel.invalidate_cache_if_needed(extended_prices.size());
    
    auto sma4 = chart_panel.calculate_sma(extended_prices, 5);
    assert(sma4.size() == extended_prices.size());
    
    std::cout << "✓ SMA cache invalidation test passed" << std::endl;
}

void test_ema_caching() {
    std::cout << "Testing EMA Caching..." << std::endl;

    // Create mock objects
    auto bridge = std::make_shared<MockHotSpineDataBridge>();
    auto processor = std::make_shared<MockMarketDataProcessor>();
    ChartManager chart_manager;
    
    PanelConfig config;
    config.title = "Test Chart";
    config.position = {0, 0};
    config.size = {800, 600};
    
    ChartPanel chart_panel(config, bridge, processor, &chart_manager);
    
    // Create test data
    std::vector<float> prices = {100.0f, 101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f, 108.0f, 109.0f};
    
    // Calculate EMA twice with same data - should use cache second time
    auto ema1 = chart_panel.calculate_ema(prices, 5);
    auto ema2 = chart_panel.calculate_ema(prices, 5);
    
    // Results should be identical
    assert(ema1.size() == ema2.size());
    for (size_t i = 0; i < ema1.size(); ++i) {
        assert(ema1[i] == ema2[i]);
    }
    
    std::cout << "✓ EMA caching test passed" << std::endl;
    
    // Test with different period - should not use cache
    auto ema3 = chart_panel.calculate_ema(prices, 3);
    assert(ema1.size() == ema3.size());
    
    std::cout << "✓ EMA different period test passed" << std::endl;
}

void test_rsi_caching() {
    std::cout << "Testing RSI Caching..." << std::endl;

    // Create mock objects
    auto bridge = std::make_shared<MockHotSpineDataBridge>();
    auto processor = std::make_shared<MockMarketDataProcessor>();
    ChartManager chart_manager;
    
    PanelConfig config;
    config.title = "Test Chart";
    config.position = {0, 0};
    config.size = {800, 600};
    
    ChartPanel chart_panel(config, bridge, processor, &chart_manager);
    
    // Create test data
    std::vector<float> prices = {100.0f, 101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f, 108.0f, 109.0f};
    
    // Calculate RSI twice with same data - should use cache second time
    auto rsi1 = chart_panel.calculate_rsi(prices, 5);
    auto rsi2 = chart_panel.calculate_rsi(prices, 5);
    
    // Results should be identical
    assert(rsi1.size() == rsi2.size());
    for (size_t i = 0; i < rsi1.size(); ++i) {
        assert(rsi1[i] == rsi2[i]);
    }
    
    std::cout << "✓ RSI caching test passed" << std::endl;
    
    // Test with different period - should not use cache
    auto rsi3 = chart_panel.calculate_rsi(prices, 3);
    assert(rsi1.size() == rsi3.size());
    
    std::cout << "✓ RSI different period test passed" << std::endl;
}

void test_cache_invalidation() {
    std::cout << "Testing Cache Invalidation..." << std::endl;

    // Create mock objects
    auto bridge = std::make_shared<MockHotSpineDataBridge>();
    auto processor = std::make_shared<MockMarketDataProcessor>();
    ChartManager chart_manager;
    
    PanelConfig config;
    config.title = "Test Chart";
    config.position = {0, 0};
    config.size = {800, 600};
    
    ChartPanel chart_panel(config, bridge, processor, &chart_manager);
    
    // Create initial test data
    std::vector<float> prices = {100.0f, 101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f, 108.0f, 109.0f};
    
    // Calculate indicators to populate cache
    auto sma1 = chart_panel.calculate_sma(prices, 5);
    auto ema1 = chart_panel.calculate_ema(prices, 5);
    auto rsi1 = chart_panel.calculate_rsi(prices, 5);
    
    // Verify cache is populated by calculating again with same data
    auto sma2 = chart_panel.calculate_sma(prices, 5);
    auto ema2 = chart_panel.calculate_ema(prices, 5);
    auto rsi2 = chart_panel.calculate_rsi(prices, 5);
    
    // Results should be identical (from cache)
    assert(sma1.size() == sma2.size());
    assert(ema1.size() == ema2.size());
    assert(rsi1.size() == rsi2.size());
    
    for (size_t i = 0; i < sma1.size(); ++i) {
        assert(sma1[i] == sma2[i]);
        if (i < ema1.size()) assert(ema1[i] == ema2[i]);
        if (i < rsi1.size()) assert(rsi1[i] == rsi2[i]);
    }
    
    // Invalidate cache by simulating new data arrival
    chart_panel.invalidate_cache_if_needed(prices.size() + 1);
    
    // Calculate again - should recalculate, not use cache
    auto sma3 = chart_panel.calculate_sma(prices, 5);
    auto ema3 = chart_panel.calculate_ema(prices, 5);
    auto rsi3 = chart_panel.calculate_rsi(prices, 5);
    
    // Results should still be identical to original since data didn't change
    // But internally they should have been recalculated due to cache invalidation
    assert(sma1.size() == sma3.size());
    assert(ema1.size() == ema3.size());
    assert(rsi1.size() == rsi3.size());
    
    for (size_t i = 0; i < sma1.size(); ++i) {
        assert(sma1[i] == sma3[i]);
        if (i < ema1.size()) assert(ema1[i] == ema3[i]);
        if (i < rsi1.size()) assert(rsi1[i] == rsi3[i]);
    }
    
    std::cout << "✓ Cache invalidation test passed" << std::endl;
}

int main() {
    std::cout << "=== Running Indicator Caching Tests ===" << std::endl;

    try {
        test_sma_caching();
        test_ema_caching();
        test_rsi_caching();
        test_cache_invalidation();

        std::cout << "=== All indicator caching tests completed successfully! ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}