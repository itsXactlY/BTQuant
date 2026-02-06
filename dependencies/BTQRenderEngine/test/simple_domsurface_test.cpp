#include <iostream>
#include <memory>

// Test that our header compiles correctly
#include "components/domsurfacepanel.h"

using namespace BTQuant::RenderEngine;

// Mock MarketDataProcessor for testing
class MockMarketDataProcessor : public MarketDataProcessor {
public:
    MockMarketDataProcessor() = default;
    
    uint64_t subscribe(uint32_t symbol_id, NotificationType type, 
                      std::function<void(uint32_t, NotificationType)> callback) override {
        return 1; // Return mock subscription ID
    }
    
    bool unsubscribe(uint64_t subscription_id) override {
        return true; // Always succeed for mock
    }
    
    std::optional<OrderbookData> getOrderbookData(uint32_t symbol_id) override {
        // Return mock orderbook data
        OrderbookData data;
        // Add some mock bids and asks
        data.bids.push_back({100.0, 10.0}); // price, size
        data.bids.push_back({99.5, 15.0});
        data.asks.push_back({101.0, 8.0});
        data.asks.push_back({101.5, 12.0});
        data.timestamp = 1234567890; // Mock timestamp
        return data;
    }
    
    std::vector<OrderbookData> getHistoricalOrderbooks(uint32_t symbol_id, size_t limit) override {
        std::vector<OrderbookData> history;
        
        // Add multiple mock snapshots
        for (int i = 0; i < 10; ++i) {
            OrderbookData data;
            data.bids.push_back({100.0 + i*0.1, 10.0 + i});
            data.asks.push_back({101.0 + i*0.1, 8.0 + i});
            data.timestamp = 1234567890 + i * 1000000; // Increment timestamp
            history.push_back(data);
        }
        
        return history;
    }
    
    void updateOrderbook(uint32_t symbol_id, const OrderbookData& data) override {
        // Mock implementation
    }
    
    void updateTrades(uint32_t symbol_id, const std::vector<TradeData>& trades) override {
        // Mock implementation
    }
    
    void updateBars(uint32_t symbol_id, const std::vector<BarData>& bars) override {
        // Mock implementation
    }
    
    std::vector<TradeData> getTradeHistory(uint32_t symbol_id, size_t limit) override {
        return {};
    }
    
    std::vector<BarData> getBarHistory(uint32_t symbol_id, size_t limit) override {
        return {};
    }
    
    void setMaxHistorySize(size_t max_size) override {
        // Mock implementation
    }
    
    size_t getMaxHistorySize() const override {
        return 1000;
    }
    
    void clearHistory(uint32_t symbol_id) override {
        // Mock implementation
    }
    
    void clearAllHistory() override {
        // Mock implementation
    }
    
    std::vector<uint32_t> getActiveSymbols() const override {
        return {1};
    }
    
    void setSymbolName(uint32_t symbol_id, const std::string& name) override {
        // Mock implementation
    }
    
    std::string getSymbolName(uint32_t symbol_id) const override {
        return "TEST";
    }
};

int main() {
    std::cout << "Testing DomSurfacePanel with Vulkan acceleration..." << std::endl;
    
    // Create mock processor
    auto processor = std::make_shared<MockMarketDataProcessor>();
    
    // Create DomSurfacePanel
    auto panel = std::make_unique<DomSurfacePanel>(processor);
    
    std::cout << "DomSurfacePanel created successfully!" << std::endl;
    
    // Set a symbol
    panel->setSymbol(1);
    std::cout << "Symbol set to 1" << std::endl;
    
    // Test configuration
    panel->setHistoryDepth(200);
    panel->setPriceRange(0.05);
    panel->setLargeOrderThreshold(5.0);
    panel->setMaxLargeOrderMarkers(50);
    panel->setLargeOrderFadeOut(true);
    
    std::cout << "Configuration applied" << std::endl;
    
    std::cout << "DomSurfacePanel test completed successfully!" << std::endl;
    
    return 0;
}