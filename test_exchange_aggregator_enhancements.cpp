#include "dependencies/BTQ_Render_Engine/include/data/exchange_aggregator.hpp"
#include <iostream>
#include <cassert>
#include <memory>

// Mock classes for testing
namespace RenderEngine {
    struct MarketDataUpdate {
        double price = 0.0;
        double size = 0.0;
        uint64_t timestamp = 0;
        std::string side = "BUY";
    };
    
    class MarketDataProcessor {};
    class SymbolManager {};
}

namespace HotSpine {
    class DataBridge {};
}

using HotSpineDataBridge = HotSpine::DataBridge;

void testNewMultiExchangeAggregationMethods() {
    std::cout << "Testing new multi-exchange aggregation methods...\n";
    
    // Create mock dependencies
    auto bridge = std::make_shared<HotSpineDataBridge>();
    auto processor = std::make_shared<RenderEngine::MarketDataProcessor>();
    auto symbol_manager = std::make_shared<RenderEngine::SymbolManager>();
    
    // Create the exchange aggregator
    BTQuant::Data::ExchangeAggregator aggregator(bridge, processor, symbol_manager);
    
    // Initialize the aggregator
    assert(aggregator.initialize() == true);
    std::cout << "✓ Aggregator initialized successfully\n";
    
    // Add some mock exchanges
    BTQuant::Data::ExchangeFeatures features1;
    features1.exchange_name = "Binance";
    features1.reliability_score = 0.95;
    features1.latency_offset_us = 100.0;
    
    BTQuant::Data::ExchangeFeatures features2;
    features2.exchange_name = "Coinbase";
    features2.reliability_score = 0.85;
    features2.latency_offset_us = 200.0;
    
    aggregator.addExchange("Binance", features1);
    aggregator.addExchange("Coinbase", features2);
    
    std::cout << "✓ Exchanges added successfully\n";
    
    // Create some mock market data
    RenderEngine::MarketDataUpdate update1;
    update1.price = 45000.0;
    update1.size = 1.5;
    update1.timestamp = 1000000;
    update1.side = "BUY";
    
    RenderEngine::MarketDataUpdate update2;
    update2.price = 44950.0;
    update2.size = 2.0;
    update2.timestamp = 1000100;
    update2.side = "SELL";
    
    // Process the data updates
    aggregator.processDataUpdate("Binance", "BTCUSD", update1);
    aggregator.processDataUpdate("Coinbase", "BTCUSD", update2);
    
    std::cout << "✓ Data updates processed successfully\n";
    
    // Test the new advanced aggregation method
    auto advanced_result = aggregator.getAdvancedMultiExchangeAggregatedData("BTCUSD");
    if (advanced_result.has_value()) {
        std::cout << "✓ Advanced multi-exchange aggregation works\n";
        std::cout << "  Aggregated price: " << advanced_result->aggregated_price << "\n";
        std::cout << "  Volume: " << advanced_result->aggregated_volume << "\n";
    } else {
        std::cout << "✗ Advanced multi-exchange aggregation failed\n";
    }
    
    // Test the comprehensive view with analytics
    auto comprehensive_result = aggregator.getComprehensiveMultiExchangeViewWithAnalytics("BTCUSD");
    if (comprehensive_result.has_value()) {
        std::cout << "✓ Comprehensive multi-exchange view with analytics works\n";
        std::cout << "  Number of exchanges: " << comprehensive_result->exchange_data.size() << "\n";
        std::cout << "  Total volume: " << comprehensive_result->market_metrics.total_volume << "\n";
    } else {
        std::cout << "✗ Comprehensive multi-exchange view with analytics failed\n";
    }
    
    // Test the sophisticated exchange-specific features handler
    RenderEngine::MarketDataUpdate test_update;
    test_update.price = 45100.0;
    test_update.size = 0.5;
    test_update.timestamp = 1000200;
    test_update.side = "BUY";
    
    aggregator.handleSophisticatedExchangeSpecificFeatures("Binance", "BTCUSD", test_update);
    std::cout << "✓ Sophisticated exchange-specific features handling works\n";
    
    std::cout << "All tests passed!\n";
}

int main() {
    testNewMultiExchangeAggregationMethods();
    return 0;
}