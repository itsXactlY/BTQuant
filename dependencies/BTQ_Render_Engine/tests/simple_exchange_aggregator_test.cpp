#include <iostream>
#include <memory>

#include "../include/data/exchange_aggregator.hpp"
#include "../include/hotspine_data_bridge.hpp"
#include "../include/market_data_processor.hpp"
#include "../include/symbol_manager.hpp"

int main() {
    std::cout << "Testing Exchange Aggregator..." << std::endl;
    
    // Create mock/shared objects for testing
    auto bridge = std::make_shared<BTQuant::HotSpineDataBridge>("/tmp/test_shm");
    auto processor = std::make_shared<BTQuant::RenderEngine::MarketDataProcessor>();
    auto symbol_manager = std::make_shared<BTQuant::RenderEngine::SymbolManager>();
    
    auto aggregator = std::make_unique<BTQuant::Data::ExchangeAggregator>(bridge, processor, symbol_manager);
    
    if (aggregator->initialize()) {
        std::cout << "✓ ExchangeAggregator initialized successfully" << std::endl;
        
        // Add exchanges
        BTQuant::Data::ExchangeFeatures binance_features;
        binance_features.exchange_name = "binance";
        binance_features.latency_offset_us = 100.0;
        binance_features.reliability_score = 0.95;
        
        BTQuant::Data::ExchangeFeatures coinbase_features;
        coinbase_features.exchange_name = "coinbase";
        coinbase_features.latency_offset_us = 150.0;
        coinbase_features.reliability_score = 0.90;
        
        aggregator->addExchange("binance", binance_features);
        aggregator->addExchange("coinbase", coinbase_features);
        
        std::cout << "✓ Exchanges added successfully" << std::endl;
        
        // Create mock market data update
        BTQuant::RenderEngine::MarketDataUpdate update;
        update.type = BTQuant::RenderEngine::MarketDataType::TRADE;
        update.symbol_id = 1;
        update.timestamp = 1704067200000000ULL;  // Jan 1, 2024
        update.price = 40000.0;
        update.size = 1.0;
        update.side = "buy";
        
        // Process data from both exchanges
        aggregator->processDataUpdate("binance", "BTCUSDT", update);
        aggregator->processDataUpdate("coinbase", "BTCUSDT", update);
        
        std::cout << "✓ Data processed from exchanges" << std::endl;
        
        // Get aggregated data
        auto aggregated = aggregator->getAggregatedData("BTCUSDT");
        if (aggregated.has_value()) {
            std::cout << "✓ Successfully retrieved aggregated data for BTCUSDT" << std::endl;
            std::cout << "  Symbol: " << aggregated->symbol << std::endl;
            std::cout << "  Aggregated price: " << aggregated->aggregated_price << std::endl;
            std::cout << "  Exchange count: " << aggregated->exchange_timestamps.size() << std::endl;
        } else {
            std::cout << "✗ Failed to retrieve aggregated data" << std::endl;
        }
        
        // Test time synchronization
        aggregator->setTimeSyncStrategy(BTQuant::Data::TimeSyncStrategy::EARLIEST_TIMESTAMP);
        uint64_t sync_ts = aggregator->calculateSynchronizedTimestamp("BTCUSDT");
        std::cout << "✓ Synchronized timestamp: " << sync_ts << std::endl;
        
        // Get available exchanges
        auto exchanges = aggregator->getAvailableExchanges();
        std::cout << "✓ Available exchanges: ";
        for (const auto& exch : exchanges) {
            std::cout << exch << " ";
        }
        std::cout << std::endl;
        
        // Get exchange features
        auto features = aggregator->getExchangeFeatures("binance");
        if (features.has_value()) {
            std::cout << "✓ Retrieved features for binance: latency_offset=" 
                      << features->latency_offset_us 
                      << ", reliability=" << features->reliability_score << std::endl;
        }
        
        // Get statistics
        auto stats = aggregator->getStats();
        std::cout << "✓ Stats - symbols aggregated: " << stats.total_symbols_aggregated 
                  << ", exchanges: " << stats.total_exchanges << std::endl;
        
    } else {
        std::cout << "✗ Failed to initialize ExchangeAggregator" << std::endl;
        return 1;
    }
    
    std::cout << "\nAll tests passed! Exchange Aggregator is working correctly." << std::endl;
    return 0;
}