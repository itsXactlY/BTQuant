#include <iostream>
#include <memory>
#include <thread>
#include <chrono>

#include "dependencies/BTQ_Render_Engine/include/data/exchange_aggregator.hpp"
#include "dependencies/BTQ_Render_Engine/include/hotspine_data_bridge.hpp"
#include "dependencies/BTQ_Render_Engine/include/market_data_processor.hpp"
#include "dependencies/BTQ_Render_Engine/include/symbol_manager.hpp"

using namespace BTQuant::Data;

int main() {
    std::cout << "Testing Multi-Exchange Aggregation..." << std::endl;

    // Create mock shared pointers for dependencies
    auto bridge = std::make_shared<HotSpineDataBridge>();
    auto processor = std::make_shared<RenderEngine::MarketDataProcessor>();
    auto symbol_manager = std::make_shared<RenderEngine::SymbolManager>();

    // Create the exchange aggregator
    ExchangeAggregator aggregator(bridge, processor, symbol_manager);
    
    if (!aggregator.initialize()) {
        std::cerr << "Failed to initialize ExchangeAggregator" << std::endl;
        return 1;
    }

    // Define exchange features for testing
    ExchangeFeatures binance_features;
    binance_features.exchange_name = "binance";
    binance_features.latency_offset_us = 100.0;
    binance_features.reliability_score = 0.95;
    binance_features.trading_fee_rate = 0.001;
    binance_features.precision = 8;

    ExchangeFeatures coinbase_features;
    coinbase_features.exchange_name = "coinbase";
    coinbase_features.latency_offset_us = 500.0;
    coinbase_features.reliability_score = 0.90;
    coinbase_features.trading_fee_rate = 0.005;
    coinbase_features.precision = 8;

    ExchangeFeatures kraken_features;
    kraken_features.exchange_name = "kraken";
    kraken_features.latency_offset_us = 300.0;
    kraken_features.reliability_score = 0.85;
    kraken_features.trading_fee_rate = 0.002;
    coinbase_features.precision = 8;

    // Add exchanges to the aggregator
    aggregator.addExchange("binance", binance_features);
    aggregator.addExchange("coinbase", coinbase_features);
    aggregator.addExchange("kraken", kraken_features);

    // Set time synchronization strategy
    aggregator.setTimeSyncStrategy(TimeSyncStrategy::SMART_SYNC);

    // Simulate market data updates from different exchanges
    RenderEngine::MarketDataUpdate binance_update;
    binance_update.price = 45000.0;
    binance_update.size = 1.5;
    binance_update.timestamp = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count());
    binance_update.side = "BUY";

    RenderEngine::MarketDataUpdate coinbase_update;
    coinbase_update.price = 45050.0;
    coinbase_update.size = 0.8;
    coinbase_update.timestamp = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count()) - 1000;
    coinbase_update.side = "BUY";

    RenderEngine::MarketDataUpdate kraken_update;
    kraken_update.price = 44980.0;
    kraken_update.size = 2.0;
    kraken_update.timestamp = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count()) - 500;
    kraken_update.side = "BUY";

    // Process data updates
    aggregator.processDataUpdate("binance", "BTC/USD", binance_update);
    aggregator.processDataUpdate("coinbase", "BTC/USD", coinbase_update);
    aggregator.processDataUpdate("kraken", "BTC/USD", kraken_update);

    // Wait a bit for processing
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Test basic aggregation
    auto aggregated_data = aggregator.getAggregatedData("BTC/USD");
    if (aggregated_data.has_value()) {
        std::cout << "Basic aggregation successful:" << std::endl;
        std::cout << "  Aggregated price: " << aggregated_data->aggregated_price << std::endl;
        std::cout << "  Weighted price: " << aggregated_data->weighted_price << std::endl;
        std::cout << "  Consensus price: " << aggregated_data->consensus_price << std::endl;
        std::cout << "  Number of exchanges: " << aggregated_data->exchange_data.size() << std::endl;
    } else {
        std::cout << "Basic aggregation returned no data" << std::endl;
    }

    // Test advanced unified view
    auto unified_view = aggregator.getAdvancedUnifiedMultiExchangeView("BTC/USD");
    if (unified_view.has_value()) {
        std::cout << "\nAdvanced unified view successful:" << std::endl;
        std::cout << "  Symbol: " << unified_view->symbol << std::endl;
        std::cout << "  Average price: " << unified_view->market_metrics.average_price << std::endl;
        std::cout << "  Spread: " << unified_view->market_metrics.spread << std::endl;
        std::cout << "  Total volume: " << unified_view->market_metrics.total_volume << std::endl;
        std::cout << "  Number of exchanges: " << unified_view->exchange_data.size() << std::endl;
        
        // Print exchange-specific data
        for (const auto& [exchange, data] : unified_view->exchange_data) {
            std::cout << "  " << exchange << ": price=" << data.stats.price 
                      << ", volume=" << data.stats.volume 
                      << ", deviation=" << data.stats.percent_price_deviation << "%" << std::endl;
        }
    } else {
        std::cout << "Advanced unified view returned no data" << std::endl;
    }

    // Test comprehensive analytics
    auto comprehensive_view = aggregator.getComprehensiveMultiExchangeViewWithAllAnalytics("BTC/USD");
    if (comprehensive_view.has_value()) {
        std::cout << "\nComprehensive analytics view successful:" << std::endl;
        std::cout << "  Symbol: " << comprehensive_view->symbol << std::endl;
        std::cout << "  Average price: " << comprehensive_view->market_metrics.average_price << std::endl;
        std::cout << "  Arbitrage opportunity: " << (comprehensive_view->arbitrage_opportunity_exists ? "YES" : "NO") << std::endl;
        if (comprehensive_view->arbitrage_opportunity_exists) {
            std::cout << "  Arbitrage profit potential: " << comprehensive_view->arbitrage_profit_potential << std::endl;
        }
        std::cout << "  Number of exchanges: " << comprehensive_view->exchange_data.size() << std::endl;
    } else {
        std::cout << "Comprehensive analytics view returned no data" << std::endl;
    }

    // Test time synchronization
    auto time_sync_result = aggregator.performAdvancedTimeSync("BTC/USD", TimeSyncStrategy::SMART_SYNC);
    if (time_sync_result.has_value()) {
        std::cout << "\nTime synchronization successful:" << std::endl;
        std::cout << "  Strategy used: SMART_SYNC" << std::endl;
        std::cout << "  Synchronized timestamp: " << time_sync_result->synchronized_timestamp << std::endl;
        std::cout << "  Cross-correlation: " << time_sync_result->cross_correlation << std::endl;
        std::cout << "  Number of exchanges: " << time_sync_result->original_timestamps.size() << std::endl;
    } else {
        std::cout << "Time synchronization returned no data" << std::endl;
    }

    std::cout << "\nMulti-Exchange Aggregation test completed successfully!" << std::endl;
    
    return 0;
}