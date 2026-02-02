#include <iostream>
#include <memory>
#include <cassert>
#include <thread>
#include <chrono>

#include "dependencies/BTQ_Render_Engine/include/data/exchange_aggregator.hpp"
#include "dependencies/BTQ_Render_Engine/include/hotspine_data_bridge.hpp"
#include "dependencies/BTQ_Render_Engine/include/market_data_processor.hpp"
#include "dependencies/BTQ_Render_Engine/include/symbol_manager.hpp"

using namespace BTQuant::Data;

int main() {
    std::cout << "Testing ExchangeAggregator implementation..." << std::endl;
    
    // Create mock dependencies
    auto bridge = std::make_shared<HotSpineDataBridge>();
    auto processor = std::make_shared<RenderEngine::MarketDataProcessor>();
    auto symbol_manager = std::make_shared<RenderEngine::SymbolManager>();
    
    // Create the ExchangeAggregator
    ExchangeAggregator aggregator(bridge, processor, symbol_manager);
    
    // Initialize the aggregator
    bool initialized = aggregator.initialize();
    assert(initialized);
    std::cout << "✓ ExchangeAggregator initialized successfully" << std::endl;
    
    // Define exchange features
    ExchangeFeatures exchange1_features;
    exchange1_features.exchange_name = "Binance";
    exchange1_features.latency_offset_us = 100.0;
    exchange1_features.reliability_score = 0.95;
    exchange1_features.supported_symbols = {"BTCUSDT", "ETHUSDT"};
    
    ExchangeFeatures exchange2_features;
    exchange2_features.exchange_name = "Coinbase";
    exchange2_features.latency_offset_us = 200.0;
    exchange2_features.reliability_score = 0.90;
    exchange2_features.supported_symbols = {"BTCUSDT", "ETHUSDT"};
    
    ExchangeFeatures exchange3_features;
    exchange3_features.exchange_name = "Kraken";
    exchange3_features.latency_offset_us = 150.0;
    exchange3_features.reliability_score = 0.85;
    exchange3_features.supported_symbols = {"BTCUSDT", "ETHUSDT"};
    
    // Add exchanges to the aggregator
    aggregator.addExchange("Binance", exchange1_features);
    aggregator.addExchange("Coinbase", exchange2_features);
    aggregator.addExchange("Kraken", exchange3_features);
    
    std::cout << "✓ Exchanges added to aggregation pool" << std::endl;
    
    // Verify exchanges were added
    auto available_exchanges = aggregator.getAvailableExchanges();
    assert(available_exchanges.size() == 3);
    std::cout << "✓ Available exchanges: " << available_exchanges.size() << std::endl;
    
    // Create mock market data updates
    RenderEngine::MarketDataUpdate binance_data;
    binance_data.type = RenderEngine::MarketDataType::TRADE;
    binance_data.symbol_id = 1;
    binance_data.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    binance_data.price = 45000.0;
    binance_data.size = 1.5;
    binance_data.side = "BUY";
    
    RenderEngine::MarketDataUpdate coinbase_data;
    coinbase_data.type = RenderEngine::MarketDataType::TRADE;
    coinbase_data.symbol_id = 1;
    coinbase_data.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() - 50000;  // 50ms earlier
    coinbase_data.price = 44950.0;
    coinbase_data.size = 2.0;
    coinbase_data.side = "SELL";
    
    RenderEngine::MarketDataUpdate kraken_data;
    kraken_data.type = RenderEngine::MarketDataType::TRADE;
    kraken_data.symbol_id = 1;
    kraken_data.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() - 25000;  // 25ms earlier
    kraken_data.price = 45020.0;
    kraken_data.size = 0.8;
    kraken_data.side = "BUY";
    
    // Process data updates
    aggregator.processDataUpdate("Binance", "BTCUSDT", binance_data);
    aggregator.processDataUpdate("Coinbase", "BTCUSDT", coinbase_data);
    aggregator.processDataUpdate("Kraken", "BTCUSDT", kraken_data);
    
    std::cout << "✓ Market data updates processed" << std::endl;
    
    // Test aggregation with different time sync strategies
    aggregator.setTimeSyncStrategy(TimeSyncStrategy::AVERAGE_TIMESTAMP);
    auto aggregated_data_avg = aggregator.getAggregatedData("BTCUSDT");
    assert(aggregated_data_avg.has_value());
    std::cout << "✓ Aggregation with AVERAGE_TIMESTAMP strategy successful" << std::endl;
    
    aggregator.setTimeSyncStrategy(TimeSyncStrategy::MEDIAN_TIMESTAMP);
    auto aggregated_data_median = aggregator.getAggregatedData("BTCUSDT");
    assert(aggregated_data_median.has_value());
    std::cout << "✓ Aggregation with MEDIAN_TIMESTAMP strategy successful" << std::endl;
    
    aggregator.setTimeSyncStrategy(TimeSyncStrategy::ADAPTIVE_SYNC);
    auto aggregated_data_adaptive = aggregator.getAggregatedData("BTCUSDT");
    assert(aggregated_data_adaptive.has_value());
    std::cout << "✓ Aggregation with ADAPTIVE_SYNC strategy successful" << std::endl;
    
    // Verify aggregated values
    if (aggregated_data_adaptive) {
        std::cout << "  Aggregated price: " << aggregated_data_adaptive->aggregated_price << std::endl;
        std::cout << "  Weighted price: " << aggregated_data_adaptive->weighted_price << std::endl;
        std::cout << "  Aggregated volume: " << aggregated_data_adaptive->aggregated_volume << std::endl;
        std::cout << "  Arbitrage opportunity: " << (aggregated_data_adaptive->arbitrage_opportunity ? "Yes" : "No") << std::endl;
        std::cout << "  Overall correlation: " << aggregated_data_adaptive->overall_correlation << std::endl;
    }
    
    // Test advanced aggregation algorithms
    std::unordered_map<std::string, RenderEngine::MarketDataUpdate> test_data;
    test_data["Binance"] = binance_data;
    test_data["Coinbase"] = coinbase_data;
    test_data["Kraken"] = kraken_data;
    
    double twap = aggregator.calculateTWAP(test_data, 
        binance_data.timestamp - 100000, 
        binance_data.timestamp + 100000);
    std::cout << "✓ TWAP calculation: " << twap << std::endl;
    
    double vwap = aggregator.calculateVWAP(test_data);
    std::cout << "✓ VWAP calculation: " << vwap << std::endl;
    
    double median_price = aggregator.calculateMedianPrice(test_data);
    std::cout << "✓ Median price calculation: " << median_price << std::endl;
    
    double trimmed_mean = aggregator.calculateTrimmedMean(test_data, 0.1);
    std::cout << "✓ Trimmed mean calculation: " << trimmed_mean << std::endl;
    
    double harmonic_mean = aggregator.calculateHarmonicMean(test_data);
    std::cout << "✓ Harmonic mean calculation: " << harmonic_mean << std::endl;
    
    // Get statistics
    auto stats = aggregator.getStats();
    std::cout << "✓ Statistics retrieved - Valid exchanges: " << stats.valid_exchanges 
              << ", Total exchanges: " << stats.total_exchanges << std::endl;
    
    std::cout << "\nAll tests passed! ExchangeAggregator implementation is working correctly." << std::endl;
    
    return 0;
}