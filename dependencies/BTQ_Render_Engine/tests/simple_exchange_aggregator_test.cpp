#include <iostream>
#include <memory>

#include "../include/data/exchange_aggregator.hpp"
#include "../include/market_data_processor.hpp"
#include "../include/symbol_manager.hpp"

int main() {
    std::cout << "Testing Exchange Aggregator..." << std::endl;
    
    // Create mock/shared objects for testing
    auto processor = std::make_shared<BTQuant::RenderEngine::MarketDataProcessor>();
    auto symbol_manager = std::make_shared<BTQuant::RenderEngine::SymbolManager>();
    
    auto aggregator = std::make_unique<BTQuant::Data::ExchangeAggregator>(processor, symbol_manager);
    
    if (!aggregator->initialize()) {
        std::cerr << "Failed to initialize ExchangeAggregator" << std::endl;
        return 1;
    }
    
    std::cout << "ExchangeAggregator initialized successfully" << std::endl;
    
    // Add an exchange
    BTQuant::Data::ExchangeFeatures features;
    features.exchange_name = "binance";
    features.latency_offset_us = 100.0;
    features.reliability_score = 0.95;
    
    aggregator->addExchange("binance", features);
    
    std::cout << "Added exchange: binance" << std::endl;
    
    // Verify exchange was added
    auto exchanges = aggregator->getAvailableExchanges();
    if (exchanges.size() == 1 && exchanges[0] == "binance") {
        std::cout << "SUCCESS: Exchange added correctly" << std::endl;
    } else {
        std::cout << "FAILURE: Exchange not added correctly" << std::endl;
        return 1;
    }
    
    // Create mock market data
    BTQuant::RenderEngine::MarketDataUpdate update;
    update.type = BTQuant::RenderEngine::MarketDataType::TRADE;
    update.symbol_id = 1;
    update.timestamp = 1704067200000000ULL;
    update.price = 40000.0;
    update.size = 1.0;
    update.side = "buy";
    
    // Process data update
    aggregator->processDataUpdate("binance", "BTCUSDT", update);
    
    std::cout << "Processed data update for BTCUSDT on binance" << std::endl;
    
    // Get aggregated data
    auto aggregated = aggregator->getAggregatedData("BTCUSDT");
    if (aggregated) {
        std::cout << "SUCCESS: Retrieved aggregated data for BTCUSDT" << std::endl;
        std::cout << "Symbol: " << aggregated->symbol << std::endl;
        std::cout << "Number of exchanges with data: " << aggregated->exchange_data.size() << std::endl;
    } else {
        std::cout << "FAILURE: Could not retrieve aggregated data" << std::endl;
        return 1;
    }
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
}