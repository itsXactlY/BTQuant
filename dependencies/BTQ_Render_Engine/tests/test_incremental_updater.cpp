#include "data/incremental_updater.hpp"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "Testing Incremental Updater..." << std::endl;
    
    // Create a symbol analytics object
    BTQuant::RenderEngine::SymbolAnalytics symbol_data;
    
    // Create a sample trade
    BTQuant::RenderEngine::TradeData trade;
    trade.symbol_id = 1;
    trade.timestamp = 1000000;  // 1 second in microseconds
    trade.price = 100.0;
    trade.size = 10.0;
    trade.is_buy = true;
    
    // Test initial state
    assert(symbol_data.trade_count == 0);
    assert(symbol_data.running_total_volume == 0.0);
    assert(symbol_data.running_total_price_volume == 0.0);
    
    // Process the first trade incrementally
    BTQuant::RenderEngine::processTradeIncrementally(symbol_data, trade);
    
    // Check that the analytics were updated correctly
    assert(symbol_data.trade_count == 1);
    assert(symbol_data.running_total_volume == 10.0);
    assert(symbol_data.running_total_price_volume == 1000.0);  // 100.0 * 10.0
    assert(symbol_data.vwap == 100.0);  // 1000.0 / 10.0
    
    // Add another trade
    trade.price = 105.0;
    trade.size = 20.0;
    trade.timestamp = 1001000;  // 1 second later
    
    BTQuant::RenderEngine::processTradeIncrementally(symbol_data, trade);
    
    // Check that the analytics were updated correctly
    assert(symbol_data.trade_count == 2);
    assert(symbol_data.running_total_volume == 30.0);  // 10.0 + 20.0
    assert(symbol_data.running_total_price_volume == 3100.0);  // 1000.0 + (105.0 * 20.0)
    assert(symbol_data.vwap == 3100.0 / 30.0);  // ~103.33
    
    std::cout << "All tests passed!" << std::endl;
    
    return 0;
}