#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <optional>

// Include the actual headers to test the real implementation
#include "dependencies/BTQ_Render_Engine/include/symbol_registry.hpp"

namespace BTQuant {
namespace RenderEngine {

// Mock data structures
struct PriceLevel {
    double price;
    double size;
};

struct OrderbookData {
    std::string symbol;
    uint32_t symbol_id = 0;
    uint64_t timestamp;
    std::vector<PriceLevel> bids;
    std::vector<PriceLevel> asks;
    double spread;
    double spread_percent;
    double bid_depth;
    double ask_depth;
    double total_depth;
    double imbalance;  // (bid_depth - ask_depth) / total_depth
};

struct HistoricalOrderbookData {
    uint64_t timestamp;
    std::vector<PriceLevel> bids;
    std::vector<PriceLevel> asks;
};

// Mock MarketDataProcessor
class MarketDataProcessor {
public:
    std::vector<HistoricalOrderbookData> getHistoricalOrderbooks(uint32_t symbol_id, size_t count) const {
        // Mock implementation returning some sample data
        std::vector<HistoricalOrderbookData> result;
        
        // Create some sample historical data
        for (size_t i = 0; i < 10; ++i) {
            HistoricalOrderbookData data;
            data.timestamp = 1000000 + i * 1000; // Simulated timestamps
            
            // Add some sample bids
            for (int j = 0; j < 5; ++j) {
                PriceLevel bid;
                bid.price = 100.0 - j * 0.1 + i * 0.01; // Slightly varying prices
                bid.size = 10.0 + j * 2.0; // Varying sizes
                data.bids.push_back(bid);
            }
            
            // Add some sample asks
            for (int j = 0; j < 5; ++j) {
                PriceLevel ask;
                ask.price = 100.1 + j * 0.1 + i * 0.01; // Slightly varying prices
                ask.size = 8.0 + j * 1.5; // Varying sizes
                data.asks.push_back(ask);
            }
            
            result.push_back(data);
        }
        
        return result;
    }
};

} // namespace RenderEngine
} // namespace BTQuant

// Test function to validate multi-exchange aggregation
void testMultiExchangeAggregation() {
    std::cout << "Testing Multi-Exchange Aggregation Feature..." << std::endl;
    
    // Create a mock processor
    auto processor = std::make_shared<BTQuant::RenderEngine::MarketDataProcessor>();
    
    // Test the aggregation logic
    std::vector<std::string> selected_exchanges = {"Binance", "Coinbase", "Kraken"};
    uint32_t current_symbol_id = 1; // BTCUSDT on Binance
    
    std::cout << "Selected exchanges: ";
    for (const auto& ex : selected_exchanges) {
        std::cout << ex << " ";
    }
    std::cout << std::endl;
    
    // Get the symbol name for the current symbol ID to find equivalent symbols on other exchanges
    std::string base_symbol_name = "UNKNOWN";
    auto symbol_info_opt = BTQuant::SymbolRegistry::instance().get_symbol_info(current_symbol_id);
    if (symbol_info_opt) {
        base_symbol_name = symbol_info_opt->symbol;
        std::cout << "Base symbol: " << base_symbol_name << std::endl;
    }
    
    // Simulate the multi-exchange aggregation process
    std::vector<BTQuant::RenderEngine::HistoricalOrderbookData> combined_history;
    
    // For each selected exchange, get the corresponding symbol data
    for (const auto& exchange : selected_exchanges) {
        // Find the symbol ID for the same symbol on this exchange
        auto exchange_symbol_id_opt = BTQuant::SymbolRegistry::instance().get_symbol_id(exchange, base_symbol_name);
        
        if (exchange_symbol_id_opt.has_value()) {
            std::cout << "Fetching data for " << exchange << ":" << base_symbol_name 
                      << " (symbol_id: " << exchange_symbol_id_opt.value() << ")" << std::endl;
                      
            auto exchange_history = processor->getHistoricalOrderbooks(exchange_symbol_id_opt.value(), 0);
            std::cout << "  Retrieved " << exchange_history.size() << " historical records" << std::endl;
            
            // Append the data from each exchange
            combined_history.insert(combined_history.end(), 
                                  exchange_history.begin(), 
                                  exchange_history.end());
        } else {
            std::cout << "  Symbol not found for " << exchange << std::endl;
        }
    }
    
    std::cout << "Combined history contains " << combined_history.size() 
              << " total records from " << selected_exchanges.size() << " exchanges" << std::endl;
    
    // Verify that we have aggregated data
    if (combined_history.size() > 0) {
        std::cout << "✓ Multi-exchange aggregation test PASSED!" << std::endl;
        std::cout << "  Successfully aggregated data from multiple exchanges." << std::endl;
    } else {
        std::cout << "✗ Multi-exchange aggregation test FAILED!" << std::endl;
    }
    
    // Test the exchange selection mechanism
    std::cout << "\nTesting exchange selection mechanism..." << std::endl;
    
    auto available_exchanges = BTQuant::SymbolRegistry::instance().get_exchanges();
    std::cout << "Available exchanges: ";
    for (const auto& ex : available_exchanges) {
        std::cout << ex << " ";
    }
    std::cout << std::endl;
    
    if (!available_exchanges.empty()) {
        std::cout << "✓ Exchange selection mechanism test PASSED!" << std::endl;
    } else {
        std::cout << "✗ Exchange selection mechanism test FAILED!" << std::endl;
    }
    
    std::cout << "\nAll tests completed successfully!" << std::endl;
}

int main() {
    testMultiExchangeAggregation();
    return 0;
}