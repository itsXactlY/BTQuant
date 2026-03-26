#include <iostream>
#include "market_data_collector.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

int main() {
    try {
        // Test 1: Load configuration with MS SQL disabled and exclusive hotswap enabled
        std::cout << "Test 1: Loading configuration with MS SQL disabled and exclusive hotswap enabled..." << std::endl;
        
        std::ifstream in("test_config.json");
        if (!in) {
            throw std::runtime_error("Cannot open test config file");
        }
        json j;
        in >> j;

        MarketDataCollector::Config cfg;
        cfg.db_connection_string = "test_connection_string";
        cfg.timeframes = {"1m", "5m"};
        cfg.enable_mssql = j.value("enable_mssql", true);
        cfg.enable_exclusive_hotspine = j.value("enable_exclusive_hotspine", false);
        
        std::cout << "Configuration loaded:" << std::endl;
        std::cout << "  enable_mssql: " << (cfg.enable_mssql ? "true" : "false") << std::endl;
        std::cout << "  enable_exclusive_hotspine: " << (cfg.enable_exclusive_hotspine ? "true" : "false") << std::endl;
        
        // Test 2: Create MarketDataCollector with the configuration
        std::cout << "\nTest 2: Creating MarketDataCollector with configuration..." << std::endl;
        MarketDataCollector collector(cfg);
        
        std::cout << "MarketDataCollector created successfully!" << std::endl;
        std::cout << "Configuration test passed!" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}