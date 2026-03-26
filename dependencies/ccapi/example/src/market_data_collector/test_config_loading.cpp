#include <iostream>
#include <fstream>
#include <cstdio>
#include "market_data_collector.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

// Forward declaration of the loadConfig function from main.cpp
MarketDataCollector::Config loadConfig(const std::string& path);

int main() {
    std::cout << "Testing configuration loading..." << std::endl;
    
    // Test 1: Load config with existing file
    std::cout << "\nTest 1: Loading config from existing file (config.json)" << std::endl;
    try {
        auto config1 = loadConfig("config.json");
        std::cout << "✓ Successfully loaded config from file" << std::endl;
        std::cout << "  enable_mssql: " << (config1.enable_mssql ? "true" : "false") << std::endl;
        std::cout << "  enable_exclusive_hotspine: " << (config1.enable_exclusive_hotspine ? "true" : "false") << std::endl;
        std::cout << "  exchanges: " << config1.exchanges.size() << std::endl;
        std::cout << "  timeframes: ";
        for (const auto& tf : config1.timeframes) {
            std::cout << tf << " ";
        }
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ Failed to load config: " << e.what() << std::endl;
    }
    
    // Test 2: Load config with non-existing file (should use defaults)
    std::cout << "\nTest 2: Loading config from non-existing file (should use defaults)" << std::endl;
    try {
        auto config2 = loadConfig("nonexistent_config.json");
        std::cout << "✓ Successfully loaded default config" << std::endl;
        std::cout << "  enable_mssql: " << (config2.enable_mssql ? "true" : "false") << std::endl;
        std::cout << "  enable_exclusive_hotspine: " << (config2.enable_exclusive_hotspine ? "true" : "false") << std::endl;
        std::cout << "  exchanges: " << config2.exchanges.size() << std::endl;
        std::cout << "  timeframes: ";
        for (const auto& tf : config2.timeframes) {
            std::cout << tf << " ";
        }
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ Failed to load default config: " << e.what() << std::endl;
    }
    
    // Test 3: Test with test_config.json (which has different settings)
    std::cout << "\nTest 3: Loading config from test_config.json (different settings)" << std::endl;
    try {
        auto config3 = loadConfig("test_config.json");
        std::cout << "✓ Successfully loaded config from test_config.json" << std::endl;
        std::cout << "  enable_mssql: " << (config3.enable_mssql ? "true" : "false") << std::endl;
        std::cout << "  enable_exclusive_hotspine: " << (config3.enable_exclusive_hotspine ? "true" : "false") << std::endl;
        std::cout << "  exchanges: " << config3.exchanges.size() << std::endl;
        std::cout << "  timeframes: ";
        for (const auto& tf : config3.timeframes) {
            std::cout << tf << " ";
        }
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ Failed to load test config: " << e.what() << std::endl;
    }
    
    std::cout << "\nConfiguration loading tests completed!" << std::endl;
    return 0;
}