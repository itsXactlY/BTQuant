/**
 * @file test_dynamic_config.cpp
 * @brief Integration test for MarketDataCollector dynamic configuration
 * 
 * This test verifies that MarketDataCollector properly integrates with:
 * - ConfigLoader for configuration management
 * - SymbolRegistry for symbol management
 * - HotSpine service for shared memory communication
 * - ExchangeConnectionManager for dynamic subscriptions
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <memory>
#include <cstdlib>

// Include MarketDataCollector headers
#include "market_data_collector.h"

// Include configuration and symbol registry from tests/new
#include "../../../tests/new/include/config/config_loader.hpp"
#include "../../../tests/new/include/symbol_registry.hpp"
#include "../../../tests/new/include/hotspine_extended_reader.hpp"

using namespace BTQuant;
using namespace BTQuant::Config;

// Helper for timestamped logging
static std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_time = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::tm tm = *std::localtime(&now_time);
    char buffer[64];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm);
    
    char ms_buffer[10];
    snprintf(ms_buffer, sizeof(ms_buffer), "%03d", static_cast<int>(now_ms.count()));
    
    return std::string(buffer) + "." + ms_buffer;
}

#define LOG_INFO(msg) std::cout << "[" << getCurrentTimestamp() << "][INFO] " << msg << std::endl
#define LOG_WARN(msg) std::cout << "[" << getCurrentTimestamp() << "][WARN] " << msg << std::endl
#define LOG_ERROR(msg) std::cout << "[" << getCurrentTimestamp() << "][ERROR] " << msg << std::endl

// ============================================================================
// Test Result Tracking
// ============================================================================

struct TestResult {
    std::string test_name;
    bool passed;
    std::string message;
};

std::vector<TestResult> test_results;

void recordTestResult(const std::string& test_name, bool passed, const std::string& message) {
    test_results.push_back({test_name, passed, message});
    std::cout << "  [" << (passed ? "PASS" : "FAIL") << "] " << test_name << ": " << message << std::endl;
}

// ============================================================================
// Test Functions
// ============================================================================

bool testConfigLoaderInitialization() {
    LOG_INFO("Testing ConfigLoader initialization...");
    
    try {
        auto& config = ConfigLoader::instance();
        bool initialized = config.initialize();
        
        if (!initialized) {
            recordTestResult("ConfigLoader Init", false, "Failed to initialize");
            return false;
        }
        
        bool loaded = config.load();
        if (!loaded) {
            recordTestResult("ConfigLoader Load", false, "Failed to load configuration");
            return false;
        }
        
        recordTestResult("ConfigLoader Init", true, "ConfigLoader initialized and loaded successfully");
        return true;
    } catch (const std::exception& e) {
        recordTestResult("ConfigLoader Init", false, std::string("Exception: ") + e.what());
        return false;
    }
}

bool testConfigLoaderValues() {
    LOG_INFO("Testing ConfigLoader value retrieval...");
    
    auto& config = ConfigLoader::instance();
    
    // Test shared_memory.name
    auto shm_name = config.get_as<std::string>("shared_memory", "name");
    if (shm_name) {
        recordTestResult("ConfigLoader Get SHM Name", true, "Value: " + *shm_name);
    } else {
        recordTestResult("ConfigLoader Get SHM Name", false, "shared_memory.name not found");
        return false;
    }
    
    // Test exchanges.enabled
    auto exchanges = config.get_as<std::vector<std::string>>("exchanges", "enabled");
    if (exchanges) {
        recordTestResult("ConfigLoader Get Exchanges", true, 
            "Found " + std::to_string(exchanges->size()) + " exchanges");
    } else {
        recordTestResult("ConfigLoader Get Exchanges", false, "exchanges.enabled not found");
        return false;
    }
    
    // Test monitoring.symbols
    auto symbols = config.get_as<std::vector<std::string>>("monitoring", "symbols");
    if (symbols) {
        recordTestResult("ConfigLoader Get Symbols", true, 
            "Found " + std::to_string(symbols->size()) + " symbols");
    } else {
        recordTestResult("ConfigLoader Get Symbols", false, "monitoring.symbols not found");
        return false;
    }
    
    return true;
}

bool testEnvironmentVariableOverride() {
    LOG_INFO("Testing environment variable override...");
    
    // Set environment variable
    #ifdef _WIN32
    _putenv_s("BTQ_HOTSPINE_SHM_NAME", "/test_hotspine_env");
    #else
    setenv("BTQ_HOTSPINE_SHM_NAME", "/test_hotspine_env", 1);
    #endif
    
    auto& config = ConfigLoader::instance();
    
    // Test that environment variable can be read
    std::string env_value = ConfigLoader::get_env_var("BTQ_HOTSPINE_SHM_NAME");
    if (env_value == "/test_hotspine_env") {
        recordTestResult("Env Override", true, "Environment variable read correctly");
    } else {
        recordTestResult("Env Override", false, "Environment variable not read correctly");
        return false;
    }
    
    // Clean up
    #ifdef _WIN32
    _putenv_s("BTQ_HOTSPINE_SHM_NAME", "");
    #else
    unsetenv("BTQ_HOTSPINE_SHM_NAME");
    #endif
    
    return true;
}

bool testSymbolRegistry() {
    LOG_INFO("Testing SymbolRegistry integration...");
    
    try {
        auto& registry = SymbolRegistry::instance();
        
        // Get all exchanges
        auto exchanges = registry.get_exchanges();
        LOG_INFO("  SymbolRegistry has " << std::to_string(exchanges.size()) << " exchanges");
        
        // Get symbols for binance
        auto binance_symbols = registry.get_exchange_symbols("binance");
        LOG_INFO("  Binance has " << std::to_string(binance_symbols.size()) << " symbols");
        
        // Test symbol registration
        uint32_t test_id = registry.register_symbol("test_exchange", "TEST_SYMBOL");
        LOG_INFO("  Registered test symbol with ID: " << std::to_string(test_id));
        
        // Verify symbol lookup
        auto symbol_info = registry.get_symbol_info(test_id);
        if (symbol_info && symbol_info->exchange == "test_exchange" && symbol_info->symbol == "TEST_SYMBOL") {
            recordTestResult("SymbolRegistry Lookup", true, "Symbol lookup successful");
        } else {
            recordTestResult("SymbolRegistry Lookup", false, "Symbol lookup failed");
            return false;
        }
        
        recordTestResult("SymbolRegistry Init", true, 
            "Registry initialized with " + std::to_string(exchanges.size()) + " exchanges");
        return true;
    } catch (const std::exception& e) {
        recordTestResult("SymbolRegistry Init", false, std::string("Exception: ") + e.what());
        return false;
    }
}

bool testMarketDataCollectorConstruction() {
    LOG_INFO("Testing MarketDataCollector dynamic construction...");
    
    try {
        // Create collector with dynamic configuration
        MarketDataCollector collector("/btquant_hotspine");
        
        recordTestResult("MarketDataCollector Init", true, "Collector created with dynamic config");
        return true;
    } catch (const std::exception& e) {
        recordTestResult("MarketDataCollector Init", false, std::string("Exception: ") + e.what());
        return false;
    }
}

bool testHotSpineServiceDiscovery() {
    LOG_INFO("Testing HotSpine service discovery...");
    
    try {
        MarketDataCollector collector("/btquant_hotspine");
        
        // Discover service info
        auto service_info = collector.discoverHotSpineService();
        
        LOG_INFO("  Shared Memory: " << service_info.shared_memory_path);
        LOG_INFO("  Is Available: " + std::string(service_info.is_available ? "YES" : "NO"));
        
        if (service_info.is_available) {
            recordTestResult("HotSpine Discovery", true, "Service discovered successfully");
        } else {
            recordTestResult("HotSpine Discovery", false, "Service not available");
            return false;
        }
        
        // Test health check
        auto health_status = collector.checkHotSpineHealth();
        LOG_INFO("  Health Status: " + std::to_string(static_cast<int>(health_status)));
        
        // Test statistics
        auto stats = collector.getHotSpineStatistics();
        LOG_INFO("  Attached: " + std::string(stats.is_attached ? "YES" : "NO"));
        
        return service_info.is_available;
    } catch (const std::exception& e) {
        recordTestResult("HotSpine Discovery", false, std::string("Exception: ") + e.what());
        return false;
    }
}

bool testExchangeValidation() {
    LOG_INFO("Testing exchange validation...");
    
    try {
        MarketDataCollector collector("/btquant_hotspine");
        
        // Get supported exchanges
        auto supported = collector.getSupportedExchanges();
        LOG_INFO("  Supported exchanges: " + std::to_string(supported.size()));
        for (const auto& ex : supported) {
            LOG_INFO("    - " + ex);
        }
        
        // Validate all exchanges
        auto validation_result = collector.validateAllExchanges();
        LOG_INFO("  Validation result: " + validation_result.message);
        
        if (validation_result.is_valid) {
            recordTestResult("Exchange Validation", true, 
                std::to_string(validation_result.valid_items.size()) + " exchanges valid");
        } else {
            recordTestResult("Exchange Validation", false, 
                std::to_string(validation_result.invalid_items.size()) + " exchanges invalid");
            return false;
        }
        
        return true;
    } catch (const std::exception& e) {
        recordTestResult("Exchange Validation", false, std::string("Exception: ") + e.what());
        return false;
    }
}

bool testSymbolValidation() {
    LOG_INFO("Testing symbol validation...");
    
    try {
        MarketDataCollector collector("/btquant_hotspine");
        
        // Test symbols for binance
        std::vector<std::string> test_symbols = {"BTCUSDT", "ETHUSDT", "SOLUSDT"};
        auto result = collector.validateExchangeSymbols("binance", test_symbols);
        
        LOG_INFO("  Validated " + std::to_string(result.valid_symbols) + "/" 
                 + std::to_string(result.total_symbols) + " symbols");
        
        if (result.is_valid) {
            recordTestResult("Symbol Validation", true, "All symbols validated");
        } else {
            recordTestResult("Symbol Validation", false, 
                std::to_string(result.invalid_items.size()) + " symbols invalid");
            return false;
        }
        
        // Test symbol ID resolution
        uint32_t btc_id = collector.resolveSymbolId("binance", "BTCUSDT");
        LOG_INFO("  BTCUSDT symbol ID: " + std::to_string(btc_id));
        
        if (btc_id > 0) {
            recordTestResult("Symbol ID Resolution", true, "ID: " + std::to_string(btc_id));
        } else {
            recordTestResult("Symbol ID Resolution", false, "Symbol ID not resolved");
            return false;
        }
        
        return true;
    } catch (const std::exception& e) {
        recordTestResult("Symbol Validation", false, std::string("Exception: ") + e.what());
        return false;
    }
}

bool testValidationSummary() {
    LOG_INFO("Testing validation summary...");
    
    try {
        MarketDataCollector collector("/btquant_hotspine");
        
        auto summary = collector.getValidationSummary();
        
        LOG_INFO("  Total Exchanges: " + std::to_string(summary.total_exchanges));
        LOG_INFO("  Valid Exchanges: " + std::to_string(summary.valid_exchanges));
        LOG_INFO("  Total Symbols: " + std::to_string(summary.total_symbols));
        LOG_INFO("  Valid Symbols: " + std::to_string(summary.valid_symbols));
        
        recordTestResult("Validation Summary", true, 
            std::to_string(summary.valid_exchanges) + "/" + std::to_string(summary.total_exchanges) 
            + " exchanges, " + std::to_string(summary.valid_symbols) + "/" 
            + std::to_string(summary.total_symbols) + " symbols");
        
        return true;
    } catch (const std::exception& e) {
        recordTestResult("Validation Summary", false, std::string("Exception: ") + e.what());
        return false;
    }
}

bool testHotSpineExtendedReader() {
    LOG_INFO("Testing HotSpineExtendedReader compatibility...");
    
    try {
        HotSpineExtendedReader reader("/btquant_hotspine");
        
        if (reader.is_attached()) {
            recordTestResult("HotSpineExtendedReader", true, "Reader attached to shared memory");
        } else {
            recordTestResult("HotSpineExtendedReader", false, "Failed to attach to shared memory");
            return false;
        }
        
        // Test price lookup (will fail if no data, but should not crash)
        auto btc_price = reader.get_latest_price("binance", "BTCUSDT");
        if (btc_price) {
            LOG_INFO("  BTC price: " + std::to_string(*btc_price));
        } else {
            LOG_INFO("  BTC price: no data");
        }
        
        return true;
    } catch (const std::exception& e) {
        recordTestResult("HotSpineExtendedReader", false, std::string("Exception: ") + e.what());
        return false;
    }
}

bool testRuntimeSymbolAddition() {
    LOG_INFO("Testing runtime symbol addition...");
    
    try {
        MarketDataCollector collector("/btquant_hotspine");
        
        // Add symbol at runtime
        bool added = collector.addSymbolAtRuntime("binance", "ADAUSDT");
        
        if (added) {
            recordTestResult("Runtime Symbol Addition", true, "Symbol added successfully");
        } else {
            recordTestResult("Runtime Symbol Addition", false, "Failed to add symbol");
            return false;
        }
        
        // Verify symbol exists
        bool exists = collector.symbolExists("binance", "ADAUSDT");
        if (exists) {
            recordTestResult("Symbol Existence Check", true, "Symbol exists after addition");
        } else {
            recordTestResult("Symbol Existence Check", false, "Symbol not found after addition");
            return false;
        }
        
        // Get symbol info
        auto info = collector.getSymbolInfo("binance", "ADAUSDT");
        if (info) {
            LOG_INFO("  Symbol ID: " + std::to_string(info->id));
            LOG_INFO("  Exchange: " + info->exchange);
            LOG_INFO("  Symbol: " + info->symbol);
            recordTestResult("Symbol Info Retrieval", true, "Info retrieved for added symbol");
        } else {
            recordTestResult("Symbol Info Retrieval", false, "Failed to get symbol info");
            return false;
        }
        
        return true;
    } catch (const std::exception& e) {
        recordTestResult("Runtime Symbol Addition", false, std::string("Exception: ") + e.what());
        return false;
    }
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  MarketDataCollector Dynamic Configuration Integration Test" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << std::endl;
    
    int passed = 0;
    int failed = 0;
    
    // Test 1: ConfigLoader initialization
    std::cout << "\n--- Test 1: ConfigLoader Integration ---\n" << std::endl;
    if (testConfigLoaderInitialization() && testConfigLoaderValues() && testEnvironmentVariableOverride()) {
        passed += 3;
    } else {
        failed += 3;
    }
    
    // Test 2: SymbolRegistry integration
    std::cout << "\n--- Test 2: SymbolRegistry Integration ---\n" << std::endl;
    if (testSymbolRegistry()) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 3: MarketDataCollector construction
    std::cout << "\n--- Test 3: MarketDataCollector Construction ---\n" << std::endl;
    if (testMarketDataCollectorConstruction()) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 4: HotSpine service discovery
    std::cout << "\n--- Test 4: HotSpine Service Discovery ---\n" << std::endl;
    if (testHotSpineServiceDiscovery()) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 5: Exchange validation
    std::cout << "\n--- Test 5: Exchange Validation ---\n" << std::endl;
    if (testExchangeValidation()) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 6: Symbol validation
    std::cout << "\n--- Test 6: Symbol Validation ---\n" << std::endl;
    if (testSymbolValidation()) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 7: Validation summary
    std::cout << "\n--- Test 7: Validation Summary ---\n" << std::endl;
    if (testValidationSummary()) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 8: HotSpineExtendedReader compatibility
    std::cout << "\n--- Test 8: HotSpineExtendedReader Compatibility ---\n" << std::endl;
    if (testHotSpineExtendedReader()) {
        passed++;
    } else {
        failed++;
    }
    
    // Test 9: Runtime symbol addition
    std::cout << "\n--- Test 9: Runtime Symbol Addition ---\n" << std::endl;
    if (testRuntimeSymbolAddition()) {
        passed++;
    } else {
        failed++;
    }
    
    // Print summary
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "  Test Summary" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "  Passed: " << passed << std::endl;
    std::cout << "  Failed: " << failed << std::endl;
    std::cout << "  Total:  " << (passed + failed) << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    // Print detailed results
    std::cout << "\n--- Detailed Test Results ---\n" << std::endl;
    for (const auto& result : test_results) {
        std::cout << "  [" << std::setw(4) << (result.passed ? "PASS" : "FAIL") << "] "
                  << result.test_name << std::endl;
        std::cout << "         " << result.message << std::endl;
    }
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    if (failed == 0) {
        std::cout << "  ALL TESTS PASSED" << std::endl;
    } else {
        std::cout << "  SOME TESTS FAILED" << std::endl;
    }
    std::cout << std::string(70, '=') << std::endl;
    
    return failed == 0 ? 0 : 1;
}
