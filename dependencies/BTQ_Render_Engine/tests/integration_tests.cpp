/**
 * BTQuant Integration Testing Suite
 * 
 * Comprehensive integration testing for the BTQuant Advanced Vulkan Dashboard
 * 
 * Test Coverage:
 * - HotSpine data integration validation
 * - Symbol registry integration testing
 * - Market data processor integration
 * - Vulkan pipeline integration testing
 * - Multi-threading coordination validation
 * - Cross-component communication testing
 */

#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <memory>
#include <atomic>
#include <random>
#include <fstream>
#include <iomanip>
#include <functional>
#include <future>

#include "../include/symbol_manager.hpp"
#include "../include/market_data_processor.hpp"
#include "../include/performance_monitor.hpp"
#include "../include/dashboard_config.hpp"
#include "../include/hotspine_data_bridge.hpp"
#include "../include/data_visualization_engine.hpp"

namespace BTQuant {
namespace IntegrationTesting {

// ============================================================================
// Integration Test Framework
// ============================================================================

enum class IntegrationTestResult {
    PASS,
    FAIL,
    SKIP,
    ERROR
};

struct IntegrationTestCase {
    std::string name;
    std::string description;
    std::function<IntegrationTestResult()> test_function;
    std::string integration_area;
    std::vector<std::string> components;
    bool requires_hotspine = false;
};

struct IntegrationTestResults {
    std::string test_name;
    IntegrationTestResult result;
    std::string error_message;
    double execution_time_ms;
    std::vector<std::string> component_status;
};

class IntegrationTestSuite {
private:
    std::vector<IntegrationTestCase> test_cases_;
    std::vector<IntegrationTestResults> results_;
    
public:
    void register_test(const IntegrationTestCase& test_case) {
        test_cases_.push_back(test_case);
    }
    
    void run_all_tests() {
        std::cout << "=== BTQuant Integration Test Suite ===\n";
        std::cout << "Testing component integration and data flow\n";
        std::cout << "==========================================\n\n";
        
        for (const auto& test_case : test_cases_) {
            std::cout << "Testing: " << test_case.name << "\n";
            std::cout << "  Integration Area: " << test_case.integration_area << "\n";
            std::cout << "  Components: ";
            for (size_t i = 0; i < test_case.components.size(); ++i) {
                std::cout << test_case.components[i];
                if (i < test_case.components.size() - 1) std::cout << ", ";
            }
            std::cout << "\n";
            
            auto start_time = std::chrono::high_resolution_clock::now();
            IntegrationTestResults result;
            result.test_name = test_case.name;
            
            try {
                result.result = test_case.test_function();
            } catch (const std::exception& e) {
                result.result = IntegrationTestResult::ERROR;
                result.error_message = e.what();
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            result.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            
            results_.push_back(result);
            
            // Print result
            std::string status;
            switch (result.result) {
                case IntegrationTestResult::PASS: status = "✓ PASS"; break;
                case IntegrationTestResult::FAIL: status = "✗ FAIL"; break;
                case IntegrationTestResult::SKIP: status = "- SKIP"; break;
                case IntegrationTestResult::ERROR: status = "! ERROR"; break;
            }
            
            std::cout << "  Result: " << status;
            if (!result.error_message.empty()) {
                std::cout << " - " << result.error_message;
            }
            std::cout << "\n\n";
        }
        
        generate_integration_report();
    }
    
private:
    void generate_integration_report() {
        std::cout << "=== Integration Test Summary ===\n";
        
        int passed = 0, failed = 0, skipped = 0, errors = 0;
        double total_time = 0.0;
        
        for (const auto& result : results_) {
            total_time += result.execution_time_ms;
            switch (result.result) {
                case IntegrationTestResult::PASS: passed++; break;
                case IntegrationTestResult::FAIL: failed++; break;
                case IntegrationTestResult::SKIP: skipped++; break;
                case IntegrationTestResult::ERROR: errors++; break;
            }
        }
        
        std::cout << "Total Tests: " << results_.size() << "\n";
        std::cout << "Passed: " << passed << "\n";
        std::cout << "Failed: " << failed << "\n";
        std::cout << "Skipped: " << skipped << "\n";
        std::cout << "Errors: " << errors << "\n";
        std::cout << "Success Rate: " << std::fixed << std::setprecision(1) 
                  << (100.0 * passed / results_.size()) << "%\n";
        std::cout << "Total Time: " << std::fixed << std::setprecision(2) 
                  << (total_time / 1000.0) << " seconds\n\n";
        
        save_integration_report();
    }
    
    void save_integration_report() {
        std::ofstream report_file("integration_test_report.json");
        if (!report_file.is_open()) return;
        
        report_file << "{\n";
        report_file << "  \"timestamp\": \"" << std::time(nullptr) << "\",\n";
        report_file << "  \"test_suite\": \"BTQuant Integration Tests\",\n";
        report_file << "  \"total_tests\": " << results_.size() << ",\n";
        report_file << "  \"results\": [\n";
        
        for (size_t i = 0; i < results_.size(); ++i) {
            const auto& result = results_[i];
            const auto& test_case = test_cases_[i];
            
            report_file << "    {\n";
            report_file << "      \"test_name\": \"" << result.test_name << "\",\n";
            report_file << "      \"integration_area\": \"" << test_case.integration_area << "\",\n";
            report_file << "      \"components\": [";
            for (size_t j = 0; j < test_case.components.size(); ++j) {
                report_file << "\"" << test_case.components[j] << "\"";
                if (j < test_case.components.size() - 1) report_file << ", ";
            }
            report_file << "],\n";
            report_file << "      \"result\": \"";
            switch (result.result) {
                case IntegrationTestResult::PASS: report_file << "PASS"; break;
                case IntegrationTestResult::FAIL: report_file << "FAIL"; break;
                case IntegrationTestResult::SKIP: report_file << "SKIP"; break;
                case IntegrationTestResult::ERROR: report_file << "ERROR"; break;
            }
            report_file << "\",\n";
            report_file << "      \"execution_time_ms\": " << result.execution_time_ms << ",\n";
            report_file << "      \"error_message\": \"" << result.error_message << "\"\n";
            report_file << "    }";
            if (i < results_.size() - 1) report_file << ",";
            report_file << "\n";
        }
        
        report_file << "  ]\n";
        report_file << "}\n";
        report_file.close();
        
        std::cout << "Integration test report saved to integration_test_report.json\n";
    }
};

// ============================================================================
// HotSpine Integration Tests
// ============================================================================

IntegrationTestResult test_hotspine_symbol_registry_integration() {
    try {
        using namespace BTQuant::RenderEngine;
        
        // Test integration between HotSpine data bridge and symbol registry
        HotSpineDataBridge bridge;
        SymbolManager symbol_manager;
        
        // Initialize components
        if (!symbol_manager.initialize()) {
            return IntegrationTestResult::SKIP;
        }
        
        // Test symbol registration through both systems
        uint32_t symbol_id_1 = symbol_manager.registerSymbol("binance", "BTCUSD");
        uint32_t symbol_id_2 = symbol_manager.registerSymbol("okx", "ETHUSD");
        
        if (symbol_id_1 == 0 || symbol_id_2 == 0) {
            return IntegrationTestResult::FAIL;
        }
        
        // Verify symbol information consistency
        auto btc_info = symbol_manager.getSymbolInfo(symbol_id_1);
        if (!btc_info || btc_info->symbol != "BTCUSD" || btc_info->exchange != "binance") {
            return IntegrationTestResult::FAIL;
        }
        
        // Test exchange filtering
        auto binance_symbols = symbol_manager.getExchangeSymbols("binance");
        auto okx_symbols = symbol_manager.getExchangeSymbols("okx");
        
        if (binance_symbols.empty() || okx_symbols.empty()) {
            return IntegrationTestResult::FAIL;
        }
        
        // Test cross-component symbol lookup
        auto symbol_id_lookup = symbol_manager.getSymbolId("binance", "BTCUSD");
        if (!symbol_id_lookup || *symbol_id_lookup != symbol_id_1) {
            return IntegrationTestResult::FAIL;
        }
        
        return IntegrationTestResult::PASS;
        
    } catch (const std::exception& e) {
        std::cerr << "HotSpine-Symbol Registry integration test error: " << e.what() << std::endl;
        return IntegrationTestResult::ERROR;
    }
}

IntegrationTestResult test_data_processor_visualization_integration() {
    try {
        using namespace BTQuant::RenderEngine;
        
        // Test integration between market data processor and visualization engine
        MarketDataProcessor processor;
        SymbolManager symbol_manager;
        
        if (!symbol_manager.initialize()) {
            return IntegrationTestResult::SKIP;
        }
        
        // Register symbols for testing
        uint32_t btc_id = symbol_manager.registerSymbol("binance", "BTCUSD");
        uint32_t eth_id = symbol_manager.registerSymbol("binance", "ETHUSD");
        
        // Generate comprehensive test data
        auto base_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        // Process trade data
        for (int i = 0; i < 100; ++i) {
            MarketDataUpdate trade_update;
            trade_update.type = MarketDataType::TRADE;
            trade_update.symbol_id = (i % 2 == 0) ? btc_id : eth_id;
            trade_update.exchange = "binance";
            trade_update.symbol = (i % 2 == 0) ? "BTCUSD" : "ETHUSD";
            trade_update.timestamp_us = base_time + i;
            trade_update.local_timestamp_us = base_time + i;
            trade_update.price = 50000.0 + (i * 10.0) + (std::sin(i * 0.1) * 100.0);
            trade_update.size = 1.0 + (i % 5) * 0.2;
            trade_update.side = (i % 2 == 0) ? "buy" : "sell";
            
            processor.processTradeUpdate(trade_update);
        }
        
        // Process orderbook data
        for (int symbol_idx = 0; symbol_idx < 2; ++symbol_idx) {
            uint32_t symbol_id = (symbol_idx == 0) ? btc_id : eth_id;
            
            MarketDataUpdate orderbook_update;
            orderbook_update.type = MarketDataType::ORDERBOOK;
            orderbook_update.symbol_id = symbol_id;
            orderbook_update.exchange = "binance";
            orderbook_update.symbol = (symbol_idx == 0) ? "BTCUSD" : "ETHUSD";
            orderbook_update.timestamp_us = base_time + 1000;
            orderbook_update.local_timestamp_us = base_time + 1000;
            
            // Generate orderbook levels
            for (int i = 0; i < 10; ++i) {
                PriceLevel bid, ask;
                bid.price = 50000.0 - (i * 10.0);
                bid.size = 1.0 + (i * 0.1);
                ask.price = 50010.0 + (i * 10.0);
                ask.size = 1.0 + (i * 0.1);
                
                orderbook_update.bids.push_back(bid);
                orderbook_update.asks.push_back(ask);
            }
            
            processor.processOrderbookUpdate(orderbook_update);
        }
        
        // Verify integrated analytics
        auto btc_analytics = processor.getSymbolAnalytics(btc_id);
        auto eth_analytics = processor.getSymbolAnalytics(eth_id);
        
        if (btc_analytics.symbol_id != btc_id || eth_analytics.symbol_id != eth_id) {
            return IntegrationTestResult::FAIL;
        }
        
        // Verify trade processing
        if (btc_analytics.trade_count == 0 || eth_analytics.trade_count == 0) {
            return IntegrationTestResult::FAIL;
        }
        
        // Verify VWAP calculations
        if (btc_analytics.vwap <= 0 || eth_analytics.vwap <= 0) {
            return IntegrationTestResult::FAIL;
        }
        
        // Verify orderbook analytics
        if (btc_analytics.current_spread <= 0 || eth_analytics.current_spread <= 0) {
            return IntegrationTestResult::FAIL;
        }
        
        // Test market summary integration
        auto market_summary = processor.getMarketSummary();
        if (market_summary.total_symbols < 2) {
            return IntegrationTestResult::FAIL;
        }
        
        return IntegrationTestResult::PASS;
        
    } catch (const std::exception& e) {
        std::cerr << "Data Processor-Visualization integration test error: " << e.what() << std::endl;
        return IntegrationTestResult::ERROR;
    }
}

IntegrationTestResult test_performance_monitoring_integration() {
    try {
        using namespace BTQuant::RenderEngine;
        
        // Test integration between performance monitor and other components
        PerformanceMonitor monitor;
        MarketDataProcessor processor;
        SymbolManager symbol_manager;
        
        // Initialize components
        if (!symbol_manager.initialize()) {
            return IntegrationTestResult::SKIP;
        }
        
        if (!monitor.startMonitoring()) {
            return IntegrationTestResult::SKIP;
        }
        
        // Simulate integrated workload
        uint32_t symbol_id = symbol_manager.registerSymbol("test_exchange", "TESTSYMBOL");
        
        // Generate workload while monitoring
        for (int i = 0; i < 50; ++i) {
            // Simulate frame rendering
            monitor.updateFrameMetrics(60.0, 16.67);
            
            // Simulate data processing
            MarketDataUpdate update;
            update.type = MarketDataType::TRADE;
            update.symbol_id = symbol_id;
            update.exchange = "test_exchange";
            update.symbol = "TESTSYMBOL";
            update.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count() + i;
            update.local_timestamp_us = update.timestamp_us;
            update.price = 50000.0 + i;
            update.size = 1.0;
            update.side = "buy";
            
            processor.processTradeUpdate(update);
            
            // Update performance metrics
            monitor.updateDataLatency(0.5); // 0.5ms latency
            monitor.updateMemoryUsage(100.0 + i); // Increasing memory usage
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        // Verify integrated metrics
        auto metrics = monitor.getCurrentMetrics();
        if (metrics.fps <= 0) {
            return IntegrationTestResult::FAIL;
        }
        
        auto statistics = monitor.getStatistics();
        if (statistics.avg_fps <= 0) {
            return IntegrationTestResult::FAIL;
        }
        
        // Verify data processor metrics
        auto processor_metrics = processor.getPerformanceMetrics();
        if (processor_metrics.total_trades_processed == 0) {
            return IntegrationTestResult::FAIL;
        }
        
        monitor.stopMonitoring();
        
        return IntegrationTestResult::PASS;
        
    } catch (const std::exception& e) {
        std::cerr << "Performance monitoring integration test error: " << e.what() << std::endl;
        return IntegrationTestResult::ERROR;
    }
}

IntegrationTestResult test_configuration_system_integration() {
    try {
        using namespace BTQuant::RenderEngine;
        
        // Test integration between configuration system and all components
        DashboardConfig config;
        PerformanceMonitor monitor;
        SymbolManager symbol_manager;
        
        // Load configuration
        config.resetToDefaults();
        
        // Test configuration propagation to components
        auto performance_config = config.getPerformanceConfig();
        auto display_config = config.getDisplayConfig();
        auto data_source_config = config.getDataSourceConfig();
        
        // Initialize components with configuration
        if (!symbol_manager.initialize(data_source_config.symbols_file)) {
            return IntegrationTestResult::SKIP;
        }
        
        if (!monitor.startMonitoring()) {
            return IntegrationTestResult::SKIP;
        }
        
        // Configure performance monitor with config values
        monitor.setAlertThresholds(
            performance_config.fps_alert_threshold,
            performance_config.latency_alert_threshold_ms,
            performance_config.memory_alert_threshold_mb
        );
        
        monitor.setMonitoringInterval(performance_config.monitoring_interval_ms);
        monitor.setHistorySize(performance_config.history_size);
        
        // Test configuration changes
        PerformanceConfig new_perf_config = performance_config;
        new_perf_config.fps_alert_threshold = 45.0;
        new_perf_config.latency_alert_threshold_ms = 5.0;
        
        config.setPerformanceConfig(new_perf_config);
        
        // Verify configuration update
        auto updated_config = config.getPerformanceConfig();
        if (updated_config.fps_alert_threshold != 45.0) {
            return IntegrationTestResult::FAIL;
        }
        
        // Test theme integration
        auto available_themes = config.getAvailableThemes();
        if (!available_themes.empty()) {
            bool theme_applied = config.applyTheme(available_themes[0]);
            if (!theme_applied) {
                return IntegrationTestResult::FAIL;
            }
            
            auto theme_config = config.getThemeConfig();
            if (theme_config.name != available_themes[0]) {
                return IntegrationTestResult::FAIL;
            }
        }
        
        monitor.stopMonitoring();
        
        return IntegrationTestResult::PASS;
        
    } catch (const std::exception& e) {
        std::cerr << "Configuration system integration test error: " << e.what() << std::endl;
        return IntegrationTestResult::ERROR;
    }
}

// ============================================================================
// Multi-Threading Integration Tests
// ============================================================================

IntegrationTestResult test_concurrent_data_processing() {
    try {
        using namespace BTQuant::RenderEngine;
        
        // Test concurrent data processing across multiple components
        MarketDataProcessor processor;
        SymbolManager symbol_manager;
        PerformanceMonitor monitor;
        
        if (!symbol_manager.initialize()) {
            return IntegrationTestResult::SKIP;
        }
        
        if (!monitor.startMonitoring()) {
            return IntegrationTestResult::SKIP;
        }
        
        // Register symbols for concurrent testing
        std::vector<uint32_t> symbol_ids;
        for (int i = 0; i < 10; ++i) {
            std::string symbol = "SYMBOL" + std::to_string(i);
            uint32_t id = symbol_manager.registerSymbol("test_exchange", symbol);
            symbol_ids.push_back(id);
        }
        
        // Concurrent data processing test
        const int num_threads = 4;
        const int updates_per_thread = 250;
        std::atomic<int> successful_updates{0};
        std::atomic<int> failed_updates{0};
        
        std::vector<std::thread> threads;
        
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                try {
                    auto base_time = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count();
                    
                    for (int i = 0; i < updates_per_thread; ++i) {
                        uint32_t symbol_id = symbol_ids[i % symbol_ids.size()];
                        
                        MarketDataUpdate update;
                        update.type = MarketDataType::TRADE;
                        update.symbol_id = symbol_id;
                        update.exchange = "test_exchange";
                        update.symbol = "SYMBOL" + std::to_string(symbol_id);
                        update.timestamp_us = base_time + (t * updates_per_thread + i);
                        update.local_timestamp_us = update.timestamp_us;
                        update.price = 50000.0 + (i * 0.1);
                        update.size = 1.0;
                        update.side = (i % 2 == 0) ? "buy" : "sell";
                        
                        processor.processTradeUpdate(update);
                        
                        // Update performance metrics
                        monitor.updateDataLatency(0.5);
                        
                        successful_updates++;
                    }
                } catch (...) {
                    failed_updates++;
                }
            });
        }
        
        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }
        
        // Verify results
        int total_expected = num_threads * updates_per_thread;
        if (successful_updates < total_expected * 0.95) { // Allow 5% failure rate
            return IntegrationTestResult::FAIL;
        }
        
        // Verify all symbols have data
        auto active_symbols = processor.getActiveSymbols();
        if (active_symbols.size() < symbol_ids.size()) {
            return IntegrationTestResult::FAIL;
        }
        
        // Verify performance metrics were updated
        auto metrics = monitor.getCurrentMetrics();
        if (metrics.data_to_display_latency_us <= 0) {
            return IntegrationTestResult::FAIL;
        }
        
        monitor.stopMonitoring();
        
        return IntegrationTestResult::PASS;
        
    } catch (const std::exception& e) {
        std::cerr << "Concurrent data processing integration test error: " << e.what() << std::endl;
        return IntegrationTestResult::ERROR;
    }
}

// ============================================================================
// Cross-Component Communication Tests
// ============================================================================

IntegrationTestResult test_component_communication() {
    try {
        using namespace BTQuant::RenderEngine;
        
        // Test communication between all major components
        SymbolManager symbol_manager;
        MarketDataProcessor processor;
        PerformanceMonitor monitor;
        DashboardConfig config;
        
        // Initialize all components
        config.resetToDefaults();
        
        if (!symbol_manager.initialize()) {
            return IntegrationTestResult::SKIP;
        }
        
        if (!monitor.startMonitoring()) {
            return IntegrationTestResult::SKIP;
        }
        
        // Test configuration propagation
        auto perf_config = config.getPerformanceConfig();
        monitor.setAlertThresholds(
            perf_config.fps_alert_threshold,
            perf_config.latency_alert_threshold_ms,
            perf_config.memory_alert_threshold_mb
        );
        
        // Test symbol registration and data flow
        uint32_t symbol_id = symbol_manager.registerSymbol("test_exchange", "TESTSYMBOL");
        
        // Generate test data flow
        for (int i = 0; i < 20; ++i) {
            // Update performance metrics
            monitor.updateFrameMetrics(60.0, 16.67);
            monitor.updateDataLatency(0.8);
            monitor.updateMemoryUsage(200.0 + i);
            
            // Process market data
            MarketDataUpdate update;
            update.type = MarketDataType::TRADE;
            update.symbol_id = symbol_id;
            update.exchange = "test_exchange";
            update.symbol = "TESTSYMBOL";
            update.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count() + i;
            update.local_timestamp_us = update.timestamp_us;
            update.price = 50000.0 + i;
            update.size = 1.0;
            update.side = "buy";
            
            processor.processTradeUpdate(update);
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        // Verify cross-component data consistency
        auto symbol_info = symbol_manager.getSymbolInfo(symbol_id);
        auto analytics = processor.getSymbolAnalytics(symbol_id);
        auto metrics = monitor.getCurrentMetrics();
        
        if (!symbol_info || analytics.symbol_id != symbol_id) {
            return IntegrationTestResult::FAIL;
        }
        
        if (metrics.fps <= 0 || metrics.data_to_display_latency_us <= 0) {
            return IntegrationTestResult::FAIL;
        }
        
        // Test statistics aggregation
        auto processor_metrics = processor.getPerformanceMetrics();
        auto monitor_stats = monitor.getStatistics();
        
        if (processor_metrics.total_trades_processed == 0) {
            return IntegrationTestResult::FAIL;
        }
        
        if (monitor_stats.avg_fps <= 0) {
            return IntegrationTestResult::FAIL;
        }
        
        monitor.stopMonitoring();
        
        return IntegrationTestResult::PASS;
        
    } catch (const std::exception& e) {
        std::cerr << "Component communication integration test error: " << e.what() << std::endl;
        return IntegrationTestResult::ERROR;
    }
}

IntegrationTestResult test_end_to_end_data_flow() {
    try {
        using namespace BTQuant::RenderEngine;
        
        // Test complete end-to-end data flow
        HotSpineDataBridge bridge;
        SymbolManager symbol_manager;
        MarketDataProcessor processor;
        PerformanceMonitor monitor;
        DashboardConfig config;
        
        // Initialize complete system
        config.resetToDefaults();
        
        if (!symbol_manager.initialize()) {
            return IntegrationTestResult::SKIP;
        }
        
        if (!monitor.startMonitoring()) {
            return IntegrationTestResult::SKIP;
        }
        
        // Test data flow: Configuration -> Symbol Registration -> Data Processing -> Visualization
        
        // 1. Configuration drives symbol selection
        auto data_config = config.getDataSourceConfig();
        
        // 2. Register symbols based on configuration
        std::vector<uint32_t> symbol_ids;
        for (size_t i = 0; i < std::min(data_config.max_symbols, size_t(10)); ++i) {
            std::string symbol = "SYMBOL" + std::to_string(i);
            uint32_t id = symbol_manager.registerSymbol("test_exchange", symbol);
            symbol_ids.push_back(id);
        }
        
        // 3. Process market data for registered symbols
        auto base_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        for (int round = 0; round < 5; ++round) {
            for (uint32_t symbol_id : symbol_ids) {
                // Trade update
                MarketDataUpdate trade_update;
                trade_update.type = MarketDataType::TRADE;
                trade_update.symbol_id = symbol_id;
                trade_update.exchange = "test_exchange";
                trade_update.symbol = "SYMBOL" + std::to_string(symbol_id);
                trade_update.timestamp_us = base_time + (round * symbol_ids.size()) + symbol_id;
                trade_update.local_timestamp_us = trade_update.timestamp_us;
                trade_update.price = 50000.0 + (round * 10.0) + (symbol_id * 100.0);
                trade_update.size = 1.0;
                trade_update.side = (round % 2 == 0) ? "buy" : "sell";
                
                processor.processTradeUpdate(trade_update);
                
                // Orderbook update
                MarketDataUpdate orderbook_update;
                orderbook_update.type = MarketDataType::ORDERBOOK;
                orderbook_update.symbol_id = symbol_id;
                orderbook_update.exchange = "test_exchange";
                orderbook_update.symbol = "SYMBOL" + std::to_string(symbol_id);
                orderbook_update.timestamp_us = trade_update.timestamp_us + 1;
                orderbook_update.local_timestamp_us = orderbook_update.timestamp_us;
                
                // Simple orderbook
                PriceLevel bid, ask;
                bid.price = trade_update.price - 5.0;
                bid.size = 10.0;
                ask.price = trade_update.price + 5.0;
                ask.size = 10.0;
                
                orderbook_update.bids.push_back(bid);
                orderbook_update.asks.push_back(ask);
                
                processor.processOrderbookUpdate(orderbook_update);
                
                // Update performance metrics
                monitor.updateFrameMetrics(60.0, 16.67);
                monitor.updateDataLatency(0.7);
            }
        }
        
        // 4. Verify end-to-end data integrity
        
        // Check symbol registration
        auto all_symbols = symbol_manager.getAllSymbols();
        if (all_symbols.size() < symbol_ids.size()) {
            return IntegrationTestResult::FAIL;
        }
        
        // Check data processing
        auto active_symbols = processor.getActiveSymbols();
        if (active_symbols.size() < symbol_ids.size()) {
            return IntegrationTestResult::FAIL;
        }
        
        // Check analytics calculation
        for (uint32_t symbol_id : symbol_ids) {
            auto analytics = processor.getSymbolAnalytics(symbol_id);
            if (analytics.symbol_id != symbol_id || analytics.trade_count == 0) {
                return IntegrationTestResult::FAIL;
            }
            
            if (analytics.vwap <= 0 || analytics.current_spread <= 0) {
                return IntegrationTestResult::FAIL;
            }
        }
        
        // Check performance monitoring
        auto metrics = monitor.getCurrentMetrics();
        if (metrics.fps <= 0 || metrics.data_to_display_latency_us <= 0) {
            return IntegrationTestResult::FAIL;
        }
        
        // Check market summary
        auto market_summary = processor.getMarketSummary();
        if (market_summary.total_symbols < symbol_ids.size()) {
            return IntegrationTestResult::FAIL;
        }
        
        monitor.stopMonitoring();
        
        return IntegrationTestResult::PASS;
        
    } catch (const std::exception& e) {
        std::cerr << "End-to-end data flow integration test error: " << e.what() << std::endl;
        return IntegrationTestResult::ERROR;
    }
}

// ============================================================================
// Test Registration
// ============================================================================

void register_integration_tests(IntegrationTestSuite& suite) {
    // HotSpine Integration Tests
    suite.register_test({
        "test_hotspine_symbol_registry_integration",
        "Test integration between HotSpine data bridge and symbol registry",
        test_hotspine_symbol_registry_integration,
        "Data Integration",
        {"HotSpineDataBridge", "SymbolManager"},
        true
    });
    
    // Data Processing Integration Tests
    suite.register_test({
        "test_data_processor_visualization_integration",
        "Test integration between market data processor and visualization engine",
        test_data_processor_visualization_integration,
        "Data Processing",
        {"MarketDataProcessor", "SymbolManager", "DataVisualizationEngine"},
        false
    });
    
    // Performance Monitoring Integration Tests
    suite.register_test({
        "test_performance_monitoring_integration",
        "Test integration between performance monitor and other components",
        test_performance_monitoring_integration,
        "Performance Monitoring",
        {"PerformanceMonitor", "MarketDataProcessor", "SymbolManager"},
        false
    });
    
    // Configuration System Integration Tests
    suite.register_test({
        "test_configuration_system_integration",
        "Test configuration system integration with all components",
        test_configuration_system_integration,
        "Configuration",
        {"DashboardConfig", "PerformanceMonitor", "SymbolManager"},
        false
    });
    
    // Multi-Threading Integration Tests
    suite.register_test({
        "test_concurrent_data_processing",
        "Test concurrent data processing across multiple threads",
        test_concurrent_data_processing,
        "Multi-Threading",
        {"MarketDataProcessor", "SymbolManager", "PerformanceMonitor"},
        false
    });
    
    // End-to-End Integration Tests
    suite.register_test({
        "test_end_to_end_data_flow",
        "Test complete end-to-end data flow through all components",
        test_end_to_end_data_flow,
        "End-to-End",
        {"HotSpineDataBridge", "SymbolManager", "MarketDataProcessor", "PerformanceMonitor", "DashboardConfig"},
        false
    });
}

} // namespace IntegrationTesting
} // namespace BTQuant

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char* argv[]) {
    try {
        std::cout << "BTQuant Integration Test Suite\n";
        std::cout << "Testing component integration and data flow\n";
        std::cout << "==========================================\n\n";
        
        BTQuant::IntegrationTesting::IntegrationTestSuite suite;
        BTQuant::IntegrationTesting::register_integration_tests(suite);
        
        suite.run_all_tests();
        
        std::cout << "\nIntegration testing completed.\n";
        std::cout << "See integration_test_report.json for detailed results.\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Integration test suite failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Integration test suite failed with unknown exception" << std::endl;
        return 1;
    }
}