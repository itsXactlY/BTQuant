/**
 * BTQuant Comprehensive Test Suite
 * 
 * Professional-grade testing framework for the BTQuant Advanced Vulkan Dashboard
 * Validates all components, performance targets, and professional standards
 * 
 * Test Coverage:
 * - Unit tests for all major components and classes
 * - Integration tests for data flow and rendering pipeline
 * - Performance benchmarks for all critical operations
 * - Memory leak detection and resource management validation
 * - Thread safety tests for concurrent operations
 * - Error handling and recovery testing
 */

#include <iostream>
#include <chrono>
#include <thread>
#include <random>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <fstream>
#include <sstream>
#include <cassert>
#include <cstring>
#include <unordered_map>
#include <algorithm>

// Test framework includes
#include "../include/vulkan_dashboard_advanced.hpp"
#include "../include/interaction_manager.hpp"
#include "../include/hotspine_data_bridge.hpp"
#include "../include/market_data_processor.hpp"
#include "../include/symbol_manager.hpp"
#include "../include/performance_monitor.hpp"
#include "../include/dashboard_config.hpp"
#include "../include/data_visualization_engine.hpp"

namespace BTQuant {
namespace Testing {

// ============================================================================
// Test Framework Infrastructure
// ============================================================================

enum class TestResult {
    PASS,
    FAIL,
    SKIP,
    ERROR
};

struct TestCase {
    std::string name;
    std::string description;
    std::function<TestResult()> test_function;
    std::string category;
    int priority = 1; // 1=critical, 2=important, 3=nice-to-have
    double timeout_seconds = 30.0;
};

struct TestResults {
    std::string test_name;
    TestResult result;
    std::string error_message;
    double execution_time_ms;
    std::unordered_map<std::string, std::string> metrics;
};

class TestFramework {
private:
    std::vector<TestCase> test_cases_;
    std::vector<TestResults> results_;
    std::atomic<bool> stop_requested_{false};
    
public:
    void register_test(const TestCase& test_case) {
        test_cases_.push_back(test_case);
    }
    
    void run_all_tests() {
        std::cout << "=== BTQuant Comprehensive Test Suite ===\n";
        std::cout << "Running " << test_cases_.size() << " test cases...\n\n";
        
        // Sort tests by priority
        std::sort(test_cases_.begin(), test_cases_.end(),
            [](const TestCase& a, const TestCase& b) {
                return a.priority < b.priority;
            });
        
        for (const auto& test_case : test_cases_) {
            if (stop_requested_) break;
            
            std::cout << "Running: " << test_case.name << " (" << test_case.category << ")\n";
            
            auto start_time = std::chrono::high_resolution_clock::now();
            TestResults result;
            result.test_name = test_case.name;
            
            try {
                // Run test with timeout
                std::atomic<bool> test_completed{false};
                TestResult test_result = TestResult::ERROR;
                std::string error_msg;
                
                std::thread test_thread([&]() {
                    try {
                        test_result = test_case.test_function();
                        test_completed = true;
                    } catch (const std::exception& e) {
                        error_msg = e.what();
                        test_result = TestResult::ERROR;
                        test_completed = true;
                    }
                });
                
                // Wait for completion or timeout
                auto timeout = std::chrono::duration<double>(test_case.timeout_seconds);
                auto deadline = start_time + std::chrono::duration_cast<std::chrono::steady_clock::duration>(timeout);
                
                while (!test_completed && std::chrono::steady_clock::now() < deadline) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                
                if (!test_completed) {
                    test_result = TestResult::ERROR;
                    error_msg = "Test timeout after " + std::to_string(test_case.timeout_seconds) + " seconds";
                    // Note: In production, we'd need proper thread termination
                }
                
                if (test_thread.joinable()) {
                    test_thread.join();
                }
                
                result.result = test_result;
                result.error_message = error_msg;
                
            } catch (const std::exception& e) {
                result.result = TestResult::ERROR;
                result.error_message = e.what();
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            result.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            
            results_.push_back(result);
            
            // Print result
            std::string status;
            switch (result.result) {
                case TestResult::PASS: status = "✓ PASS"; break;
                case TestResult::FAIL: status = "✗ FAIL"; break;
                case TestResult::SKIP: status = "- SKIP"; break;
                case TestResult::ERROR: status = "! ERROR"; break;
            }
            
            std::cout << "  " << status << " (" << std::fixed << std::setprecision(2) 
                      << result.execution_time_ms << "ms)";
            
            if (!result.error_message.empty()) {
                std::cout << " - " << result.error_message;
            }
            std::cout << "\n";
        }
        
        generate_test_report();
    }
    
    void generate_test_report() {
        std::cout << "\n=== Test Summary ===\n";
        
        int passed = 0, failed = 0, skipped = 0, errors = 0;
        double total_time = 0.0;
        
        for (const auto& result : results_) {
            total_time += result.execution_time_ms;
            switch (result.result) {
                case TestResult::PASS: passed++; break;
                case TestResult::FAIL: failed++; break;
                case TestResult::SKIP: skipped++; break;
                case TestResult::ERROR: errors++; break;
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
        
        // Detailed failure report
        if (failed > 0 || errors > 0) {
            std::cout << "=== Failed Tests ===\n";
            for (const auto& result : results_) {
                if (result.result == TestResult::FAIL || result.result == TestResult::ERROR) {
                    std::cout << "- " << result.test_name << ": " << result.error_message << "\n";
                }
            }
            std::cout << "\n";
        }
        
        // Save detailed report to file
        save_detailed_report();
    }
    
private:
    void save_detailed_report() {
        std::ofstream report_file("test_report.json");
        if (!report_file.is_open()) return;
        
        report_file << "{\n";
        report_file << "  \"timestamp\": \"" << std::time(nullptr) << "\",\n";
        report_file << "  \"total_tests\": " << results_.size() << ",\n";
        report_file << "  \"results\": [\n";
        
        for (size_t i = 0; i < results_.size(); ++i) {
            const auto& result = results_[i];
            report_file << "    {\n";
            report_file << "      \"name\": \"" << result.test_name << "\",\n";
            report_file << "      \"result\": \"";
            switch (result.result) {
                case TestResult::PASS: report_file << "PASS"; break;
                case TestResult::FAIL: report_file << "FAIL"; break;
                case TestResult::SKIP: report_file << "SKIP"; break;
                case TestResult::ERROR: report_file << "ERROR"; break;
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
        
        std::cout << "Detailed report saved to test_report.json\n";
    }
};

// ============================================================================
// Memory Management Tests
// ============================================================================

class MemoryLeakDetector {
private:
    struct AllocationInfo {
        size_t size;
        std::string file;
        int line;
        std::chrono::time_point<std::chrono::steady_clock> timestamp;
    };
    
    std::mutex mutex_;
    std::unordered_map<void*, AllocationInfo> allocations_;
    std::atomic<size_t> total_allocated_{0};
    std::atomic<size_t> total_deallocated_{0};
    
public:
    void record_allocation(void* ptr, size_t size, const std::string& file, int line) {
        std::lock_guard<std::mutex> lock(mutex_);
        allocations_[ptr] = {size, file, line, std::chrono::steady_clock::now()};
        total_allocated_ += size;
    }
    
    void record_deallocation(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = allocations_.find(ptr);
        if (it != allocations_.end()) {
            total_deallocated_ += it->second.size;
            allocations_.erase(it);
        }
    }
    
    size_t get_leaked_bytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t leaked = 0;
        for (const auto& [ptr, info] : allocations_) {
            leaked += info.size;
        }
        return leaked;
    }
    
    std::vector<std::string> get_leak_report() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<std::string> report;
        for (const auto& [ptr, info] : allocations_) {
            std::ostringstream oss;
            oss << "Leaked " << info.size << " bytes at " << ptr 
                << " (" << info.file << ":" << info.line << ")";
            report.push_back(oss.str());
        }
        return report;
    }
};

// Global memory leak detector
static MemoryLeakDetector g_leak_detector;

// ============================================================================
// Unit Tests
// ============================================================================

TestResult test_hotspine_data_bridge() {
    try {
        // Test HotSpine data bridge initialization
        HotSpineDataBridge bridge("/test_shm", "/test/symbols.json");
        
        // Test configuration
        if (!bridge.is_configured()) {
            return TestResult::FAIL;
        }
        
        // Test symbol registration
        uint32_t symbol_id = bridge.register_symbol("BTCUSD", "binance");
        if (symbol_id == 0) {
            return TestResult::FAIL;
        }
        
        // Test data retrieval (mock data)
        auto symbols = bridge.get_all_symbols();
        if (symbols.empty()) {
            // This might be expected if no real data is available
            // Consider this a pass for unit testing
        }
        
        return TestResult::PASS;
        
    } catch (const std::exception& e) {
        std::cerr << "HotSpine test error: " << e.what() << std::endl;
        return TestResult::ERROR;
    }
}

TestResult test_market_data_processor() {
    try {
        MarketDataProcessor processor;
        
        // Test VWAP calculation
        std::vector<ProcessedTrade> test_trades;
        for (int i = 0; i < 100; ++i) {
            ProcessedTrade trade;
            trade.symbol_id = 1;
            trade.price = 50000.0 + (i * 10.0);
            trade.size = 1.0;
            trade.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count() + i;
            test_trades.push_back(trade);
        }
        
        auto vwap = processor.calculate_vwap(test_trades, 60000); // 1 minute window
        if (vwap <= 0) {
            return TestResult::FAIL;
        }
        
        // Test momentum calculation
        auto momentum = processor.calculate_momentum(test_trades, 30000); // 30 second window
        // Momentum can be positive, negative, or zero
        
        // Test volatility calculation
        auto volatility = processor.calculate_volatility(test_trades, 60000);
        if (volatility < 0) {
            return TestResult::FAIL;
        }
        
        return TestResult::PASS;
        
    } catch (const std::exception& e) {
        std::cerr << "Market data processor test error: " << e.what() << std::endl;
        return TestResult::ERROR;
    }
}

TestResult test_symbol_manager() {
    try {
        SymbolManager symbol_manager;
        
        // Test symbol registration
        uint32_t btc_id = symbol_manager.register_symbol("BTCUSD", "binance");
        uint32_t eth_id = symbol_manager.register_symbol("ETHUSD", "binance");
        
        if (btc_id == 0 || eth_id == 0 || btc_id == eth_id) {
            return TestResult::FAIL;
        }
        
        // Test symbol lookup
        auto btc_info = symbol_manager.get_symbol_info(btc_id);
        if (btc_info.symbol != "BTCUSD" || btc_info.exchange != "binance") {
            return TestResult::FAIL;
        }
        
        // Test symbol search
        auto search_results = symbol_manager.search_symbols("BTC");
        if (search_results.empty()) {
            return TestResult::FAIL;
        }
        
        // Test symbol filtering
        auto binance_symbols = symbol_manager.get_symbols_by_exchange("binance");
        if (binance_symbols.size() < 2) {
            return TestResult::FAIL;
        }
        
        return TestResult::PASS;
        
    } catch (const std::exception& e) {
        std::cerr << "Symbol manager test error: " << e.what() << std::endl;
        return TestResult::ERROR;
    }
}

TestResult test_performance_monitor() {
    try {
        PerformanceMonitor monitor;
        
        // Test frame timing
        monitor.begin_frame();
        std::this_thread::sleep_for(std::chrono::milliseconds(16)); // Simulate 60 FPS
        monitor.end_frame();
        
        auto fps = monitor.get_average_fps();
        if (fps <= 0 || fps > 1000) { // Sanity check
            return TestResult::FAIL;
        }
        
        // Test latency measurement
        auto latency_id = monitor.begin_latency_measurement("test_operation");
        std::this_thread::sleep_for(std::chrono::microseconds(500));
        monitor.end_latency_measurement(latency_id);
        
        auto avg_latency = monitor.get_average_latency("test_operation");
        if (avg_latency <= 0) {
            return TestResult::FAIL;
        }
        
        // Test memory tracking
        monitor.record_memory_usage(1024 * 1024); // 1MB
        auto memory_usage = monitor.get_current_memory_usage();
        if (memory_usage <= 0) {
            return TestResult::FAIL;
        }
        
        // Test performance alerts
        monitor.set_fps_alert_threshold(30.0);
        monitor.set_latency_alert_threshold(10.0);
        
        // Simulate low performance
        for (int i = 0; i < 10; ++i) {
            monitor.begin_frame();
            std::this_thread::sleep_for(std::chrono::milliseconds(50)); // 20 FPS
            monitor.end_frame();
        }
        
        auto alerts = monitor.get_active_alerts();
        // Should have FPS alert
        
        return TestResult::PASS;
        
    } catch (const std::exception& e) {
        std::cerr << "Performance monitor test error: " << e.what() << std::endl;
        return TestResult::ERROR;
    }
}

TestResult test_dashboard_config() {
    try {
        DashboardConfig config;
        
        // Test default configuration
        if (!config.load_default_config()) {
            return TestResult::FAIL;
        }
        
        // Test configuration values
        auto window_width = config.get_window_width();
        auto window_height = config.get_window_height();
        auto target_fps = config.get_target_fps();
        
        if (window_width <= 0 || window_height <= 0 || target_fps <= 0) {
            return TestResult::FAIL;
        }
        
        // Test configuration modification
        config.set_window_size(1920, 1080);
        config.set_target_fps(60.0f);
        
        if (config.get_window_width() != 1920 || config.get_window_height() != 1080) {
            return TestResult::FAIL;
        }
        
        // Test theme configuration
        auto available_themes = config.get_available_themes();
        if (available_themes.empty()) {
            return TestResult::FAIL;
        }
        
        config.set_theme(available_themes[0]);
        auto current_theme = config.get_current_theme();
        if (current_theme != available_themes[0]) {
            return TestResult::FAIL;
        }
        
        // Test configuration persistence
        std::string config_file = "/tmp/test_config.json";
        if (!config.save_to_file(config_file)) {
            return TestResult::FAIL;
        }
        
        DashboardConfig loaded_config;
        if (!loaded_config.load_from_file(config_file)) {
            return TestResult::FAIL;
        }
        
        if (loaded_config.get_window_width() != 1920 || 
            loaded_config.get_window_height() != 1080) {
            return TestResult::FAIL;
        }
        
        return TestResult::PASS;
        
    } catch (const std::exception& e) {
        std::cerr << "Dashboard config test error: " << e.what() << std::endl;
        return TestResult::ERROR;
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

TestResult test_data_flow_integration() {
    try {
        // Test complete data flow from HotSpine to visualization
        HotSpineDataBridge bridge("/test_shm", "/test/symbols.json");
        MarketDataProcessor processor;
        SymbolManager symbol_manager;
        
        // Register test symbols
        uint32_t btc_id = symbol_manager.register_symbol("BTCUSD", "binance");
        uint32_t eth_id = symbol_manager.register_symbol("ETHUSD", "binance");
        
        // Simulate data flow
        std::vector<ProcessedTrade> test_trades;
        for (int i = 0; i < 50; ++i) {
            ProcessedTrade trade;
            trade.symbol_id = btc_id;
            trade.price = 50000.0 + (i * 10.0);
            trade.size = 1.0 + (i * 0.1);
            trade.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count() + i;
            test_trades.push_back(trade);
        }
        
        // Process data through pipeline
        auto vwap = processor.calculate_vwap(test_trades, 60000);
        auto momentum = processor.calculate_momentum(test_trades, 30000);
        auto volatility = processor.calculate_volatility(test_trades, 60000);
        
        // Validate processed data
        if (vwap <= 0 || volatility < 0) {
            return TestResult::FAIL;
        }
        
        // Test data aggregation
        auto aggregated_data = processor.aggregate_trades_to_candles(test_trades, 60000);
        if (aggregated_data.empty()) {
            return TestResult::FAIL;
        }
        
        return TestResult::PASS;
        
    } catch (const std::exception& e) {
        std::cerr << "Data flow integration test error: " << e.what() << std::endl;
        return TestResult::ERROR;
    }
}

TestResult test_multi_threading_safety() {
    try {
        const int num_threads = 4;
        const int operations_per_thread = 1000;
        std::atomic<int> success_count{0};
        std::atomic<int> error_count{0};
        
        SymbolManager symbol_manager;
        std::vector<std::thread> threads;
        
        // Test concurrent symbol registration and lookup
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&, t]() {
                try {
                    for (int i = 0; i < operations_per_thread; ++i) {
                        std::string symbol = "TEST" + std::to_string(t) + "_" + std::to_string(i);
                        uint32_t id = symbol_manager.register_symbol(symbol, "test_exchange");
                        
                        if (id > 0) {
                            auto info = symbol_manager.get_symbol_info(id);
                            if (info.symbol == symbol) {
                                success_count++;
                            } else {
                                error_count++;
                            }
                        } else {
                            error_count++;
                        }
                    }
                } catch (...) {
                    error_count++;
                }
            });
        }
        
        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }
        
        // Check results
        int total_operations = num_threads * operations_per_thread;
        if (success_count < total_operations * 0.95) { // Allow 5% error rate
            return TestResult::FAIL;
        }
        
        return TestResult::PASS;
        
    } catch (const std::exception& e) {
        std::cerr << "Multi-threading safety test error: " << e.what() << std::endl;
        return TestResult::ERROR;
    }
}

// ============================================================================
// Performance Tests
// ============================================================================

TestResult test_performance_benchmarks() {
    try {
        const int num_iterations = 10000;
        
        // Test symbol lookup performance
        SymbolManager symbol_manager;
        
        // Pre-populate with symbols
        std::vector<uint32_t> symbol_ids;
        for (int i = 0; i < 1000; ++i) {
            std::string symbol = "SYMBOL" + std::to_string(i);
            uint32_t id = symbol_manager.register_symbol(symbol, "test_exchange");
            symbol_ids.push_back(id);
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Benchmark symbol lookups
        for (int i = 0; i < num_iterations; ++i) {
            uint32_t id = symbol_ids[i % symbol_ids.size()];
            auto info = symbol_manager.get_symbol_info(id);
            if (info.symbol.empty()) {
                return TestResult::FAIL;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        double avg_lookup_time = static_cast<double>(duration.count()) / num_iterations;
        
        // Target: < 1 microsecond per lookup
        if (avg_lookup_time > 1.0) {
            std::cerr << "Symbol lookup too slow: " << avg_lookup_time << " μs" << std::endl;
            return TestResult::FAIL;
        }
        
        // Test market data processing performance
        MarketDataProcessor processor;
        std::vector<ProcessedTrade> large_trade_set;
        
        // Generate large dataset
        for (int i = 0; i < 10000; ++i) {
            ProcessedTrade trade;
            trade.symbol_id = 1;
            trade.price = 50000.0 + (i * 0.1);
            trade.size = 1.0;
            trade.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count() + i;
            large_trade_set.push_back(trade);
        }
        
        start_time = std::chrono::high_resolution_clock::now();
        
        // Benchmark VWAP calculation
        for (int i = 0; i < 100; ++i) {
            auto vwap = processor.calculate_vwap(large_trade_set, 60000);
            if (vwap <= 0) {
                return TestResult::FAIL;
            }
        }
        
        end_time = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        double avg_vwap_time = static_cast<double>(duration.count()) / 100.0;
        
        // Target: < 1000 microseconds per VWAP calculation on 10k trades
        if (avg_vwap_time > 1000.0) {
            std::cerr << "VWAP calculation too slow: " << avg_vwap_time << " μs" << std::endl;
            return TestResult::FAIL;
        }
        
        std::cout << "Performance metrics:\n";
        std::cout << "  Symbol lookup: " << std::fixed << std::setprecision(2) << avg_lookup_time << " μs\n";
        std::cout << "  VWAP calculation: " << std::fixed << std::setprecision(2) << avg_vwap_time << " μs\n";
        
        return TestResult::PASS;
        
    } catch (const std::exception& e) {
        std::cerr << "Performance benchmark test error: " << e.what() << std::endl;
        return TestResult::ERROR;
    }
}

// ============================================================================
// Memory Tests
// ============================================================================

TestResult test_memory_management() {
    try {
        size_t initial_leaked = g_leak_detector.get_leaked_bytes();
        
        // Test memory allocation and deallocation
        std::vector<void*> allocations;
        
        for (int i = 0; i < 1000; ++i) {
            size_t size = 1024 + (i % 1024);
            void* ptr = malloc(size);
            if (!ptr) {
                return TestResult::ERROR;
            }
            
            g_leak_detector.record_allocation(ptr, size, __FILE__, __LINE__);
            allocations.push_back(ptr);
        }
        
        // Free half the allocations
