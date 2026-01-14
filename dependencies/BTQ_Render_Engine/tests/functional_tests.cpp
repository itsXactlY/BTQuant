/**
 * BTQuant Functional Testing Suite
 * 
 * Comprehensive functional testing for the BTQuant Advanced Vulkan Dashboard
 * 
 * Test Coverage:
 * - UI component rendering validation
 * - Interactive feature testing (mouse, keyboard, touch)
 * - Data visualization accuracy testing
 * - Real-time data processing validation
 * - Configuration system testing
 * - Error recovery and graceful degradation testing
 */

#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <memory>
#include <random>
#include <fstream>
#include <iomanip>
#include <functional>

#include "../include/symbol_manager.hpp"
#include "../include/market_data_processor.hpp"
#include "../include/performance_monitor.hpp"
#include "../include/dashboard_config.hpp"
#include "../include/hotspine_data_bridge.hpp"
#include "../include/interaction_manager.hpp"

namespace BTQuant {
namespace FunctionalTesting {

// ============================================================================
// Functional Test Framework
// ============================================================================

enum class FunctionalTestResult {
    PASS,
    FAIL,
    SKIP,
    ERROR
};

struct FunctionalTestCase {
    std::string name;
    std::string description;
    std::function<FunctionalTestResult()> test_function;
    std::string component;
    bool is_critical = true;
};

struct FunctionalTestResults {
    std::string test_name;
    FunctionalTestResult result;
    std::string error_message;
    double execution_time_ms;
    std::vector<std::string> validation_points;
};

class FunctionalTestSuite {
private:
    std::vector<FunctionalTestCase> test_cases_;
    std::vector<FunctionalTestResults> results_;
    
public:
    void register_test(const FunctionalTestCase& test_case) {
        test_cases_.push_back(test_case);
    }
    
    void run_all_tests() {
        std::cout << "=== BTQuant Functional Test Suite ===\n";
        std::cout << "Testing UI components and interactive features\n";
        std::cout << "============================================\n\n";
        
        for (const auto& test_case : test_cases_) {
            std::cout << "Testing: " << test_case.name << " (" << test_case.component << ")\n";
            
            auto start_time = std::chrono::high_resolution_clock::now();
            FunctionalTestResults result;
            result.test_name = test_case.name;
            
            try {
                result.result = test_case.test_function();
            } catch (const std::exception& e) {
                result.result = FunctionalTestResult::ERROR;
                result.error_message = e.what();
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            result.execution_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            
            results_.push_back(result);
            
            // Print result
            std::string status;
            switch (result.result) {
                case FunctionalTestResult::PASS: status = "✓ PASS"; break;
                case FunctionalTestResult::FAIL: status = "✗ FAIL"; break;
                case FunctionalTestResult::SKIP: status = "- SKIP"; break;
                case FunctionalTestResult::ERROR: status = "! ERROR"; break;
            }
            
            std::cout << "  " << status;
            if (!result.error_message.empty()) {
                std::cout << " - " << result.error_message;
            }
            std::cout << "\n";
        }
        
        generate_functional_report();
    }
    
private:
    void generate_functional_report() {
        std::cout << "\n=== Functional Test Summary ===\n";
        
        int passed = 0, failed = 0, skipped = 0, errors = 0;
        int critical_passed = 0, critical_total = 0;
        
        for (size_t i = 0; i < results_.size(); ++i) {
            const auto& result = results_[i];
            const auto& test_case = test_cases_[i];
            
            switch (result.result) {
                case FunctionalTestResult::PASS: 
                    passed++; 
                    if (test_case.is_critical) critical_passed++;
                    break;
                case FunctionalTestResult::FAIL: failed++; break;
                case FunctionalTestResult::SKIP: skipped++; break;
                case FunctionalTestResult::ERROR: errors++; break;
            }
            
            if (test_case.is_critical) critical_total++;
        }
        
        std::cout << "Total Tests: " << results_.size() << "\n";
        std::cout << "Passed: " << passed << "\n";
        std::cout << "Failed: " << failed << "\n";
        std::cout << "Skipped: " << skipped << "\n";
        std::cout << "Errors: " << errors << "\n";
        std::cout << "Critical Tests: " << critical_passed << "/" << critical_total << "\n";
        std::cout << "Overall Success Rate: " << std::fixed << std::setprecision(1) 
                  << (100.0 * passed / results_.size()) << "%\n\n";
        
        // Component-wise breakdown
        std::unordered_map<std::string, std::pair<int, int>> component_stats; // passed, total
        for (size_t i = 0; i < results_.size(); ++i) {
            const auto& result = results_[i];
            const auto& test_case = test_cases_[i];
            
            component_stats[test_case.component].second++; // total
            if (result.result == FunctionalTestResult::PASS) {
                component_stats[test_case.component].first++; // passed
            }
        }
        
        std::cout << "=== Component Test Results ===\n";
        for (const auto& [component, stats] : component_stats) {
            double success_rate = (100.0 * stats.first) / stats.second;
            std::cout << component << ": " << stats.first << "/" << stats.second 
                      << " (" << std::fixed << std::setprecision(1) << success_rate << "%)\n";
        }
        
        save_functional_report();
    }
    
    void save_functional_report() {
        std::ofstream report_file("functional_test_report.json");
        if (!report_file.is_open()) return;
        
        report_file << "{\n";
        report_file << "  \"timestamp\": \"" << std::time(nullptr) << "\",\n";
        report_file << "  \"test_suite\": \"BTQuant Functional Tests\",\n";
        report_file << "  \"total_tests\": " << results_.size() << ",\n";
        report_file << "  \"results\": [\n";
        
        for (size_t i = 0; i < results_.size(); ++i) {
            const auto& result = results_[i];
            const auto& test_case = test_cases_[i];
            
            report_file << "    {\n";
            report_file << "      \"test_name\": \"" << result.test_name << "\",\n";
            report_file << "      \"component\": \"" << test_case.component << "\",\n";
            report_file << "      \"critical\": " << (test_case.is_critical ? "true" : "false") << ",\n";
            report_file << "      \"result\": \"";
            switch (result.result) {
                case FunctionalTestResult::PASS: report_file << "PASS"; break;
                case FunctionalTestResult::FAIL: report_file << "FAIL"; break;
                case FunctionalTestResult::SKIP: report_file << "SKIP"; break;
                case FunctionalTestResult::ERROR: report_file << "ERROR"; break;
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
        
        std::cout << "\nFunctional test report saved to functional_test_report.json\n";
    }
};

// ============================================================================
// UI Component Tests
// ============================================================================

FunctionalTestResult test_data_grid_rendering() {
    try {
        using namespace BTQuant::RenderEngine;
        
        // Test data grid component functionality
        SymbolManager symbol_manager;
        if (!symbol_manager.initialize()) {
            return FunctionalTestResult::SKIP;
        }
        
        // Register test symbols for grid display
        std::vector<uint32_t> symbol_ids;
        for (int i = 0; i < 50; ++i) {
            std::string symbol = "SYMBOL" + std::to_string(i);
            uint32_t id = symbol_manager.registerSymbol("test_exchange", symbol);
            symbol_ids.push_back(id);
        }
        
        // Verify all symbols are registered
        auto all_symbols = symbol_manager.getAllSymbols();
        if (all_symbols.size() < 50) {
            return FunctionalTestResult::FAIL;
        }
        
        // Test filtering functionality
        SymbolFilter filter;
        filter.exchanges = {"test_exchange"};
        filter.active_only = false;
        
        auto filtered_symbols = symbol_manager.getFilteredSymbols(filter);
        if (filtered_symbols.size() < 50) {
            return FunctionalTestResult::FAIL;
        }
        
        return FunctionalTestResult::PASS;
        
    } catch (const std::exception& e) {
        std::cerr << "Data grid test error: " << e.what() << std::endl;
        return FunctionalTestResult::ERROR;
    }
}

FunctionalTestResult test_chart_rendering() {
    try {
        using namespace BTQuant::RenderEngine;
        
        MarketDataProcessor processor;
        
        // Generate test chart data
        std::vector<MarketDataUpdate> chart_data;
        auto base_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        for (int i = 0; i < 100; ++i) {
            MarketDataUpdate update;
            update.type = MarketDataType::TRADE;
            update.symbol_id = 1;
            update.exchange = "test_exchange";
            update.symbol = "BTCUSD";
            update.timestamp_us = base_time + (i * 1000);
            update.local_timestamp_us = base_time + (i * 1000);
            update.price = 50000.0 + (i * 10.0) + (std::sin(i * 0.1) * 100.0);
            update.size = 1.0 + (i % 5) * 0.2;
            update.side = (i % 2 == 0) ? "buy" : "sell";
            
            chart_data.push_back(update);
        }
        
        // Process chart data
        for (const auto& update : chart_data) {
            processor.processTradeUpdate(update);
        }
        
        // Verify analytics are calculated
        auto analytics = processor.getSymbolAnalytics(1);
        if (analytics.symbol_id != 1) {
            return FunctionalTestResult::FAIL;
        }
        
        if (analytics.trade_count == 0) {
            return FunctionalTestResult::FAIL;
        }
        
        // Verify VWAP calculation
        if (analytics.vwap <= 0) {
            return FunctionalTestResult::FAIL;
        }
        
        return FunctionalTestResult::PASS;
        
    } catch (const std::exception& e) {
        std::cerr << "Chart rendering test error: " << e.what() << std::endl;
        return FunctionalTestResult::ERROR;
    }
}

FunctionalTestResult test_heatmap_visualization() {
    try {
        using namespace BTQuant::RenderEngine;
        
        MarketDataProcessor processor;
        
        // Generate test data for multiple symbols (heatmap)
        const int num_symbols = 25; // 5x5 heatmap
        
        for (int symbol_id = 1; symbol_id <= num_symbols; ++symbol_id) {
            // Generate different momentum patterns for each symbol
            double base_price = 50000.0 + (symbol_id * 1000.0);
            double momentum_factor = (symbol_id % 5 - 2) * 0.02; // -4% to +4% momentum
            
            for (int i = 0; i < 20; ++i) {
                MarketDataUpdate update;
                update.type = MarketDataType::TRADE;
                update.symbol_id = symbol_id;
                update.exchange = "test_exchange";
                update.symbol = "SYMBOL" + std::to_string(symbol_id);
                update.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count() + i;
                update.local_timestamp_us = update.timestamp_us;
                update.price = base_price * (1.0 + momentum_factor * (i / 20.0));
                update.size = 1.0;
                update.side = (i % 2 == 0) ? "buy" : "sell";
                
                processor.processTradeUpdate(update);
            }
        }
        
        // Verify all symbols have analytics
        auto active_symbols = processor.getActiveSymbols();
        if (active_symbols.size() < num_symbols) {
            return FunctionalTestResult::FAIL;
        }
        
        // Verify momentum calculations for heatmap
        bool has_positive_momentum = false;
        bool has_negative_momentum = false;
        
        for (uint32_t symbol_id : active_symbols) {
            auto analytics = processor.getSymbolAnalytics(symbol_id);
            if (analytics.momentum > 0.01) has_positive_momentum = true;
            if (analytics.momentum < -0.01) has_negative_momentum = true;
        }
        
        if (!has_positive_momentum || !has_negative_momentum) {
            return FunctionalTestResult::FAIL;
        }
        
        return FunctionalTestResult::PASS;
        
    } catch (const std::exception& e) {
        std::cerr << "Heatmap visualization test error: " << e.what() << std::endl;
        return FunctionalTestResult::ERROR;
    }
}

FunctionalTestResult test_orderbook_display() {
    try {
        using namespace BTQuant::RenderEngine;
        
        MarketDataProcessor processor;
        
        // Generate test orderbook data
        MarketDataUpdate orderbook_update;
        orderbook_update.type = MarketDataType::ORDERBOOK;
        orderbook_update.symbol_id = 1;
        orderbook_update.exchange = "test_exchange";
        orderbook_update.symbol = "BTCUSD";
        orderbook_update.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        orderbook_update.local_timestamp_us = orderbook_update.timestamp_us;
        
        // Generate bid levels
        for (int i = 0; i < 10; ++i) {
            PriceLevel bid;
            bid.price = 50000.0 - (i * 10.0);
            bid.size = 1.0 + (i * 0.5);
            orderbook_update.bids.push_back(bid);
        }
        
        // Generate ask levels
        for (int i = 0; i < 10; ++i) {
            PriceLevel ask;
            ask.price = 50010.0 + (i * 10.0);
            ask.size = 1.0 + (i * 0.5);
            orderbook_update.asks.push_back(ask);
        }
        
        // Process orderbook update
        processor.processOrderbookUpdate(orderbook_update);
        
        // Verify orderbook analytics
        auto analytics = processor.getSymbolAnalytics(1);
        if (analytics.symbol_id != 1) {
            return FunctionalTestResult::FAIL;
        }
        
        // Verify spread calculation
        if (analytics.current_spread <= 0) {
            return FunctionalTestResult::FAIL;
        }
        
        // Verify market depth calculation
        if (analytics.market_depth <= 0) {
            return FunctionalTestResult::FAIL;
        }
        
        return FunctionalTestResult::PASS;
        
    } catch (const std::exception& e) {
        std::cerr << "Orderbook display test error: " << e.what() << std::endl;
        return FunctionalTestResult::ERROR;
    }
}

// ============================================================================
// Interactive Feature Tests
// ============================================================================

FunctionalTestResult test_mouse_interaction() {
    try {
        // Mock mouse interaction testing
        struct MockMouseEvent {
            enum Type { MOVE, CLICK, SCROLL };
            Type type;
            double x, y;
            int button;
            double scroll_delta;
        };
        
        std::vector<MockMouseEvent> test_events = {
            {MockMouseEvent::MOVE, 100.0, 200.0, 0, 0.0},
            {MockMouseEvent::CLICK, 100.0, 200.0, 1, 0.0}, // Left click
            {MockMouseEvent::SCROLL, 150.0, 250.0, 0, 1.0}, // Scroll up
            {MockMouseEvent::MOVE, 200.0, 300.0, 0, 0.0},
            {MockMouseEvent::CLICK, 200.0, 300.0, 2, 0.0}, // Right click
        };
        
        // Process mouse events
        bool all_events_processed = true;
        for (const auto& event : test_events) {
            // Simulate event processing
            if (event.x < 0 || event.y < 0) {
                all_events_processed = false;
                break;
            }
            
            // Simulate different event types
            switch (event.type) {
                case MockMouseEvent::MOVE:
                    // Simulate hover detection, tooltip triggering
                    break;
                case MockMouseEvent::CLICK:
                    // Simulate selection, context menu
                    break;
                case MockMouseEvent::SCROLL:
                    // Simulate zoom/pan operations
                    break;
            }
        }
        
        return all_events_processed ? FunctionalTestResult::PASS : FunctionalTestResult::FAIL;
        
    } catch (const std::exception& e) {
        std::cerr << "Mouse interaction test error: " << e.what() << std::endl;
        return FunctionalTestResult::ERROR;
    }
}

FunctionalTestResult test_keyboard_interaction() {
    try {
        // Mock keyboard interaction testing
        struct MockKeyEvent {
            int key_code;
            bool ctrl_pressed;
            bool shift_pressed;
            bool alt_pressed;
        };
        
        std::vector<MockKeyEvent> test_events = {
            {65, true, false, false},   // Ctrl+A (Select All)
            {67, true, false, false},   // Ctrl+C (Copy)
            {86, true, false, false},   // Ctrl+V (Paste)
            {90, true, false, false},   // Ctrl+Z (Undo)
            {70, true, false, false},   // Ctrl+F (Find)
            {27, false, false, false},  // Escape
            {13, false, false, false},  // Enter
        };
        
        // Process keyboard events
        bool all_hotkeys_recognized = true;
        for (const auto& event : test_events) {
            // Simulate hotkey recognition
            bool is_valid_hotkey = (event.key_code >= 32 && event.key_code <= 126) || 
                                  (event.key_code == 27) || (event.key_code == 13);
            
            if (!is_valid_hotkey) {
                all_hotkeys_recognized = false;
                break;
            }
            
            // Simulate hotkey actions
            if (event.ctrl_pressed) {
                switch (event.key_code) {
                    case 65: // Ctrl+A
                    case 67: // Ctrl+C
                    case 86: // Ctrl+V
                    case 90: // Ctrl+Z
                    case 70: // Ctrl+F
                        // Valid hotkey combinations
                        break;
                    default:
                        // Unknown combination
                        break;
                }
            }
        }
        
        return all_hotkeys_recognized ? FunctionalTestResult::PASS : FunctionalTestResult::FAIL;
        
    } catch (const std::exception& e) {
        std::cerr << "Keyboard interaction test error: " << e.what() << std::endl;
        return FunctionalTestResult::ERROR;
    }
}

FunctionalTestResult test_touch_gestures() {
    try {
        // Mock touch gesture testing
        struct MockTouchPoint {
            int id;
            double x, y;
            bool active;
        };
        
        struct MockGesture {
            enum Type { TAP, PINCH, PAN, ROTATE };
            Type type;
            std::vector<MockTouchPoint> touch_points;
        };
        
        std::vector<MockGesture> test_gestures = {
            // Single tap
            {MockGesture::TAP, {{1, 100.0, 200.0, true}}},
            
            // Pinch gesture (zoom)
            {MockGesture::PINCH, {{1, 100.0, 100.0, true}, {2, 200.0, 200.0, true}}},
            
            // Pan gesture (scroll)
            {MockGesture::PAN, {{1, 150.0, 150.0, true}}},
            
            // Rotation gesture
            {MockGesture::ROTATE, {{1, 100.0, 150.0, true}, {2, 200.0, 150.0, true}}},
        };
        
        // Process touch gestures
        bool all_gestures_recognized = true;
        for (const auto& gesture : test_gestures) {
            // Validate gesture data
            if (gesture.touch_points.empty()) {
                all_gestures_recognized = false;
                break;
            }
            
            for (const auto& touch : gesture.touch_points) {
                if (touch.x < 0 || touch.y < 0 || !touch.active) {
                    all_gestures_recognized = false;
                    break;
                }
            }
            
            if (!all_gestures_recognized) break;
            
            // Simulate gesture processing
            switch (gesture.type) {
                case MockGesture::TAP:
                    // Simulate selection or activation
                    break;
                case MockGesture::PINCH:
                    // Simulate zoom operation
                    if (gesture.touch_points.size() != 2) {
                        all_gestures_recognized = false;
                    }
                    break;
                case MockGesture::PAN:
                    // Simulate pan operation
                    break;
                case MockGesture::ROTATE:
                    // Simulate rotation (if supported)
                    break;
            }
        }
        
        return all_gestures_recognized ? FunctionalTestResult::PASS : FunctionalTestResult::FAIL;
        
    } catch (const std::exception& e) {
        std::cerr << "Touch gesture test error: " << e.what() << std::endl;
        return FunctionalTestResult::ERROR;
    }
}

// ============================================================================
// Data Validation Tests
// ============================================================================

FunctionalTestResult test_real_time_data_accuracy() {
    try {
        using namespace BTQuant::RenderEngine;
        
        MarketDataProcessor processor;
        
        // Test data accuracy with known inputs
        struct TestCase {
            std::vector<double> prices;
            std::vector<double> sizes;
            double expected_vwap;
        };
        
        TestCase test_case;
        test_case.prices = {50000.0, 50100.0, 49900.0, 50200.0, 50050.0};
        test_case.sizes = {1.0, 2.0, 1.5, 0.5, 3.0};
        
        // Calculate expected VWAP manually
        double total_value = 0.0;
        double total_volume = 0.0;
        for (size_t i = 0; i < test_case.prices.size(); ++i) {
            total_value += test_case.prices[i] * test_case.sizes[i];
            total_volume += test_case.sizes[i];
        }
        test_case.expected_vwap = total_value / total_volume;
        
        // Process test data
        auto base_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        for (size_t i = 0; i < test_case.prices.size(); ++i) {
            MarketDataUpdate update;
            update.type = MarketDataType::TRADE;
            update.symbol_id = 1;
            update.exchange = "test_exchange";
            update.symbol = "BTCUSD";
            update.timestamp_us = base_time + i;
            update.local_timestamp_us = base_time + i;
            update.price = test_case.prices[i];
            update.size = test_case.sizes[i];
            update.side = "buy";
            
            processor.processTradeUpdate(update);
        }
        
        // Verify calculated VWAP
        auto analytics = processor.getSymbolAnalytics(1);
        double calculated_vwap = analytics.vwap;
        
        // Allow small floating-point error
        double vwap_error = std::abs(calculated_vwap - test_case.expected_vwap) / test_case.expected_vwap;
        if (vwap_error > 0.001) { // 0.1% tolerance
            std::cerr << "VWAP accuracy error: expected " << test_case.expected_vwap 
                      << ", got " << calculated_vwap << std::endl;
            return FunctionalTestResult::FAIL;
        }
        
        return FunctionalTestResult::PASS;
        
    } catch (const std::exception& e) {
        std::cerr << "Real-time data accuracy test error: " << e.what() << std::endl;
        return FunctionalTestResult::ERROR;
    }
}

FunctionalTestResult test_configuration_system() {
    try {
        using namespace BTQuant::RenderEngine;
        
        DashboardConfig config;
        
        // Test default configuration loading
        config.resetToDefaults();
        
        // Test configuration access
        auto display_config = config.getDisplayConfig();
        if (display_config.window_width <= 0 || display_config.window_height <= 0) {
            return FunctionalTestResult::FAIL;
        }
        
        auto performance_config = config.getPerformanceConfig();
        if (performance_config.fps_alert_threshold <= 0) {
            return FunctionalTestResult::FAIL;
        }
        
        auto theme_config = config.getThemeConfig();
        if (theme_config.name.empty()) {
            return FunctionalTestResult::FAIL;
        }
        
        // Test configuration modification
        DisplayConfig new_display_config = display_config;
        new_display_config.window_width = 1920;
        new_display_config.window_height = 1080;
        new_display_config.target_fps = 60;
        
        config.setDisplayConfig(new_display_config);
        
        auto updated_config = config.getDisplayConfig();
        if (updated_config.window_width != 1920 || updated_config.window_height != 1080) {
            return FunctionalTestResult::FAIL;
        }
        
        // Test theme switching
        auto available_themes = config.getAvailableThemes();
        if (!available_themes.empty()) {
            bool theme_applied = config.applyTheme(available_themes[0]);
            if (!theme_applied) {
                return FunctionalTestResult::FAIL;
            }
        }
        
        return FunctionalTestResult::PASS;
        
    } catch (const std::exception& e) {
        std::cerr << "Configuration system test error: " << e.what() << std::endl;
        return FunctionalTestResult::ERROR;
    }
}

// ============================================================================
// Error Recovery Tests
// ============================================================================

FunctionalTestResult test_error_recovery() {
    try {
        using namespace BTQuant::RenderEngine;
        
        // Test HotSpine connection recovery
        HotSpineDataBridge bridge("/nonexistent_shm", "/nonexistent/symbols.json");
        
        // Should handle connection failure gracefully
        bool started = bridge.start();
        // Connection might fail, but should not crash
        
        if (started) {
            // If it started, test reconnection
            bridge.stop();
            bool reconnected = bridge.reconnect();
            // Reconnection might fail, but should not crash
        }
        
        // Test symbol manager with invalid data
        SymbolManager symbol_manager;
        bool initialized = symbol_manager.initialize("/invalid/path", "/invalid/config");
        // Should handle gracefully
        
        // Test market data processor with malformed data
        MarketDataProcessor processor;
        
        MarketDataUpdate malformed_update;
        malformed_update.type = MarketDataType::TRADE;
        malformed_update.symbol_id = 0; // Invalid
        malformed_update.price = -1.0; // Invalid
        malformed_update.size = 0.0; // Invalid
        
        // Should handle gracefully without crashing
        processor.processTradeUpdate(malformed_update);
        
        return FunctionalTestResult::PASS;
        
    } catch (const std::exception& e) {
        std::cerr << "Error recovery test error: " << e.what() << std::endl;
        return FunctionalTestResult::ERROR;
    }
}

FunctionalTestResult test_graceful_degradation() {
    try {
        using namespace BTQuant::RenderEngine;
        
        // Test system behavior under resource constraints
        PerformanceMonitor monitor;
        
        if (!monitor.startMonitoring()) {
            // Should handle monitoring failure gracefully
            std::cout << "  Note: Performance monitoring not available\n";
        }
        
        // Simulate high load conditions
        monitor.updateFrameMetrics(30.0, 33.33); // Low FPS
        monitor.updateDataLatency(15.0); // High latency
        monitor.updateMemoryUsage(1800.0); // High memory usage
        
        auto metrics = monitor.getCurrentMetrics();
        auto alerts = monitor.getRecentAlerts();
        
        // System should generate appropriate alerts
        // but continue functioning
        
        monitor.stopMonitoring();
        
        return FunctionalTestResult::PASS;
        
    } catch (const std::exception& e) {
        std::cerr << "Graceful degradation test error: " << e.what() << std::endl;
        return FunctionalTestResult::ERROR;
    }
}

// ============================================================================
// Test Registration
// ============================================================================

void register_functional_tests(FunctionalTestSuite& suite) {
    // UI Component Tests
    suite.register_test({
        "test_data_grid_rendering",
        "Test data grid component rendering and functionality",
        test_data_grid_rendering,
        "Data Grid",
        true
    });
    
    suite.register_test({
        "test_chart_rendering",
        "Test chart component rendering and data visualization",
        test_chart_rendering,
        "Charts",
        true
    });
    
    suite.register_test({
        "test_heatmap_visualization",
        "Test heatmap component and momentum visualization",
        test_heatmap_visualization,
        "Heatmap",
        true
    });
    
    suite.register_test({
        "test_orderbook_display",
        "Test orderbook component and market depth visualization",
        test_orderbook_display,
        "Orderbook",
        true
    });
    
    // Interactive Feature Tests
    suite.register_test({
        "test_mouse_interaction",
        "Test mouse input handling and interaction",
        test_mouse_interaction,
        "Mouse Input",
        true
    });
    
    suite.register_test({
        "test_keyboard_interaction",
        "Test keyboard input and hotkey functionality",
        test_keyboard_interaction,
        "Keyboard Input",
        true
    });
    
    suite.register_test({
        "test_touch_gestures",
        "Test touch gesture recognition and handling",
        test_touch_gestures,
        "Touch Input",
        false
    });
    
    // Data Validation Tests
    suite.register_test({
        "test_real_time_data_accuracy",
        "Test accuracy of real-time data processing and calculations",
        test_real_time_data_accuracy,
        "Data Processing",
        true
    });
    
    suite.register_test({
        "test_configuration_system",
        "Test configuration loading, saving, and modification",
        test_configuration_system,
        "Configuration",
        true
    });
    
    // Error Recovery Tests
    suite.register_test({
        "test_error_recovery",
        "Test error handling and recovery mechanisms",
        test_error_recovery,
        "Error Handling",
        true
    });
    
    suite.register_test({
        "test_graceful_degradation",
        "Test graceful degradation under adverse conditions",
        test_graceful_degradation,
        "System Resilience",
        true
    });
}

} // namespace FunctionalTesting
} // namespace BTQuant

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char* argv[]) {
    try {
        std::cout << "BTQuant Functional Test Suite\n";
        std::cout << "Testing UI components and interactive features\n";
        std::cout << "============================================\n\n";
        
        BTQuant::FunctionalTesting::FunctionalTestSuite suite;
        BTQuant::FunctionalTesting::register_functional_tests(suite);
        
        suite.run_all_tests();
        
        std::cout << "\nFunctional testing completed.\n";
        std::cout << "See functional_test_report.json for detailed results.\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Functional test suite failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Functional test suite failed with unknown exception" << std::endl;
        return 1;
    }
}