/**
 * Enhanced test for Data Quality Monitor
 * Tests the new enhanced features
 */

#include "data/data_quality_monitor.hpp"
#include <iostream>
#include <thread>
#include <chrono>

using namespace BTQuant::Data;

int main() {
    std::cout << "Testing Enhanced Data Quality Monitor Features..." << std::endl;

    // Set up an alert callback to receive notifications
    g_data_quality_monitor.set_alert_callback([](const DataQualityIssue& issue) {
        std::cout << "ALERT: " << issue.description
                  << " (Severity: " << issue.severity << ")" << std::endl;
    });

    // Get current time for realistic testing
    auto current_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    std::string symbol = "BTCUSD";
    
    // Test 1: Valid trades to establish baseline
    std::cout << "\n--- Test 1: Establishing baseline with valid trades ---" << std::endl;
    for (int i = 0; i < 5; ++i) {
        TradeData trade(current_time_ms + i * 100, 45000.0 + i * 10.0, 1.0f + i * 0.1f, TradeSide::BUY, 1, 0);
        auto issues = g_data_quality_monitor.process_trade(trade, symbol);
        std::cout << "Processed trade " << i+1 << ", issues: " << issues.size() << std::endl;
    }

    // Test 2: Large gap to trigger missing data detection
    std::cout << "\n--- Test 2: Simulating missing data (large time gap) ---" << std::endl;
    TradeData gap_trade(current_time_ms + 10000, 45050.0, 1.5f, TradeSide::SELL, 1, 0);  // 10 second gap
    auto issues = g_data_quality_monitor.process_trade(gap_trade, symbol);
    std::cout << "Processed gap trade, issues: " << issues.size() << std::endl;

    // Test 3: Duplicate trade detection
    std::cout << "\n--- Test 3: Testing duplicate trade detection ---" << std::endl;
    TradeData dup_trade(current_time_ms + 10100, 45050.0, 1.5f, TradeSide::SELL, 1, 0);  // Same as above
    issues = g_data_quality_monitor.process_trade(dup_trade, symbol);
    std::cout << "Processed duplicate trade, issues: " << issues.size() << std::endl;

    // Test 4: Out-of-order timestamp
    std::cout << "\n--- Test 4: Testing out-of-order timestamp ---" << std::endl;
    TradeData ooo_trade(current_time_ms + 5000, 45040.0, 0.8f, TradeSide::BUY, 1, 0);  // Earlier than expected
    issues = g_data_quality_monitor.process_trade(ooo_trade, symbol);
    std::cout << "Processed out-of-order trade, issues: " << issues.size() << std::endl;

    // Test 5: Latency simulation
    std::cout << "\n--- Test 5: Testing latency detection ---" << std::endl;
    // Sleep to simulate processing delay
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    TradeData latency_trade(current_time_ms + 10200, 45060.0, 2.0f, TradeSide::BUY, 1, 0);
    issues = g_data_quality_monitor.process_trade(latency_trade, symbol);
    std::cout << "Processed latency test trade, issues: " << issues.size() << std::endl;

    // Get metrics
    auto metrics = g_data_quality_monitor.get_metrics();
    std::cout << "\n--- Data Quality Metrics ---" << std::endl;
    std::cout << "  Total trades processed: " << metrics.total_trades_processed << std::endl;
    std::cout << "  Missing data issues: " << metrics.missing_data_issues << std::endl;
    std::cout << "  Duplicate trade issues: " << metrics.duplicate_trade_issues << std::endl;
    std::cout << "  Out-of-order timestamp issues: " << metrics.out_of_order_timestamp_issues << std::endl;
    std::cout << "  Latency issues: " << metrics.latency_issues << std::endl;
    std::cout << "  Invalid price issues: " << metrics.invalid_price_issues << std::endl;
    std::cout << "  Invalid volume issues: " << metrics.invalid_volume_issues << std::endl;
    std::cout << "  Average latency: " << metrics.average_latency_ms << " ms" << std::endl;

    // Get quality summary
    std::cout << "\n--- Quality Summary ---" << std::endl;
    std::cout << g_data_quality_monitor.get_quality_summary() << std::endl;

    // Get high severity issues
    auto high_severity_issues = g_data_quality_monitor.get_high_severity_issues(0.5);
    std::cout << "High severity issues (threshold 0.5): " << high_severity_issues.size() << std::endl;

    // Get recent issues
    auto recent_issues = g_data_quality_monitor.get_recent_issues();
    std::cout << "\n--- Recent Issues (" << recent_issues.size() << " total) ---" << std::endl;
    for (const auto& issue : recent_issues) {
        std::cout << "  - [" << static_cast<int>(issue.type) << "] " << issue.description 
                  << " at " << issue.timestamp << " (Severity: " << issue.severity << ")" << std::endl;
    }

    std::cout << "\nEnhanced Data Quality Monitor test completed!" << std::endl;

    return 0;
}