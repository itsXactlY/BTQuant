/**
 * Test for Data Quality Monitor
 */

#include "data/data_quality_monitor.hpp"
#include <iostream>
#include <thread>
#include <chrono>

using namespace BTQuant::Data;

int main() {
    std::cout << "Testing Data Quality Monitor..." << std::endl;
    
    // Set up an alert callback to receive notifications
    g_data_quality_monitor.set_alert_callback([](const DataQualityIssue& issue) {
        std::cout << "ALERT: " << issue.description 
                  << " (Severity: " << issue.severity << ")" << std::endl;
    });
    
    // Get current time for realistic testing
    auto current_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    // Test with a valid trade
    std::string symbol = "BTCUSD";
    TradeData valid_trade(current_time_ms - 100, 45000.0, 1.5f, TradeSide::BUY, 1, 0);

    std::cout << "\nProcessing valid trade..." << std::endl;
    auto issues1 = g_data_quality_monitor.process_trade(valid_trade, symbol);
    std::cout << "Issues detected: " << issues1.size() << std::endl;

    // Test with a duplicate trade
    std::cout << "\nProcessing duplicate trade..." << std::endl;
    auto issues2 = g_data_quality_monitor.process_trade(valid_trade, symbol);
    std::cout << "Issues detected: " << issues2.size() << std::endl;

    // Test with an out-of-order timestamp
    TradeData out_of_order_trade(current_time_ms - 200, 45001.0, 0.5f, TradeSide::SELL, 1, 0);  // Earlier timestamp

    std::cout << "\nProcessing out-of-order trade..." << std::endl;
    auto issues3 = g_data_quality_monitor.process_trade(out_of_order_trade, symbol);
    std::cout << "Issues detected: " << issues3.size() << std::endl;

    // Test with an invalid price
    TradeData invalid_price_trade(current_time_ms - 50, -100.0, 1.0f, TradeSide::BUY, 1, 0);

    std::cout << "\nProcessing invalid price trade..." << std::endl;
    auto issues4 = g_data_quality_monitor.process_trade(invalid_price_trade, symbol);
    std::cout << "Issues detected: " << issues4.size() << std::endl;

    // Test with an invalid volume
    TradeData invalid_volume_trade(current_time_ms - 25, 45002.0, -1.0f, TradeSide::BUY, 1, 0);

    std::cout << "\nProcessing invalid volume trade..." << std::endl;
    auto issues5 = g_data_quality_monitor.process_trade(invalid_volume_trade, symbol);
    std::cout << "Issues detected: " << issues5.size() << std::endl;
    
    // Get metrics
    auto metrics = g_data_quality_monitor.get_metrics();
    std::cout << "\nData Quality Metrics:" << std::endl;
    std::cout << "  Total trades processed: " << metrics.total_trades_processed << std::endl;
    std::cout << "  Duplicate trade issues: " << metrics.duplicate_trade_issues << std::endl;
    std::cout << "  Out-of-order timestamp issues: " << metrics.out_of_order_timestamp_issues << std::endl;
    std::cout << "  Invalid price issues: " << metrics.invalid_price_issues << std::endl;
    std::cout << "  Invalid volume issues: " << metrics.invalid_volume_issues << std::endl;
    
    // Get recent issues
    auto recent_issues = g_data_quality_monitor.get_recent_issues();
    std::cout << "\nRecent Issues (" << recent_issues.size() << " total):" << std::endl;
    for (const auto& issue : recent_issues) {
        std::cout << "  - " << issue.description << " at " << issue.timestamp << std::endl;
    }
    
    std::cout << "\nData Quality Monitor test completed!" << std::endl;
    
    return 0;
}