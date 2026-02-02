#include <iostream>
#include <thread>
#include <chrono>
#include "dependencies/BTQ_Render_Engine/include/data/data_quality_monitor.hpp"
#include "dependencies/BTQ_Render_Engine/include/data/TradeData.h"

using namespace BTQuant::Data;

int main() {
    std::cout << "Testing Data Quality Monitor Enhancements..." << std::endl;
    
    // Get reference to global data quality monitor
    auto& monitor = g_data_quality_monitor;
    
    // Enable console alerts for testing
    monitor.set_console_alerts_enabled(true);
    
    std::cout << "1. Testing normal trade processing..." << std::endl;
    
    // Test normal trades
    TradeData trade1(1000, 50000.0, 1.0f, TradeSide::BUY, 1, 0);
    TradeData trade2(1001, 50001.0, 0.5f, TradeSide::SELL, 1, 0);
    TradeData trade3(1002, 50002.0, 2.0f, TradeSide::BUY, 1, 0);
    
    auto issues1 = monitor.process_trade(trade1, "BTCUSD");
    auto issues2 = monitor.process_trade(trade2, "BTCUSD");
    auto issues3 = monitor.process_trade(trade3, "BTCUSD");
    
    std::cout << "   Processed 3 normal trades. Issues found: " 
              << (issues1.size() + issues2.size() + issues3.size()) << std::endl;
    
    std::cout << "\n2. Testing duplicate trade detection..." << std::endl;
    
    // Test duplicate detection
    TradeData duplicate_trade(1003, 50003.0, 1.5f, TradeSide::BUY, 1, 0);
    auto issues4 = monitor.process_trade(duplicate_trade, "BTCUSD");
    auto issues5 = monitor.process_trade(duplicate_trade, "BTCUSD"); // This should be detected as duplicate
    
    std::cout << "   Processed duplicate trade. Issues found: " << issues5.size() << std::endl;
    
    std::cout << "\n3. Testing out-of-order timestamp detection..." << std::endl;
    
    // Test out-of-order timestamp (earlier timestamp after later one)
    TradeData out_of_order_trade(999, 50004.0, 1.0f, TradeSide::SELL, 1, 0); // Earlier timestamp
    auto issues6 = monitor.process_trade(out_of_order_trade, "BTCUSD");
    
    std::cout << "   Processed out-of-order trade. Issues found: " << issues6.size() << std::endl;
    
    std::cout << "\n4. Testing missing data detection..." << std::endl;
    
    // Test missing data (large gap in timestamps)
    TradeData gap_trade(10000, 50005.0, 1.0f, TradeSide::BUY, 1, 0); // Much later timestamp
    auto issues7 = monitor.process_trade(gap_trade, "BTCUSD");
    
    std::cout << "   Processed trade with large gap. Issues found: " << issues7.size() << std::endl;
    
    std::cout << "\n5. Testing invalid data detection..." << std::endl;
    
    // Test invalid price
    TradeData invalid_price_trade(10001, -1.0, 1.0f, TradeSide::BUY, 1, 0); // Invalid price
    auto issues8 = monitor.process_trade(invalid_price_trade, "BTCUSD");
    
    std::cout << "   Processed invalid price trade. Issues found: " << issues8.size() << std::endl;
    
    std::cout << "\n6. Testing user alert functionality..." << std::endl;
    
    // Test manual user alert
    monitor.alert_user_to_data_problems("BTCUSD", "Manual test alert for duplicate trades", 0.7);
    
    std::cout << "\n7. Generating comprehensive report..." << std::endl;
    
    // Generate comprehensive report
    monitor.generate_comprehensive_alert_report();
    
    std::cout << "\n8. Getting data quality metrics..." << std::endl;
    
    auto metrics = monitor.get_metrics();
    std::cout << "   Total trades processed: " << metrics.total_trades_processed << std::endl;
    std::cout << "   Missing data issues: " << metrics.missing_data_issues << std::endl;
    std::cout << "   Duplicate trade issues: " << metrics.duplicate_trade_issues << std::endl;
    std::cout << "   Out-of-order timestamp issues: " << metrics.out_of_order_timestamp_issues << std::endl;
    std::cout << "   Latency issues: " << metrics.latency_issues << std::endl;
    std::cout << "   Invalid price issues: " << metrics.invalid_price_issues << std::endl;
    std::cout << "   Invalid volume issues: " << metrics.invalid_volume_issues << std::endl;
    
    std::cout << "\nData Quality Monitor Enhancement Test Completed!" << std::endl;
    
    return 0;
}