#include "hotspine_writer.hpp"
#include "hotspine_layout.hpp"
#include "market_data_types.h"

#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <cassert>

using namespace std::chrono;

// Simple validation test
bool testHotSpineIntegration() {
    std::cout << "=== HotSpine Integration Validation Test ===" << std::endl;
    
    // Create writer
    HotSpine::HotSpineWriter writer("/btquant_hotspine_test");
    
    if (!writer.isHealthy()) {
        std::cerr << "ERROR: HotSpine writer is not healthy" << std::endl;
        return false;
    }
    
    // Configure for testing
    writer.setBatchingEnabled(true);
    writer.setBatchSize(10);
    
    // Create test trades
    std::vector<MarketData::Trade> test_trades;
    
    for (int i = 0; i < 100; ++i) {
        MarketData::Trade trade;
        trade.timestamp_us = system_clock::now().time_since_epoch().count();
        trade.exchange = "binance";
        trade.symbol = "BTC/USDT";
        trade.market_type = "spot";
        trade.trade_id = "test_" + std::to_string(i);
        trade.price = 50000.0 + i;
        trade.quantity = 0.01 + (i * 0.001);
        trade.side = (i % 2 == 0) ? "buy" : "sell";
        trade.is_buyer_maker = (i % 2 == 0);
        
        test_trades.push_back(trade);
    }
    
    std::cout << "Writing " << test_trades.size() << " test trades..." << std::endl;
    
    // Write test trades
    auto start_time = high_resolution_clock::now();
    
    for (const auto& trade : test_trades) {
        if (!writer.writeTrade(trade)) {
            std::cerr << "ERROR: Failed to write trade " << trade.trade_id << std::endl;
            return false;
        }
    }
    
    // Flush any remaining trades
    writer.flushBatch();
    
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time);
    
    std::cout << "Successfully wrote " << test_trades.size() << " trades in " 
              << duration.count() << " ms" << std::endl;
    
    // Check statistics
    uint64_t trades_written = writer.getTradesWritten();
    uint64_t write_errors = writer.getWriteErrors();
    
    std::cout << "Statistics:" << std::endl;
    std::cout << "  Trades written: " << trades_written << std::endl;
    std::cout << "  Write errors: " << write_errors << std::endl;
    std::cout << "  Throughput: " 
              << (trades_written * 1000.0 / std::max(duration.count(), 1L)) 
              << " trades/sec" << std::endl;
    
    // Validate results
    if (trades_written != test_trades.size()) {
        std::cerr << "ERROR: Expected " << test_trades.size() 
                  << " trades, but got " << trades_written << std::endl;
        return false;
    }
    
    if (write_errors > 0) {
        std::cerr << "ERROR: " << write_errors << " write errors occurred" << std::endl;
        return false;
    }
    
    std::cout << "✓ All validation tests passed!" << std::endl;
    return true;
}

// Performance benchmark
void benchmarkHotSpinePerformance() {
    std::cout << "\n=== HotSpine Performance Benchmark ===" << std::endl;
    
    HotSpine::HotSpineWriter writer("/btquant_hotspine_bench");
    writer.setBatchingEnabled(true);
    writer.setBatchSize(100); // Larger batch for benchmark
    
    const size_t num_trades = 10000;
    std::cout << "Benchmarking with " << num_trades << " trades..." << std::endl;
    
    auto start_time = high_resolution_clock::now();
    
    for (size_t i = 0; i < num_trades; ++i) {
        MarketData::Trade trade;
        trade.timestamp_us = system_clock::now().time_since_epoch().count();
        trade.exchange = "binance";
        trade.symbol = "BTC/USDT";
        trade.market_type = "spot";
        trade.trade_id = "bench_" + std::to_string(i);
        trade.price = 50000.0 + (i % 1000);
        trade.quantity = 0.01 + ((i % 100) * 0.001);
        trade.side = (i % 2 == 0) ? "buy" : "sell";
        
        writer.writeTrade(trade);
    }
    
    writer.flushBatch();
    
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time);
    
    uint64_t trades_written = writer.getTradesWritten();
    
    std::cout << "Performance Results:" << std::endl;
    std::cout << "  Total time: " << duration.count() << " ms" << std::endl;
    std::cout << "  Trades written: " << trades_written << std::endl;
    std::cout << "  Throughput: " 
              << (trades_written * 1000.0 / std::max(duration.count(), 1L)) 
              << " trades/sec" << std::endl;
    std::cout << "  Latency per trade: " 
              << (duration.count() * 1000.0 / std::max(trades_written, 1ULL)) 
              << " µs" << std::endl;
}

int main() {
    try {
        if (!testHotSpineIntegration()) {
            std::cerr << "HotSpine integration validation FAILED" << std::endl;
            return 1;
        }
        
        benchmarkHotSpinePerformance();
        
        std::cout << "\n=== All tests completed successfully ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        return 1;
    }
}