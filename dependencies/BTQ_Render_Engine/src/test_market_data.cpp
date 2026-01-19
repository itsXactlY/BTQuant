#include "market_data_processor.hpp"
#include <iostream>
#include <chrono>
#include <random>

using namespace BTQuant::RenderEngine;

int main() {
    std::cout << "=== Market Data Processor Quick Test ===" << std::endl;
    
    try {
        MarketDataProcessor processor;
        
        // Create test market data updates
        std::vector<MarketDataUpdate> test_updates;
        
        // Generate test trade data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> price_dist(50000.0, 51000.0);
        std::uniform_real_distribution<> size_dist(0.001, 1.0);
        
        for (int i = 0; i < 100; ++i) {
            MarketDataUpdate update;
            update.type = MarketDataType::TRADE;
            update.symbol_id = 12345;
            update.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            update.price = price_dist(gen);
            update.size = size_dist(gen);
            update.side = (i % 2 == 0) ? "buy" : "sell";
            
            test_updates.push_back(update);
        }
        
        // Process test data
        auto start_time = std::chrono::high_resolution_clock::now();
        for (const auto& update : test_updates) {
            processor.processTradeUpdate(update);
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        std::cout << "✅ Processed " << test_updates.size() << " trades in " << duration.count() << "µs" << std::endl;
        std::cout << "   Average: " << (double)duration.count() / test_updates.size() << "µs per trade" << std::endl;
        
        // Test analytics
        auto analytics = processor.getSymbolAnalytics(12345);
        std::cout << "✅ Trade count: " << analytics.trade_count << std::endl;
        std::cout << "✅ VWAP: " << std::fixed << std::setprecision(2) << analytics.vwap << std::endl;
        std::cout << "✅ Buy/Sell ratio: " << std::fixed << std::setprecision(2) << analytics.buy_sell_ratio << std::endl;
        
        // Test OHLCV candles
        auto candles_1min = processor.getCandles(12345, TimeFrame::TF_1MIN);
        auto current_candle = processor.getCurrentCandle(12345, TimeFrame::TF_1MIN);
        
        if (current_candle) {
            std::cout << "✅ Current 1-min candle: Open=" << current_candle->open 
                      << ", High=" << current_candle->high 
                      << ", Low=" << current_candle->low 
                      << ", Close=" << current_candle->close 
                      << ", Volume=" << current_candle->volume << std::endl;
        }
        
        // Test market summary
        auto summary = processor.getMarketSummary();
        std::cout << "✅ Market summary: " << summary.total_symbols << " symbols" << std::endl;
        
        // Test rankings
        auto rankings = processor.getRankings(RankingCriteria::VOLUME, 10);
        std::cout << "✅ Volume rankings: " << rankings.size() << " symbols" << std::endl;
        
        std::cout << "\n🎉 Market Data Processor test passed!" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
}
