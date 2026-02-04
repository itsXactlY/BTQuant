#include "dependencies/BTQ_Render_Engine/include/task_scheduler.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

int main() {
    std::cout << "Testing multi-threaded calculations..." << std::endl;
    
    // Create a TaskScheduler instance
    btq::TaskScheduler scheduler(4); // Use 4 threads
    
    // Generate sample trade data for testing
    std::vector<btq::Trade> trades;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<> price_dist(100.0, 200.0);
    std::uniform_real_distribution<> volume_dist(10.0, 100.0);
    
    auto start_time = std::chrono::system_clock::now();
    for (int i = 0; i < 50000; ++i) {
        btq::Trade trade;
        trade.timestamp = start_time + std::chrono::seconds(i);
        trade.price = price_dist(gen);
        trade.volume = volume_dist(gen);
        trades.push_back(trade);
    }
    
    std::cout << "Generated " << trades.size() << " trades" << std::endl;
    
    // Test volume calculations
    std::cout << "\nTesting volume calculations..." << std::endl;
    
    auto volume_profile_future = scheduler.calculate_volume_profile_async(trades, 90.0, 210.0, 100);
    auto vwap_future = scheduler.calculate_volume_weighted_average_price_async(trades);
    auto volume_by_time_future = scheduler.calculate_volume_by_time_async(trades, 5);
    
    // Test enhanced volume calculations
    auto rolling_volume_future = scheduler.calculate_rolling_volume_profile_async(trades, 90.0, 210.0, 100, 1000);
    auto time_based_volume_future = scheduler.calculate_time_based_volume_async(trades, 60);
    
    std::cout << "Volume calculation tasks enqueued" << std::endl;
    
    // Test indicator computations
    std::cout << "\nTesting indicator computations..." << std::endl;
    
    // Extract prices from trades for indicator calculations
    std::vector<double> prices;
    for (const auto& trade : trades) {
        prices.push_back(trade.price);
    }
    
    auto sma_future = scheduler.calculate_sma_async(prices, 20);
    auto ema_future = scheduler.calculate_ema_async(prices, 20);
    auto rsi_future = scheduler.calculate_rsi_async(prices, 14);
    
    // Test enhanced indicator computations
    auto adaptive_sma_future = scheduler.calculate_adaptive_sma_async(prices, 10, 50);
    
    std::cout << "Indicator calculation tasks enqueued" << std::endl;
    
    // Test data processing
    std::cout << "\nTesting data processing..." << std::endl;
    
    auto candle_future = scheduler.aggregate_candles_async(trades, std::chrono::seconds(300)); // 5-minute candles
    
    // Test enhanced data processing
    auto dynamic_candle_future = scheduler.create_dynamic_timeframe_candles_async(trades, std::chrono::seconds(60), 50.0);
    
    std::cout << "Data processing tasks enqueued" << std::endl;
    
    // Wait for results
    std::cout << "\nWaiting for results..." << std::endl;
    
    try {
        auto volume_profile = volume_profile_future.get();
        std::cout << "Volume profile calculated, size: " << volume_profile.size() << std::endl;
        
        auto vwap = vwap_future.get();
        std::cout << "VWAP calculated: " << vwap << std::endl;
        
        auto volume_by_time = volume_by_time_future.get();
        std::cout << "Volume by time calculated, size: " << volume_by_time.size() << std::endl;
        
        auto rolling_volume = rolling_volume_future.get();
        std::cout << "Rolling volume profile calculated, size: " << rolling_volume.size() << std::endl;
        
        auto time_based_volume = time_based_volume_future.get();
        std::cout << "Time-based volume calculated, size: " << time_based_volume.size() << std::endl;
        
        auto sma_values = sma_future.get();
        std::cout << "SMA calculated, size: " << sma_values.size() << std::endl;
        
        auto ema_values = ema_future.get();
        std::cout << "EMA calculated, size: " << ema_values.size() << std::endl;
        
        auto rsi_values = rsi_future.get();
        std::cout << "RSI calculated, size: " << rsi_values.size() << std::endl;
        
        auto adaptive_sma_values = adaptive_sma_future.get();
        std::cout << "Adaptive SMA calculated, size: " << adaptive_sma_values.size() << std::endl;
        
        auto candles = candle_future.get();
        std::cout << "Candles aggregated, count: " << candles.size() << std::endl;
        
        auto dynamic_candles = dynamic_candle_future.get();
        std::cout << "Dynamic timeframe candles created, count: " << dynamic_candles.size() << std::endl;
        
        std::cout << "\nAll multi-threaded calculations completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}