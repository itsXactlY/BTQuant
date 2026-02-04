#include "task_scheduler.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>

/**
 * @brief Example implementation of technical indicator calculations
 * Demonstrates how to use the TaskScheduler for common technical indicators
 */
int main() {
    std::cout << "=== Technical Indicators Example ===" << std::endl;

    // Initialize the TaskScheduler with 4 threads
    btq::TaskScheduler scheduler(4);

    // Generate sample price data
    std::cout << "Generating sample price data..." << std::endl;
    std::vector<double> prices;
    std::mt19937 gen(42); // Fixed seed for reproducible results
    std::normal_distribution<> price_noise(0.0, 0.5); // Small random noise
    
    // Create a base price with some trend
    double base_price = 150.0;
    for (int i = 0; i < 1000; ++i) {
        base_price += price_noise(gen); // Add some random movement
        if (i % 100 == 0) base_price += 0.5; // Small upward trend every 100 periods
        prices.push_back(base_price);
    }

    std::cout << "Generated " << prices.size() << " price points" << std::endl;

    // Example 1: Calculate Simple Moving Average (SMA)
    std::cout << "\n1. Calculating Simple Moving Average (SMA)..." << std::endl;
    auto sma_future = scheduler.calculate_sma_async(prices, 20);
    
    // Example 2: Calculate Exponential Moving Average (EMA)
    std::cout << "2. Calculating Exponential Moving Average (EMA)..." << std::endl;
    auto ema_future = scheduler.calculate_ema_async(prices, 20);
    
    // Example 3: Calculate RSI (Relative Strength Index)
    std::cout << "3. Calculating RSI..." << std::endl;
    auto rsi_future = scheduler.calculate_rsi_async(prices, 14);
    
    // Example 4: Calculate Bollinger Bands
    std::cout << "4. Calculating Bollinger Bands..." << std::endl;
    auto bollinger_future = scheduler.calculate_bollinger_bands_async(prices, 20, 2.0);
    
    // Example 5: Calculate MACD (Moving Average Convergence Divergence)
    std::cout << "5. Calculating MACD..." << std::endl;
    auto macd_future = scheduler.calculate_macd_async(prices, 12, 26, 9);
    
    // Example 6: Calculate Adaptive SMA
    std::cout << "6. Calculating Adaptive SMA..." << std::endl;
    auto adaptive_sma_future = scheduler.calculate_adaptive_sma_async(prices, 10, 50);
    
    // Example 7: Calculate Hull Moving Average
    std::cout << "7. Calculating Hull Moving Average..." << std::endl;
    auto hull_ma_future = scheduler.calculate_hull_moving_average_async(prices, 16);
    
    // Example 8: Calculate Keltner Channels
    std::cout << "8. Calculating Keltner Channels..." << std::endl;
    // First, we need to generate some candle data for Keltner channels
    std::vector<btq::Candle> candles;
    for (size_t i = 0; i < prices.size(); ++i) {
        btq::Candle candle;
        candle.timestamp = std::chrono::system_clock::now() + std::chrono::minutes(i);
        candle.open = prices[i];
        candle.high = prices[i] + 0.5;
        candle.low = prices[i] - 0.5;
        candle.close = prices[i];
        candle.volume = 1000.0 + (i % 100) * 10;
        candles.push_back(candle);
    }
    auto keltner_future = scheduler.calculate_keltner_channels_async(candles, 20, 2.0);

    // Wait for all calculations to complete and display results
    std::cout << "\nWaiting for calculations to complete..." << std::endl;

    try {
        // Retrieve SMA
        auto sma_values = sma_future.get();
        std::cout << "SMA calculated with " << sma_values.size() << " values" << std::endl;
        
        // Display first few SMA values
        std::cout << "Sample SMA values:" << std::endl;
        for (size_t i = 0; i < std::min(static_cast<size_t>(5), sma_values.size()); ++i) {
            std::cout << "  Period " << i << ": " << std::fixed << std::setprecision(4) << sma_values[i] << std::endl;
        }

        // Retrieve EMA
        auto ema_values = ema_future.get();
        std::cout << "\nEMA calculated with " << ema_values.size() << " values" << std::endl;
        
        // Display first few EMA values
        std::cout << "Sample EMA values:" << std::endl;
        for (size_t i = 0; i < std::min(static_cast<size_t>(5), ema_values.size()); ++i) {
            std::cout << "  Period " << i << ": " << std::fixed << std::setprecision(4) << ema_values[i] << std::endl;
        }

        // Retrieve RSI
        auto rsi_values = rsi_future.get();
        std::cout << "\nRSI calculated with " << rsi_values.size() << " values" << std::endl;
        
        // Display first few RSI values
        std::cout << "Sample RSI values:" << std::endl;
        for (size_t i = 0; i < std::min(static_cast<size_t>(5), rsi_values.size()); ++i) {
            std::cout << "  Period " << i << ": " << std::fixed << std::setprecision(2) << rsi_values[i] << std::endl;
        }

        // Retrieve Bollinger Bands
        auto bollinger_result = bollinger_future.get();
        const auto& upper_band = std::get<0>(bollinger_result);
        const auto& middle_band = std::get<1>(bollinger_result);
        const auto& lower_band = std::get<2>(bollinger_result);
        std::cout << "\nBollinger Bands calculated with " << upper_band.size() << " values each" << std::endl;
        
        // Display first few Bollinger Band values
        std::cout << "Sample Bollinger Band values:" << std::endl;
        for (size_t i = 0; i < std::min(static_cast<size_t>(3), upper_band.size()); ++i) {
            std::cout << "  Period " << i << ": Upper=" << std::fixed << std::setprecision(4) << upper_band[i] 
                      << ", Middle=" << middle_band[i] << ", Lower=" << lower_band[i] << std::endl;
        }

        // Retrieve MACD
        auto macd_values = macd_future.get();
        std::cout << "\nMACD calculated with " << macd_values.size() << " values" << std::endl;
        
        // Display first few MACD values
        std::cout << "Sample MACD values:" << std::endl;
        for (size_t i = 0; i < std::min(static_cast<size_t>(5), macd_values.size()); ++i) {
            std::cout << "  Period " << i << ": " << std::fixed << std::setprecision(6) << macd_values[i] << std::endl;
        }

        // Retrieve Adaptive SMA
        auto adaptive_sma_values = adaptive_sma_future.get();
        std::cout << "\nAdaptive SMA calculated with " << adaptive_sma_values.size() << " values" << std::endl;

        // Retrieve Hull Moving Average
        auto hull_ma_values = hull_ma_future.get();
        std::cout << "\nHull Moving Average calculated with " << hull_ma_values.size() << " values" << std::endl;

        // Retrieve Keltner Channels
        auto keltner_values = keltner_future.get();
        std::cout << "\nKeltner Channels calculated with " << keltner_values.size() << " values" << std::endl;

        std::cout << "\nAll technical indicator calculations completed successfully!" << std::endl;
        std::cout << "This example demonstrates how to use the TaskScheduler for common technical analysis indicators." << std::endl;
        std::cout << "These indicators can be used for charting, trading signals, and market analysis." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error during calculations: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}