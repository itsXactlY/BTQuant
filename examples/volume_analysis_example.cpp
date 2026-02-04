#include "task_scheduler.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>

/**
 * @brief Example implementation of volume analysis calculations
 * Demonstrates how to use the TaskScheduler for common volume-based calculations
 */
int main() {
    std::cout << "=== Volume Analysis Example ===" << std::endl;

    // Initialize the TaskScheduler with 4 threads
    btq::TaskScheduler scheduler(4);

    // Generate sample trade data
    std::cout << "Generating sample trade data..." << std::endl;
    std::vector<btq::Trade> trades;
    std::mt19937 gen(42); // Fixed seed for reproducible results
    std::uniform_real_distribution<> price_dist(100.0, 200.0);
    std::uniform_real_distribution<> volume_dist(10.0, 100.0);

    auto start_time = std::chrono::system_clock::now();
    for (int i = 0; i < 10000; ++i) {
        btq::Trade trade;
        trade.timestamp = start_time + std::chrono::milliseconds(i * 100); // 100ms intervals
        trade.price = price_dist(gen);
        trade.volume = volume_dist(gen);
        trades.push_back(trade);
    }

    std::cout << "Generated " << trades.size() << " trades" << std::endl;

    // Example 1: Calculate volume profile
    std::cout << "\n1. Calculating volume profile..." << std::endl;
    auto volume_profile_future = scheduler.calculate_volume_profile_async(trades, 90.0, 210.0, 120);
    
    // Example 2: Calculate VWAP (Volume Weighted Average Price)
    std::cout << "2. Calculating VWAP..." << std::endl;
    auto vwap_future = scheduler.calculate_volume_weighted_average_price_async(trades);
    
    // Example 3: Calculate volume by time
    std::cout << "3. Calculating volume by time (5-minute intervals)..." << std::endl;
    auto volume_by_time_future = scheduler.calculate_volume_by_time_async(trades, 5);
    
    // Example 4: Calculate rolling volume profile
    std::cout << "4. Calculating rolling volume profile..." << std::endl;
    auto rolling_volume_future = scheduler.calculate_rolling_volume_profile_async(trades, 90.0, 210.0, 120, 1000);
    
    // Example 5: Calculate time-weighted volume
    std::cout << "5. Calculating time-weighted volume..." << std::endl;
    auto time_weighted_volume_future = scheduler.calculate_time_weighted_volume_async(trades, 10);

    // Wait for all calculations to complete and display results
    std::cout << "\nWaiting for calculations to complete..." << std::endl;

    try {
        // Retrieve volume profile
        auto volume_profile = volume_profile_future.get();
        std::cout << "Volume profile calculated with " << volume_profile.size() << " price levels" << std::endl;
        
        // Display first few and last few values of volume profile
        std::cout << "Sample volume profile values:" << std::endl;
        for (size_t i = 0; i < std::min(static_cast<size_t>(5), volume_profile.size()); ++i) {
            std::cout << "  Level " << i << ": " << std::fixed << std::setprecision(2) << volume_profile[i] << std::endl;
        }
        if (volume_profile.size() > 5) {
            std::cout << "  ..." << std::endl;
            for (size_t i = std::max(volume_profile.size() - 5, static_cast<size_t>(5)); i < volume_profile.size(); ++i) {
                std::cout << "  Level " << i << ": " << std::fixed << std::setprecision(2) << volume_profile[i] << std::endl;
            }
        }

        // Retrieve VWAP
        auto vwap = vwap_future.get();
        std::cout << "\nVWAP: " << std::fixed << std::setprecision(4) << vwap << std::endl;

        // Retrieve volume by time
        auto volume_by_time = volume_by_time_future.get();
        std::cout << "Volume by time calculated with " << volume_by_time.size() << " time intervals" << std::endl;
        
        // Display sample time-based volumes
        std::cout << "Sample volume by time values:" << std::endl;
        for (size_t i = 0; i < std::min(static_cast<size_t>(3), volume_by_time.size()); ++i) {
            std::cout << "  Interval " << i << ": " << std::fixed << std::setprecision(2) << volume_by_time[i] << std::endl;
        }

        // Retrieve rolling volume profile
        auto rolling_volume = rolling_volume_future.get();
        std::cout << "Rolling volume profile calculated with " << rolling_volume.size() << " price levels" << std::endl;

        // Retrieve time-weighted volume
        auto time_weighted_vol = time_weighted_volume_future.get();
        std::cout << "Time-weighted volume calculated" << std::endl;

        std::cout << "\nAll volume analysis calculations completed successfully!" << std::endl;
        std::cout << "This example demonstrates how to use the TaskScheduler for common volume-based calculations." << std::endl;
        std::cout << "These calculations can be used for footprint charts, volume profile analysis, and VWAP calculations." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error during calculations: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}