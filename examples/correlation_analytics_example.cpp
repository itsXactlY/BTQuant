#include "task_scheduler.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <algorithm>

/**
 * @brief Example implementation of correlation and advanced analytics
 * Demonstrates how to use the TaskScheduler for advanced analytical calculations
 */
int main() {
    std::cout << "=== Correlation and Advanced Analytics Example ===" << std::endl;

    // Initialize the TaskScheduler with 4 threads
    btq::TaskScheduler scheduler(4);

    // Generate sample trade data for two different instruments
    std::cout << "Generating sample trade data for two instruments..." << std::endl;
    std::vector<btq::Trade> instrument1_trades;
    std::vector<btq::Trade> instrument2_trades;
    std::mt19937 gen(42); // Fixed seed for reproducible results
    std::normal_distribution<> price_noise1(0.0, 0.5); // Noise for instrument 1
    std::normal_distribution<> price_noise2(0.0, 0.4); // Noise for instrument 2 (correlated with instrument 1)

    auto start_time = std::chrono::system_clock::now();
    double base_price1 = 150.0;
    double base_price2 = 200.0;
    
    for (int i = 0; i < 1000; ++i) {
        // Generate correlated prices
        double noise1 = price_noise1(gen);
        double noise2 = price_noise2(gen) + 0.3 * noise1; // Add correlation
        
        btq::Trade trade1, trade2;
        trade1.timestamp = start_time + std::chrono::seconds(i);
        trade1.price = base_price1 + noise1;
        trade1.volume = 10.0 + (i % 50);
        instrument1_trades.push_back(trade1);
        
        trade2.timestamp = start_time + std::chrono::seconds(i);
        trade2.price = base_price2 + noise2;
        trade2.volume = 15.0 + (i % 40);
        instrument2_trades.push_back(trade2);
        
        base_price1 += 0.01; // Small trend
        base_price2 += 0.02; // Slightly different trend
    }

    std::cout << "Generated " << instrument1_trades.size() << " trades for instrument 1" << std::endl;
    std::cout << "Generated " << instrument2_trades.size() << " trades for instrument 2" << std::endl;

    // Extract price series
    std::vector<double> prices1, prices2;
    for (const auto& trade : instrument1_trades) {
        prices1.push_back(trade.price);
    }
    for (const auto& trade : instrument2_trades) {
        prices2.push_back(trade.price);
    }

    // Example 1: Calculate correlation between two price series
    std::cout << "\n1. Calculating correlation between two instruments..." << std::endl;
    auto correlation_future = scheduler.calculate_correlation_async(prices1, prices2);
    
    // Example 2: Calculate ATR (Average True Range)
    std::cout << "2. Calculating ATR for instrument 1..." << std::endl;
    // Create candles from trades for ATR calculation
    std::vector<btq::Candle> candles1;
    for (size_t i = 0; i < instrument1_trades.size(); ++i) {
        btq::Candle candle;
        candle.timestamp = instrument1_trades[i].timestamp;
        candle.open = instrument1_trades[i].price;
        candle.high = instrument1_trades[i].price + 0.5;
        candle.low = instrument1_trades[i].price - 0.5;
        candle.close = instrument1_trades[i].price + ((i % 3) - 1) * 0.2; // Small variation
        candle.volume = instrument1_trades[i].volume;
        candles1.push_back(candle);
    }
    auto atr_future = scheduler.calculate_atr_async(candles1, 14);
    
    // Example 3: Calculate Stochastic Oscillator
    std::cout << "3. Calculating Stochastic Oscillator for instrument 1..." << std::endl;
    auto stochastic_future = scheduler.calculate_stochastic_oscillator_async(candles1, 14, 3);
    
    // Example 4: Calculate On-Balance Volume (OBV)
    std::cout << "4. Calculating On-Balance Volume for instrument 1..." << std::endl;
    auto obv_future = scheduler.calculate_on_balance_volume_async(candles1);
    
    // Example 5: Calculate batch indicators
    std::cout << "5. Calculating batch indicators..." << std::endl;
    std::vector<std::vector<double>> price_series = {prices1, prices2};
    std::vector<std::pair<std::string, int>> indicator_configs = {
        {"SMA", 10},
        {"SMA", 20},
        {"RSI", 14}
    };
    auto batch_indicators_future = scheduler.calculate_batch_indicators_async(
        price_series, indicator_configs);
    
    // Example 6: Calculate multiple timeframe indicators
    std::cout << "6. Calculating multiple timeframe indicators for instrument 1..." << std::endl;
    std::vector<std::pair<std::string, std::vector<int>>> mt_configs = {
        {"SMA", {10, 20, 50}},
        {"RSI", {7, 14}}
    };
    auto mt_indicators_future = scheduler.calculate_multiple_timeframe_indicators_async(
        prices1, mt_configs);
    
    // Example 7: Calculate normalized correlation matrix
    std::cout << "7. Calculating normalized correlation matrix..." << std::endl;
    auto correlation_matrix_future = scheduler.calculate_normalized_correlation_matrix_async(
        price_series);
    
    // Example 8: Apply market microstructure filters
    std::cout << "8. Applying market microstructure filters..." << std::endl;
    auto filtered_trades_future = scheduler.apply_market_microstructure_filters_async(
        instrument1_trades, 0.01); // Tick size of 0.01
    
    // Example 9: Calculate cumulative volume delta
    std::cout << "9. Calculating cumulative volume delta..." << std::endl;
    auto cum_vol_delta_future = scheduler.calculate_cumulative_volume_delta_async(
        instrument1_trades, prices1);
    
    // Example 10: Calculate Volume Price Confirmation Indicator (VPCI)
    std::cout << "10. Calculating VPCI..." << std::endl;
    auto vpci_future = scheduler.calculate_volume_price_confirmation_indicator_async(
        instrument1_trades, 10);

    // Wait for all calculations to complete and display results
    std::cout << "\nWaiting for calculations to complete..." << std::endl;

    try {
        // Retrieve correlation
        auto correlation = correlation_future.get();
        std::cout << "Correlation between instruments: " << std::fixed << std::setprecision(4) 
                  << correlation << std::endl;

        // Retrieve ATR
        auto atr_values = atr_future.get();
        std::cout << "ATR calculated with " << atr_values.size() << " values" << std::endl;
        
        // Display first few ATR values
        std::cout << "Sample ATR values:" << std::endl;
        for (size_t i = 0; i < std::min(static_cast<size_t>(5), atr_values.size()); ++i) {
            std::cout << "  Period " << i << ": " << std::fixed << std::setprecision(4) 
                      << atr_values[i] << std::endl;
        }

        // Retrieve Stochastic Oscillator
        auto stochastic_values = stochastic_future.get();
        std::cout << "\nStochastic Oscillator calculated with " << stochastic_values.size() << " values" << std::endl;
        
        // Display first few stochastic values
        std::cout << "Sample Stochastic values:" << std::endl;
        for (size_t i = 0; i < std::min(static_cast<size_t>(5), stochastic_values.size()); ++i) {
            std::cout << "  Period " << i << ": " << std::fixed << std::setprecision(2) 
                      << stochastic_values[i] << std::endl;
        }

        // Retrieve OBV
        auto obv_values = obv_future.get();
        std::cout << "\nOn-Balance Volume calculated with " << obv_values.size() << " values" << std::endl;
        
        // Display first few OBV values
        std::cout << "Sample OBV values:" << std::endl;
        for (size_t i = 0; i < std::min(static_cast<size_t>(5), obv_values.size()); ++i) {
            std::cout << "  Period " << i << ": " << std::fixed << std::setprecision(2) 
                      << obv_values[i] << std::endl;
        }

        // Retrieve batch indicators
        auto batch_indicators = batch_indicators_future.get();
        std::cout << "\nBatch indicators calculated with " << batch_indicators.size() 
                  << " series and configurations" << std::endl;
        for (size_t i = 0; i < batch_indicators.size(); ++i) {
            std::cout << "  Series " << i << ": " << batch_indicators[i].size() << " values" << std::endl;
        }

        // Retrieve multiple timeframe indicators
        auto mt_indicators = mt_indicators_future.get();
        std::cout << "\nMultiple timeframe indicators calculated with " << mt_indicators.size() 
                  << " indicator types" << std::endl;

        // Retrieve correlation matrix
        auto correlation_matrix = correlation_matrix_future.get();
        std::cout << "\nNormalized correlation matrix calculated with " << correlation_matrix.size() 
                  << " rows" << std::endl;
        if (!correlation_matrix.empty()) {
            std::cout << "Matrix dimensions: " << correlation_matrix.size() << "x" 
                      << correlation_matrix[0].size() << std::endl;
        }

        // Retrieve filtered trades
        auto filtered_trades = filtered_trades_future.get();
        std::cout << "\nFiltered trades: " << filtered_trades.size() << " remaining after microstructure filters" << std::endl;

        // Retrieve cumulative volume delta
        auto cum_vol_delta = cum_vol_delta_future.get();
        std::cout << "\nCumulative volume delta calculated with " << cum_vol_delta.size() << " values" << std::endl;

        // Retrieve VPCI
        auto vpci = vpci_future.get();
        std::cout << "\nVPCI calculated with " << vpci.size() << " values" << std::endl;

        std::cout << "\nAll advanced analytics calculations completed successfully!" << std::endl;
        std::cout << "This example demonstrates how to use the TaskScheduler for advanced analytical calculations." << std::endl;
        std::cout << "These analyses are useful for portfolio management, risk assessment, and advanced trading strategies." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error during calculations: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}