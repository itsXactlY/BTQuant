#include "task_scheduler.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <algorithm>

/**
 * @brief Example implementation of data processing and candle aggregation
 * Demonstrates how to use the TaskScheduler for common data processing tasks
 */
int main() {
    std::cout << "=== Data Processing and Candle Aggregation Example ===" << std::endl;

    // Initialize the TaskScheduler with 4 threads
    btq::TaskScheduler scheduler(4);

    // Generate sample trade data
    std::cout << "Generating sample trade data..." << std::endl;
    std::vector<btq::Trade> trades;
    std::mt19937 gen(42); // Fixed seed for reproducible results
    std::uniform_real_distribution<> price_dist(100.0, 200.0);
    std::uniform_real_distribution<> volume_dist(10.0, 100.0);

    auto start_time = std::chrono::system_clock::now();
    for (int i = 0; i < 50000; ++i) {
        btq::Trade trade;
        trade.timestamp = start_time + std::chrono::milliseconds(i * 10); // 10ms intervals
        trade.price = price_dist(gen);
        trade.volume = volume_dist(gen);
        trades.push_back(trade);
    }

    std::cout << "Generated " << trades.size() << " trades" << std::endl;

    // Example 1: Aggregate candles (1-minute timeframe)
    std::cout << "\n1. Aggregating 1-minute candles..." << std::endl;
    auto candles_future = scheduler.aggregate_candles_async(trades, std::chrono::minutes(1));
    
    // Example 2: Filter trades (by volume threshold)
    std::cout << "2. Filtering trades with volume > 50..." << std::endl;
    auto filtered_trades_future = scheduler.filter_trades_async(trades, 
        [](const btq::Trade& trade) { return trade.volume > 50.0; });
    
    // Example 3: Calculate histogram of prices
    std::cout << "3. Calculating price histogram..." << std::endl;
    std::vector<double> prices;
    for (const auto& trade : trades) {
        prices.push_back(trade.price);
    }
    auto histogram_future = scheduler.calculate_histogram_async(prices, 50);
    
    // Example 4: Transform data (square root of volumes)
    std::cout << "4. Transforming data (square root of volumes)..." << std::endl;
    auto transformed_data_future = scheduler.transform_data_parallel_async(
        prices, [](double price) { return std::sqrt(price); });
    
    // Example 5: Create dynamic timeframe candles
    std::cout << "5. Creating dynamic timeframe candles..." << std::endl;
    auto dynamic_candles_future = scheduler.create_dynamic_timeframe_candles_async(
        trades, std::chrono::seconds(30), 1000.0); // 30s base timeframe, 1000 volume threshold
    
    // Example 6: Process batch trades
    std::cout << "6. Processing batch trades..." << std::endl;
    std::vector<std::vector<btq::Trade>> trade_batches;
    // Split trades into batches of 10000 each
    for (size_t i = 0; i < trades.size(); i += 10000) {
        std::vector<btq::Trade> batch;
        size_t end = std::min(i + 10000, trades.size());
        for (size_t j = i; j < end; ++j) {
            batch.push_back(trades[j]);
        }
        trade_batches.push_back(batch);
    }
    auto processed_batches_future = scheduler.process_batch_trades_async(
        trade_batches, 
        [](const std::vector<btq::Trade>& batch) -> std::vector<btq::Trade> {
            // Example: filter trades with price > 150
            std::vector<btq::Trade> filtered;
            for (const auto& trade : batch) {
                if (trade.price > 150.0) {
                    filtered.push_back(trade);
                }
            }
            return filtered;
        });
    
    // Example 7: Partition and process trades
    std::cout << "7. Partitioning and processing trades..." << std::endl;
    auto partitioned_future = scheduler.partition_and_process_trades_async(
        trades,
        [](const std::vector<btq::Trade>& partition) -> std::vector<btq::Trade> {
            // Example: sort trades by volume in descending order
            auto sorted_partition = partition;
            std::sort(sorted_partition.begin(), sorted_partition.end(),
                     [](const btq::Trade& a, const btq::Trade& b) {
                         return a.volume > b.volume;
                     });
            return sorted_partition;
        },
        4); // 4 partitions

    // Wait for all calculations to complete and display results
    std::cout << "\nWaiting for calculations to complete..." << std::endl;

    try {
        // Retrieve aggregated candles
        auto candles = candles_future.get();
        std::cout << "Aggregated " << candles.size() << " 1-minute candles" << std::endl;
        
        // Display first few candles
        std::cout << "Sample candle values:" << std::endl;
        for (size_t i = 0; i < std::min(static_cast<size_t>(3), candles.size()); ++i) {
            std::cout << "  Candle " << i << ": Open=" << std::fixed << std::setprecision(2) 
                      << candles[i].open << ", High=" << candles[i].high 
                      << ", Low=" << candles[i].low << ", Close=" << candles[i].close 
                      << ", Vol=" << candles[i].volume << std::endl;
        }

        // Retrieve filtered trades
        auto filtered_trades = filtered_trades_future.get();
        std::cout << "\nFiltered trades: " << filtered_trades.size() << " trades with volume > 50" << std::endl;

        // Retrieve histogram
        auto histogram = histogram_future.get();
        std::cout << "\nPrice histogram calculated with " << histogram.size() << " bins" << std::endl;
        
        // Display first few histogram values
        std::cout << "Sample histogram values:" << std::endl;
        for (size_t i = 0; i < std::min(static_cast<size_t>(5), histogram.size()); ++i) {
            std::cout << "  Bin " << i << ": " << std::fixed << std::setprecision(2) << histogram[i].first
                      << "-" << histogram[i].second << std::endl;
        }

        // Retrieve transformed data
        auto transformed_data = transformed_data_future.get();
        std::cout << "\nTransformed data calculated with " << transformed_data.size() << " values" << std::endl;
        
        // Display first few transformed values
        std::cout << "Sample transformed values (sqrt of prices):" << std::endl;
        for (size_t i = 0; i < std::min(static_cast<size_t>(5), transformed_data.size()); ++i) {
            std::cout << "  Value " << i << ": " << std::fixed << std::setprecision(4) 
                      << transformed_data[i] << std::endl;
        }

        // Retrieve dynamic candles
        auto dynamic_candles = dynamic_candles_future.get();
        std::cout << "\nCreated " << dynamic_candles.size() << " dynamic timeframe candles" << std::endl;

        // Retrieve processed batch trades
        auto processed_batches = processed_batches_future.get();
        std::cout << "\nProcessed " << processed_batches.size() << " trade batches" << std::endl;
        size_t total_filtered_in_batches = 0;
        for (const auto& batch : processed_batches) {
            total_filtered_in_batches += batch.size();
        }
        std::cout << "Total filtered trades in batches: " << total_filtered_in_batches << std::endl;

        // Retrieve partitioned trades
        auto partitioned_trades = partitioned_future.get();
        std::cout << "\nPartitioned and processed trades: " << partitioned_trades.size() << " total trades" << std::endl;

        std::cout << "\nAll data processing calculations completed successfully!" << std::endl;
        std::cout << "This example demonstrates how to use the TaskScheduler for common data processing tasks." << std::endl;
        std::cout << "These operations are essential for real-time trading data analysis and charting." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error during calculations: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}