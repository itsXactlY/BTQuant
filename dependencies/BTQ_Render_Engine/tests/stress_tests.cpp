#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <vector>
#include <random>
#include <atomic>
#include <future>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <numeric>

#include "analytics/technical_analysis.hpp"
#include "indicators/rolling_vwap.hpp"
#include "indicators/anchored_vwap.hpp"
#include "market_data_processor.hpp"
#include "performance/memory_tracker.hpp"
#include "performance/cpu_profiler.hpp"
#include "performance_monitor.hpp"

namespace BTQuant {
namespace StressTests {

// Stress test fixture for high-volume trading simulations
class TradingStressTests : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize performance monitors
        g_performance_monitor.set_enabled(true);
        g_cpu_profiler.set_enabled(true);
        
        // Start memory tracking
        memory_tracker_.startTracking();
    }

    void TearDown() override {
        // Stop memory tracking
        memory_tracker_.stopTracking();
        
        // Print final statistics
        printFinalStatistics();
    }

    btq::performance::MemoryTracker& memory_tracker_ = btq::performance::getGlobalMemoryTracker();

    void printFinalStatistics() {
        std::cout << "\n=== Final Stress Test Statistics ===" << std::endl;
        std::cout << "FPS: " << g_performance_monitor.get_fps() << std::endl;
        std::cout << "Frame Time: " << g_performance_monitor.get_frame_time_ms() << " ms" << std::endl;
        std::cout << "Peak Memory Usage: " << memory_tracker_.getPeakMemoryUsage() << " bytes" << std::endl;
        std::cout << "Current Memory Usage: " << memory_tracker_.getCurrentMemoryUsage() << " bytes" << std::endl;

        // Generate CPU profiler report
        std::string cpu_report = g_cpu_profiler.generate_report();
        std::cout << "CPU Profiler Summary:\n" << cpu_report.substr(0, 500) << "..." << std::endl; // Truncate for brevity

        // Additional performance metrics
        std::cout << "Additional Performance Metrics:" << std::endl;
        std::cout << "  Average FPS: " << g_performance_monitor.get_avg_fps() << std::endl;
        std::cout << "  Min FPS: " << g_performance_monitor.get_min_fps() << std::endl;
        std::cout << "  Max FPS: " << g_performance_monitor.get_max_fps() << std::endl;
        std::cout << "  Average Frame Time: " << g_performance_monitor.get_avg_frame_time_ms() << " ms" << std::endl;
        std::cout << "  Min Frame Time: " << g_performance_monitor.get_min_frame_time() << " ms" << std::endl;
        std::cout << "  Max Frame Time: " << g_performance_monitor.get_max_frame_time() << " ms" << std::endl;
        std::cout << "  Total Data Processed: " << g_performance_monitor.get_data_processed_count() << std::endl;
        std::cout << "  Indicators Calculated: " << g_performance_monitor.get_indicators_calculated_count() << std::endl;

        // Memory statistics
        std::cout << "Memory Statistics:" << std::endl;
        std::cout << "  Total Allocated Bytes: " << memory_tracker_.getTotalAllocatedBytes() << std::endl;
        std::cout << "  Total Deallocated Bytes: " << memory_tracker_.getTotalDeallocatedBytes() << std::endl;
        std::cout << "  Current Allocation Count: " << memory_tracker_.getCurrentAllocationCount() << std::endl;
        std::cout << "  Average Memory Growth Rate: " << memory_tracker_.getAverageMemoryGrowthRate() << " bytes/sec" << std::endl;
    }
};

// Test high-frequency trading data processing under extreme load
TEST_F(TradingStressTests, HighFrequencyDataProcessing) {
    const int num_threads = std::thread::hardware_concurrency();
    const int iterations_per_thread = 10000;
    const int batch_size = 100;

    std::vector<std::thread> threads;
    std::atomic<int> processed_count{0};
    std::atomic<bool> should_stop{false};

    // Lambda for processing trading data
    auto process_trading_data = [&]() {
        TechnicalIndicators ti;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> price_dist(90.0, 110.0);
        std::uniform_real_distribution<> volume_dist(100.0, 1000.0);

        for (int i = 0; i < iterations_per_thread && !should_stop.load(); ++i) {
            // Generate synthetic OHLCV data
            std::vector<TechnicalIndicators::OHLCV> ohlcv_batch;
            for (int j = 0; j < batch_size; ++j) {
                double open = price_dist(gen);
                double high = open + std::abs(price_dist(gen) - open) + 0.01; // Ensure high >= open
                double low = open - std::abs(price_dist(gen) - open) - 0.01;  // Ensure low <= open
                double close = price_dist(gen);
                double volume = volume_dist(gen);

                ohlcv_batch.push_back({open, high, low, close, volume,
                                      static_cast<uint64_t>(std::time(nullptr)) + i * batch_size + j});
            }

            // Calculate various indicators under stress
            auto vwap_result = ti.calculate_vwap(ohlcv_batch, 0);
            auto vwap_stddev_result = ti.calculate_vwap_standard_deviation(ohlcv_batch, 0);

            // Simulate real-time updates
            processed_count.fetch_add(batch_size);

            // Occasionally yield to allow other threads to run
            if (i % 100 == 0) {
                std::this_thread::yield();
            }
        }
    };

    // Start multiple threads to simulate concurrent trading activity
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(process_trading_data);
    }

    // Monitor performance during stress test
    auto start_time = std::chrono::steady_clock::now();
    const auto duration = std::chrono::minutes(2); // Run for 2 minutes

    while (std::chrono::steady_clock::now() - start_time < duration &&
           processed_count.load() < num_threads * iterations_per_thread) {
        // Print periodic status
        if (processed_count.load() % 50000 == 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time).count();
            std::cout << "Processed " << processed_count.load()
                      << " data points in " << elapsed << " seconds" << std::endl;

            // Check performance metrics
            double current_fps = g_performance_monitor.get_fps();
            size_t current_memory = memory_tracker_.getCurrentMemoryUsage();

            // Fail test if performance degrades significantly
            ASSERT_GT(current_fps, 10.0) << "FPS dropped below acceptable threshold";
            ASSERT_LT(current_memory, 100UL * 1024 * 1024) << "Memory usage exceeded 100MB"; // 100MB limit
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Signal threads to stop
    should_stop.store(true);

    // Wait for all threads to complete
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // Verify final performance metrics
    EXPECT_GT(processed_count.load(), 0) << "No data was processed";
    std::cout << "Total data points processed: " << processed_count.load() << std::endl;
}

// Test ultra-high frequency trading simulation with millions of trades
TEST_F(TradingStressTests, UltraHighFrequencySimulation) {
    const int num_threads = std::thread::hardware_concurrency();
    const int iterations_per_thread = 50000;  // More iterations for UHF simulation
    const int batch_size = 50;  // Smaller batches for higher frequency

    std::vector<std::thread> threads;
    std::atomic<int64_t> total_ticks{0};
    std::atomic<int64_t> total_orders_processed{0};
    std::atomic<bool> should_stop{false};

    // Lambda for ultra-high frequency trading simulation
    auto simulate_uhf_trading = [&](int thread_id) {
        TechnicalIndicators ti;
        std::random_device rd;
        std::mt19937 gen(rd() + thread_id);  // Unique seed per thread
        std::uniform_real_distribution<> price_dist(95.0, 105.0);
        std::uniform_real_distribution<> volume_dist(1.0, 50.0);  // Lower volume for HFT
        std::uniform_int_distribution<> order_type_dist(0, 3);    // Buy, Sell, Modify, Cancel

        for (int i = 0; i < iterations_per_thread && !should_stop.load(); ++i) {
            // Generate tick-by-tick data representing rapid market movements
            std::vector<TechnicalIndicators::OHLCV> ticks;
            for (int j = 0; j < batch_size; ++j) {
                // Simulate micro-movements in price
                static double current_price = price_dist(gen);
                double movement = (price_dist(gen) - current_price) * 0.001; // Very small movements
                current_price += movement;

                double open = current_price;
                double high = open + std::abs(movement) * 0.5;
                double low = open - std::abs(movement) * 0.5;
                double close = open + movement;
                double volume = volume_dist(gen);

                ticks.push_back({open, high, low, close, volume,
                                static_cast<uint64_t>(std::time(nullptr)) * 1000000 +
                                (i * batch_size + j)});  // Microsecond precision timestamp

                total_ticks.fetch_add(1);
            }

            // Calculate indicators rapidly
            auto vwap_result = ti.calculate_vwap(ticks, 0);
            auto vwap_stddev_result = ti.calculate_vwap_standard_deviation(ticks, 0);

            // Simulate order processing based on indicators
            if (!vwap_result.values.empty() && !ticks.empty()) {
                double current_price = ticks.back().close;
                double vwap_value = vwap_result.values.back();

                // Simple trading logic: buy if price below VWAP, sell if above
                if (current_price < vwap_value * 0.995) {  // 0.5% below VWAP
                    total_orders_processed.fetch_add(1);
                } else if (current_price > vwap_value * 1.005) {  // 0.5% above VWAP
                    total_orders_processed.fetch_add(1);
                }
            }

            // Occasional yield
            if (i % 500 == 0) {
                std::this_thread::yield();
            }
        }
    };

    // Launch UHF simulation threads
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(simulate_uhf_trading, t);
    }

    // Monitor performance during UHF test
    auto start_time = std::chrono::steady_clock::now();
    const auto duration = std::chrono::minutes(3); // Run for 3 minutes for UHF

    while (std::chrono::steady_clock::now() - start_time < duration) {
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start_time).count();

        if (elapsed % 30 == 0) {  // Log every 30 seconds
            int64_t current_ticks = total_ticks.load();
            int64_t current_orders = total_orders_processed.load();

            std::cout << "UHF Simulation - Elapsed: " << elapsed << "s | Ticks: "
                      << current_ticks << " | Orders: " << current_orders << std::endl;

            // Check performance metrics
            double current_fps = g_performance_monitor.get_fps();
            size_t current_memory = memory_tracker_.getCurrentMemoryUsage();

            // Ensure system remains responsive under UHF load
            ASSERT_GT(current_fps, 5.0) << "FPS dropped below acceptable threshold during UHF simulation";
            ASSERT_LT(current_memory, 150UL * 1024 * 1024) << "Memory usage exceeded 150MB during UHF simulation";
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Signal threads to stop
    should_stop.store(true);

    // Wait for all threads to complete
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // Final verification for UHF test
    int64_t final_ticks = total_ticks.load();
    int64_t final_orders = total_orders_processed.load();

    std::cout << "UHF Simulation completed:" << std::endl;
    std::cout << "Total ticks processed: " << final_ticks << std::endl;
    std::cout << "Total orders processed: " << final_orders << std::endl;

    EXPECT_GT(final_ticks, 0) << "No ticks were processed in UHF simulation";
    EXPECT_GT(final_orders, 0) << "No orders were processed in UHF simulation";
}

// Test memory stability under prolonged high-volume trading
TEST_F(TradingStressTests, MemoryStabilityUnderLoad) {
    const int iterations = 50000;
    std::vector<std::unique_ptr<std::vector<TechnicalIndicators::OHLCV>>> data_holders;
    
    // Reserve space to minimize reallocations during test
    data_holders.reserve(iterations / 100);
    
    size_t initial_memory = memory_tracker_.getCurrentMemoryUsage();
    
    for (int i = 0; i < iterations; ++i) {
        // Create synthetic trading data
        auto data = std::make_unique<std::vector<TechnicalIndicators::OHLCV>>();
        data->reserve(50); // Pre-allocate for efficiency
        
        std::random_device rd;
        std::mt19937 gen(rd() + i); // Different seed for each iteration
        std::uniform_real_distribution<> price_dist(50.0, 150.0);
        std::uniform_real_distribution<> volume_dist(50.0, 500.0);
        
        for (int j = 0; j < 50; ++j) {
            double open = price_dist(gen);
            double high = open + std::abs(price_dist(gen) - open) * 0.02 + 0.01;
            double low = open - std::abs(price_dist(gen) - open) * 0.02 - 0.01;
            double close = price_dist(gen);
            double volume = volume_dist(gen);
            
            data->push_back({open, high, low, close, volume, 
                           static_cast<uint64_t>(std::time(nullptr)) + i * 50 + j});
        }
        
        // Process the data with technical indicators
        TechnicalIndicators ti;
        auto vwap_result = ti.calculate_vwap(*data, 0);
        auto vwap_stddev_result = ti.calculate_vwap_standard_deviation(*data, 0);
        
        // Occasionally clear some data to simulate cleanup
        if (i % 100 == 0 && !data_holders.empty()) {
            data_holders.erase(data_holders.begin(), data_holders.begin() + 10);
        }
        
        // Hold onto some data to create memory pressure
        if (i % 5 == 0) {
            data_holders.push_back(std::move(data));
        }
        
        // Check memory usage periodically
        if (i % 5000 == 0) {
            size_t current_memory = memory_tracker_.getCurrentMemoryUsage();
            double memory_growth = static_cast<double>(current_memory - initial_memory) / initial_memory;
            
            // Ensure memory growth stays within acceptable bounds (10% growth tolerance)
            EXPECT_LT(memory_growth, 0.10) << "Memory growth exceeded 10% threshold at iteration " << i;
            
            std::cout << "Iteration " << i << ": Memory usage " << current_memory 
                      << " bytes, growth: " << (memory_growth * 100) << "%" << std::endl;
        }
    }
    
    // Clear all held data at the end
    data_holders.clear();
    
    // Final memory check
    size_t final_memory = memory_tracker_.getCurrentMemoryUsage();
    double final_growth = static_cast<double>(final_memory - initial_memory) / initial_memory;
    
    EXPECT_LT(final_growth, 0.15) << "Final memory growth exceeded 15% threshold";
    std::cout << "Final memory growth: " << (final_growth * 100) << "%" << std::endl;
}

// Test system stability over extended periods with continuous trading simulation
TEST_F(TradingStressTests, ExtendedStabilityOverTime) {
    const auto test_duration = std::chrono::minutes(10); // 10-minute extended test
    auto start_time = std::chrono::steady_clock::now();

    std::atomic<int> processed_batches{0};
    std::atomic<double> avg_processing_time_ms{0.0};

    // Thread to continuously generate and process trading data
    std::thread data_processor([&]() {
        TechnicalIndicators ti;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> price_dist(80.0, 120.0);
        std::uniform_real_distribution<> volume_dist(100.0, 1000.0);

        while (std::chrono::steady_clock::now() - start_time < test_duration) {
            auto batch_start = std::chrono::high_resolution_clock::now();

            // Generate a batch of OHLCV data
            std::vector<TechnicalIndicators::OHLCV> batch_data;
            const int batch_size = 250; // Larger batch size for more stress

            for (int i = 0; i < batch_size; ++i) {
                double open = price_dist(gen);
                double high = std::max(open, price_dist(gen)); // Ensure high >= open
                double low = std::min(open, price_dist(gen));  // Ensure low <= open
                double close = price_dist(gen);
                double volume = volume_dist(gen);

                // Add some realistic price relationships
                high = std::max({high, open, close});
                low = std::min({low, open, close});

                batch_data.push_back({open, high, low, close, volume,
                                    static_cast<uint64_t>(std::time(nullptr)) * 1000 +
                                    std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::steady_clock::now() - start_time).count()});
            }

            // Calculate multiple indicators on the batch
            auto vwap_result = ti.calculate_vwap(batch_data, 0);
            auto vwap_stddev_result = ti.calculate_vwap_standard_deviation(batch_data, 0);

            // Simulate additional processing that might happen in a real trading system
            std::vector<double> processed_values;
            if (!vwap_result.values.empty()) {
                processed_values.insert(processed_values.end(),
                                      vwap_result.values.begin(),
                                      vwap_result.values.end());
            }

            // Calculate processing time for this batch
            auto batch_end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                batch_end - batch_start).count() / 1000.0; // Convert to milliseconds

            // Update average processing time
            double current_avg = avg_processing_time_ms.load();
            int count = processed_batches.load() + 1;
            double new_avg = (current_avg * (count - 1) + duration) / count;
            avg_processing_time_ms.store(new_avg);

            processed_batches.fetch_add(1);

            // Small delay to prevent overwhelming the system completely
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    // Monitor the system during the extended test
    auto last_check = start_time;
    int check_interval = 30; // seconds

    while (std::chrono::steady_clock::now() - start_time < test_duration) {
        auto current_time = std::chrono::steady_clock::now();

        if (std::chrono::duration_cast<std::chrono::seconds>(current_time - last_check).count() >= check_interval) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
            int batches_processed = processed_batches.load();
            double avg_proc_time = avg_processing_time_ms.load();

            // Get performance metrics
            double current_fps = g_performance_monitor.get_fps();
            size_t current_memory = memory_tracker_.getCurrentMemoryUsage();

            std::cout << "Time: " << elapsed << "s | Batches: " << batches_processed
                      << " | Avg proc time: " << avg_proc_time << "ms"
                      << " | FPS: " << current_fps
                      << " | Memory: " << current_memory / (1024 * 1024) << "MB" << std::endl;

            // Check that system remains stable
            EXPECT_GT(current_fps, 5.0) << "FPS dropped below minimum acceptable level at " << elapsed << " seconds";
            EXPECT_LT(avg_proc_time, 50.0) << "Average processing time exceeded 50ms at " << elapsed << " seconds";
            EXPECT_LT(current_memory, 200UL * 1024 * 1024) << "Memory usage exceeded 200MB at " << elapsed << " seconds";

            last_check = current_time;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Wait for the data processor thread to finish
    if (data_processor.joinable()) {
        data_processor.join();
    }

    // Final verification
    int total_batches = processed_batches.load();
    double final_avg_time = avg_processing_time_ms.load();

    std::cout << "\nExtended test completed:" << std::endl;
    std::cout << "Total batches processed: " << total_batches << std::endl;
    std::cout << "Final average processing time: " << final_avg_time << "ms" << std::endl;

    EXPECT_GT(total_batches, 0) << "No batches were processed during extended test";
    EXPECT_LT(final_avg_time, 100.0) << "Final average processing time too high";
}

// Test long-running stability with variable load patterns
TEST_F(TradingStressTests, VariableLoadStabilityTest) {
    const auto test_duration = std::chrono::minutes(15); // 15-minute test with variable load
    auto start_time = std::chrono::steady_clock::now();

    std::atomic<int> total_operations{0};
    std::atomic<double> min_fps{1000.0};  // Initialize to high value
    std::atomic<double> max_memory{0.0};  // Initialize to low value

    // Thread to simulate variable load patterns
    std::thread variable_load_thread([&]() {
        TechnicalIndicators ti;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> price_dist(75.0, 125.0);
        std::uniform_real_distribution<> volume_dist(50.0, 800.0);
        std::uniform_int_distribution<> load_pattern_dist(1, 5);  // Different load patterns

        while (std::chrono::steady_clock::now() - start_time < test_duration) {
            // Determine current load pattern (1-5, where 5 is highest load)
            int current_load = load_pattern_dist(gen);

            // Generate data based on current load level
            int batch_size = 50 * current_load;  // Higher load = larger batches
            int num_batches = 10 * current_load;  // Higher load = more batches

            for (int batch = 0; batch < num_batches; ++batch) {
                std::vector<TechnicalIndicators::OHLCV> batch_data;
                for (int i = 0; i < batch_size; ++i) {
                    double open = price_dist(gen);
                    double high = std::max(open, price_dist(gen));
                    double low = std::min(open, price_dist(gen));
                    double close = price_dist(gen);
                    double volume = volume_dist(gen);

                    high = std::max({high, open, close});
                    low = std::min({low, open, close});

                    batch_data.push_back({open, high, low, close, volume,
                                         static_cast<uint64_t>(std::time(nullptr)) * 1000 +
                                         std::chrono::duration_cast<std::chrono::milliseconds>(
                                             std::chrono::steady_clock::now() - start_time).count() +
                                         batch * batch_size + i});
                }

                // Calculate multiple indicators
                auto vwap_result = ti.calculate_vwap(batch_data, 0);
                auto vwap_stddev_result = ti.calculate_vwap_standard_deviation(batch_data, 0);

                total_operations.fetch_add(1);

                // Small delay based on load (lower load = more delay)
                int delay_ms = 5 / current_load;  // Inverse relationship
                std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            }

            // Update performance metrics
            double current_fps = g_performance_monitor.get_fps();
            size_t current_memory = memory_tracker_.getCurrentMemoryUsage();

            // Update min/max values atomically
            double current_min_fps = min_fps.load();
            while (current_fps < current_min_fps &&
                   !min_fps.compare_exchange_weak(current_min_fps, current_fps));

            double current_max_memory = max_memory.load();
            while (current_memory > current_max_memory &&
                   !max_memory.compare_exchange_weak(current_max_memory, static_cast<double>(current_memory)));

            // Brief pause between load cycles
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    // Monitor system during variable load test
    auto last_check = start_time;
    int check_interval = 60; // seconds

    while (std::chrono::steady_clock::now() - start_time < test_duration) {
        auto current_time = std::chrono::steady_clock::now();

        if (std::chrono::duration_cast<std::chrono::seconds>(current_time - last_check).count() >= check_interval) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
            int ops_completed = total_operations.load();

            double current_fps = g_performance_monitor.get_fps();
            size_t current_memory = memory_tracker_.getCurrentMemoryUsage();

            std::cout << "Variable Load Test - Time: " << elapsed << "s | Operations: " << ops_completed
                      << " | Current FPS: " << current_fps
                      << " | Current Memory: " << current_memory / (1024 * 1024) << "MB" << std::endl;

            // Check that system remains within acceptable bounds
            EXPECT_GT(current_fps, 2.0) << "FPS dropped below minimum acceptable during variable load test";
            EXPECT_LT(current_memory, 250UL * 1024 * 1024) << "Memory usage exceeded 250MB during variable load test";

            last_check = current_time;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    // Wait for the variable load thread to finish
    if (variable_load_thread.joinable()) {
        variable_load_thread.join();
    }

    // Final verification
    int final_ops = total_operations.load();
    double final_min_fps = min_fps.load();
    double final_max_memory = max_memory.load();

    std::cout << "\nVariable load stability test completed:" << std::endl;
    std::cout << "Total operations completed: " << final_ops << std::endl;
    std::cout << "Minimum FPS recorded: " << final_min_fps << std::endl;
    std::cout << "Maximum memory usage: " << final_max_memory / (1024 * 1024) << "MB" << std::endl;

    EXPECT_GT(final_ops, 0) << "No operations were completed during variable load test";
    EXPECT_GT(final_min_fps, 1.0) << "Minimum FPS fell below acceptable threshold";
    EXPECT_LT(final_max_memory, 300UL * 1024 * 1024) << "Maximum memory usage exceeded 300MB";
}

// Test concurrent access to shared trading resources
TEST_F(TradingStressTests, ConcurrentResourceAccess) {
    const int num_threads = std::thread::hardware_concurrency();
    const int operations_per_thread = 5000;
    
    std::vector<std::thread> worker_threads;
    std::vector<std::future<void>> futures;
    
    // Shared resources that will be accessed concurrently
    std::vector<btq::RollingVWAP> shared_indicators;
    for (int i = 0; i < 5; ++i) {
        shared_indicators.emplace_back(20); // 20-period rolling VWAP
    }
    
    std::mutex shared_mutex;
    std::atomic<int> successful_operations{0};
    std::atomic<int> failed_operations{0};
    
    // Lambda for concurrent operations
    auto perform_concurrent_operations = [&](int thread_id) {
        TechnicalIndicators ti;
        std::random_device rd;
        std::mt19937 gen(rd() + thread_id);
        std::uniform_real_distribution<> price_dist(75.0, 125.0);
        std::uniform_real_distribution<> volume_dist(50.0, 500.0);
        
        for (int op = 0; op < operations_per_thread; ++op) {
            try {
                // Generate data for this operation
                std::vector<TechnicalIndicators::OHLCV> data;
                for (int i = 0; i < 10; ++i) {
                    double open = price_dist(gen);
                    double high = std::max(open, price_dist(gen));
                    double low = std::min(open, price_dist(gen));
                    double close = price_dist(gen);
                    double volume = volume_dist(gen);
                    
                    data.push_back({open, high, low, close, volume, 
                                  static_cast<uint64_t>(std::time(nullptr)) * 1000 + op * 10 + i});
                }
                
                // Acquire lock to access shared resources safely
                std::lock_guard<std::mutex> lock(shared_mutex);
                
                // Perform operations on shared indicators
                for (auto& indicator : shared_indicators) {
                    // Convert TechnicalIndicators::OHLCV to RenderEngine::OHLCVCandle
                    std::vector<BTQuant::RenderEngine::OHLCVCandle> candle_data;
                    for (const auto& ohlcv : data) {
                        BTQuant::RenderEngine::OHLCVCandle candle;
                        candle.timestamp = ohlcv.timestamp;
                        candle.open = ohlcv.open;
                        candle.high = ohlcv.high;
                        candle.low = ohlcv.low;
                        candle.close = ohlcv.close;
                        candle.volume = ohlcv.volume;
                        candle_data.push_back(candle);
                    }
                    indicator.calculate(candle_data);
                }
                
                // Calculate indicators using the TI object
                auto vwap_result = ti.calculate_vwap(data, 0);
                auto vwap_stddev_result = ti.calculate_vwap_standard_deviation(data, 0);
                
                successful_operations.fetch_add(1);
            } catch (const std::exception& e) {
                failed_operations.fetch_add(1);
                std::cerr << "Operation failed on thread " << thread_id << ": " << e.what() << std::endl;
            }
            
            // Yield occasionally to allow other threads to access resources
            if (op % 100 == 0) {
                std::this_thread::yield();
            }
        }
    };
    
    // Launch worker threads
    for (int t = 0; t < num_threads; ++t) {
        worker_threads.emplace_back(perform_concurrent_operations, t);
    }
    
    // Wait for all threads to complete
    for (auto& thread : worker_threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    // Final verification
    int total_successful = successful_operations.load();
    int total_failed = failed_operations.load();
    int total_operations = total_successful + total_failed;
    
    std::cout << "Concurrent resource access results:" << std::endl;
    std::cout << "Successful operations: " << total_successful << std::endl;
    std::cout << "Failed operations: " << total_failed << std::endl;
    std::cout << "Success rate: " << (static_cast<double>(total_successful) / total_operations * 100) << "%" << std::endl;
    
    // Ensure most operations succeeded (allowing for some failures due to extreme stress)
    EXPECT_GT(static_cast<double>(total_successful) / total_operations, 0.95) 
        << "Success rate fell below 95% threshold";
}

// Test system recovery after high-stress periods
TEST_F(TradingStressTests, SystemRecoveryAfterStress) {
    // Phase 1: Apply high stress
    std::cout << "Applying high stress to the system..." << std::endl;

    const int stress_iterations = 20000;
    std::vector<std::unique_ptr<std::vector<TechnicalIndicators::OHLCV>>> temporary_data;

    for (int i = 0; i < stress_iterations; ++i) {
        // Create and hold onto data to increase memory pressure
        auto data = std::make_unique<std::vector<TechnicalIndicators::OHLCV>>();

        std::random_device rd;
        std::mt19937 gen(rd() + i);
        std::uniform_real_distribution<> price_dist(100.0, 110.0);

        for (int j = 0; j < 50; ++j) {
            double open = price_dist(gen);
            data->push_back({open, open + 0.1, open - 0.1, open + 0.05, 100.0,
                           static_cast<uint64_t>(std::time(nullptr)) + i * 50 + j});
        }

        // Process the data
        TechnicalIndicators ti;
        auto result = ti.calculate_vwap(*data, 0);

        // Occasionally hold onto data to increase memory pressure
        if (i % 10 == 0) {
            temporary_data.push_back(std::move(data));
        }

        if (i % 5000 == 0) {
            std::cout << "Stress phase: " << i << " / " << stress_iterations << std::endl;
        }
    }

    // Record metrics after stress
    size_t memory_after_stress = memory_tracker_.getCurrentMemoryUsage();
    double fps_after_stress = g_performance_monitor.get_fps();
    std::cout << "Metrics after stress - Memory: " << memory_after_stress / (1024 * 1024)
              << "MB, FPS: " << fps_after_stress << std::endl;

    // Phase 2: Clear resources and allow system to recover
    std::cout << "Clearing resources and allowing recovery..." << std::endl;
    temporary_data.clear();

    // Wait for garbage collection and system stabilization
    std::this_thread::sleep_for(std::chrono::seconds(5));

    // Phase 3: Verify system recovery
    size_t memory_after_recovery = memory_tracker_.getCurrentMemoryUsage();
    double fps_after_recovery = g_performance_monitor.get_fps();

    std::cout << "Metrics after recovery - Memory: " << memory_after_recovery / (1024 * 1024)
              << "MB, FPS: " << fps_after_recovery << std::endl;

    // Verify that the system has recovered reasonably well
    EXPECT_LT(memory_after_recovery, memory_after_stress * 1.2)  // Memory should be reduced significantly
        << "System did not recover adequately from memory pressure";

    // FPS should have recovered somewhat (may not be exactly the same due to ongoing background processes)
    EXPECT_GT(fps_after_recovery, fps_after_stress * 0.5)  // At least 50% of stressed FPS
        << "System performance did not recover adequately";

    std::cout << "System recovery verified successfully." << std::endl;
}

// Comprehensive end-to-end stress test combining all aspects
TEST_F(TradingStressTests, ComprehensiveEndToEndStressTest) {
    std::cout << "Starting comprehensive end-to-end stress test..." << std::endl;

    const auto test_duration = std::chrono::minutes(20); // 20-minute comprehensive test
    auto start_time = std::chrono::steady_clock::now();

    // Atomic counters for tracking various metrics
    std::atomic<int64_t> total_ticks_processed{0};
    std::atomic<int64_t> total_indicators_calculated{0};
    std::atomic<int64_t> total_orders_generated{0};
    std::atomic<int> peak_active_threads{0};

    // Vector to hold worker threads
    std::vector<std::thread> worker_pool;
    const int max_worker_threads = std::thread::hardware_concurrency() * 2; // More threads than CPU cores for stress
    std::atomic<bool> should_continue{true};

    // Create a thread pool for processing
    for (int i = 0; i < max_worker_threads; ++i) {
        worker_pool.emplace_back([&, i]() {
            TechnicalIndicators ti;
            std::random_device rd;
            std::mt19937 gen(rd() + i * 1000); // Unique seed per thread
            std::uniform_real_distribution<> price_dist(50.0, 150.0);
            std::uniform_real_distribution<> volume_dist(10.0, 1000.0);
            std::uniform_int_distribution<> indicator_choice(0, 3);

            int local_peak = 0;
            int active_count = 0;

            while (should_continue.load()) {
                // Generate synthetic trading data
                std::vector<TechnicalIndicators::OHLCV> data_batch;
                const int batch_size = 20 + (i % 30); // Vary batch size by thread ID

                for (int j = 0; j < batch_size; ++j) {
                    double open = price_dist(gen);
                    double high = open + std::abs(price_dist(gen) - open) * 0.05 + 0.01;
                    double low = open - std::abs(price_dist(gen) - open) * 0.05 - 0.01;
                    double close = price_dist(gen);
                    double volume = volume_dist(gen);

                    data_batch.push_back({open, high, low, close, volume,
                                        static_cast<uint64_t>(std::time(nullptr)) * 1000000 +
                                        std::chrono::duration_cast<std::chrono::microseconds>(
                                            std::chrono::steady_clock::now() - start_time).count()});
                }

                // Calculate different indicators based on random selection
                switch(indicator_choice(gen) % 2) {  // Only use 2 options since we only have 2 functions
                    case 0: {
                        auto result = ti.calculate_vwap(data_batch, 0);
                        total_indicators_calculated.fetch_add(result.values.size());
                        break;
                    }
                    case 1: {
                        auto result = ti.calculate_vwap_standard_deviation(data_batch, 0);
                        total_indicators_calculated.fetch_add(result.values.size());
                        break;
                    }
                }

                // Simulate order generation based on indicator values
                if (!data_batch.empty()) {
                    double current_price = data_batch.back().close;
                    // Simple strategy: if price is above 1% of first price, generate sell order
                    if (current_price > data_batch.front().open * 1.01) {
                        total_orders_generated.fetch_add(1);
                    } else if (current_price < data_batch.front().open * 0.99) {
                        total_orders_generated.fetch_add(1);
                    }
                }

                total_ticks_processed.fetch_add(batch_size);

                // Small delay to prevent overwhelming the system
                std::this_thread::sleep_for(std::chrono::microseconds(100));

                // Update thread activity counters
                active_count++;
                if (active_count > local_peak) {
                    local_peak = active_count;
                    int current_peak = peak_active_threads.load();
                    while (local_peak > current_peak &&
                           !peak_active_threads.compare_exchange_weak(current_peak, local_peak));
                }
            }
        });
    }

    // Monitor the comprehensive test
    auto last_report = start_time;
    const int report_interval = 60; // seconds

    while (std::chrono::steady_clock::now() - start_time < test_duration) {
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();

        if (std::chrono::duration_cast<std::chrono::seconds>(current_time - last_report).count() >= report_interval) {
            int64_t ticks = total_ticks_processed.load();
            int64_t indicators = total_indicators_calculated.load();
            int64_t orders = total_orders_generated.load();
            double fps = g_performance_monitor.get_fps();
            size_t memory = memory_tracker_.getCurrentMemoryUsage();
            int active_threads = peak_active_threads.load();

            std::cout << "Comprehensive Test - Elapsed: " << elapsed << "s" << std::endl;
            std::cout << "  Ticks processed: " << ticks << std::endl;
            std::cout << "  Indicators calculated: " << indicators << std::endl;
            std::cout << "  Orders generated: " << orders << std::endl;
            std::cout << "  FPS: " << fps << std::endl;
            std::cout << "  Memory: " << memory / (1024 * 1024) << "MB" << std::endl;
            std::cout << "  Peak active threads: " << active_threads << std::endl;

            // Verify system remains stable under comprehensive stress
            EXPECT_GT(fps, 1.0) << "FPS dropped below acceptable threshold at " << elapsed << " seconds";
            EXPECT_LT(memory, 500UL * 1024 * 1024) << "Memory usage exceeded 500MB at " << elapsed << " seconds";

            last_report = current_time;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // Signal all threads to stop
    should_continue.store(false);

    // Wait for all threads to complete
    for (auto& thread : worker_pool) {
        if (thread.joinable()) {
            thread.join();
        }
    }

    // Final verification of comprehensive test
    int64_t final_ticks = total_ticks_processed.load();
    int64_t final_indicators = total_indicators_calculated.load();
    int64_t final_orders = total_orders_generated.load();
    int final_active_threads = peak_active_threads.load();

    std::cout << "\nComprehensive end-to-end stress test completed:" << std::endl;
    std::cout << "Final metrics:" << std::endl;
    std::cout << "  Total ticks processed: " << final_ticks << std::endl;
    std::cout << "  Total indicators calculated: " << final_indicators << std::endl;
    std::cout << "  Total orders generated: " << final_orders << std::endl;
    std::cout << "  Peak active threads: " << final_active_threads << std::endl;
    std::cout << "  Final FPS: " << g_performance_monitor.get_fps() << std::endl;
    std::cout << "  Final memory usage: " << memory_tracker_.getCurrentMemoryUsage() / (1024 * 1024) << "MB" << std::endl;

    // Ensure all metrics show substantial activity
    EXPECT_GT(final_ticks, 0) << "No ticks were processed in comprehensive test";
    EXPECT_GT(final_indicators, 0) << "No indicators were calculated in comprehensive test";
    EXPECT_GT(final_orders, 0) << "No orders were generated in comprehensive test";
    EXPECT_GT(final_active_threads, 0) << "No threads were active during comprehensive test";

    std::cout << "Comprehensive stress test passed successfully!" << std::endl;
}

} // namespace StressTests
} // namespace BTQuant