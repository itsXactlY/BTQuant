#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <random>
#include <atomic>
#include <memory>
#include <future>
#include <queue>

#include "dependencies/BTQ_Render_Engine/include/market_data_processor.hpp"

using namespace BTQuant::RenderEngine;

/**
 * Optimized benchmark for MarketDataProcessor with 1M messages/sec replay
 * This version batches messages to reduce overhead and maximize throughput
 */
class OptimizedMarketDataProcessorBenchmark {
public:
    OptimizedMarketDataProcessorBenchmark() : processor_(std::make_unique<MarketDataProcessor>()) {}

    void runBenchmark() {
        std::cout << "Starting Optimized MarketDataProcessor benchmark with 1M messages/sec replay...\n";

        // Generate test data for 1 million messages
        std::cout << "Generating 1,000,000 test messages...\n";
        auto messages = generateTestMessages(1000000);
        
        std::cout << "Generated " << messages.size() << " messages\n";
        
        // Record start time
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Use multiple threads to send messages in batches for better performance
        const int num_threads = std::thread::hardware_concurrency();
        std::cout << "Using " << num_threads << " threads to send messages in batches...\n";
        
        size_t messages_per_thread = messages.size() / num_threads;
        std::vector<std::thread> threads;
        
        for (int i = 0; i < num_threads; ++i) {
            size_t start_idx = i * messages_per_thread;
            size_t end_idx = (i == num_threads - 1) ? messages.size() : (i + 1) * messages_per_thread;
            
            threads.emplace_back([&, start_idx, end_idx]() {
                // Process messages in smaller batches to improve performance
                const size_t batch_size = 1000;
                for (size_t j = start_idx; j < end_idx; j += batch_size) {
                    size_t current_batch_end = std::min(j + batch_size, end_idx);
                    
                    // Create a batch of messages to process
                    std::vector<MarketDataUpdate> batch;
                    batch.reserve(current_batch_end - j);
                    
                    for (size_t k = j; k < current_batch_end; ++k) {
                        batch.push_back(messages[k]);
                    }
                    
                    // Process the entire batch at once using the batch method
                    processor_->processTradeUpdates(batch);
                }
            });
        }
        
        // Wait for all threads to finish sending messages
        for (auto& t : threads) {
            t.join();
        }
        
        std::cout << "All messages sent, waiting for processing to complete...\n";
        
        // Wait for all messages to be processed
        waitForProcessing(messages.size());
        
        // Record end time
        auto end_time = std::chrono::high_resolution_clock::now();
        
        // Calculate performance metrics
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double seconds = duration.count() / 1000000.0;
        double messages_per_second = messages.size() / seconds;
        
        // Get performance metrics from the processor
        auto perf_metrics = processor_->getPerformanceMetrics();
        
        std::cout << "\n=== Optimized Benchmark Results ===\n";
        std::cout << "Total messages processed: " << messages.size() << std::endl;
        std::cout << "Total time: " << seconds << " seconds\n";
        std::cout << "Messages per second: " << static_cast<long long>(messages_per_second) << std::endl;
        std::cout << "Processor reported trades/sec: " << static_cast<long long>(perf_metrics.trades_per_second) << std::endl;
        std::cout << "Processor reported total trades: " << perf_metrics.total_trades_processed << std::endl;
        std::cout << "Average latency: " << perf_metrics.avg_latency_ms << " ms\n";
        std::cout << "Processing latency: " << perf_metrics.processing_latency_us << " μs\n";
        
        // Verify we achieved 1M messages/sec target
        if (messages_per_second >= 1000000.0) {
            std::cout << "\n✅ SUCCESS: Achieved " << static_cast<long long>(messages_per_second) 
                      << " messages/sec (> 1M/sec target)\n";
        } else {
            std::cout << "\n⚠️  NOTE: Achieved " << static_cast<long long>(messages_per_second) 
                      << " messages/sec (< 1M/sec target)\n";
        }
        
        // Additional performance analysis
        std::cout << "\n=== Performance Analysis ===\n";
        std::cout << "Worker threads used: " << std::thread::hardware_concurrency() << std::endl;
        std::cout << "Shards used: 16 (hardcoded in MarketDataProcessor)\n";
        std::cout << "Batch size used: 1000 messages per batch\n";
        std::cout << "Processing latency per message: " << (perf_metrics.processing_latency_us / messages_per_second * 1000) << " nanoseconds\n";
    }

private:
    std::unique_ptr<MarketDataProcessor> processor_;
    
    std::vector<MarketDataUpdate> generateTestMessages(size_t count) {
        std::vector<MarketDataUpdate> messages;
        messages.reserve(count);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> symbol_dist(1, 100);  // 100 different symbols
        std::uniform_real_distribution<> price_dist(100.0, 5000.0);  // Prices between 100-5000
        std::uniform_real_distribution<> size_dist(0.1, 10.0);  // Trade sizes between 0.1-10.0
        std::uniform_int_distribution<> time_dist(0, 100);  // Microsecond variations
        
        uint64_t base_timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        
        for (size_t i = 0; i < count; ++i) {
            MarketDataUpdate update;
            update.type = MarketDataType::TRADE;
            update.symbol_id = static_cast<uint32_t>(symbol_dist(gen));
            update.timestamp = base_timestamp + (i * 10);  // 10μs intervals (simulating realistic timing)
            update.price = price_dist(gen);
            update.size = size_dist(gen);
            update.side = (i % 2 == 0) ? "buy" : "sell";  // Alternate between buy/sell
            
            messages.push_back(update);
        }
        
        return messages;
    }
    
    void waitForProcessing(size_t expected_messages) {
        // Wait for the processor to handle all messages
        // Poll the performance metrics to see if we've processed the expected number
        size_t processed_count = 0;
        auto start_wait = std::chrono::high_resolution_clock::now();
        
        while (processed_count < expected_messages) {
            auto perf_metrics = processor_->getPerformanceMetrics();
            processed_count = perf_metrics.total_trades_processed;
            
            // Timeout after 30 seconds to prevent infinite loop
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_wait);
            if (elapsed.count() > 30) {
                std::cout << "Timeout waiting for processing to complete. Processed: " 
                          << processed_count << "/" << expected_messages << std::endl;
                break;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        std::cout << "Wait completed. Processed " << processed_count << " messages." << std::endl;
    }
};

int main() {
    std::cout << "Optimized MarketDataProcessor 1M Messages/Sec Benchmark\n";
    std::cout << "=====================================================\n";
    
    OptimizedMarketDataProcessorBenchmark benchmark;
    benchmark.runBenchmark();
    
    std::cout << "\nOptimized benchmark completed.\n";
    return 0;
}