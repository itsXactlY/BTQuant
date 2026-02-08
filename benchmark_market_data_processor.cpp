#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <random>
#include <atomic>
#include <memory>
#include <future>

#include "dependencies/BTQ_Render_Engine/include/market_data_processor.hpp"

using namespace BTQuant::RenderEngine;

/**
 * Benchmark for MarketDataProcessor with 1M messages/sec replay
 */
class MarketDataProcessorBenchmark {
public:
    MarketDataProcessorBenchmark() : processor_(std::make_unique<MarketDataProcessor>()) {}

    void runBenchmark() {
        std::cout << "Starting MarketDataProcessor benchmark with 1M messages/sec replay...\n";

        // Generate test data for 1 million messages
        std::cout << "Generating 1,000,000 test messages...\n";
        auto messages = generateTestMessages(1000000);
        
        std::cout << "Generated " << messages.size() << " messages\n";
        
        // Record start time
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Send all messages to the processor
        std::cout << "Sending messages to MarketDataProcessor...\n";
        for (const auto& msg : messages) {
            processor_->processTradeUpdate(msg);
        }
        
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
        
        std::cout << "\n=== Benchmark Results ===\n";
        std::cout << "Total messages processed: " << messages.size() << std::endl;
        std::cout << "Total time: " << seconds << " seconds\n";
        std::cout << "Messages per second: " << messages_per_second << std::endl;
        std::cout << "Processor reported trades/sec: " << perf_metrics.trades_per_second << std::endl;
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
        std::uniform_int_distribution<> time_dist(0, 1000000);  // Microsecond increments
        
        uint64_t base_timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        
        for (size_t i = 0; i < count; ++i) {
            MarketDataUpdate update;
            update.type = MarketDataType::TRADE;
            update.symbol_id = static_cast<uint32_t>(symbol_dist(gen));
            update.timestamp = base_timestamp + (i * 1000);  // 1ms intervals (simulating realistic timing)
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
    std::cout << "MarketDataProcessor 1M Messages/Sec Benchmark\n";
    std::cout << "==============================================\n";
    
    MarketDataProcessorBenchmark benchmark;
    benchmark.runBenchmark();
    
    std::cout << "\nBenchmark completed.\n";
    return 0;
}