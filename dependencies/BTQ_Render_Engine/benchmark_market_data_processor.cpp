#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <random>
#include <memory>
#include <atomic>
#include <future>
#include <algorithm>

#include "market_data_processor.hpp"
#include "data/data_types.hpp"

using namespace BTQuant::RenderEngine;

/**
 * Benchmark for MarketDataProcessor with 1M messages/sec replay
 * This benchmark tests the performance of the MarketDataProcessor under high load conditions
 */
class MarketDataProcessorBenchmark {
public:
    MarketDataProcessorBenchmark() 
        : processor_(std::make_shared<MarketDataProcessor>()) {}

    void runBenchmark() {
        std::cout << "Starting MarketDataProcessor benchmark with 1M messages/sec replay...\n";
        
        // Generate 1 million market data updates (mix of trades and orderbooks)
        std::cout << "Generating 1,000,000 market data updates...\n";
        auto updates = generateMarketDataUpdates(1000000);
        
        std::cout << "Starting benchmark...\n";
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Send all updates to the processor
        sendUpdates(updates);
        
        // Wait for processing to complete
        waitForProcessing();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        
        // Calculate and report metrics
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto seconds = duration.count() / 1000.0;
        
        auto metrics = processor_->getPerformanceMetrics();
        uint64_t total_messages = metrics.total_trades_processed + metrics.total_orderbooks_processed;
        
        std::cout << "\n=== Benchmark Results ===\n";
        std::cout << "Time taken: " << seconds << " seconds\n";
        std::cout << "Messages processed: " << total_messages << "\n";
        std::cout << "Target throughput: 1,000,000 messages/sec\n";
        std::cout << "Actual throughput: " << total_messages / seconds << " messages/sec\n";
        std::cout << "Trades processed: " << metrics.total_trades_processed << "\n";
        std::cout << "Orderbooks processed: " << metrics.total_orderbooks_processed << "\n";
        std::cout << "Avg processing latency: " << metrics.processing_latency_us << " μs\n";
        std::cout << "Avg overall latency: " << metrics.avg_latency_ms << " ms\n";
        std::cout << "Trades per second: " << metrics.trades_per_second << "\n";
        std::cout << "Orderbooks per second: " << metrics.orderbooks_per_second << "\n";
        
        // Calculate efficiency
        double efficiency = (static_cast<double>(total_messages) / seconds) / 1000000.0 * 100.0;
        std::cout << "Throughput efficiency: " << efficiency << "% of target\n";
        
        // Check if we achieved at least 90% of target throughput
        bool passed = (total_messages / seconds >= 900000) && (total_messages >= 950000);
        
        if (passed) {
            std::cout << "\nBenchmark PASSED: Achieved " << efficiency << "% of target throughput!\n";
        } else {
            std::cout << "\nBenchmark FAILED: Throughput below acceptable threshold!\n";
        }
        
        // Additional metrics
        std::cout << "\n=== Additional Metrics ===\n";
        if (total_messages > 0) {
            std::cout << "Average message processing time: " << (duration.count() / static_cast<double>(total_messages)) << " ms/message\n";
            std::cout << "Processing utilization estimate: " << std::min(100.0, (total_messages * metrics.processing_latency_us) / (duration.count() * 1000.0)) << "%\n";
        }
    }

private:
    std::shared_ptr<MarketDataProcessor> processor_;
    
    std::vector<MarketDataUpdate> generateMarketDataUpdates(size_t count) {
        std::vector<MarketDataUpdate> updates;
        updates.reserve(count);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> symbol_dist(1, 100);  // 100 different symbols
        std::uniform_int_distribution<> type_dist(0, 1);      // 0=trade, 1=orderbook
        std::uniform_real_distribution<> price_dist(100.0, 200.0);
        std::uniform_real_distribution<> size_dist(1.0, 100.0);
        
        uint64_t timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        
        for (size_t i = 0; i < count; ++i) {
            MarketDataUpdate update;
            update.symbol_id = symbol_dist(gen);
            update.timestamp = timestamp + i;  // Slightly increasing timestamps
            
            // Alternate between trade and orderbook updates
            if (type_dist(gen) == 0) {
                update.type = MarketDataType::TRADE;
                update.price = price_dist(gen);
                update.size = size_dist(gen);
                update.side = (i % 2 == 0) ? "buy" : "sell";
            } else {
                update.type = MarketDataType::ORDERBOOK;
                
                // Generate some bids and asks
                for (int j = 0; j < 10; ++j) {
                    PriceLevel bid;
                    bid.price = price_dist(gen) - (j * 0.01);
                    bid.size = size_dist(gen);
                    update.bids.push_back(bid);
                    
                    PriceLevel ask;
                    ask.price = price_dist(gen) + (j * 0.01);
                    ask.size = size_dist(gen);
                    update.asks.push_back(ask);
                }
            }
            
            updates.push_back(update);
        }
        
        return updates;
    }
    
    void sendUpdates(const std::vector<MarketDataUpdate>& updates) {
        std::cout << "Sending updates to processor at 1M/sec rate...\n";
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Send messages at approximately 1M/sec rate
        const size_t batch_size = 10000;  // Send in batches of 10k
        const auto target_interval = std::chrono::milliseconds(10);  // 10k messages per 10ms = 1M/sec
        
        for (size_t i = 0; i < updates.size(); i += batch_size) {
            auto batch_start = std::chrono::high_resolution_clock::now();
            
            size_t end = std::min(i + batch_size, updates.size());
            
            // Send batch of updates
            for (size_t j = i; j < end; ++j) {
                const auto& update = updates[j];
                if (update.type == MarketDataType::TRADE) {
                    processor_->processTradeUpdate(update);
                } else {
                    processor_->processOrderbookUpdate(update);
                }
            }
            
            // Calculate elapsed time and sleep if needed to maintain target rate
            auto batch_end = std::chrono::high_resolution_clock::now();
            auto batch_duration = std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start);
            
            if (batch_duration < target_interval) {
                std::this_thread::sleep_for(target_interval - batch_duration);
            }
            
            // Print progress
            if (i % 100000 == 0) {
                std::cout << "Sent " << i << " updates...\n";
            }
        }
        
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_time);
        
        std::cout << "All updates sent to processor in " << total_time.count() << " ms.\n";
    }
    
    void waitForProcessing() {
        // Wait until all messages are processed
        std::cout << "Waiting for processing to complete...\n";
        
        auto initial_metrics = processor_->getPerformanceMetrics();
        uint64_t last_processed = initial_metrics.total_trades_processed + initial_metrics.total_orderbooks_processed;
        int no_progress_count = 0;
        auto start_wait_time = std::chrono::high_resolution_clock::now();
        const auto max_wait_time = std::chrono::seconds(30); // Max wait time of 30 seconds
        
        while (true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            
            auto current_metrics = processor_->getPerformanceMetrics();
            uint64_t current_processed = current_metrics.total_trades_processed + current_metrics.total_orderbooks_processed;
            
            if (current_processed >= 1000000) {
                // We've processed at least the target amount
                std::cout << "Target number of messages processed.\n";
                break;
            }
            
            auto elapsed = std::chrono::high_resolution_clock::now() - start_wait_time;
            if (elapsed > max_wait_time) {
                std::cout << "Maximum wait time exceeded, stopping wait.\n";
                break;
            }
            
            if (current_processed == last_processed) {
                no_progress_count++;
                if (no_progress_count > 100) {  // 5 seconds of no progress (with 50ms sleep)
                    std::cout << "No progress detected for 5 seconds, assuming processing complete.\n";
                    break;
                }
            } else {
                no_progress_count = 0;  // Reset counter if we see progress
            }
            
            last_processed = current_processed;
        }
        
        std::cout << "Processing appears to be complete.\n";
    }
};

int main() {
    std::cout << "MarketDataProcessor 1M Messages/Sec Benchmark\n";
    std::cout << "================================================\n";
    
    MarketDataProcessorBenchmark benchmark;
    benchmark.runBenchmark();
    
    return 0;
}