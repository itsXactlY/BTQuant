#include "lockfreesnapshotpipeline.h"
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <random>

int main() {
    std::cout << "Testing Lock-Free Snapshot Pipeline\n";

    // Create a lock-free snapshot pipeline for 1000 symbols
    LockFreeSnapshotPipeline pipeline(1000);

    // Test 1: Single symbol write and read
    std::cout << "\n=== Test 1: Single Symbol Write/Read ===\n";
    
    AtomicMarketData test_data(100.5, 1000.0, 100.4, 100.6, 500.0, 600.0);
    test_data.timestamp.store(std::chrono::system_clock::now());
    
    bool write_result = pipeline.write_market_data(0, test_data);
    std::cout << "Write result: " << (write_result ? "Success" : "Failed") << std::endl;
    
    AtomicMarketData read_data;
    bool read_result = pipeline.read_market_data_snapshot(0, read_data);
    std::cout << "Read result: " << (read_result ? "Success" : "Failed") << std::endl;
    
    if (read_result) {
        std::cout << "Price: " << read_data.price.load() << std::endl;
        std::cout << "Volume: " << read_data.volume.load() << std::endl;
        std::cout << "Bid: " << read_data.bid_price.load() << "@" << read_data.bid_volume.load() << std::endl;
        std::cout << "Ask: " << read_data.ask_price.load() << "@" << read_data.ask_volume.load() << std::endl;
    }

    // Test 2: Batch read
    std::cout << "\n=== Test 2: Batch Read ===\n";
    
    // Write data to multiple symbols
    for (int i = 1; i <= 5; ++i) {
        AtomicMarketData data(100.0 + i*0.1, 1000.0 + i*100, 
                             99.9 + i*0.1, 100.1 + i*0.1, 
                             500.0 + i*50, 600.0 + i*50);
        pipeline.write_market_data(i, data);
    }
    
    uint32_t symbols[] = {1, 2, 3, 4, 5};
    AtomicMarketData batch_results[5];
    
    size_t batch_count = pipeline.read_batch_snapshot(symbols, batch_results, 5);
    std::cout << "Batch read count: " << batch_count << std::endl;
    
    for (size_t i = 0; i < batch_count; ++i) {
        std::cout << "Symbol " << symbols[i] << " - Price: " << batch_results[i].price.load() 
                  << ", Volume: " << batch_results[i].volume.load() << std::endl;
    }

    // Test 3: Concurrent producer/consumer simulation
    std::cout << "\n=== Test 3: Concurrent Producer/Consumer Simulation ===\n";
    
    // Producer thread - continuously writes data
    std::atomic<bool> stop_producer(false);
    std::thread producer([&pipeline, &stop_producer]() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> price_dist(95.0, 105.0);
        std::uniform_int_distribution<> symbol_dist(0, 99); // Symbols 0-99
        
        int counter = 0;
        while (!stop_producer.load()) {
            uint32_t symbol_id = symbol_dist(gen);
            AtomicMarketData data(
                price_dist(gen),           // Random price
                1000.0 + (counter % 1000), // Volume
                price_dist(gen) - 0.05,    // Bid price
                price_dist(gen) + 0.05,    // Ask price
                500.0 + (counter % 500),   // Bid volume
                600.0 + (counter % 600)    // Ask volume
            );
            
            pipeline.write_market_data(symbol_id, data);
            
            // Commit snapshot every 1000 updates to make data available to readers
            if (++counter % 1000 == 0) {
                pipeline.commit_snapshot();
            }
            
            // Yield occasionally to allow other threads to run
            if (counter % 10 == 0) {
                std::this_thread::yield();
            }
        }
    });
    
    // Consumer thread - periodically reads snapshots
    std::thread consumer([&pipeline, &stop_producer]() {
        int read_counter = 0;
        while (!stop_producer.load()) {
            AtomicMarketData snapshot;
            bool success = pipeline.read_market_data_snapshot(0, snapshot);
            
            if (success && ++read_counter % 1000 == 0) {
                std::cout << "Consumer read symbol 0 - Price: " << snapshot.price.load() 
                          << ", Seq: " << snapshot.sequence_number.load() << std::endl;
            }
            
            // Sleep briefly to avoid busy-waiting
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });
    
    // Let the threads run for a few seconds
    std::this_thread::sleep_for(std::chrono::seconds(3));
    
    // Stop the threads
    stop_producer.store(true);
    producer.join();
    consumer.join();
    
    // Print final statistics
    auto stats = pipeline.get_stats();
    std::cout << "\n=== Pipeline Statistics ===\n";
    std::cout << "Total updates: " << stats.total_updates << std::endl;
    std::cout << "Dropped updates: " << stats.dropped_updates << std::endl;
    std::cout << "Write head: " << stats.write_head << std::endl;
    std::cout << "Read tail: " << stats.read_tail << std::endl;
    std::cout << "Buffer capacity: " << stats.buffer_capacity << std::endl;

    // Test 4: Using MarketDataPoller
    std::cout << "\n=== Test 4: MarketDataPoller ===\n";
    
    LockFreeSnapshotPipeline poll_pipeline(100);
    
    // Write some test data
    for (int i = 0; i < 5; ++i) {
        AtomicMarketData data(100.0 + i*0.5, 1000.0 + i*100, 
                             99.9 + i*0.5, 100.1 + i*0.5, 
                             500.0 + i*50, 600.0 + i*50);
        poll_pipeline.write_market_data(i, data);
    }
    
    // Create a poller and add some symbols to watch
    MarketDataPoller poller(&poll_pipeline);
    for (int i = 0; i < 3; ++i) {
        poller.add_symbol_to_watch(i);
    }
    
    // Poll the watched symbols
    std::cout << "Polling watched symbols:" << std::endl;
    poller.poll_watched_symbols([](uint32_t symbol_id, const AtomicMarketData& data) {
        std::cout << "Symbol " << symbol_id << " - Price: " << data.price.load() 
                  << ", Volume: " << data.volume.load() << std::endl;
    });

    std::cout << "\nLock-Free Snapshot Pipeline test completed successfully!\n";

    return 0;
}