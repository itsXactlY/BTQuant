#include "../include/market_data_processor.hpp"
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <random>
#include <cstring> // for memset

int main() {
    std::cout << "Starting Latency Test for AtomicMarketDataStorage...\n";

    // Create storage instance
    BTQuant::HotspineDataStorage storage;

    // Prepare test data
    BTQuant::RenderEngine::HotspineData testData;
    testData.timestamp = 1234567890;
    testData.symbolId = 1;
    testData.eventType = 0;
    testData.price = 100.5;
    testData.volume = 10.0;
    testData.flags = 0;
    testData.sequenceNumber = 1;
    testData.payloadSize = 0;

    // Initialize padding to zero
    memset(testData.padding, 0, sizeof(testData.padding));

    // Test 1: Single-threaded store/load latency
    std::cout << "\n=== Single-threaded Store/Latency Test ===\n";
    
    const int iterations = 100000;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<double> store_latencies;
    std::vector<double> load_latencies;
    store_latencies.reserve(iterations);
    load_latencies.reserve(iterations);
    
    for (int i = 0; i < iterations; ++i) {
        // Measure store latency
        auto store_start = std::chrono::high_resolution_clock::now();
        bool stored = storage.store(testData);
        auto store_end = std::chrono::high_resolution_clock::now();
        
        if (!stored) {
            std::cerr << "Failed to store data at iteration " << i << std::endl;
            return 1;
        }
        
        auto store_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(store_end - store_start);
        store_latencies.push_back(store_duration.count());
        
        // Measure load latency
        BTQuant::RenderEngine::HotspineData loadedData;
        auto load_start = std::chrono::high_resolution_clock::now();
        bool loaded = storage.load(loadedData);
        auto load_end = std::chrono::high_resolution_clock::now();
        
        if (!loaded) {
            std::cerr << "Failed to load data at iteration " << i << std::endl;
            return 1;
        }
        
        auto load_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(load_end - load_start);
        load_latencies.push_back(load_duration.count());
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // Calculate statistics
    double avg_store_latency = 0.0;
    double avg_load_latency = 0.0;
    double min_store_latency = store_latencies[0];
    double max_store_latency = store_latencies[0];
    double min_load_latency = load_latencies[0];
    double max_load_latency = load_latencies[0];
    
    for (double lat : store_latencies) {
        avg_store_latency += lat;
        if (lat < min_store_latency) min_store_latency = lat;
        if (lat > max_store_latency) max_store_latency = lat;
    }
    avg_store_latency /= store_latencies.size();
    
    for (double lat : load_latencies) {
        avg_load_latency += lat;
        if (lat < min_load_latency) min_load_latency = lat;
        if (lat > max_load_latency) max_load_latency = lat;
    }
    avg_load_latency /= load_latencies.size();
    
    std::cout << "Total operations: " << iterations * 2 << std::endl;
    std::cout << "Total time: " << total_duration.count() << " microseconds" << std::endl;
    std::cout << "Operations per second: " << (iterations * 2.0 * 1000000.0) / total_duration.count() << std::endl;
    std::cout << "\nStore Latency:" << std::endl;
    std::cout << "  Average: " << avg_store_latency << " ns" << std::endl;
    std::cout << "  Min: " << min_store_latency << " ns" << std::endl;
    std::cout << "  Max: " << max_store_latency << " ns" << std::endl;
    std::cout << "\nLoad Latency:" << std::endl;
    std::cout << "  Average: " << avg_load_latency << " ns" << std::endl;
    std::cout << "  Min: " << min_load_latency << " ns" << std::endl;
    std::cout << "  Max: " << max_load_latency << " ns" << std::endl;

    // Test 2: Concurrent producer-consumer latency test
    std::cout << "\n=== Concurrent Producer-Consumer Latency Test ===\n";
    
    const int num_producers = 2;
    const int num_consumers = 2;
    const int ops_per_thread = 50000;
    
    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;
    
    // Vectors to collect latencies from different threads
    std::vector<std::vector<double>> producer_latencies(num_producers);
    std::vector<std::vector<double>> consumer_latencies(num_consumers);
    
    // Producer threads with latency measurement
    for (int t = 0; t < num_producers; ++t) {
        producers.emplace_back([&storage, t, ops_per_thread, &producer_latencies]() {
            producer_latencies[t].reserve(ops_per_thread);
            
            for (int i = 0; i < ops_per_thread; ++i) {
                BTQuant::RenderEngine::HotspineData data;
                data.timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
                data.symbolId = t;
                data.eventType = i % 2; // Alternate between 0 and 1
                data.price = 100.0 + i + t;
                data.volume = i * 2.0;
                data.flags = 0;
                data.sequenceNumber = i;
                data.payloadSize = 0;

                // Initialize padding to zero
                memset(data.padding, 0, sizeof(data.padding));

                auto start = std::chrono::high_resolution_clock::now();
                storage.store(data);
                auto end = std::chrono::high_resolution_clock::now();
                
                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
                producer_latencies[t].push_back(duration.count());
            }
        });
    }

    // Consumer threads with latency measurement
    for (int t = 0; t < num_consumers; ++t) {
        consumers.emplace_back([&storage, t, ops_per_thread, &consumer_latencies]() {
            consumer_latencies[t].reserve(ops_per_thread);
            
            for (int i = 0; i < ops_per_thread; ++i) {
                BTQuant::RenderEngine::HotspineData data;
                auto start = std::chrono::high_resolution_clock::now();
                bool success = storage.load(data);
                auto end = std::chrono::high_resolution_clock::now();
                
                if (success) {
                    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
                    consumer_latencies[t].push_back(duration.count());
                } else {
                    // If no data available, record a high latency to indicate waiting
                    consumer_latencies[t].push_back(10000); // 10 microseconds as indication of waiting
                }
            }
        });
    }

    // Wait for all threads to complete
    for (auto& producer : producers) {
        producer.join();
    }

    for (auto& consumer : consumers) {
        consumer.join();
    }

    // Calculate concurrent test statistics
    double avg_producer_latency = 0.0;
    double min_producer_latency = std::numeric_limits<double>::max();
    double max_producer_latency = 0.0;
    int total_producer_ops = 0;
    
    for (const auto& thread_latencies : producer_latencies) {
        for (double lat : thread_latencies) {
            avg_producer_latency += lat;
            if (lat < min_producer_latency) min_producer_latency = lat;
            if (lat > max_producer_latency) max_producer_latency = lat;
            total_producer_ops++;
        }
    }
    avg_producer_latency /= total_producer_ops;
    
    double avg_consumer_latency = 0.0;
    double min_consumer_latency = std::numeric_limits<double>::max();
    double max_consumer_latency = 0.0;
    int total_consumer_ops = 0;
    
    for (const auto& thread_latencies : consumer_latencies) {
        for (double lat : thread_latencies) {
            avg_consumer_latency += lat;
            if (lat < min_consumer_latency) min_consumer_latency = lat;
            if (lat > max_consumer_latency) max_consumer_latency = lat;
            total_consumer_ops++;
        }
    }
    avg_consumer_latency /= total_consumer_ops > 0 ? total_consumer_ops : 1;
    
    std::cout << "Producer operations: " << total_producer_ops << std::endl;
    std::cout << "Consumer operations: " << total_consumer_ops << std::endl;
    std::cout << "\nProducer Latency (concurrent):" << std::endl;
    std::cout << "  Average: " << avg_producer_latency << " ns" << std::endl;
    std::cout << "  Min: " << min_producer_latency << " ns" << std::endl;
    std::cout << "  Max: " << max_producer_latency << " ns" << std::endl;
    std::cout << "\nConsumer Latency (concurrent):" << std::endl;
    std::cout << "  Average: " << avg_consumer_latency << " ns" << std::endl;
    std::cout << "  Min: " << min_consumer_latency << " ns" << std::endl;
    std::cout << "  Max: " << max_consumer_latency << " ns" << std::endl;

    // Print final statistics
    std::cout << "\n=== Final Statistics ===" << std::endl;
    std::cout << "Storage capacity: " << storage.capacity() << std::endl;
    std::cout << "Final available count: " << storage.available_count() << std::endl;
    std::cout << "Dropped count: " << storage.dropped_count() << std::endl;

    std::cout << "\nLatency Test Completed Successfully!" << std::endl;

    return 0;
}