#include "../include/market_data_processor.hpp"
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <cstring> // for memset

int main() {
    BTQuant::HotspineDataStorage storage; // Using the atomic storage implementation
    
    std::cout << "Testing AtomicMarketDataStorage...\n";
    
    // Test basic functionality
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
    
    // Store data
    bool stored = storage.store(testData);
    if (!stored) {
        std::cerr << "Failed to store data\n";
        return 1;
    }
    
    // Load data
    BTQuant::RenderEngine::HotspineData loadedData;
    bool loaded = storage.load(loadedData);
    if (!loaded) {
        std::cerr << "Failed to load data\n";
        return 1;
    }
    
    // Verify data integrity
    if (loadedData.timestamp != testData.timestamp ||
        loadedData.symbolId != testData.symbolId ||
        loadedData.eventType != testData.eventType ||
        loadedData.price != testData.price ||
        loadedData.volume != testData.volume) {
        std::cerr << "Data mismatch after store/load\n";
        return 1;
    }
    
    std::cout << "Basic store/load test passed.\n";
    
    // Test concurrent operations
    std::cout << "Testing concurrent operations...\n";
    
    const int num_threads = 4;
    const int num_operations = 1000;
    
    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;
    
    // Producer threads
    for (int t = 0; t < num_threads; ++t) {
        producers.emplace_back([&storage, t, num_operations]() {
            for (int i = 0; i < num_operations; ++i) {
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
                
                storage.store(data);
            }
        });
    }
    
    // Consumer threads
    for (int t = 0; t < num_threads; ++t) {
        consumers.emplace_back([&storage, t, num_operations]() {
            for (int i = 0; i < num_operations; ++i) {
                BTQuant::RenderEngine::HotspineData data;
                storage.load(data);
                // Just consume the data, no validation in this test
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
    
    std::cout << "Concurrent operations test completed.\n";
    
    // Print statistics
    std::cout << "Statistics:\n";
    std::cout << "  Available count: " << storage.available_count() << "\n";
    std::cout << "  Dropped count: " << storage.dropped_count() << "\n";
    std::cout << "  Capacity: " << storage.capacity() << "\n";
    std::cout << "  Is empty: " << (storage.is_empty() ? "true" : "false") << "\n";
    
    std::cout << "All tests passed!\n";
    
    return 0;
}