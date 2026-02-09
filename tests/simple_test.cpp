#include "market_data_processor.hpp"
#include <iostream>

int main() {
    // Test that the type is available
    BTQuant::HotspineDataStorage storage;
    
    std::cout << "HotspineDataStorage type is available!" << std::endl;
    
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
    
    // Store and load test
    bool stored = storage.store(testData);
    if (!stored) {
        std::cout << "Store failed!" << std::endl;
        return 1;
    }
    
    BTQuant::RenderEngine::HotspineData loadedData;
    bool loaded = storage.load(loadedData);
    if (!loaded) {
        std::cout << "Load failed!" << std::endl;
        return 1;
    }
    
    std::cout << "Store and load successful!" << std::endl;
    std::cout << "Available count: " << storage.available_count() << std::endl;
    std::cout << "Capacity: " << storage.capacity() << std::endl;
    
    return 0;
}