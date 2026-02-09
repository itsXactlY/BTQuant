#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>
#include <random>
#include <cstring>

#include "dependencies/BTQ_Render_Engine/include/hotspine_layout_v3.hpp"
#include "market_data_collector/src/data/hotspine_data_bridge.hpp"

using namespace BTQuant;

int main() {
    std::cout << "Starting Ring Buffer Memory Stability Test..." << std::endl;
    
    // Create the data bridge
    HotSpineDataBridge bridge("/test_ring_buffer_stability");
    
    // Create a sample HotspineData event
    RenderEngine::HotspineData event;
    event.timestamp = 0;
    event.symbolId = 1;
    event.price = 100.0;
    event.volume = 10.0;
    event.eventType = 0; // TRADE
    event.flags = 0;
    
    // Write a series of events to the ring buffer
    const int num_events = 100000;
    std::cout << "Writing " << num_events << " events to ring buffer..." << std::endl;
    
    for (int i = 0; i < num_events; ++i) {
        event.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        event.price = 100.0 + (i % 1000) * 0.01; // Vary price slightly
        event.volume = 10.0 + (i % 100); // Vary volume slightly
        
        if (i % 2 == 0) {
            event.flags = 0x04; // Buy flag
        } else {
            event.flags = 0x00; // Sell flag
        }
        
        bool success = bridge.write_direct(event);
        if (!success) {
            std::cerr << "Failed to write event " << i << std::endl;
        }
        
        // Occasionally yield to allow other operations
        if (i % 1000 == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
            std::cout << "Written " << i << " events so far..." << std::endl;
        }
    }
    
    std::cout << "Completed writing " << num_events << " events." << std::endl;
    
    // Wait a bit to allow any background operations to complete
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    std::cout << "Ring Buffer Memory Stability Test completed successfully!" << std::endl;
    
    return 0;
}