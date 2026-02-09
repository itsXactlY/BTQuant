#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <csignal>

#include "data/hotspine_data_bridge.hpp"
#include "../../dependencies/BTQ_Render_Engine/include/trading/HotspineData.h"

namespace BTQuant {

// Global flag to handle graceful shutdown
std::atomic<bool> g_running{true};

// Signal handler for graceful shutdown
void signalHandler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down gracefully..." << std::endl;
    g_running.store(false);
}

} // namespace BTQuant

int main() {
    using namespace BTQuant;
    
    // Register signal handlers for graceful shutdown
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    
    std::cout << "Starting Market Data Collector with Warm-Up Generator..." << std::endl;
    
    // Initialize the HotSpine data bridge
    auto bridge = std::make_shared<HotSpineDataBridge>();
    
    std::cout << "HotSpine data bridge initialized." << std::endl;
    
    // Main warm-up generation loop
    std::cout << "Starting warm-up generator (injecting events every 100ms)..." << std::endl;
    
    uint64_t event_counter = 0;
    
    while (g_running.load()) {
        // Create a warm-up event
        auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        
        RenderEngine::HotspineData warmup_event;
        warmup_event.timestamp = now;
        warmup_event.symbolId = 0;  // Use symbol ID 0 for warm-up events
        warmup_event.eventType = 0; // Use event type 0 for warm-up events
        warmup_event.price = 0.0;   // Price is irrelevant for warm-up
        warmup_event.volume = 0.0;  // Volume is irrelevant for warm-up
        warmup_event.flags = RenderEngine::HotspineData::IS_WARMUP; // Set the warm-up flag
        warmup_event.reserved_flags[0] = 0;
        warmup_event.reserved_flags[1] = 0;
        warmup_event.reserved_flags[2] = 0;
        warmup_event.sequenceNumber = static_cast<uint32_t>(event_counter++);
        warmup_event.payloadSize = 0;
        
        // Initialize padding to zero
        memset(warmup_event.padding, 0, sizeof(warmup_event.padding));
        
        // Write the warm-up event directly to the shared memory ring buffer
        if (!bridge->write_direct(warmup_event)) {
            std::cerr << "Failed to write warm-up event to shared memory" << std::endl;
        }
        
        // Sleep for 100ms before next warm-up event
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    std::cout << "Warm-up generator stopped." << std::endl;
    
    return 0;
}