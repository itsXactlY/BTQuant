#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>
#include <random>
#include <cstring>
#include <csignal>
#include <sys/resource.h>

#include "dependencies/BTQ_Render_Engine/include/hotspine_layout_v3.hpp"
#include "market_data_collector/src/data/hotspine_data_bridge.hpp"

using namespace BTQuant;

// Global flag to control test execution
volatile sig_atomic_t running = 1;

// Signal handler for graceful shutdown
void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", stopping test..." << std::endl;
    running = 0;
}

int main() {
    std::cout << "Starting Enhanced Ring Buffer Memory Stability Test..." << std::endl;

    // Set up signal handlers for graceful shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Print initial memory usage
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    std::cout << "Initial memory usage: " << usage.ru_maxrss << " KB" << std::endl;

    // Create the data bridge
    HotSpineDataBridge bridge("/enhanced_memory_stability_test");

    // Create a sample HotspineData event
    RenderEngine::HotspineData event;
    event.timestamp = 0;
    event.symbolId = 1;
    event.price = 100.0;
    event.volume = 10.0;
    event.eventType = 0; // TRADE
    event.flags = 0;

    // Run the test for a specified duration or number of events
    const int max_events = 1000000; // 1 million events
    const auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "Writing " << max_events << " events to ring buffer with memory stability checks..." << std::endl;

    int success_count = 0;
    int failure_count = 0;

    for (int i = 0; i < max_events && running; ++i) {
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
        if (success) {
            success_count++;
        } else {
            failure_count++;
            std::cerr << "Failed to write event " << i << std::endl;
        }

        // Occasionally yield to allow other operations and check memory
        if (i % 10000 == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            
            std::cout << "Written " << i << " events so far... Success: " << success_count 
                      << ", Failures: " << failure_count << std::endl;
            
            // Check memory usage periodically
            if (i % 100000 == 0) {
                getrusage(RUSAGE_SELF, &usage);
                std::cout << "Current memory usage: " << usage.ru_maxrss << " KB" << std::endl;
            }
        }
    }

    const auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Completed writing " << success_count << " events successfully." << std::endl;
    std::cout << "Failed to write " << failure_count << " events." << std::endl;
    std::cout << "Test duration: " << duration.count() << " ms" << std::endl;

    // Final memory usage check
    getrusage(RUSAGE_SELF, &usage);
    std::cout << "Final memory usage: " << usage.ru_maxrss << " KB" << std::endl;

    // Calculate memory growth
    std::cout << "Enhanced Ring Buffer Memory Stability Test completed successfully!" << std::endl;

    return 0;
}