#include "../include/performance/memory_tracker.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <memory>

using namespace btq::performance;

void testBasicFunctionality() {
    std::cout << "Testing basic memory tracker functionality...\n";
    
    MemoryTracker& tracker = getGlobalMemoryTracker();
    
    // Start tracking
    tracker.startTracking();
    
    // Allocate some memory
    const size_t alloc_size = 1024 * 1024; // 1 MB
    void* ptr1 = malloc(alloc_size);
    tracker.trackAllocation(alloc_size, ptr1, "test_allocation_1");
    
    // Check memory usage increased
    size_t current_usage = tracker.getCurrentMemoryUsage();
    std::cout << "Current memory usage after allocation: " << current_usage << " bytes\n";
    
    // Deallocate memory - track deallocation before freeing
    tracker.trackDeallocation(ptr1);
    free(ptr1);
    
    // Stop tracking
    tracker.stopTracking();
    
    std::cout << "Basic functionality test completed.\n\n";
}

void testLeakDetection() {
    std::cout << "Testing leak detection...\n";
    
    MemoryTracker& tracker = getGlobalMemoryTracker();
    tracker.setLeakThreshold(1.0); // Set threshold to 1 second for testing
    
    tracker.startTracking();
    
    // Allocate memory that will be considered a leak
    std::vector<void*> allocations;
    for (int i = 0; i < 5; ++i) {
        void* ptr = malloc(1024 * 100); // 100KB
        allocations.push_back(ptr);
        tracker.trackAllocation(1024 * 100, ptr, "leak_candidate_" + std::to_string(i));
    }
    
    // Wait for allocations to exceed leak threshold
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Check for leaks
    auto leaks = tracker.identifyLeaks(1.0);
    std::cout << "Number of leaks detected: " << leaks.size() << std::endl;
    
    // Clean up allocations
    for (auto ptr : allocations) {
        tracker.trackDeallocation(ptr);
        free(ptr);
    }
    
    tracker.stopTracking();
    
    std::cout << "Leak detection test completed.\n\n";
}

void testTrendTracking() {
    std::cout << "Testing trend tracking...\n";
    
    MemoryTracker& tracker = getGlobalMemoryTracker();
    tracker.setSamplingInterval(50); // Sample every 50ms
    
    tracker.startTracking();
    
    // Simulate memory usage changes over time
    for (int i = 0; i < 10; ++i) {
        // Allocate some memory
        void* ptr = malloc(1024 * 50); // 50KB
        tracker.trackAllocation(1024 * 50, ptr, "trend_test_" + std::to_string(i));
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Deallocate some memory occasionally
        if (i % 3 == 0) {
            tracker.trackDeallocation(ptr);
            free(ptr);
        }
    }
    
    // Get memory samples and growth rate
    auto samples = tracker.getMemorySamples();
    std::cout << "Number of memory samples collected: " << samples.size() << std::endl;
    
    double growth_rate = tracker.getAverageMemoryGrowthRate();
    std::cout << "Average memory growth rate: " << growth_rate << " bytes/sec" << std::endl;
    
    tracker.stopTracking();
    
    std::cout << "Trend tracking test completed.\n\n";
}

void testReportGeneration() {
    std::cout << "Testing report generation...\n";
    
    MemoryTracker& tracker = getGlobalMemoryTracker();
    
    // Perform some allocations to populate data
    tracker.startTracking();
    
    for (int i = 0; i < 3; ++i) {
        void* ptr = malloc(1024 * 10); // 10KB
        tracker.trackAllocation(1024 * 10, ptr, "report_test_" + std::to_string(i));
    }
    
    // Wait a bit to allow sampling
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // Export report
    tracker.exportMemoryReport("memory_report_test.txt");
    
    tracker.stopTracking();
    
    std::cout << "Report generation test completed. Report saved to memory_report_test.txt\n\n";
}

int main() {
    std::cout << "=== Memory Tracker Test Suite ===\n\n";
    
    testBasicFunctionality();
    testLeakDetection();
    testTrendTracking();
    testReportGeneration();
    
    std::cout << "=== All tests completed ===\n";
    
    return 0;
}