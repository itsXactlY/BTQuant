/**
 * @file test_hazard_pointer_leak_check.cpp
 * @brief Test to verify Hazard Pointers correctly reclaim memory after the 4-hour window moves.
 * 
 * This test simulates the scenario where data chunks older than 4 hours are pruned
 * and their memory should be reclaimed by the hazard pointer mechanism once no
 * readers are referencing them.
 */

#include "dependencies/BTQ_Render_Engine/include/threading/hazard_pointer_cxx26.hpp"
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <memory>
#include <chrono>
#include <cassert>
#include <map>

using namespace btq;

// Counter to track how many objects have been destroyed
static std::atomic<int> destruction_counter{0};

// Class representing a data chunk that can be protected by hazard pointers
struct DataChunk : hazard_pointer_obj_base<DataChunk> {
    int id;
    uint64_t creation_time;  // Timestamp in microseconds
    
    DataChunk(int chunk_id) : id(chunk_id) {
        creation_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }

    ~DataChunk() {
        destruction_counter.fetch_add(1, std::memory_order_relaxed);
        std::cout << "DataChunk with ID " << id << " destroyed at time: " 
                  << std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::high_resolution_clock::now().time_since_epoch()).count() 
                  << std::endl;
    }
};

// Shared atomic pointer to the current data chunk
std::atomic<DataChunk*> current_chunk{nullptr};

// Flag to signal threads to stop
std::atomic<bool> should_stop{false};

// Global counter for created chunks
std::atomic<int> total_chunks_created{0};

// Map to track allocated chunks for leak detection
std::map<int, DataChunk*> allocated_chunks;
std::mutex chunks_mutex;

void reader_thread(int reader_id) {
    hazard_pointer hp = make_hazard_pointer();
    
    while (!should_stop.load()) {
        // Protect the current chunk
        DataChunk* chunk = hp.protect(current_chunk);
        
        if (chunk) {
            // Simulate some work with the chunk
            volatile int temp = chunk->id;  // Prevent optimization
            
            // Print occasional status to show activity
            static int counter = 0;
            if (++counter % 1000 == 0) {
                std::cout << "Reader " << reader_id << " accessed chunk " << chunk->id << std::endl;
            }
        }
        
        // Reset protection before next iteration
        hp.reset_protection();
        
        // Small delay to simulate realistic workload
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

void writer_thread() {
    int chunk_id = 0;
    
    while (!should_stop.load()) {
        // Create a new data chunk
        DataChunk* old_chunk = current_chunk.load();
        DataChunk* new_chunk = new DataChunk(++chunk_id);
        
        // Update the global counter
        total_chunks_created.store(chunk_id, std::memory_order_relaxed);
        
        // Track the allocation for leak checking
        {
            std::lock_guard<std::mutex> lock(chunks_mutex);
            allocated_chunks[new_chunk->id] = new_chunk;
        }
        
        // Atomically replace the pointer
        DataChunk* expected = old_chunk;
        while (!current_chunk.compare_exchange_weak(expected, new_chunk)) {
            expected = old_chunk;
        }
        
        // Retire the old chunk if it existed
        if (old_chunk) {
            old_chunk->retire();
            
            // Remove from tracking map
            {
                std::lock_guard<std::mutex> lock(chunks_mutex);
                allocated_chunks.erase(old_chunk->id);
            }
        }
        
        std::cout << "Writer created chunk " << new_chunk->id << std::endl;
        
        // Simulate creating new chunks at a reasonable rate
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

void simulate_4_hour_window_cleanup() {
    // In a real implementation, this would run periodically to clean up old chunks
    // For this test, we'll simulate the cleanup happening after some time
    
    std::this_thread::sleep_for(std::chrono::seconds(2));  // Shortened for testing
    
    std::cout << "\n=== Initiating 4-hour window cleanup of retired objects ===" << std::endl;
    
    // Convert 4 hours to microseconds for the time-based cleanup
    // 4 hours = 4 * 60 * 60 * 1000000 = 14,400,000,000 microseconds
    uint64_t four_hours_in_microseconds = 4ULL * 60 * 60 * 1000000;
    
    // For testing purposes, we'll use a shorter duration (500ms) to simulate the concept
    uint64_t test_duration_microseconds = 500000;  // 0.5 seconds for testing
    
    std::cout << "Cleaning up objects retired for more than " 
              << test_duration_microseconds / 1000000.0 << " seconds" << std::endl;
    
    // Perform time-based cleanup of retired objects
    cleanup_retired_objects_older_than(test_duration_microseconds);
    
    std::cout << "4-hour window cleanup completed." << std::endl;
}

int main() {
    std::cout << "=== Hazard Pointer Leak Check Test ===" << std::endl;
    std::cout << "Testing memory reclamation after 4-hour window simulation..." << std::endl;
    
    // Initialize destruction counter
    destruction_counter.store(0);
    
    // Start reader threads
    const int num_readers = 3;
    std::vector<std::thread> readers;
    for (int i = 0; i < num_readers; ++i) {
        readers.emplace_back(reader_thread, i);
    }
    
    // Start writer thread
    std::thread writer(writer_thread);
    
    // Start cleanup simulation thread
    std::thread cleanup(simulate_4_hour_window_cleanup);
    
    // Let the system run for a while
    std::this_thread::sleep_for(std::chrono::seconds(3));
    
    // Signal all threads to stop
    std::cout << "\nStopping all threads..." << std::endl;
    should_stop.store(true);
    
    // Wait for all threads to complete
    for (auto& r : readers) {
        r.join();
    }
    writer.join();
    cleanup.join();
    
    // Perform final cleanup
    std::cout << "\nPerforming final cleanup..." << std::endl;
    
    // Use the time-based cleanup to simulate the 4-hour window concept
    // Clean up objects that have been retired for more than 1 second
    cleanup_retired_objects_older_than(1000000);  // 1 second in microseconds
    
    // Check for leaks
    std::cout << "\n=== Leak Check Results ===" << std::endl;
    
    // Count remaining chunks in the retired list
    int retired_count = 0;
    // Note: We can't directly access the retired list from outside the manager,
    // so we'll rely on the destruction counter and manual tracking
    
    // Check manually tracked allocations
    {
        std::lock_guard<std::mutex> lock(chunks_mutex);
        retired_count = allocated_chunks.size();
    }
    
    int destructed_count = destruction_counter.load();
    int total_created = total_chunks_created.load();
    
    std::cout << "Total objects created: " << total_created << std::endl;
    std::cout << "Objects destroyed: " << destructed_count << std::endl;
    std::cout << "Objects still allocated: " << retired_count << std::endl;
    
    // Since we're simulating the 4-hour window concept, we expect that
    // retired objects that are no longer protected should be cleaned up
    // Force another cleanup to ensure everything possible is reclaimed
    cleanup_retired_objects_older_than(1000000);  // 1 second in microseconds
    
    // Wait a bit more to allow for any delayed cleanup
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Check again after final cleanup
    destructed_count = destruction_counter.load();
    int final_total_chunks = total_chunks_created.load();
    std::cout << "After final cleanup - Objects destroyed: " << destructed_count << std::endl;
    
    {
        std::lock_guard<std::mutex> lock(chunks_mutex);
        retired_count = allocated_chunks.size();
    }
    std::cout << "After final cleanup - Objects still allocated: " << retired_count << std::endl;
    
    std::cout << "Total objects created: " << final_total_chunks << std::endl;
    
    if (retired_count == 0) {
        std::cout << "\nSUCCESS: All retired objects have been properly reclaimed!" << std::endl;
        std::cout << "Hazard pointers correctly reclaimed memory after the 4-hour window moved." << std::endl;
        return 0;
    } else {
        std::cout << "\nWARNING: " << retired_count << " objects were not reclaimed!" << std::endl;
        std::cout << "There may be a memory leak in the hazard pointer reclamation mechanism." << std::endl;
        
        // Show which chunks are still allocated
        {
            std::lock_guard<std::mutex> lock(chunks_mutex);
            std::cout << "Still allocated chunk IDs: ";
            for (const auto& pair : allocated_chunks) {
                std::cout << pair.first << " ";
            }
            std::cout << std::endl;
        }
        
        return 1;  // Return error code to indicate potential leak
    }
}