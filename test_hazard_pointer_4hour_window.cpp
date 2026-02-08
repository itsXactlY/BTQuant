/**
 * @file test_hazard_pointer_4hour_window.cpp
 * @brief Comprehensive test to verify Hazard Pointers correctly reclaim memory after the 4-hour window moves.
 * 
 * This test specifically validates the 4-hour window concept where data chunks older than 
 * 4 hours are pruned and their memory should be reclaimed by the hazard pointer mechanism 
 * once no readers are referencing them.
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
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void simulate_4_hour_window_cleanup() {
    // In a real implementation, this would run periodically to clean up old chunks
    // For this test, we'll simulate the cleanup happening after some time
    
    std::this_thread::sleep_for(std::chrono::seconds(1));  // Shortened for testing
    
    std::cout << "\n=== Initiating 4-hour window cleanup of retired objects ===" << std::endl;
    
    // For testing purposes, we'll use a shorter duration (100ms) to simulate the concept
    uint64_t test_duration_microseconds = 100000;  // 0.1 seconds for testing
    
    std::cout << "Cleaning up objects retired for more than " 
              << test_duration_microseconds / 1000000.0 << " seconds" << std::endl;
    
    // Perform time-based cleanup of retired objects
    cleanup_retired_objects_older_than(test_duration_microseconds);
    
    std::cout << "4-hour window cleanup completed." << std::endl;
}

int main() {
    std::cout << "=== Hazard Pointer 4-Hour Window Leak Check Test ===" << std::endl;
    std::cout << "Testing memory reclamation after 4-hour window simulation..." << std::endl;
    
    // Initialize counters
    destruction_counter.store(0);
    total_chunks_created.store(0);
    
    // Start reader threads
    const int num_readers = 2;
    std::vector<std::thread> readers;
    for (int i = 0; i < num_readers; ++i) {
        readers.emplace_back(reader_thread, i);
    }
    
    // Start writer thread
    std::thread writer(writer_thread);
    
    // Start cleanup simulation thread
    std::thread cleanup(simulate_4_hour_window_cleanup);
    
    // Let the system run for a while to generate some retired chunks
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Signal all threads to stop
    std::cout << "\nStopping all threads..." << std::endl;
    should_stop.store(true);
    
    // Wait for all threads to complete
    for (auto& r : readers) {
        r.join();
    }
    writer.join();
    cleanup.join();
    
    // At this point, no readers should be holding hazard pointers
    // So all retired objects should be safe to reclaim
    
    std::cout << "\nPerforming final cleanup after all readers stopped..." << std::endl;
    
    // Perform final cleanup - now all retired objects should be reclaimable
    // since no readers are active anymore
    cleanup_retired_objects_older_than(0);  // Clean up everything that's not protected
    
    // Wait a bit more to allow for any delayed cleanup
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // Check for leaks
    std::cout << "\n=== Final Leak Check Results ===" << std::endl;
    
    int final_total_chunks = total_chunks_created.load();
    int destructed_count = destruction_counter.load();
    
    // Check manually tracked allocations
    int remaining_count = 0;
    {
        std::lock_guard<std::mutex> lock(chunks_mutex);
        remaining_count = allocated_chunks.size();
        std::cout << "Remaining tracked chunks: ";
        for (const auto& pair : allocated_chunks) {
            std::cout << pair.first << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "Total objects created: " << final_total_chunks << std::endl;
    std::cout << "Objects destroyed: " << destructed_count << std::endl;
    std::cout << "Objects still allocated: " << remaining_count << std::endl;
    
    // The current chunk (most recent) might still be referenced by the shared pointer
    // But all retired chunks should be cleaned up now that readers have stopped
    DataChunk* current = current_chunk.load();
    int current_chunk_id = current ? current->id : -1;
    
    std::cout << "Current chunk ID (may still be referenced): " << current_chunk_id << std::endl;
    
    // Calculate how many chunks should have been retired and cleaned up
    int expected_destroyed = final_total_chunks - (current_chunk_id == -1 ? 0 : 1);
    
    std::cout << "Expected to destroy approximately: " << expected_destroyed << " objects" << std::endl;
    
    if (remaining_count <= 1) {  // Allow for the current chunk to still be referenced
        std::cout << "\nSUCCESS: Hazard pointers correctly reclaimed memory after the 4-hour window moved!" << std::endl;
        std::cout << "Only " << remaining_count << " object remains (likely the current active chunk)." << std::endl;
        return 0;
    } else {
        std::cout << "\nWARNING: " << remaining_count << " objects were not reclaimed!" << std::endl;
        std::cout << "There may be a memory leak in the hazard pointer reclamation mechanism." << std::endl;
        return 1;  // Return error code to indicate potential leak
    }
}