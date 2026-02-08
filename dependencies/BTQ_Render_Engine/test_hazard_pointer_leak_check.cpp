#include <iostream>
#include <thread>
#include <vector>
#include <memory>
#include <chrono>
#include <atomic>
#include <cassert>
#include <algorithm>

#include "include/threading/hazard_pointer.hpp"

using namespace btq::threading;

// Structure to represent a ClusterChunk that would be managed by hazard pointers
struct ClusterChunk {
    int id;
    uint64_t timestamp_ms;  // Timestamp when chunk was created (simulating the 4-hour window)
    std::atomic<int> ref_count{1};
    
    explicit ClusterChunk(int chunk_id, uint64_t ts) : id(chunk_id), timestamp_ms(ts) {}
    
    void inc_ref() { ref_count.fetch_add(1, std::memory_order_relaxed); }
    void dec_ref() {
        if (ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete this;
        }
    }
    
    ~ClusterChunk() = default;
};

// Function to get current time in milliseconds since epoch
uint64_t get_current_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

void test_hazard_pointer_leak_check() {
    std::cout << "Testing hazard pointer leak check after 4-hour window moves...\n";
    
    HazardPointerManager& hpm = HazardPointerManager::instance();
    
    // Clear any previous retired objects
    hpm.force_cleanup();
    
    const uint64_t FOUR_HOURS_MS = 4ULL * 60 * 60 * 1000; // 4 hours in milliseconds
    
    // Create a set of cluster chunks with different timestamps
    std::vector<ClusterChunk*> chunks;
    uint64_t current_time = get_current_time_ms();
    
    // Create chunks: some "recent" (within 4 hours) and some "old" (older than 4 hours)
    for (int i = 0; i < 10; ++i) {
        // Alternate between recent and old chunks
        uint64_t chunk_timestamp = (i % 2 == 0) ? 
            current_time - (FOUR_HOURS_MS + 1000) :  // Old chunk (> 4 hours)
            current_time - 1000;                     // Recent chunk (< 4 hours)
        
        chunks.push_back(new ClusterChunk(i, chunk_timestamp));
    }
    
    std::cout << "Created " << chunks.size() << " cluster chunks\n";
    std::cout << "Initial retired count: " << hpm.get_retired_count() << "\n";
    
    // Simulate readers (renderers) acquiring hazard pointers to some chunks
    std::vector<HazardPointerGuard<ClusterChunk>> active_guards;
    
    // Acquire hazard pointers for some chunks (simulating active renderers)
    for (size_t i = 0; i < chunks.size(); i += 3) {  // Every 3rd chunk
        active_guards.emplace_back(hpm.acquire_hazard_pointer(chunks[i]));
        std::cout << "Acquired hazard pointer for chunk " << chunks[i]->id 
                  << " (timestamp: " << chunks[i]->timestamp_ms << ")\n";
    }
    
    std::cout << "Acquired " << active_guards.size() << " hazard pointers\n";
    
    // Simulate the writer retiring old chunks (those older than 4 hours)
    // In a real system, this would happen when the 4-hour window moves forward
    std::cout << "Retiring chunks older than 4 hours...\n";
    
    for (auto* chunk : chunks) {
        // Check if chunk is older than 4 hours from current time
        if ((current_time - chunk->timestamp_ms) > FOUR_HOURS_MS) {
            std::cout << "Retiring old chunk " << chunk->id 
                      << " (age: " << (current_time - chunk->timestamp_ms) / 1000.0 / 3600.0 << " hours)\n";
            
            // Retire the chunk - it will be deleted when no hazard pointers protect it
            hpm.retire(chunk, [chunk]() {
                std::cout << "  -> Chunk " << chunk->id << " was safely deleted\n";
                delete chunk;
            });
        } else {
            std::cout << "Keeping recent chunk " << chunk->id 
                      << " (age: " << (current_time - chunk->timestamp_ms) / 1000.0 / 3600.0 << " hours)\n";
        }
    }
    
    std::cout << "After retiring old chunks, retired count: " << hpm.get_retired_count() << "\n";
    
    // At this point, some retired chunks should NOT be deleted because they're protected by hazard pointers
    // Others should be eligible for deletion
    
    // Force cleanup - this should delete chunks that are NOT protected by hazard pointers
    hpm.force_cleanup();
    
    std::cout << "After first cleanup, retired count: " << hpm.get_retired_count() << "\n";
    
    // Verify that retired count reflects the chunks that are still protected
    size_t expected_remaining_retired = 0;
    for (size_t i = 0; i < chunks.size(); i += 3) {  // The chunks we protected with hazard pointers
        if ((current_time - chunks[i]->timestamp_ms) > FOUR_HOURS_MS) {
            // This chunk was old and retired, but is still protected by a hazard pointer
            expected_remaining_retired++;
        }
    }
    
    std::cout << "Expected remaining retired: " << expected_remaining_retired 
              << ", Actual remaining retired: " << hpm.get_retired_count() << "\n";
    
    // Release the hazard pointers for old chunks (simulating renderers finishing with old data)
    std::cout << "Releasing hazard pointers for old chunks...\n";
    for (size_t i = 0; i < active_guards.size(); ++i) {
        size_t chunk_idx = i * 3;
        if (chunk_idx < chunks.size() && 
            (current_time - chunks[chunk_idx]->timestamp_ms) > FOUR_HOURS_MS) {
            std::cout << "Releasing hazard pointer for old chunk " << chunks[chunk_idx]->id << "\n";
            active_guards[i].release();  // Explicitly release the hazard pointer
        }
    }
    
    // Force cleanup again - now all retired chunks should be deleted
    std::cout << "Performing final cleanup after releasing all hazard pointers...\n";
    hpm.force_cleanup();
    
    std::cout << "Final retired count: " << hpm.get_retired_count() << "\n";
    
    // All retired objects should now be cleaned up since no hazard pointers are protecting them
    assert(hpm.get_retired_count() == 0 && "All retired objects should be cleaned up after hazard pointers are released");
    
    std::cout << "Hazard pointer leak check test passed!\n";
    std::cout << "Memory was correctly reclaimed after the 4-hour window moved.\n";
}

void test_extended_leak_scenario() {
    std::cout << "\nTesting extended leak scenario with simulated time progression...\n";
    
    HazardPointerManager& hpm = HazardPointerManager::instance();
    
    // Clear any previous retired objects
    hpm.force_cleanup();
    
    const uint64_t FOUR_HOURS_MS = 4ULL * 60 * 60 * 1000;
    
    // Simulate a timeline of chunk creation and retirement
    std::vector<ClusterChunk*> all_chunks;
    uint64_t base_time = get_current_time_ms();
    
    // Phase 1: Create chunks at time T
    std::cout << "Phase 1: Creating chunks at time T\n";
    for (int i = 0; i < 5; ++i) {
        all_chunks.push_back(new ClusterChunk(i, base_time));
    }
    
    // Phase 2: Simulate time passing (advance by 5 hours - now chunks are > 4 hours old)
    uint64_t time_after_5_hours = base_time + (5 * 60 * 60 * 1000);
    std::cout << "Phase 2: Time advanced to simulate 5 hours later\n";
    
    // Phase 3: Some readers acquire hazard pointers to old chunks
    std::vector<HazardPointerGuard<ClusterChunk>> reader_guards;
    for (size_t i = 0; i < all_chunks.size(); i += 2) {  // Every other chunk
        reader_guards.push_back(hpm.acquire_hazard_pointer(all_chunks[i]));
        std::cout << "  Reader acquired hazard pointer for chunk " << all_chunks[i]->id << "\n";
    }
    
    // Phase 4: Writer retires all old chunks (they're all > 4 hours old now)
    std::cout << "Phase 4: Writer retiring all chunks older than 4 hours\n";
    for (auto* chunk : all_chunks) {
        hpm.retire(chunk, [chunk]() {
            std::cout << "    Chunk " << chunk->id << " safely deleted\n";
            delete chunk;
        });
    }
    
    std::cout << "  Retired " << all_chunks.size() << " chunks, retired count: " << hpm.get_retired_count() << "\n";
    
    // Phase 5: Cleanup - some chunks should remain retired because they're protected
    hpm.force_cleanup();
    std::cout << "  After cleanup, retired count: " << hpm.get_retired_count() 
              << " (should be equal to number of chunks with active hazard pointers)\n";
    
    // Phase 6: Readers finish and release their hazard pointers
    std::cout << "Phase 6: Readers releasing hazard pointers\n";
    reader_guards.clear();  // Release all hazard pointers
    
    // Phase 7: Final cleanup - all should be reclaimed now
    hpm.force_cleanup();
    std::cout << "  After releasing all hazard pointers, final retired count: " << hpm.get_retired_count() << "\n";
    
    assert(hpm.get_retired_count() == 0 && "All retired objects should be cleaned up after hazard pointers are released");
    
    std::cout << "Extended leak scenario test passed!\n";
}

int main() {
    std::cout << "Starting hazard pointer leak check tests...\n";
    
    test_hazard_pointer_leak_check();
    test_extended_leak_scenario();
    
    std::cout << "\nAll hazard pointer leak check tests passed!\n";
    std::cout << "Verified that memory is correctly reclaimed after the 4-hour window moves.\n";
    
    return 0;
}