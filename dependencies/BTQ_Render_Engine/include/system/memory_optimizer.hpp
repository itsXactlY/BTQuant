#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace BTQuant {

// ============================================================================
// Memory Optimization System
// ============================================================================

class MemoryOptimizer {
 public:
  MemoryOptimizer();
  ~MemoryOptimizer();

  // Initialize the memory optimization system
  void initialize();

  // Optimize memory allocations
  void optimize_allocations();

  // Compact memory pools
  void compact_memory_pools();

  // Release unused memory
  void release_unused_memory();

  // Track memory usage
  void track_allocation(size_t size, const char* tag = nullptr);
  void track_deallocation(size_t size, const char* tag = nullptr);

  // Get memory statistics
  struct MemoryStats {
    size_t total_allocated = 0;
    size_t total_deallocated = 0;
    size_t current_allocated = 0;
    size_t peak_usage = 0;
    size_t fragmentation = 0;          // Estimated fragmentation in bytes
    float fragmentation_ratio = 0.0f;  // Fragmentation as percentage of total allocated
    size_t pool_count = 0;
    size_t reserved_memory = 0;
    size_t committed_memory = 0;
  };

  MemoryStats get_memory_stats() const;

  // Set optimization parameters
  void set_compaction_threshold(size_t threshold_bytes);
  void set_release_threshold(size_t threshold_bytes);

  // Force garbage collection
  void force_garbage_collection();

  // Enable/disable memory tracking
  void enable_tracking(bool enabled);

  // Clear all memory allocation records and reset statistics
  void clear();

  // Get allocation history
  struct AllocationRecord {
    size_t size;
    const char* tag;
    void* ptr;
  };

  std::vector<AllocationRecord> get_recent_allocations(size_t count = 100) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace BTQuant