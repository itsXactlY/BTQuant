#include "threading/lockfree_queue.hpp"

// Lock-free queue implementation for thread-safe data passing between calculation and UI threads
// Implements the Michael & Scott algorithm for lock-free queues using atomic operations
// All implementation is in the header file due to template nature.
//
// This lock-free queue provides:
// - Wait-free enqueue operations (single consumer)
// - Lock-free dequeue operations (single producer)
// - Memory-order optimized for performance
// - Cache-line alignment to prevent false sharing
// - Safe memory management with proper cleanup
//
// Designed specifically for passing data between calculation threads and UI thread
// in the BTQ Render Engine, ensuring thread-safe and efficient data transfer.