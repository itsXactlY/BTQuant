#include "performance/global_sync.hpp"

// ============================================================================
// GLOBAL ATOMIC VARIABLES DEFINITIONS
// ============================================================================

// Crosshair price synchronized across all Charts, DOMs, and TPOs
// C++26: lock-free atomic for cross-thread communication
std::atomic<double> g_crosshair_price{0.0};

// Timestamp when crosshair price was last updated
std::atomic<uint64_t> g_crosshair_timestamp{0};
