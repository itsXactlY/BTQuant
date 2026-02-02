#include "threading/lockfree_queue.hpp"
#include "task_scheduler.hpp"  // For btq::Trade and btq::Candle
#include <string>
#include <vector>
#include <chrono>

// Explicit template instantiations for commonly used types in the BTQ Render Engine
// This helps reduce compilation times and ensures proper linking of template implementations

namespace btq {
namespace threading {

// Common data types used for passing data between calculation and UI threads
template class LockFreeQueue<int>;
template class LockFreeQueue<double>;
template class LockFreeQueue<std::string>;
template class LockFreeQueue<std::vector<double>>;
template class LockFreeQueue<std::vector<int>>;

// Trading-specific data structures
template class LockFreeQueue<btq::Trade>;
template class LockFreeQueue<btq::Candle>;
template class LockFreeQueue<std::vector<btq::Trade>>;
template class LockFreeQueue<std::vector<btq::Candle>>;

} // namespace threading
} // namespace btq