#include "threading/lockfree_queue.hpp"
#include "task_scheduler.hpp"  // For btq::Trade and btq::Candle
#include <string>
#include <vector>
#include <chrono>
#include <tuple>
#include <optional>
#include <utility>

// Explicit template instantiations for commonly used types in the BTQ Render Engine
// This helps reduce compilation times and ensures proper linking of template implementations

namespace btq {
namespace threading {

// Common data types used for passing data between calculation and UI threads
template class LockFreeQueue<int>;
template class LockFreeQueue<double>;
template class LockFreeQueue<float>;
template class LockFreeQueue<long>;
template class LockFreeQueue<bool>;
template class LockFreeQueue<std::string>;
template class LockFreeQueue<std::vector<double>>;
template class LockFreeQueue<std::vector<float>>;
template class LockFreeQueue<std::vector<int>>;
template class LockFreeQueue<std::vector<long>>;
template class LockFreeQueue<std::vector<bool>>;

// Trading-specific data structures
template class LockFreeQueue<btq::Trade>;
template class LockFreeQueue<btq::Candle>;
template class LockFreeQueue<std::vector<btq::Trade>>;
template class LockFreeQueue<std::vector<btq::Candle>>;

// Additional data structures for UI updates and indicator data
template class LockFreeQueue<std::pair<double, double>>;  // For histogram data
template class LockFreeQueue<std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>>;  // For Bollinger Bands

// New additions for enhanced functionality
template class LockFreeQueue<std::optional<int>>;
template class LockFreeQueue<std::optional<double>>;
template class LockFreeQueue<std::optional<btq::Trade>>;
template class LockFreeQueue<std::optional<btq::Candle>>;

// Additional template instantiations for new functionality
template class LockFreeQueue<std::pair<btq::Trade, btq::Candle>>;
template class LockFreeQueue<std::vector<std::pair<double, double>>>;

} // namespace threading
} // namespace btq