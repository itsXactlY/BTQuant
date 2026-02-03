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

// Template instantiations for the new LockFreeStack
template class LockFreeStack<int>;
template class LockFreeStack<double>;
template class LockFreeStack<float>;
template class LockFreeStack<long>;
template class LockFreeStack<bool>;
template class LockFreeStack<std::string>;
template class LockFreeStack<std::vector<double>>;
template class LockFreeStack<std::vector<float>>;
template class LockFreeStack<std::vector<int>>;
template class LockFreeStack<std::vector<long>>;
template class LockFreeStack<std::vector<bool>>;

// Trading-specific data structures for LockFreeStack
template class LockFreeStack<btq::Trade>;
template class LockFreeStack<btq::Candle>;
template class LockFreeStack<std::vector<btq::Trade>>;
template class LockFreeStack<std::vector<btq::Candle>>;

// Additional data structures for UI updates and indicator data for LockFreeStack
template class LockFreeStack<std::pair<double, double>>;  // For histogram data
template class LockFreeStack<std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>>;  // For Bollinger Bands

// New additions for enhanced functionality for LockFreeStack
template class LockFreeStack<std::optional<int>>;
template class LockFreeStack<std::optional<double>>;
template class LockFreeStack<std::optional<btq::Trade>>;
template class LockFreeStack<std::optional<btq::Candle>>;

// Additional template instantiations for new functionality for LockFreeStack
template class LockFreeStack<std::pair<btq::Trade, btq::Candle>>;
template class LockFreeStack<std::vector<std::pair<double, double>>>;

// Template instantiations for the new SPSCRingBuffer
template class SPSCRingBuffer<int>;
template class SPSCRingBuffer<double>;
template class SPSCRingBuffer<float>;
template class SPSCRingBuffer<long>;
template class SPSCRingBuffer<bool>;
template class SPSCRingBuffer<std::string>;
template class SPSCRingBuffer<std::vector<double>>;
template class SPSCRingBuffer<std::vector<float>>;
template class SPSCRingBuffer<std::vector<int>>;
template class SPSCRingBuffer<std::vector<long>>;
template class SPSCRingBuffer<std::vector<bool>>;

// Trading-specific data structures for SPSCRingBuffer
template class SPSCRingBuffer<btq::Trade>;
template class SPSCRingBuffer<btq::Candle>;
template class SPSCRingBuffer<std::vector<btq::Trade>>;
template class SPSCRingBuffer<std::vector<btq::Candle>>;

// Additional data structures for UI updates and indicator data for SPSCRingBuffer
template class SPSCRingBuffer<std::pair<double, double>>;  // For histogram data
template class SPSCRingBuffer<std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>>;  // For Bollinger Bands

// New additions for enhanced functionality for SPSCRingBuffer
template class SPSCRingBuffer<std::optional<int>>;
template class SPSCRingBuffer<std::optional<double>>;
template class SPSCRingBuffer<std::optional<btq::Trade>>;
template class SPSCRingBuffer<std::optional<btq::Candle>>;

// Additional template instantiations for new functionality for SPSCRingBuffer
template class SPSCRingBuffer<std::pair<btq::Trade, btq::Candle>>;
template class SPSCRingBuffer<std::vector<std::pair<double, double>>>;

} // namespace threading
} // namespace btq