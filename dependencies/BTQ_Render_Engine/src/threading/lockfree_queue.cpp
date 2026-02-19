#include "../../include/threading/lockfree_queue.hpp"
#include "../../include/task_scheduler.hpp"  // For btq::Trade and btq::Candle
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

// Template instantiations for the new MPSCQueue
template class MPSCQueue<int>;
template class MPSCQueue<double>;
template class MPSCQueue<float>;
template class MPSCQueue<long>;
template class MPSCQueue<bool>;
template class MPSCQueue<std::string>;
template class MPSCQueue<std::vector<double>>;
template class MPSCQueue<std::vector<float>>;
template class MPSCQueue<std::vector<int>>;
template class MPSCQueue<std::vector<long>>;
template class MPSCQueue<std::vector<bool>>;

// Trading-specific data structures for MPSCQueue
template class MPSCQueue<btq::Trade>;
template class MPSCQueue<btq::Candle>;
template class MPSCQueue<std::vector<btq::Trade>>;
template class MPSCQueue<std::vector<btq::Candle>>;

// Additional data structures for UI updates and indicator data for MPSCQueue
template class MPSCQueue<std::pair<double, double>>;  // For histogram data
template class MPSCQueue<std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>>;  // For Bollinger Bands

// New additions for enhanced functionality for MPSCQueue
template class MPSCQueue<std::optional<int>>;
template class MPSCQueue<std::optional<double>>;
template class MPSCQueue<std::optional<btq::Trade>>;
template class MPSCQueue<std::optional<btq::Candle>>;

// Additional template instantiations for new functionality for MPSCQueue
template class MPSCQueue<std::pair<btq::Trade, btq::Candle>>;
template class MPSCQueue<std::vector<std::pair<double, double>>>;

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

// Template instantiations for the new AtomicWrapper
template class AtomicWrapper<int>;
template class AtomicWrapper<double>;
template class AtomicWrapper<float>;
template class AtomicWrapper<long>;
template class AtomicWrapper<bool>;

// Template instantiations for the new UIUpdateQueue
template class UIUpdateQueue<int>;
template class UIUpdateQueue<double>;
template class UIUpdateQueue<float>;
template class UIUpdateQueue<long>;
template class UIUpdateQueue<bool>;
template class UIUpdateQueue<std::string>;
template class UIUpdateQueue<std::vector<double>>;
template class UIUpdateQueue<std::vector<float>>;
template class UIUpdateQueue<std::vector<int>>;
template class UIUpdateQueue<std::vector<long>>;
template class UIUpdateQueue<std::vector<bool>>;

// Trading-specific data structures for UIUpdateQueue
template class UIUpdateQueue<btq::Trade>;
template class UIUpdateQueue<btq::Candle>;
template class UIUpdateQueue<std::vector<btq::Trade>>;
template class UIUpdateQueue<std::vector<btq::Candle>>;

// Additional data structures for UI updates and indicator data for UIUpdateQueue
template class UIUpdateQueue<std::pair<double, double>>;  // For histogram data
template class UIUpdateQueue<std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>>;  // For Bollinger Bands

// New additions for enhanced functionality for UIUpdateQueue
template class UIUpdateQueue<std::optional<int>>;
template class UIUpdateQueue<std::optional<double>>;
template class UIUpdateQueue<std::optional<btq::Trade>>;
template class UIUpdateQueue<std::optional<btq::Candle>>;

// Additional template instantiations for new functionality for UIUpdateQueue
template class UIUpdateQueue<std::pair<btq::Trade, btq::Candle>>;
template class UIUpdateQueue<std::vector<std::pair<double, double>>>;

// Template instantiations for the new HighFrequencyUpdateQueue
template class HighFrequencyUpdateQueue<int>;
template class HighFrequencyUpdateQueue<double>;
template class HighFrequencyUpdateQueue<float>;
template class HighFrequencyUpdateQueue<long>;
template class HighFrequencyUpdateQueue<bool>;
template class HighFrequencyUpdateQueue<std::string>;
template class HighFrequencyUpdateQueue<std::vector<double>>;
template class HighFrequencyUpdateQueue<std::vector<float>>;
template class HighFrequencyUpdateQueue<std::vector<int>>;
template class HighFrequencyUpdateQueue<std::vector<long>>;
template class HighFrequencyUpdateQueue<std::vector<bool>>;

// Trading-specific data structures for HighFrequencyUpdateQueue
template class HighFrequencyUpdateQueue<btq::Trade>;
template class HighFrequencyUpdateQueue<btq::Candle>;
template class HighFrequencyUpdateQueue<std::vector<btq::Trade>>;
template class HighFrequencyUpdateQueue<std::vector<btq::Candle>>;

// Additional data structures for UI updates and indicator data for HighFrequencyUpdateQueue
template class HighFrequencyUpdateQueue<std::pair<double, double>>;  // For histogram data
template class HighFrequencyUpdateQueue<std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>>;  // For Bollinger Bands

// New additions for enhanced functionality for HighFrequencyUpdateQueue
template class HighFrequencyUpdateQueue<std::optional<int>>;
template class HighFrequencyUpdateQueue<std::optional<double>>;
template class HighFrequencyUpdateQueue<std::optional<btq::Trade>>;
template class HighFrequencyUpdateQueue<std::optional<btq::Candle>>;

// Additional template instantiations for new functionality for HighFrequencyUpdateQueue
template class HighFrequencyUpdateQueue<std::pair<btq::Trade, btq::Candle>>;
template class HighFrequencyUpdateQueue<std::vector<std::pair<double, double>>>;

// Template instantiations for the new CalculationToUIQueue
template class CalculationToUIQueue<int>;
template class CalculationToUIQueue<double>;
template class CalculationToUIQueue<float>;
template class CalculationToUIQueue<long>;
template class CalculationToUIQueue<bool>;
template class CalculationToUIQueue<std::string>;
template class CalculationToUIQueue<std::vector<double>>;
template class CalculationToUIQueue<std::vector<float>>;
template class CalculationToUIQueue<std::vector<int>>;
template class CalculationToUIQueue<std::vector<long>>;
template class CalculationToUIQueue<std::vector<bool>>;

// Trading-specific data structures for CalculationToUIQueue
template class CalculationToUIQueue<btq::Trade>;
template class CalculationToUIQueue<btq::Candle>;
template class CalculationToUIQueue<std::vector<btq::Trade>>;
template class CalculationToUIQueue<std::vector<btq::Candle>>;

// Additional data structures for UI updates and indicator data for CalculationToUIQueue
template class CalculationToUIQueue<std::pair<double, double>>;  // For histogram data
template class CalculationToUIQueue<std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>>;  // For Bollinger Bands

// New additions for enhanced functionality for CalculationToUIQueue
template class CalculationToUIQueue<std::optional<int>>;
template class CalculationToUIQueue<std::optional<double>>;
template class CalculationToUIQueue<std::optional<btq::Trade>>;
template class CalculationToUIQueue<std::optional<btq::Candle>>;

// Additional template instantiations for new functionality for CalculationToUIQueue
template class CalculationToUIQueue<std::pair<btq::Trade, btq::Candle>>;
template class CalculationToUIQueue<std::vector<std::pair<double, double>>>;

// Template instantiations for the new PriorityUIUpdateQueue
template class PriorityUIUpdateQueue<int>;
template class PriorityUIUpdateQueue<double>;
template class PriorityUIUpdateQueue<float>;
template class PriorityUIUpdateQueue<long>;
template class PriorityUIUpdateQueue<bool>;
template class PriorityUIUpdateQueue<std::string>;
template class PriorityUIUpdateQueue<std::vector<double>>;
template class PriorityUIUpdateQueue<std::vector<float>>;
template class PriorityUIUpdateQueue<std::vector<int>>;
template class PriorityUIUpdateQueue<std::vector<long>>;
template class PriorityUIUpdateQueue<std::vector<bool>>;

// Trading-specific data structures for PriorityUIUpdateQueue
template class PriorityUIUpdateQueue<btq::Trade>;
template class PriorityUIUpdateQueue<btq::Candle>;
template class PriorityUIUpdateQueue<std::vector<btq::Trade>>;
template class PriorityUIUpdateQueue<std::vector<btq::Candle>>;

// Additional data structures for UI updates and indicator data for PriorityUIUpdateQueue
template class PriorityUIUpdateQueue<std::pair<double, double>>;  // For histogram data
template class PriorityUIUpdateQueue<std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>>;  // For Bollinger Bands

// New additions for enhanced functionality for PriorityUIUpdateQueue
template class PriorityUIUpdateQueue<std::optional<int>>;
template class PriorityUIUpdateQueue<std::optional<double>>;
template class PriorityUIUpdateQueue<std::optional<btq::Trade>>;
template class PriorityUIUpdateQueue<std::optional<btq::Candle>>;

// Additional template instantiations for new functionality for PriorityUIUpdateQueue
template class PriorityUIUpdateQueue<std::pair<btq::Trade, btq::Candle>>;
template class PriorityUIUpdateQueue<std::vector<std::pair<double, double>>>;

} // namespace threading
} // namespace btq