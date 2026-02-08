#ifndef BTQ_PARALLEL_EXECUTION_HPP
#define BTQ_PARALLEL_EXECUTION_HPP

#include <vector>
#include <functional>
#include <thread>
#include <future>
#include <algorithm>
#include <numeric>
#include <execution>
#include <type_traits>
#include <iterator>
#include <cmath>

#include "../include/task_scheduler.hpp"

namespace btq {
namespace execution {

// Parallel execution policies
struct parallel_policy {};
struct parallel_unsequenced_policy {};
struct sequenced_policy {};

inline constexpr parallel_policy par{};
inline constexpr parallel_unsequenced_policy par_unseq{};
inline constexpr sequenced_policy seq{};

// Parallel for implementation
template<typename Iterator, typename Function>
void parallel_for(Iterator first, Iterator last, Function func, size_t num_threads = 0) {
    if (first == last) return;
    
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 1;
    }
    
    size_t total_size = std::distance(first, last);
    if (total_size < num_threads) {
        // If there are fewer elements than threads, just use sequential execution
        std::for_each(first, last, func);
        return;
    }
    
    size_t chunk_size = total_size / num_threads;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    Iterator current = first;
    for (size_t i = 0; i < num_threads; ++i) {
        Iterator chunk_end = current;
        std::advance(chunk_end, (i == num_threads - 1) ? total_size : chunk_size); // Last chunk gets remainder
        
        threads.emplace_back([current, chunk_end, func]() {
            std::for_each(current, chunk_end, func);
        });
        
        current = chunk_end;
        total_size -= std::distance(current, chunk_end);
    }
    
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

// Parallel transform implementation
template<typename InputIterator, typename OutputIterator, typename UnaryOperation>
void parallel_transform(InputIterator first, InputIterator last, OutputIterator result, 
                      UnaryOperation op, size_t num_threads = 0) {
    if (first == last) return;
    
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 1;
    }
    
    size_t total_size = std::distance(first, last);
    if (total_size < num_threads) {
        // If there are fewer elements than threads, just use sequential execution
        std::transform(first, last, result, op);
        return;
    }
    
    size_t chunk_size = total_size / num_threads;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    InputIterator input_current = first;
    OutputIterator output_current = result;
    for (size_t i = 0; i < num_threads; ++i) {
        InputIterator input_chunk_end = input_current;
        OutputIterator output_chunk_end = output_current;
        std::advance(input_chunk_end, (i == num_threads - 1) ? total_size : chunk_size); // Last chunk gets remainder
        std::advance(output_chunk_end, (i == num_threads - 1) ? total_size : chunk_size);
        
        threads.emplace_back([input_current, input_chunk_end, output_current, op]() {
            std::transform(input_current, input_chunk_end, output_current, op);
        });
        
        input_current = input_chunk_end;
        output_current = output_chunk_end;
        total_size -= std::distance(input_current, input_chunk_end);
    }
    
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

// Parallel reduce implementation
template<typename Iterator, typename T>
T parallel_reduce(Iterator first, Iterator last, T init, 
                 std::function<T(T, T)> reducer = std::plus<T>{}, 
                 size_t num_threads = 0) {
    if (first == last) return init;
    
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 1;
    }
    
    size_t total_size = std::distance(first, last);
    if (total_size < num_threads) {
        // If there are fewer elements than threads, just use sequential execution
        return std::reduce(first, last, init, reducer);
    }
    
    size_t chunk_size = total_size / num_threads;
    std::vector<std::thread> threads;
    std::vector<T> partial_results(num_threads);
    threads.reserve(num_threads);
    
    Iterator current = first;
    for (size_t i = 0; i < num_threads; ++i) {
        Iterator chunk_end = current;
        std::advance(chunk_end, (i == num_threads - 1) ? total_size : chunk_size); // Last chunk gets remainder
        
        threads.emplace_back([current, chunk_end, init, &reducer, &partial_results, i]() {
            partial_results[i] = std::reduce(current, chunk_end, init, reducer);
        });
        
        current = chunk_end;
        total_size -= std::distance(current, chunk_end);
    }
    
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    // Reduce the partial results
    return std::reduce(partial_results.begin(), partial_results.end(), init, reducer);
}

// Parallel execution context for managing thread pools
class ExecutionContext {
private:
    std::unique_ptr<TaskScheduler> scheduler_;
    size_t num_threads_;

public:
    explicit ExecutionContext(size_t num_threads = 0) 
        : num_threads_(num_threads == 0 ? std::thread::hardware_concurrency() : num_threads) {
        scheduler_ = std::make_unique<TaskScheduler>(num_threads_);
    }
    
    ~ExecutionContext() = default;
    
    // Delete copy constructor and assignment operator
    ExecutionContext(const ExecutionContext&) = delete;
    ExecutionContext& operator=(const ExecutionContext&) = delete;
    
    // Move constructor and assignment operator
    ExecutionContext(ExecutionContext&&) = default;
    ExecutionContext& operator=(ExecutionContext&&) = default;
    
    // Submit a task for asynchronous execution
    template<typename Func, typename... Args>
    auto submit(Func&& f, Args&&... args) 
        -> std::future<typename std::result_of<Func(Args...)>::type> {
        return scheduler_->enqueue_task(std::bind(std::forward<Func>(f), std::forward<Args>(args)...));
    }
    
    // Parallel for using the execution context
    template<typename Iterator, typename Function>
    void parallel_for(Iterator first, Iterator last, Function func) {
        btq::execution::parallel_for(first, last, func, num_threads_);
    }
    
    // Parallel transform using the execution context
    template<typename InputIterator, typename OutputIterator, typename UnaryOperation>
    void parallel_transform(InputIterator first, InputIterator last, OutputIterator result, 
                          UnaryOperation op) {
        btq::execution::parallel_transform(first, last, result, op, num_threads_);
    }
    
    // Parallel reduce using the execution context
    template<typename Iterator, typename T>
    T parallel_reduce(Iterator first, Iterator last, T init, 
                     std::function<T(T, T)> reducer = std::plus<T>{}) {
        return btq::execution::parallel_reduce(first, last, init, reducer, num_threads_);
    }
    
    size_t get_num_threads() const { return num_threads_; }
};

// Utility function to create an execution context
inline ExecutionContext make_execution_context(size_t num_threads = 0) {
    return ExecutionContext(num_threads);
}

// Parallel algorithms with execution policy support
template<class ExecutionPolicy, class Iterator, class Function>
void for_each(ExecutionPolicy&& policy, Iterator first, Iterator last, Function f) {
    if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, parallel_policy>) {
        parallel_for(first, last, f);
    } else if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, parallel_unsequenced_policy>) {
        // For now, treat parallel_unsequenced as parallel
        parallel_for(first, last, f);
    } else {
        // Sequential execution
        std::for_each(first, last, f);
    }
}

template<class ExecutionPolicy, class InputIterator, class OutputIterator, class UnaryOperation>
OutputIterator transform(ExecutionPolicy&& policy, InputIterator first, InputIterator last, 
                       OutputIterator result, UnaryOperation op) {
    if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, parallel_policy>) {
        parallel_transform(first, last, result, op);
    } else if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, parallel_unsequenced_policy>) {
        // For now, treat parallel_unsequenced as parallel
        parallel_transform(first, last, result, op);
    } else {
        // Sequential execution
        return std::transform(first, last, result, op);
    }
    return result + std::distance(first, last);
}

template<class ExecutionPolicy, class Iterator, class T, class BinaryOperation>
T reduce(ExecutionPolicy&& policy, Iterator first, Iterator last, T init, 
         BinaryOperation binary_op) {
    if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, parallel_policy>) {
        return parallel_reduce(first, last, init, binary_op);
    } else if constexpr (std::is_same_v<std::decay_t<ExecutionPolicy>, parallel_unsequenced_policy>) {
        // For now, treat parallel_unsequenced as parallel
        return parallel_reduce(first, last, init, binary_op);
    } else {
        // Sequential execution
        return std::reduce(first, last, init, binary_op);
    }
}

// Specialization for default binary operation (plus)
template<class ExecutionPolicy, class Iterator, class T>
T reduce(ExecutionPolicy&& policy, Iterator first, Iterator last, T init) {
    return reduce(policy, first, last, init, std::plus<T>{});
}

} // namespace execution
} // namespace btq

#endif // BTQ_PARALLEL_EXECUTION_HPP