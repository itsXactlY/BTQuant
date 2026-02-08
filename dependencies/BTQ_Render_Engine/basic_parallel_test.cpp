#include <iostream>
#include <vector>
#include <thread>
#include <future>
#include <algorithm>
#include <numeric>
#include <execution>
#include <chrono>
#include <cmath>

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

} // namespace execution
} // namespace btq

int main() {
    std::cout << "Testing BTQ Parallel Execution Capabilities\n";
    std::cout << "=========================================\n";
    
    // Test basic parallel execution
    std::cout << "Testing basic parallel execution...\n";
    
    // Create a large vector of numbers
    const size_t size = 100000;
    std::vector<double> data(size);
    std::iota(data.begin(), data.end(), 1.0);
    
    // Test parallel_for
    auto start = std::chrono::high_resolution_clock::now();
    btq::execution::parallel_for(data.begin(), data.end(), [](double& x) {
        x = x * x; // Square each element
    });
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Parallel for took: " << duration.count() << " ms\n";
    
    // Test parallel_reduce
    start = std::chrono::high_resolution_clock::now();
    double sum = btq::execution::parallel_reduce(data.begin(), data.end(), 0.0);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Parallel reduce took: " << duration.count() << " ms\n";
    std::cout << "Sum: " << sum << "\n";
    
    std::cout << "\nBasic tests completed successfully!\n";
    
    return 0;
}