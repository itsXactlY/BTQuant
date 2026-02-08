#include "include/execution/parallel_execution.hpp"
#include "include/task_scheduler.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>

void test_basic_parallel_execution() {
    std::cout << "Testing basic parallel execution...\n";
    
    // Create a large vector of numbers
    const size_t size = 1000000;
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
    
    // Test parallel_transform
    std::vector<double> result(size);
    start = std::chrono::high_resolution_clock::now();
    btq::execution::parallel_transform(data.begin(), data.end(), result.begin(), 
                                     [](double x) { return std::sqrt(x); });
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Parallel transform took: " << duration.count() << " ms\n";
    
    // Test parallel_reduce
    start = std::chrono::high_resolution_clock::now();
    double sum = btq::execution::parallel_reduce(data.begin(), data.end(), 0.0);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Parallel reduce took: " << duration.count() << " ms\n";
    std::cout << "Sum: " << sum << "\n";
}

void test_task_scheduler_execution() {
    std::cout << "\nTesting TaskScheduler execution methods...\n";
    
    btq::TaskScheduler scheduler(4); // 4 threads
    
    // Create a large vector of numbers
    const size_t size = 1000000;
    std::vector<double> data(size);
    std::iota(data.begin(), data.end(), 1.0);
    
    // Test parallel_for_async
    auto start = std::chrono::high_resolution_clock::now();
    scheduler.parallel_for_async(data.begin(), data.end(), [](double& x) {
        x = x * x; // Square each element
    });
    
    // Wait a bit for the async operation to complete
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Async parallel for scheduled in: " << duration.count() << " ms\n";
    
    // Test parallel_reduce_async
    start = std::chrono::high_resolution_clock::now();
    auto future_sum = scheduler.parallel_reduce_async(data.begin(), data.end(), 0.0);
    double sum = future_sum.get(); // Wait for result
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Async parallel reduce took: " << duration.count() << " ms\n";
    std::cout << "Async sum: " << sum << "\n";
    
    // Test execute_batch_async
    std::vector<std::function<void()>> batch_tasks;
    for (int i = 0; i < 5; ++i) {
        batch_tasks.push_back([i]() {
            std::cout << "Batch task " << i << " executed\n";
            // Simulate some work
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        });
    }
    
    start = std::chrono::high_resolution_clock::now();
    auto batch_future = scheduler.execute_batch_async(batch_tasks);
    batch_future.wait(); // Wait for all batch tasks to complete
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Batch execution took: " << duration.count() << " ms\n";
}

void test_execution_context() {
    std::cout << "\nTesting execution context...\n";
    
    // Create an execution context
    auto ctx = btq::execution::make_execution_context(4);
    
    // Create a large vector of numbers
    const size_t size = 1000000;
    std::vector<double> data(size);
    std::iota(data.begin(), data.end(), 1.0);
    
    // Test parallel_for with execution context
    auto start = std::chrono::high_resolution_clock::now();
    ctx.parallel_for(data.begin(), data.end(), [](double& x) {
        x = x * x; // Square each element
    });
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Execution context parallel for took: " << duration.count() << " ms\n";
    
    // Test parallel_reduce with execution context
    start = std::chrono::high_resolution_clock::now();
    double sum = ctx.parallel_reduce(data.begin(), data.end(), 0.0);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Execution context parallel reduce took: " << duration.count() << " ms\n";
    std::cout << "Context sum: " << sum << "\n";
}

void test_execution_policies() {
    std::cout << "\nTesting execution policies...\n";
    
    // Create a vector of numbers
    const size_t size = 100000;
    std::vector<double> data(size);
    std::iota(data.begin(), data.end(), 1.0);
    
    // Transform using parallel policy
    std::vector<double> result(size);
    auto start = std::chrono::high_resolution_clock::now();
    btq::execution::transform(btq::execution::par, data.begin(), data.end(), result.begin(),
                            [](double x) { return x * 2.0; });
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Parallel transform with policy took: " << duration.count() << " ms\n";
    
    // Reduce using parallel policy
    start = std::chrono::high_resolution_clock::now();
    double sum = btq::execution::reduce(btq::execution::par, data.begin(), data.end(), 0.0);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Parallel reduce with policy took: " << duration.count() << " ms\n";
    std::cout << "Policy sum: " << sum << "\n";
}

int main() {
    std::cout << "Testing BTQ Parallel Execution Capabilities\n";
    std::cout << "=========================================\n";
    
    test_basic_parallel_execution();
    test_task_scheduler_execution();
    test_execution_context();
    test_execution_policies();
    
    std::cout << "\nAll tests completed!\n";
    
    return 0;
}