#include "include/execution/parallel_execution.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <cmath>

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
    
    // Test execution context
    std::cout << "\nTesting execution context...\n";
    auto ctx = btq::execution::make_execution_context(4);
    
    std::vector<double> data2(size);
    std::iota(data2.begin(), data2.end(), 1.0);
    
    start = std::chrono::high_resolution_clock::now();
    ctx.parallel_for(data2.begin(), data2.end(), [](double& x) {
        x = std::sqrt(x); // Square root each element
    });
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Execution context parallel for took: " << duration.count() << " ms\n";
    
    std::cout << "\nBasic tests completed successfully!\n";
    
    return 0;
}