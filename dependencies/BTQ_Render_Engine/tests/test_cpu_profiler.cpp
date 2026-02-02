#include "../include/performance/cpu_profiler.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>

using namespace BTQuant;

// Simulate some functions to profile
void function_a() {
    CPUProfiler::ProfileScope scope("function_a");
    // Simulate some work
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

void function_b() {
    CPUProfiler::ProfileScope scope("function_b");
    // Simulate some work
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
}

void function_c() {
    CPUProfiler::ProfileScope scope("function_c");
    // Simulate some work
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
}

void nested_function_call() {
    CPUProfiler::ProfileScope scope("nested_function_call");
    
    function_a();
    function_b();
    function_c();
}

int main() {
    std::cout << "Starting CPU Profiler Test...\n";
    
    // Enable profiling
    g_cpu_profiler.set_enabled(true);
    
    // Run some test functions
    std::cout << "Running test functions...\n";
    
    for (int i = 0; i < 5; ++i) {
        function_a();
        function_b();
        function_c();
        nested_function_call();
    }
    
    // Generate and print report
    std::string report = g_cpu_profiler.generate_report();
    std::cout << "\n" << report << std::endl;
    
    // Get top functions by total time
    std::cout << "\nTop 5 functions by total time:\n";
    auto top_functions = g_cpu_profiler.get_top_functions_by_total_time(5);
    for (const auto& [name, data] : top_functions) {
        std::cout << "- " << name << ": " << data.get_total_duration_ms() << " ms total, "
                  << data.call_count << " calls, " << data.get_average_duration_ms() << " ms avg\n";
    }
    
    // Get top functions by average time
    std::cout << "\nTop 5 functions by average time:\n";
    auto top_avg_functions = g_cpu_profiler.get_top_functions_by_average_time(5);
    for (const auto& [name, data] : top_avg_functions) {
        std::cout << "- " << name << ": " << data.get_average_duration_ms() << " ms avg, "
                  << data.get_total_duration_ms() << " ms total, " << data.call_count << " calls\n";
    }
    
    // Test multi-threaded profiling
    std::cout << "\nTesting multi-threaded profiling...\n";
    
    std::vector<std::thread> threads;
    for (int t = 0; t < 3; ++t) {
        threads.emplace_back([t]() {
            for (int i = 0; i < 3; ++i) {
                CPUProfiler::ProfileScope scope("thread_" + std::to_string(t) + "_function");
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Print final report
    std::cout << "\nFinal report after multi-threaded test:\n";
    std::string final_report = g_cpu_profiler.generate_report();
    std::cout << final_report << std::endl;
    
    // Reset profiler
    g_cpu_profiler.reset();
    std::cout << "Profiler reset. Report after reset:\n";
    std::cout << g_cpu_profiler.generate_report() << std::endl;
    
    std::cout << "CPU Profiler Test Completed Successfully!\n";
    
    return 0;
}