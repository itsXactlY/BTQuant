#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <random>
#include <algorithm>
#include <future>
#include <iomanip>
#include "../include/performance_monitor.hpp"
#include "../include/performance/cpu_profiler.hpp"
#include "../include/performance/memory_tracker.hpp"

using namespace BTQuant;
using namespace btq::performance;

/**
 * @brief Comprehensive performance benchmark suite for BTQ Render Engine
 * Measures frame rates, memory usage, CPU consumption under various loads
 */

class PerformanceBenchmarkSuite {
public:
    PerformanceBenchmarkSuite() {
        // Initialize global performance monitors
        g_performance_monitor.set_enabled(true);
        g_cpu_profiler.set_enabled(true);
    }

    void run_all_benchmarks() {
        std::cout << "=== BTQ Render Engine Performance Benchmark Suite ===" << std::endl;
        
        run_frame_rate_benchmarks();
        run_memory_usage_benchmarks();
        run_cpu_consumption_benchmarks();
        run_load_variation_tests();
        
        std::cout << "\n=== Performance Benchmark Suite Completed ===" << std::endl;
    }

private:
    // Frame rate benchmarking
    void run_frame_rate_benchmarks() {
        std::cout << "\n--- Frame Rate Benchmarks ---" << std::endl;
        
        // Reset performance monitor
        g_performance_monitor.reset();
        
        // Light load test (minimal rendering work)
        std::cout << "\nLight Load Test (Minimal Rendering):" << std::endl;
        simulate_rendering_work(100, 1); // 100 frames, 1ms work per frame
        
        // Moderate load test
        std::cout << "\nModerate Load Test (Normal Rendering):" << std::endl;
        simulate_rendering_work(100, 8); // 100 frames, 8ms work per frame
        
        // Heavy load test
        std::cout << "\nHeavy Load Test (Intensive Rendering):" << std::endl;
        simulate_rendering_work(100, 16); // 100 frames, 16ms work per frame
        
        // Print final statistics
        print_frame_statistics();
    }

    // Memory usage benchmarking
    void run_memory_usage_benchmarks() {
        std::cout << "\n--- Memory Usage Benchmarks ---" << std::endl;
        
        // Start memory tracking
        MemoryTracker& tracker = getGlobalMemoryTracker();
        tracker.startTracking();
        
        // Simulate memory allocations under different scenarios
        std::cout << "\nMemory Allocation Under Different Scenarios:" << std::endl;
        
        // Scenario 1: Small frequent allocations
        std::cout << "Scenario 1: Small frequent allocations..." << std::endl;
        simulate_memory_allocations(1000, 1024); // 1000 allocations of 1KB each
        
        // Scenario 2: Large infrequent allocations
        std::cout << "Scenario 2: Large infrequent allocations..." << std::endl;
        simulate_memory_allocations(100, 1024 * 100); // 100 allocations of 100KB each
        
        // Scenario 3: Mixed allocation patterns
        std::cout << "Scenario 3: Mixed allocation patterns..." << std::endl;
        simulate_mixed_memory_allocations();
        
        // Print memory statistics
        print_memory_statistics();
        
        tracker.stopTracking();
    }

    // CPU consumption benchmarking
    void run_cpu_consumption_benchmarks() {
        std::cout << "\n--- CPU Consumption Benchmarks ---" << std::endl;
        
        // Reset CPU profiler
        g_cpu_profiler.reset();
        
        // Profile different types of workloads
        std::cout << "\nCPU Profiling Different Workloads:" << std::endl;
        
        // Mathematical computation workload
        std::cout << "Mathematical computation workload..." << std::endl;
        profile_math_computations();
        
        // String manipulation workload
        std::cout << "String manipulation workload..." << std::endl;
        profile_string_operations();
        
        // Sorting algorithm workload
        std::cout << "Sorting algorithm workload..." << std::endl;
        profile_sorting_algorithms();
        
        // Print CPU statistics
        print_cpu_statistics();
    }

    // Load variation tests
    void run_load_variation_tests() {
        std::cout << "\n--- Load Variation Tests ---" << std::endl;
        
        // Variable load simulation
        std::cout << "\nSimulating Variable Load Over Time:" << std::endl;
        
        g_performance_monitor.reset();
        
        // Simulate varying load over time
        for (int period = 0; period < 5; ++period) {
            std::cout << "Period " << (period + 1) << ": ";
            
            // Different load levels for each period
            int work_ms = 5 + (period * 5); // 5ms, 10ms, 15ms, 20ms, 25ms
            std::cout << work_ms << "ms work per frame" << std::endl;
            
            simulate_rendering_work(50, work_ms);
            
            // Print stats for this period
            std::cout << "  FPS: " << std::fixed << std::setprecision(2) 
                      << g_performance_monitor.get_fps() 
                      << ", Frame Time: " << g_performance_monitor.get_frame_time_ms() << "ms" << std::endl;
        }
        
        print_final_statistics();
    }

    // Helper methods
    void simulate_rendering_work(int num_frames, int work_ms_per_frame) {
        for (int i = 0; i < num_frames; ++i) {
            // Start frame timing
            g_performance_monitor.start_frame();
            
            // Simulate rendering work
            std::this_thread::sleep_for(std::chrono::milliseconds(work_ms_per_frame));
            
            // Simulate some computational work
            volatile double result = 0;
            for (int j = 0; j < work_ms_per_frame * 1000; ++j) {
                result += j * 0.001;
            }
            
            // End frame timing
            g_performance_monitor.end_frame();
            
            // Occasionally print progress
            if ((i + 1) % 25 == 0) {
                std::cout << "  Processed " << (i + 1) << "/" << num_frames 
                          << " frames - FPS: " << g_performance_monitor.get_fps() << std::endl;
            }
        }
    }

    void simulate_memory_allocations(size_t num_allocations, size_t allocation_size) {
        std::vector<void*> allocations;
        std::vector<size_t> sizes;
        
        // Record initial memory usage
        size_t initial_usage = getGlobalMemoryTracker().getCurrentMemoryUsage();
        
        // Perform allocations
        for (size_t i = 0; i < num_allocations; ++i) {
            void* ptr = malloc(allocation_size);
            allocations.push_back(ptr);
            sizes.push_back(allocation_size);
            
            // Track allocation
            getGlobalMemoryTracker().trackAllocation(allocation_size, ptr, 
                "benchmark_alloc_" + std::to_string(i));
        }
        
        // Record peak memory usage
        size_t peak_usage = getGlobalMemoryTracker().getCurrentMemoryUsage();
        
        // Deallocate memory
        for (size_t i = 0; i < allocations.size(); ++i) {
            free(allocations[i]);
            getGlobalMemoryTracker().trackDeallocation(allocations[i]);
        }
        
        // Record final memory usage
        size_t final_usage = getGlobalMemoryTracker().getCurrentMemoryUsage();
        
        std::cout << "    Allocations: " << num_allocations 
                  << ", Size: " << allocation_size << " bytes each" << std::endl;
        std::cout << "    Memory delta: " << (peak_usage - initial_usage) << " bytes" << std::endl;
    }

    void simulate_mixed_memory_allocations() {
        std::vector<void*> allocations;
        std::vector<size_t> sizes;
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> size_dist(64, 1024 * 1024); // 64 bytes to 1MB
        
        // Perform mixed allocations
        for (int i = 0; i < 500; ++i) {
            size_t size = size_dist(gen);
            void* ptr = malloc(size);
            allocations.push_back(ptr);
            sizes.push_back(size);
            
            getGlobalMemoryTracker().trackAllocation(size, ptr, 
                "mixed_alloc_" + std::to_string(i));
        }
        
        // Deallocate half of them randomly
        std::shuffle(allocations.begin(), allocations.end(), gen);
        size_t deallocate_count = allocations.size() / 2;
        
        for (size_t i = 0; i < deallocate_count; ++i) {
            free(allocations[i]);
            getGlobalMemoryTracker().trackDeallocation(allocations[i]);
        }
        
        std::cout << "    Mixed allocations: " << allocations.size() 
                  << ", deallocated: " << deallocate_count << std::endl;
    }

    void profile_math_computations() {
        CPUProfiler::ProfileScope scope("math_computations");
        
        // Simulate intensive mathematical computations
        std::vector<double> data(10000);
        for (int i = 0; i < 100; ++i) {
            // Fill with random data
            for (auto& val : data) {
                val = static_cast<double>(rand()) / RAND_MAX;
            }
            
            // Perform mathematical operations
            for (size_t j = 0; j < data.size(); ++j) {
                data[j] = std::sin(data[j]) * std::cos(data[j]) + std::sqrt(std::abs(data[j]));
            }
        }
    }

    void profile_string_operations() {
        CPUProfiler::ProfileScope scope("string_operations");
        
        // Simulate intensive string operations
        std::vector<std::string> strings;
        for (int i = 0; i < 1000; ++i) {
            std::string s = "Test string number " + std::to_string(i) + " with some content";
            for (int j = 0; j < 10; ++j) {
                s += " additional content " + std::to_string(j);
            }
            strings.push_back(s);
        }
        
        // Perform string operations
        for (auto& str : strings) {
            std::transform(str.begin(), str.end(), str.begin(), ::toupper);
            str += " - processed";
        }
    }

    void profile_sorting_algorithms() {
        CPUProfiler::ProfileScope scope("sorting_algorithms");
        
        // Generate random data for sorting
        std::vector<int> data(50000);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, 100000);
        
        for (auto& val : data) {
            val = dis(gen);
        }
        
        // Sort multiple times to get meaningful profiling data
        for (int i = 0; i < 10; ++i) {
            std::vector<int> copy = data;
            std::sort(copy.begin(), copy.end());
        }
    }

    void print_frame_statistics() {
        std::cout << "\nFrame Rate Statistics:" << std::endl;
        std::cout << "  Current FPS: " << std::fixed << std::setprecision(2) 
                  << g_performance_monitor.get_fps() << std::endl;
        std::cout << "  Average FPS: " << g_performance_monitor.get_avg_fps() << std::endl;
        std::cout << "  Min FPS: " << g_performance_monitor.get_min_fps() << std::endl;
        std::cout << "  Max FPS: " << g_performance_monitor.get_max_fps() << std::endl;
        std::cout << "  Current Frame Time: " << g_performance_monitor.get_frame_time_ms() << " ms" << std::endl;
        std::cout << "  Average Frame Time: " << g_performance_monitor.get_avg_frame_time_ms() << " ms" << std::endl;
        std::cout << "  Min Frame Time: " << g_performance_monitor.get_min_frame_time() << " ms" << std::endl;
        std::cout << "  Max Frame Time: " << g_performance_monitor.get_max_frame_time() << " ms" << std::endl;
    }

    void print_memory_statistics() {
        std::cout << "\nMemory Statistics:" << std::endl;
        std::cout << "  Peak Memory Usage: " << getGlobalMemoryTracker().getPeakMemoryUsage() << " bytes" << std::endl;
        std::cout << "  Current Memory Usage: " << getGlobalMemoryTracker().getCurrentMemoryUsage() << " bytes" << std::endl;
        std::cout << "  Total Allocated Bytes: " << getGlobalMemoryTracker().getTotalAllocatedBytes() << std::endl;
        std::cout << "  Total Deallocated Bytes: " << getGlobalMemoryTracker().getTotalDeallocatedBytes() << std::endl;
        std::cout << "  Current Allocation Count: " << getGlobalMemoryTracker().getCurrentAllocationCount() << std::endl;

        // Get memory samples for trend analysis
        auto samples = getGlobalMemoryTracker().getMemorySamples();
        if (!samples.empty()) {
            std::cout << "  Memory Samples Collected: " << samples.size() << std::endl;
            std::cout << "  Average Growth Rate: " << getGlobalMemoryTracker().getAverageMemoryGrowthRate()
                      << " bytes/sec" << std::endl;
        }

        // Get active allocations
        auto active_allocations = getGlobalMemoryTracker().getActiveAllocations();
        std::cout << "  Active Allocations: " << active_allocations.size() << std::endl;
    }

    void print_cpu_statistics() {
        std::cout << "\nCPU Statistics:" << std::endl;
        
        // Generate CPU profiler report
        std::string report = g_cpu_profiler.generate_report();
        std::cout << "CPU Profiler Report:\n" << report << std::endl;
        
        // Top functions by total time
        std::cout << "\nTop 5 Functions by Total Time:" << std::endl;
        auto top_functions = g_cpu_profiler.get_top_functions_by_total_time(5);
        for (const auto& [name, data] : top_functions) {
            std::cout << "  - " << name << ": " << std::fixed << std::setprecision(3)
                      << data.get_total_duration_ms() << " ms total, "
                      << data.call_count << " calls, "
                      << data.get_average_duration_ms() << " ms avg" << std::endl;
        }
        
        // Top functions by average time
        std::cout << "\nTop 5 Functions by Average Time:" << std::endl;
        auto top_avg_functions = g_cpu_profiler.get_top_functions_by_average_time(5);
        for (const auto& [name, data] : top_avg_functions) {
            std::cout << "  - " << name << ": " << std::fixed << std::setprecision(3)
                      << data.get_average_duration_ms() << " ms avg, "
                      << data.get_total_duration_ms() << " ms total, "
                      << data.call_count << " calls" << std::endl;
        }
    }

    void print_final_statistics() {
        std::cout << "\nFinal Performance Statistics:" << std::endl;
        std::cout << "  Final FPS: " << std::fixed << std::setprecision(2) 
                  << g_performance_monitor.get_fps() << std::endl;
        std::cout << "  Final Frame Time: " << g_performance_monitor.get_frame_time_ms() << " ms" << std::endl;
        std::cout << "  Total Data Processed: " << g_performance_monitor.get_data_processed_count() << std::endl;
        std::cout << "  Indicators Calculated: " << g_performance_monitor.get_indicators_calculated_count() << std::endl;
        std::cout << "  Data Processing Time: " << g_performance_monitor.get_data_processing_time_ms() << " ms" << std::endl;
        
        // Memory usage at the end
        std::cout << "  Final Memory Usage: " << getGlobalMemoryTracker().getCurrentMemoryUsage() << " bytes" << std::endl;
    }
};

int main() {
    PerformanceBenchmarkSuite benchmark_suite;
    benchmark_suite.run_all_benchmarks();
    
    return 0;
}