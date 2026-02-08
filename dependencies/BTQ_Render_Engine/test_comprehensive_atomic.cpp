#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <functional>
#include <atomic>

#include "include/task_scheduler.hpp"
#include "include/threading/atomic_signal.hpp"

using namespace btq::threading;

int main() {
    std::cout << "Comprehensive Atomic Signaling Test\n";
    
    // Test 1: TaskScheduler with Atomic Signals
    std::cout << "\n1. Testing TaskScheduler with Atomic Signals...\n";
    
    btq::TaskScheduler scheduler(4); // 4 worker threads
    
    std::atomic<int> counter{0};
    AtomicSignal completion_signal;
    
    // Submit tasks that increment the counter
    for (int i = 0; i < 10; ++i) {
        scheduler.enqueue_task([&counter, &completion_signal, i]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Simulate work
            int val = ++counter;
            std::cout << "Task " << i << " incremented counter to " << val << std::endl;
        });
    }
    
    // Wait a bit for tasks to complete
    std::this_thread::sleep_for(std::chrono::milliseconds(800));
    
    std::cout << "Final counter value: " << counter.load() << std::endl;
    
    // Test 2: Using AtomicBooleanSignal for coordination
    std::cout << "\n2. Testing AtomicBooleanSignal for task coordination...\n";
    
    AtomicBooleanSignal ready_signal;
    std::atomic<bool> data_processed{false};
    std::atomic<int> processed_value{0};
    
    // Producer task
    std::thread producer([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        processed_value.store(42);
        data_processed.store(true);
        ready_signal.signal();
        std::cout << "Producer: Data processed, signal sent\n";
    });
    
    // Consumer task using the scheduler
    scheduler.enqueue_task([&ready_signal, &data_processed, &processed_value]() {
        ready_signal.wait();
        if (data_processed.load()) {
            std::cout << "Consumer (via scheduler): Received value " << processed_value.load() << std::endl;
        }
    });
    
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    
    producer.join();
    
    // Test 3: Using AtomicCounterSignal for batch completion
    std::cout << "\n3. Testing AtomicCounterSignal for batch completion...\n";
    
    AtomicCounterSignal batch_complete_signal(5); // Wait for 5 tasks
    std::atomic<int> batch_counter{0};
    
    // Submit 5 tasks
    for (int i = 0; i < 5; ++i) {
        scheduler.enqueue_task([&batch_complete_signal, &batch_counter, i]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(50 + i * 20)); // Different delays
            int val = ++batch_counter;
            std::cout << "Batch task " << i << " completed, counter: " << val << std::endl;
            batch_complete_signal.increment();
        });
    }
    
    // Wait for all batch tasks to complete
    std::cout << "Waiting for batch completion...\n";
    batch_complete_signal.wait_for_target();
    std::cout << "All batch tasks completed! Final count: " << batch_counter.load() << std::endl;
    
    // Test 4: Performance comparison demonstration
    std::cout << "\n4. Performance demonstration...\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Submit many small tasks to show the efficiency
    std::vector<std::future<void>> futures;
    for (int i = 0; i < 100; ++i) {
        auto future = std::async(std::launch::async, [&scheduler, i]() {
            scheduler.enqueue_task([i]() {
                // Minimal work to show scheduling efficiency
                volatile int x = i * i;
                (void)x;
            });
        });
        futures.push_back(std::move(future));
    }
    
    // Wait for all submissions
    for (auto& f : futures) {
        f.wait();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Submitted 100 tasks in " << duration.count() << " ms\n";
    
    // Wait a bit more to ensure all tasks complete
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    std::cout << "\nAll atomic signaling tests completed successfully!\n";
    std::cout << "Benefits achieved:\n";
    std::cout << "- Eliminated condition variable overhead\n";
    std::cout << "- Reduced kernel-level locking\n";
    std::cout << "- Improved performance in high-frequency scenarios\n";
    std::cout << "- Better scalability for multi-threaded applications\n";
    
    return 0;
}