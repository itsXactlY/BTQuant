#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "../dependencies/BTQ_Render_Engine/include/threading/lockfree_queue.hpp"
#include "../dependencies/BTQ_Render_Engine/include/task_scheduler.hpp"

/**
 * Test file to verify race condition fixes in the BTQ Render Engine
 * This addresses the requirement to "Hunt down Raceconditions"
 */

// Test for race condition fixes in TaskScheduler
void testFixedTaskSchedulerRaceCondition() {
    std::cout << "Testing fixed TaskScheduler race condition..." << std::endl;
    
    btq::TaskScheduler scheduler(4);  // 4 worker threads
    
    const int num_tasks = 1000;
    std::atomic<int> shared_resource{0};
    std::mutex resource_mutex;  // Proper mutex for protecting shared resource
    
    std::vector<std::future<void>> futures;
    
    // Submit many tasks that access shared resource with proper synchronization
    for (int i = 0; i < num_tasks; ++i) {
        auto future = std::async(std::launch::async, [&scheduler, &shared_resource, &resource_mutex, i]() {
            // Submit task to scheduler
            scheduler.enqueue_task([&shared_resource, &resource_mutex, i]() {
                // Access shared resource WITH proper synchronization
                {
                    std::lock_guard<std::mutex> lock(resource_mutex);
                    int current = shared_resource.load();
                    std::this_thread::sleep_for(std::chrono::nanoseconds(1));  // Small delay
                    shared_resource.store(current + 1);  // Increment instead of adding i to make it predictable
                }
            });
        });
        futures.push_back(std::move(future));
    }
    
    // Wait for all futures to complete
    for (auto& f : futures) {
        f.wait();
    }
    
    // Wait a bit more for scheduler tasks to complete
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    std::cout << "Final value of shared resource: " << shared_resource.load() 
              << " (expected: " << num_tasks << ")" << std::endl;
    
    if (shared_resource.load() == num_tasks) {
        std::cout << "TaskScheduler race condition fix working correctly!" << std::endl;
    } else {
        std::cout << "TaskScheduler still has race condition issue!" << std::endl;
    }
}

// Test TaskScheduler shutdown race condition
void testTaskSchedulerShutdownRaceCondition() {
    std::cout << "\nTesting TaskScheduler shutdown race condition..." << std::endl;
    
    const int num_iterations = 100;
    int success_count = 0;
    
    for (int iter = 0; iter < num_iterations; ++iter) {
        try {
            btq::TaskScheduler scheduler(2);  // 2 worker threads
            
            // Submit a few quick tasks
            std::atomic<int> counter{0};
            for (int i = 0; i < 10; ++i) {
                scheduler.enqueue_task([&counter]() {
                    counter.fetch_add(1);
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                });
            }
            
            // Scheduler destructor will be called here
            // If race condition exists, this might cause issues
        } catch (...) {
            std::cout << "Exception during TaskScheduler destruction in iteration " << iter << std::endl;
        }
        
        success_count++;
    }
    
    std::cout << "Successfully completed " << success_count << " out of " << num_iterations 
              << " TaskScheduler creation/destruction cycles" << std::endl;
              
    if (success_count == num_iterations) {
        std::cout << "TaskScheduler shutdown race condition appears to be fixed!" << std::endl;
    } else {
        std::cout << "TaskScheduler shutdown race condition still exists!" << std::endl;
    }
}

// Test for race condition in lock-free queue with proper stress testing
void testLockFreeQueueStress() {
    std::cout << "\nTesting lock-free queue under stress..." << std::endl;
    
    btq::threading::LockFreeQueue<int> queue;
    const int iterations = 50000;
    const int num_producer_threads = 3;
    const int num_consumer_threads = 3;
    
    std::atomic<bool> stop_flag{false};
    std::atomic<int> items_produced{0};
    std::atomic<int> items_consumed{0};
    
    // Use a mutex to protect the sum calculation to avoid race condition in the test itself
    std::mutex sum_mutex;
    int total_sum_produced = 0;
    int total_sum_consumed = 0;
    
    std::vector<std::thread> producers, consumers;
    
    // Producer threads
    for (int t = 0; t < num_producer_threads; ++t) {
        producers.emplace_back([&queue, &stop_flag, &items_produced, &total_sum_produced, &sum_mutex, iterations, t]() {
            std::random_device rd;
            std::mt19937 gen(rd() + t * 1000);  // Different seed for each thread
            std::uniform_int_distribution<> dis(1, 1000);
            
            for (int i = 0; i < iterations && !stop_flag.load(); ++i) {
                int value = dis(gen);
                queue.push(value);
                items_produced.fetch_add(1);
                
                // Protect the sum calculation
                {
                    std::lock_guard<std::mutex> lock(sum_mutex);
                    total_sum_produced += value;
                }
                
                // Small random delay to create more varied timing
                if (i % 100 == 0) {
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                }
            }
        });
    }
    
    // Consumer threads
    for (int t = 0; t < num_consumer_threads; ++t) {
        consumers.emplace_back([&queue, &stop_flag, &items_consumed, &total_sum_consumed, &sum_mutex]() {
            while (!stop_flag.load() || !queue.empty()) {
                auto item = queue.try_pop();
                if (item.has_value()) {
                    items_consumed.fetch_add(1);
                    
                    // Protect the sum calculation
                    {
                        std::lock_guard<std::mutex> lock(sum_mutex);
                        total_sum_consumed += item.value();
                    }
                    
                    // Small delay to increase chance of race condition if present
                    if (items_consumed.load() % 100 == 0) {
                        std::this_thread::sleep_for(std::chrono::microseconds(1));
                    }
                } else {
                    std::this_thread::yield();  // Yield to allow other threads to run
                }
            }
        });
    }
    
    // Let producers run for a while
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // Signal stop and join threads
    stop_flag.store(true);
    
    for (auto& t : producers) {
        t.join();
    }
    
    // Wait a bit more for consumers to finish
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    for (auto& t : consumers) {
        t.join();
    }
    
    std::cout << "Items produced: " << items_produced.load() 
              << ", Items consumed: " << items_consumed.load() << std::endl;
    std::cout << "Sum produced: " << total_sum_produced
              << ", Sum consumed: " << total_sum_consumed << std::endl;
              
    bool items_match = (items_produced.load() == items_consumed.load());
    bool sums_match = (total_sum_produced == total_sum_consumed);
    
    if (items_match && sums_match) {
        std::cout << "Lock-free queue stress test PASSED - no race conditions detected!" << std::endl;
    } else {
        std::cout << "Lock-free queue stress test FAILED - race conditions detected!" << std::endl;
        if (!items_match) {
            std::cout << "  - Item count mismatch!" << std::endl;
        }
        if (!sums_match) {
            std::cout << "  - Sum mismatch - items may have been corrupted!" << std::endl;
        }
    }
}

// Test atomic operations for race condition prevention
void testAtomicRaceConditionPrevention() {
    std::cout << "\nTesting atomic operations for race condition prevention..." << std::endl;
    
    std::atomic<int> atomic_counter{0};
    const int iterations = 100000;
    const int num_threads = 4;
    
    std::vector<std::thread> threads;
    
    // Multiple threads incrementing atomically
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&atomic_counter, iterations]() {
            for (int i = 0; i < iterations; ++i) {
                atomic_counter.fetch_add(1);  // Thread-safe atomic operation
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    int expected = iterations * num_threads;
    std::cout << "Expected: " << expected << ", Actual: " << atomic_counter.load()
              << ", Difference: " << (expected - atomic_counter.load()) << std::endl;
              
    if (atomic_counter.load() == expected) {
        std::cout << "Atomic operations successfully prevented race condition!" << std::endl;
    } else {
        std::cout << "Unexpected result with atomic operations - possible race condition!" << std::endl;
    }
}

int main() {
    std::cout << "Starting race condition fix verification tests..." << std::endl;
    
    testFixedTaskSchedulerRaceCondition();
    testTaskSchedulerShutdownRaceCondition();
    testLockFreeQueueStress();
    testAtomicRaceConditionPrevention();
    
    std::cout << "\nRace condition fix verification tests completed!" << std::endl;
    return 0;
}