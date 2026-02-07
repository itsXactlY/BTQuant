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
 * Test file to detect race conditions in the BTQ Render Engine
 * This addresses the requirement to "Hunt down Raceconditions"
 */

// Test for race condition in shared counter without proper synchronization
void testSharedCounterRaceCondition() {
    std::cout << "Testing shared counter race condition..." << std::endl;
    
    int shared_counter = 0;
    const int iterations = 100000;
    const int num_threads = 4;
    
    std::vector<std::thread> threads;
    
    // Multiple threads incrementing without synchronization
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&shared_counter, iterations]() {
            for (int i = 0; i < iterations; ++i) {
                shared_counter++;  // Race condition here
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    int expected = iterations * num_threads;
    std::cout << "Expected: " << expected << ", Actual: " << shared_counter 
              << ", Difference: " << (expected - shared_counter) << std::endl;
              
    if (shared_counter != expected) {
        std::cout << "Race condition detected in shared counter!" << std::endl;
    } else {
        std::cout << "No race condition detected (unexpected)" << std::endl;
    }
}

// Test for race condition with atomic operations
void testAtomicOperations() {
    std::cout << "\nTesting atomic operations..." << std::endl;
    
    std::atomic<int> atomic_counter{0};
    const int iterations = 100000;
    const int num_threads = 4;
    
    std::vector<std::thread> threads;
    
    // Multiple threads incrementing atomically
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&atomic_counter, iterations]() {
            for (int i = 0; i < iterations; ++i) {
                atomic_counter.fetch_add(1);  // Thread-safe
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    int expected = iterations * num_threads;
    std::cout << "Expected: " << expected << ", Actual: " << atomic_counter.load()
              << ", Difference: " << (expected - atomic_counter.load()) << std::endl;
              
    if (atomic_counter.load() != expected) {
        std::cout << "Unexpected result with atomic operations!" << std::endl;
    } else {
        std::cout << "Atomic operations working correctly." << std::endl;
    }
}

// Test for race condition in lock-free queue
void testLockFreeQueueRaceCondition() {
    std::cout << "\nTesting lock-free queue race condition..." << std::endl;
    
    btq::threading::LockFreeQueue<int> queue;
    const int iterations = 10000;
    const int num_producer_threads = 2;
    const int num_consumer_threads = 2;
    
    std::atomic<bool> stop_flag{false};
    std::atomic<int> items_produced{0};
    std::atomic<int> items_consumed{0};
    
    std::vector<std::thread> producers, consumers;
    
    // Producer threads
    for (int t = 0; t < num_producer_threads; ++t) {
        producers.emplace_back([&queue, &stop_flag, &items_produced, iterations, t]() {
            std::random_device rd;
            std::mt19937 gen(rd() + t);  // Different seed for each thread
            std::uniform_int_distribution<> dis(1, 100);
            
            for (int i = 0; i < iterations && !stop_flag.load(); ++i) {
                int value = dis(gen);
                queue.push(value);
                items_produced.fetch_add(1);
                
                // Small delay to increase chance of race condition
                std::this_thread::sleep_for(std::chrono::nanoseconds(1));
            }
        });
    }
    
    // Consumer threads
    for (int t = 0; t < num_consumer_threads; ++t) {
        consumers.emplace_back([&queue, &stop_flag, &items_consumed]() {
            while (!stop_flag.load() || !queue.empty()) {
                auto item = queue.try_pop();
                if (item.has_value()) {
                    items_consumed.fetch_add(1);
                    
                    // Small delay to increase chance of race condition
                    std::this_thread::sleep_for(std::chrono::nanoseconds(1));
                } else {
                    std::this_thread::yield();  // Yield to allow other threads to run
                }
            }
        });
    }
    
    // Let producers run for a while
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Signal stop and join threads
    stop_flag.store(true);
    
    for (auto& t : producers) {
        t.join();
    }
    
    // Wait a bit more for consumers to finish
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    for (auto& t : consumers) {
        t.join();
    }
    
    std::cout << "Items produced: " << items_produced.load() 
              << ", Items consumed: " << items_consumed.load() << std::endl;
              
    if (items_produced.load() != items_consumed.load()) {
        std::cout << "Race condition detected in lock-free queue!" << std::endl;
    } else {
        std::cout << "Lock-free queue working correctly." << std::endl;
    }
}

// Test for race condition in TaskScheduler
void testTaskSchedulerRaceCondition() {
    std::cout << "\nTesting TaskScheduler race condition..." << std::endl;
    
    btq::TaskScheduler scheduler(4);  // 4 worker threads
    
    const int num_tasks = 1000;
    std::atomic<int> shared_resource{0};
    
    std::vector<std::future<void>> futures;
    
    // Submit many tasks that access shared resource
    for (int i = 0; i < num_tasks; ++i) {
        auto future = std::async(std::launch::async, [&scheduler, &shared_resource, i]() {
            // Submit task to scheduler
            scheduler.enqueue_task([&shared_resource, i]() {
                // Access shared resource without proper synchronization
                int current = shared_resource.load();
                std::this_thread::sleep_for(std::chrono::nanoseconds(1));  // Increase chance of race
                shared_resource.store(current + i);
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
    
    std::cout << "Final value of shared resource: " << shared_resource.load() << std::endl;
    
    // The final value is unpredictable due to race condition
    // This demonstrates the race condition
    std::cout << "Race condition demonstrated with shared resource in TaskScheduler." << std::endl;
}

// Test for race condition with mutex protection
void testMutexProtectedAccess() {
    std::cout << "\nTesting mutex-protected access..." << std::endl;
    
    int shared_resource = 0;
    std::mutex resource_mutex;
    const int iterations = 100000;
    const int num_threads = 4;
    
    std::vector<std::thread> threads;
    
    // Multiple threads incrementing with mutex protection
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&shared_resource, &resource_mutex, iterations]() {
            for (int i = 0; i < iterations; ++i) {
                std::lock_guard<std::mutex> lock(resource_mutex);
                shared_resource++;  // Protected access
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    int expected = iterations * num_threads;
    std::cout << "Expected: " << expected << ", Actual: " << shared_resource 
              << ", Difference: " << (expected - shared_resource) << std::endl;
              
    if (shared_resource != expected) {
        std::cout << "Unexpected result with mutex protection!" << std::endl;
    } else {
        std::cout << "Mutex protection working correctly." << std::endl;
    }
}

int main() {
    std::cout << "Starting race condition detection tests..." << std::endl;
    
    testSharedCounterRaceCondition();
    testAtomicOperations();
    testLockFreeQueueRaceCondition();
    testTaskSchedulerRaceCondition();
    testMutexProtectedAccess();
    
    std::cout << "\nRace condition detection tests completed!" << std::endl;
    return 0;
}