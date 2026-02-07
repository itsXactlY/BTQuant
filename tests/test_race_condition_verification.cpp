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
 * Test to verify that race conditions have been fixed
 */
void testLockFreeQueueCorrectness() {
    std::cout << "Testing lock-free queue correctness after fixes..." << std::endl;

    btq::threading::LockFreeQueue<int> queue;
    const int iterations = 50000;
    const int num_producer_threads = 3;
    const int num_consumer_threads = 3;

    std::atomic<bool> stop_flag{false};
    std::atomic<int> items_produced{0};
    std::atomic<int> items_consumed{0};

    // Use a mutex to protect the sum calculation to avoid race condition in the test itself
    std::mutex sum_mutex;
    std::atomic<long long> total_sum_produced{0};
    std::atomic<long long> total_sum_consumed{0};

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
                
                // Atomically update the sum
                total_sum_produced.fetch_add(value);

                // Small random delay to create more varied timing
                if (i % 100 == 0) {
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                }
            }
        });
    }

    // Consumer threads
    for (int t = 0; t < num_consumer_threads; ++t) {
        consumers.emplace_back([&queue, &stop_flag, &items_consumed, &total_sum_consumed]() {
            while (!stop_flag.load() || !queue.empty()) {
                auto item = queue.try_pop();
                if (item.has_value()) {
                    items_consumed.fetch_add(1);
                    
                    // Atomically update the sum
                    total_sum_consumed.fetch_add(item.value());

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
    std::cout << "Sum produced: " << total_sum_produced.load()
              << ", Sum consumed: " << total_sum_consumed.load() << std::endl;

    bool items_match = (items_produced.load() == items_consumed.load());
    bool sums_match = (total_sum_produced.load() == total_sum_consumed.load());

    if (items_match && sums_match) {
        std::cout << "Lock-free queue test PASSED - no race conditions detected!" << std::endl;
    } else {
        std::cout << "Lock-free queue test FAILED - race conditions detected!" << std::endl;
        if (!items_match) {
            std::cout << "  - Item count mismatch!" << std::endl;
        }
        if (!sums_match) {
            std::cout << "  - Sum mismatch - items may have been corrupted!" << std::endl;
        }
    }
}

int main() {
    std::cout << "Running race condition fix verification tests..." << std::endl;
    
    testLockFreeQueueCorrectness();
    
    std::cout << "Verification tests completed!" << std::endl;
    return 0;
}