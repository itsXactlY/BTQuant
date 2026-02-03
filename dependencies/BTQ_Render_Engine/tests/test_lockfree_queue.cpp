#include "../include/threading/lockfree_queue.hpp"
#include <thread>
#include <vector>
#include <cassert>
#include <iostream>
#include <chrono>

using namespace btq::threading;

void test_basic_operations() {
    std::cout << "Testing basic operations..." << std::endl;

    LockFreeQueue<int> queue;

    // Test push and pop
    queue.push(42);
    auto result = queue.pop();
    assert(result != nullptr);
    assert(*result == 42);

    // Test empty queue
    auto empty_result = queue.pop();
    assert(empty_result == nullptr);

    // Test try_pop
    queue.push(100);
    auto optional_result = queue.try_pop();
    assert(optional_result.has_value());
    assert(optional_result.value() == 100);

    // Test empty try_pop
    optional_result = queue.try_pop();
    assert(!optional_result.has_value());

    std::cout << "Basic operations test passed!" << std::endl;
}

void test_concurrent_operations() {
    std::cout << "Testing concurrent operations..." << std::endl;

    LockFreeQueue<int> queue;
    const int num_items = 1000;
    const int num_producers = 2;
    const int num_consumers = 2;

    std::vector<std::thread> threads;
    std::atomic<int> produced_count{0};
    std::atomic<int> consumed_count{0};

    // Producer threads
    for (int p = 0; p < num_producers; ++p) {
        threads.emplace_back([&queue, &produced_count, num_items, p]() {
            int start = p * (num_items / num_producers);
            int end = (p + 1) * (num_items / num_producers);

            for (int i = start; i < end; ++i) {
                queue.push(i);
                produced_count++;
            }
        });
    }

    // Consumer threads
    for (int c = 0; c < num_consumers; ++c) {
        threads.emplace_back([&queue, &consumed_count, num_items, num_consumers]() {
            int items_per_consumer = num_items / num_consumers;
            int consumed = 0;

            while (consumed < items_per_consumer) {
                auto result = queue.try_pop();
                if (result.has_value()) {
                    consumed++;
                    consumed_count++;
                } else {
                    // Brief pause to allow other threads to work
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                }
            }
        });
    }

    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }

    // Drain any remaining items
    while (auto result = queue.pop()) {
        consumed_count++;
    }

    assert(produced_count == num_items);
    assert(consumed_count == num_items);

    std::cout << "Concurrent operations test passed! Produced: " << produced_count
              << ", Consumed: " << consumed_count << std::endl;
}

void test_move_operations() {
    std::cout << "Testing move operations..." << std::endl;

    LockFreeQueue<std::string> queue;

    std::string test_str = "Hello, Lock-Free World!";
    queue.push(std::move(test_str));

    auto result = queue.pop();
    assert(result != nullptr);
    assert(*result == "Hello, Lock-Free World!");
    assert(test_str.empty()); // Original string should be moved from

    // Test emplace
    queue.emplace("Emplaced String");
    result = queue.pop();
    assert(result != nullptr);
    assert(*result == "Emplaced String");

    std::cout << "Move operations test passed!" << std::endl;
}

void test_empty_check() {
    std::cout << "Testing empty check..." << std::endl;

    LockFreeQueue<int> queue;
    assert(queue.empty());

    queue.push(1);
    assert(!queue.empty());

    queue.pop();
    assert(queue.empty());

    std::cout << "Empty check test passed!" << std::endl;
}

int main() {
    std::cout << "Starting LockFreeQueue tests..." << std::endl;

    test_basic_operations();
    test_move_operations();
    test_empty_check();
    test_concurrent_operations();

    std::cout << "All tests passed!" << std::endl;
    return 0;
}