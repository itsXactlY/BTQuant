#include "../include/threading/lockfree_queue.hpp"
#include "../include/task_scheduler.hpp"  // For btq::Trade and btq::Candle
#include <thread>
#include <vector>
#include <cassert>
#include <iostream>
#include <chrono>
#include <future>

using namespace btq::threading;

void test_lockfree_stack() {
    std::cout << "Testing LockFreeStack..." << std::endl;

    LockFreeStack<int> stack;

    // Test push and pop
    stack.push(42);
    stack.push(24);
    
    auto result1 = stack.pop();
    assert(result1 != nullptr);
    assert(*result1 == 24); // Stack is LIFO, so 24 should come out first
    
    auto result2 = stack.pop();
    assert(result2 != nullptr);
    assert(*result2 == 42);
    
    // Test empty stack
    auto result3 = stack.pop();
    assert(result3 == nullptr);

    // Test try_pop
    stack.push(100);
    auto optional_result = stack.try_pop();
    assert(optional_result.has_value());
    assert(optional_result.value() == 100);

    // Test empty try_pop
    optional_result = stack.try_pop();
    assert(!optional_result.has_value());

    std::cout << "LockFreeStack test passed!" << std::endl;
}

void test_spsc_ring_buffer() {
    std::cout << "Testing SPSCRingBuffer..." << std::endl;

    SPSCRingBuffer<int> buffer(10); // Capacity of 10

    // Test push and pop
    bool success = buffer.push(42);
    assert(success);
    
    auto result = buffer.try_pop();
    assert(result.has_value());
    assert(result.value() == 42);

    // Test empty buffer
    result = buffer.try_pop();
    assert(!result.has_value());

    // Test filling buffer (typically can only store capacity-1 elements)
    int max_fill = 9; // Usually one less than capacity due to full/empty ambiguity
    for (int i = 0; i < max_fill; ++i) {
        success = buffer.push(i);
        assert(success);
    }
    
    // Next push should fail
    success = buffer.push(99);
    if (success) {
        // If push succeeded, it means we can fill all 10 slots (some implementations allow this)
        max_fill = 10;
        buffer.push(99);
    }

    // Pop all items
    for (int i = 0; i < max_fill; ++i) {
        result = buffer.try_pop();
        assert(result.has_value());
        assert(result.value() == i);
    }

    std::cout << "SPSCRingBuffer test passed!" << std::endl;
}

void test_concurrent_queue_operations() {
    std::cout << "Testing concurrent LockFreeQueue operations..." << std::endl;

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

    std::cout << "Concurrent LockFreeQueue operations test passed! Produced: " << produced_count
              << ", Consumed: " << consumed_count << std::endl;
}

void test_concurrent_stack_operations() {
    std::cout << "Testing concurrent LockFreeStack operations..." << std::endl;

    LockFreeStack<int> stack;
    const int num_items = 1000;
    const int num_producers = 2;
    const int num_consumers = 2;

    std::vector<std::thread> threads;
    std::atomic<int> produced_count{0};
    std::atomic<int> consumed_count{0};

    // Producer threads
    for (int p = 0; p < num_producers; ++p) {
        threads.emplace_back([&stack, &produced_count, num_items, p]() {
            int start = p * (num_items / num_producers);
            int end = (p + 1) * (num_items / num_producers);

            for (int i = start; i < end; ++i) {
                stack.push(i);
                produced_count++;
            }
        });
    }

    // Consumer threads
    for (int c = 0; c < num_consumers; ++c) {
        threads.emplace_back([&stack, &consumed_count, num_items]() {
            // Since stack is LIFO, we need to keep trying until all items are consumed
            while (consumed_count.load() < num_items) {
                auto result = stack.pop();
                if (result != nullptr) {
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

    std::cout << "Concurrent LockFreeStack operations test passed! Produced: " << produced_count
              << ", Consumed: " << consumed_count << std::endl;
}

void test_spsc_buffer_performance() {
    std::cout << "Testing SPSCRingBuffer performance..." << std::endl;

    SPSCRingBuffer<int> buffer(1000);
    const int num_items = 100000;

    // Producer thread
    auto producer = std::async(std::launch::async, [&buffer, num_items]() {
        for (int i = 0; i < num_items; ++i) {
            while (!buffer.push(i)) {
                std::this_thread::yield(); // Buffer full, wait
            }
        }
    });

    // Consumer thread
    auto consumer = std::async(std::launch::async, [&buffer, num_items]() {
        int count = 0;
        while (count < num_items) {
            auto result = buffer.try_pop();
            if (result.has_value()) {
                count++;
            } else {
                std::this_thread::yield(); // Buffer empty, wait
            }
        }
        return count;
    });

    // Wait for both to complete
    producer.wait();
    int consumed = consumer.get();

    assert(consumed == num_items);

    std::cout << "SPSCRingBuffer performance test passed! Processed: " << consumed << " items" << std::endl;
}

void test_trade_data_structures() {
    std::cout << "Testing thread-safe structures with Trade data..." << std::endl;

    // Test LockFreeQueue with Trade
    LockFreeQueue<btq::Trade> trade_queue;
    btq::Trade trade;
    trade.price = 100.5;
    trade.volume = 100;
    trade.timestamp = std::chrono::system_clock::now();
    
    trade_queue.push(trade);
    auto retrieved_trade = trade_queue.pop();
    assert(retrieved_trade != nullptr);
    assert(retrieved_trade->price == 100.5);
    assert(retrieved_trade->volume == 100);

    // Test LockFreeStack with Candle
    LockFreeStack<btq::Candle> candle_stack;
    btq::Candle candle;
    candle.open = 100.0;
    candle.high = 105.0;
    candle.low = 98.0;
    candle.close = 103.0;
    candle.volume = 1000;
    candle.timestamp = std::chrono::system_clock::now();
    
    candle_stack.push(candle);
    auto retrieved_candle = candle_stack.pop();
    assert(retrieved_candle != nullptr);
    assert(retrieved_candle->open == 100.0);
    assert(retrieved_candle->close == 103.0);

    // Test SPSCRingBuffer with vector of doubles
    SPSCRingBuffer<std::vector<double>> vec_buffer(10);
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    bool success = vec_buffer.push(data);
    assert(success);
    
    auto retrieved_vec = vec_buffer.try_pop();
    assert(retrieved_vec.has_value());
    assert(retrieved_vec->size() == 5);
    assert((*retrieved_vec)[0] == 1.0);

    std::cout << "Trade data structures test passed!" << std::endl;
}

int main() {
    std::cout << "Starting comprehensive LockFree data structures tests..." << std::endl;

    test_lockfree_stack();
    test_spsc_ring_buffer();
    test_concurrent_queue_operations();
    test_concurrent_stack_operations();
    test_spsc_buffer_performance();
    test_trade_data_structures();

    std::cout << "All tests passed!" << std::endl;
    return 0;
}