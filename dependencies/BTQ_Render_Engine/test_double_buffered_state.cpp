#include "threading/double_buffered_state.hpp"
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <cassert>
#include <algorithm>

using namespace btq::threading;

// Simple test structure to use with DoubleBufferedState
struct TestData {
    int value = 0;
    std::vector<int> data;
    
    TestData() = default;
    TestData(int v, const std::vector<int>& d) : value(v), data(d) {}
    
    TestData& operator=(const TestData& other) {
        value = other.value;
        data = other.data;
        return *this;
    }
    
    bool operator==(const TestData& other) const {
        return value == other.value && data == other.data;
    }
};

void test_basic_functionality() {
    std::cout << "Testing basic functionality...\n";
    
    // Test construction with default value
    DoubleBufferedState<int> state1;
    assert(state1.read() == 0);
    
    // Test construction with initial value
    DoubleBufferedState<int> state2(42);
    assert(state2.read() == 42);
    
    // Test write and swap
    state2.write() = 100;
    state2.swap();
    assert(state2.read() == 100);
    
    // Test update_and_swap
    state2.update_and_swap(200);
    assert(state2.read() == 200);
    
    // Test with complex data
    DoubleBufferedState<TestData> complex_state(TestData{10, {1, 2, 3}});
    assert(complex_state.read().value == 10);
    assert(complex_state.read().data.size() == 3);
    
    complex_state.write().value = 20;
    complex_state.write().data.push_back(4);
    complex_state.swap();
    
    assert(complex_state.read().value == 20);
    assert(complex_state.read().data.size() == 4);
    assert(complex_state.read().data[3] == 4);
    
    std::cout << "Basic functionality tests passed!\n\n";
}

void test_concurrent_access() {
    std::cout << "Testing concurrent access...\n";
    
    DoubleBufferedState<int> state(0);
    const int num_iterations = 10000;
    bool success = true;
    
    // Writer thread
    std::thread writer([&state, num_iterations, &success]() {
        for (int i = 1; i <= num_iterations; ++i) {
            state.write() = i;
            state.swap();
            
            // Small delay to allow reader to catch up
            std::this_thread::sleep_for(std::chrono::nanoseconds(1));
        }
    });
    
    // Reader thread
    std::thread reader([&state, num_iterations, &success]() {
        int last_read = 0;
        for (int i = 0; i < num_iterations * 2; ++i) {  // More iterations for reader
            int current = state.read();
            // Values should be non-decreasing since we're only incrementing in writer
            if (current < last_read) {
                std::cout << "ERROR: Read value decreased from " << last_read << " to " << current << "\n";
                success = false;
            }
            last_read = std::max(last_read, current);
            
            // Small delay to avoid busy-waiting
            std::this_thread::sleep_for(std::chrono::nanoseconds(1));
        }
    });
    
    writer.join();
    reader.join();
    
    assert(success);
    assert(state.read() == num_iterations);
    
    std::cout << "Concurrent access tests passed!\n\n";
}

void test_move_semantics() {
    std::cout << "Testing move semantics...\n";
    
    TestData original_data{42, {1, 2, 3, 4, 5}};
    
    // Test move constructor
    DoubleBufferedState<TestData> original(std::move(original_data));
    assert(original.read().value == 42);
    assert(original.read().data.size() == 5);
    
    // Test move assignment
    DoubleBufferedState<TestData> moved_to;
    moved_to = std::move(original);
    assert(moved_to.read().value == 42);
    assert(moved_to.read().data.size() == 5);
    
    std::cout << "Move semantics tests passed!\n\n";
}

void test_copy_operations() {
    std::cout << "Testing copy operations...\n";
    
    DoubleBufferedState<int> original(123);
    original.write() = 456;
    original.swap();  // Now read() returns 456
    
    // Test copy constructor
    DoubleBufferedState<int> copied(original);
    assert(copied.read() == 456);
    
    // Test copy assignment
    DoubleBufferedState<int> assigned;
    assigned = original;
    assert(assigned.read() == 456);
    
    // Modify original, ensure copies are independent
    original.write() = 789;
    original.swap();
    assert(original.read() == 789);
    assert(copied.read() == 456);   // Should remain unchanged
    assert(assigned.read() == 456); // Should remain unchanged
    
    std::cout << "Copy operations tests passed!\n\n";
}

void test_stress_test() {
    std::cout << "Running stress test...\n";
    
    DoubleBufferedState<std::vector<int>> state({1, 2, 3});
    const int num_threads = 4;
    const int iterations_per_thread = 5000;
    std::vector<std::thread> threads;
    
    // Multiple writers
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&state, t, iterations_per_thread]() {
            for (int i = 0; i < iterations_per_thread; ++i) {
                auto& back_buffer = state.write();
                back_buffer.clear();
                for (int j = 0; j < 5; ++j) {
                    back_buffer.push_back(t * 1000 + i + j);
                }
                state.swap();
                
                std::this_thread::yield(); // Allow other threads to run
            }
        });
    }
    
    // Single reader thread
    std::atomic<bool> reader_done{false};
    std::thread reader([&state, &reader_done, iterations_per_thread, num_threads]() {
        int read_count = 0;
        while (read_count < iterations_per_thread * num_threads * 2 && !reader_done.load()) {
            const auto& data = state.read();
            // Just read the data to ensure no crashes
            volatile size_t size = data.size();
            (void)size; // Suppress unused warning
            
            std::this_thread::sleep_for(std::chrono::nanoseconds(100));
            read_count++;
        }
        reader_done = true;
    });
    
    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }
    reader.join();
    
    std::cout << "Stress test passed!\n\n";
}

int main() {
    std::cout << "=== Double Buffered State Tests ===\n\n";
    
    test_basic_functionality();
    test_move_semantics();
    test_copy_operations();
    test_concurrent_access();
    test_stress_test();
    
    std::cout << "=== All tests passed! ===\n";
    return 0;
}