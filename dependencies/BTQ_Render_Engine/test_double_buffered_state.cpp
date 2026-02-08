#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cassert>

#include "include/threading/double_buffered_state.hpp"

void test_basic_functionality() {
    std::cout << "Testing basic double buffered state functionality...\n";

    // Create a double buffered state with an initial vector
    btq::threading::DoubleBufferedState<std::vector<int>> db_state({1, 2, 3});

    // Read initial state
    const auto& front = db_state.read();
    assert(front.size() == 3);
    assert(front[0] == 1);
    assert(front[1] == 2);
    assert(front[2] == 3);

    // Write to back buffer
    auto& back = db_state.write();
    back.push_back(4);
    assert(back.size() == 4);

    // Front buffer should still have the old values
    assert(db_state.read().size() == 3);

    // Swap the buffers
    db_state.swap();

    // Now front buffer should have the new values
    const auto& new_front = db_state.read();
    assert(new_front.size() == 4);
    assert(new_front[3] == 4);

    std::cout << "Basic functionality test passed!\n";
}

void test_concurrent_access() {
    std::cout << "Testing concurrent access...\n";

    btq::threading::DoubleBufferedState<std::vector<int>> db_state({});

    // Writer thread: continuously adds elements and swaps
    std::thread writer([&db_state]() {
        for (int i = 0; i < 100; ++i) {
            auto& back = db_state.write();
            back.push_back(i);
            db_state.swap();
            std::this_thread::sleep_for(std::chrono::microseconds(10)); // Brief pause
        }
    });

    // Reader thread: continuously reads from the front buffer
    std::thread reader([&db_state]() {
        for (int i = 0; i < 100; ++i) {
            const auto& front = db_state.read();
            // Just read, don't validate since the content is changing
            volatile int size = front.size(); // Prevent compiler optimization
            std::this_thread::sleep_for(std::chrono::microseconds(10)); // Brief pause
        }
    });

    writer.join();
    reader.join();

    std::cout << "Concurrent access test passed!\n";
}

void test_update_and_swap() {
    std::cout << "Testing update_and_swap functionality...\n";

    btq::threading::DoubleBufferedState<std::vector<int>> db_state({1, 2, 3});

    // Use update_and_swap to replace the content
    std::vector<int> new_data = {10, 20, 30, 40};
    db_state.update_and_swap(new_data);

    // Check that the front buffer now has the new data
    const auto& front = db_state.read();
    assert(front.size() == 4);
    assert(front[0] == 10);
    assert(front[1] == 20);
    assert(front[2] == 30);
    assert(front[3] == 40);

    std::cout << "Update and swap test passed!\n";
}

void test_move_semantics() {
    std::cout << "Testing move semantics...\n";

    btq::threading::DoubleBufferedState<std::vector<int>> db_state({1, 2, 3});

    // Use move version of update_and_swap
    std::vector<int> new_data = {100, 200, 300};
    db_state.update_and_swap(std::move(new_data));

    // Check that the front buffer has the moved data
    const auto& front = db_state.read();
    assert(front.size() == 3);
    assert(front[0] == 100);
    assert(front[1] == 200);
    assert(front[2] == 300);

    std::cout << "Move semantics test passed!\n";
}

int main() {
    std::cout << "Starting double buffered state tests...\n\n";

    test_basic_functionality();
    test_concurrent_access();
    test_update_and_swap();
    test_move_semantics();

    std::cout << "\nAll tests passed successfully!\n";

    return 0;
}