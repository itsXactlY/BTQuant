#include "threading/double_buffered_state.hpp"
#include <iostream>
#include <cassert>

using namespace btq::threading;

int main() {
    std::cout << "Simple test to isolate the issue...\n";
    
    // Test with simple int
    DoubleBufferedState<int> state1(42);
    assert(state1.read() == 42);
    std::cout << "Int test passed\n";
    
    // Test with simple vector
    DoubleBufferedState<std::vector<int>> state2(std::vector<int>{1, 2, 3});
    assert(state2.read().size() == 3);
    std::cout << "Vector test passed\n";
    
    // Test copy
    DoubleBufferedState<int> copied_state(state1);
    assert(copied_state.read() == 42);
    std::cout << "Copy test passed\n";
    
    // Test move
    DoubleBufferedState<int> moved_state = std::move(state1);
    std::cout << "Move test passed\n";
    
    std::cout << "All simple tests passed!\n";
    return 0;
}