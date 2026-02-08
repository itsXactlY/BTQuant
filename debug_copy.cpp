#include "threading/double_buffered_state.hpp"
#include <iostream>

using namespace btq::threading;

int main() {
    std::cout << "Debugging copy operations...\n";
    
    DoubleBufferedState<int> original(123);
    std::cout << "After creation, original.read() = " << original.read() << std::endl;
    
    original.write() = 456;
    std::cout << "After write, original.read() still = " << original.read() << std::endl;
    std::cout << "After write, original.write() = " << original.write() << std::endl;
    
    original.swap();
    std::cout << "After swap, original.read() = " << original.read() << std::endl;
    
    // Test copy constructor
    DoubleBufferedState<int> copied(original);
    std::cout << "After copy, copied.read() = " << copied.read() << std::endl;
    std::cout << "Expected: 456, Actual: " << copied.read() << std::endl;
    
    if (copied.read() == 456) {
        std::cout << "SUCCESS: Copy operation worked correctly!" << std::endl;
    } else {
        std::cout << "FAILURE: Copy operation failed!" << std::endl;
    }
    
    return 0;
}