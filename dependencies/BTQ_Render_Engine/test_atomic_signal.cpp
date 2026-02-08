#include "include/threading/atomic_signal.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>

using namespace btq::threading;

int main() {
    std::cout << "Testing Atomic Signal Implementation...\n";

    // Test 1: Basic AtomicSignal
    {
        std::cout << "\n1. Testing AtomicSignal...\n";
        AtomicSignal signal;
        
        bool signal_received = false;
        std::thread t([&signal, &signal_received]() {
            signal.wait();
            signal_received = true;
            std::cout << "   Signal received in thread\n";
        });
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        signal.notify_one();
        
        t.join();
        if (signal_received) {
            std::cout << "   ✓ AtomicSignal test passed\n";
        } else {
            std::cout << "   ✗ AtomicSignal test failed\n";
        }
    }

    // Test 2: AtomicBooleanSignal
    {
        std::cout << "\n2. Testing AtomicBooleanSignal...\n";
        AtomicBooleanSignal bool_signal;
        
        bool signal_received = false;
        std::thread t([&bool_signal, &signal_received]() {
            bool_signal.wait();
            signal_received = true;
            std::cout << "   Boolean signal received in thread\n";
        });
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        bool_signal.signal();
        
        t.join();
        if (signal_received) {
            std::cout << "   ✓ AtomicBooleanSignal test passed\n";
        } else {
            std::cout << "   ✗ AtomicBooleanSignal test failed\n";
        }
    }

    // Test 3: AtomicCounterSignal
    {
        std::cout << "\n3. Testing AtomicCounterSignal...\n";
        AtomicCounterSignal counter_signal(3); // Wait for 3 increments
        
        bool target_reached = false;
        std::thread t([&counter_signal, &target_reached]() {
            counter_signal.wait_for_target();
            target_reached = true;
            std::cout << "   Counter target reached in thread\n";
        });
        
        // Increment 3 times
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        counter_signal.increment();
        std::cout << "   Increment 1\n";
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        counter_signal.increment();
        std::cout << "   Increment 2\n";
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        counter_signal.increment();
        std::cout << "   Increment 3\n";
        
        t.join();
        if (target_reached) {
            std::cout << "   ✓ AtomicCounterSignal test passed\n";
        } else {
            std::cout << "   ✗ AtomicCounterSignal test failed\n";
        }
    }

    // Test 4: Timeout functionality
    {
        std::cout << "\n4. Testing timeout functionality...\n";
        AtomicSignal signal;
        
        auto start = std::chrono::steady_clock::now();
        bool timeout_occurred = !signal.wait_for(std::chrono::milliseconds(100));
        auto end = std::chrono::steady_clock::now();
        
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        if (timeout_occurred && elapsed.count() >= 95 && elapsed.count() <= 150) { // Allow some tolerance
            std::cout << "   ✓ Timeout test passed (elapsed: " << elapsed.count() << "ms)\n";
        } else {
            std::cout << "   ✗ Timeout test failed (elapsed: " << elapsed.count() << "ms)\n";
        }
    }

    std::cout << "\nAll tests completed!\n";
    return 0;
}