#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <functional>

#include "include/threading/atomic_signal.hpp"

using namespace btq::threading;

int main() {
    std::cout << "Testing Atomic Signal Implementation\n";
    
    // Test 1: Basic AtomicSignal
    std::cout << "\n1. Testing basic AtomicSignal...\n";
    
    AtomicSignal signal;
    std::atomic<bool> data_ready{false};
    std::atomic<int> shared_data{0};
    
    // Producer thread
    std::thread producer([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Simulate work
        shared_data.store(42);
        data_ready.store(true);
        signal.notify_one();
        std::cout << "Producer: Data is ready, signal sent.\n";
    });
    
    // Consumer thread
    std::thread consumer([&]() {
        signal.wait([&]() { return data_ready.load(); });
        std::cout << "Consumer: Received signal, data = " << shared_data.load() << "\n";
    });
    
    producer.join();
    consumer.join();
    
    // Test 2: AtomicBooleanSignal
    std::cout << "\n2. Testing AtomicBooleanSignal...\n";
    
    AtomicBooleanSignal bool_signal;
    std::atomic<int> result{0};
    
    std::thread setter([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        result.store(123);
        bool_signal.signal();
        std::cout << "Setter: Result set to 123, signal sent.\n";
    });
    
    std::thread waiter([&]() {
        bool_signal.wait();
        std::cout << "Waiter: Received signal, result = " << result.load() << "\n";
    });
    
    setter.join();
    waiter.join();
    
    // Test 3: AtomicCounterSignal
    std::cout << "\n3. Testing AtomicCounterSignal...\n";
    
    AtomicCounterSignal counter_signal(3); // Wait for 3 increments
    
    std::vector<std::thread> incrementors;
    for (int i = 0; i < 3; ++i) {
        incrementors.emplace_back([&, i]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(50 * (i + 1)));
            counter_signal.increment();
            std::cout << "Incrementor " << i << ": Incremented counter\n";
        });
    }
    
    std::thread counter_waiter([&]() {
        std::cout << "Counter waiter: Waiting for 3 increments...\n";
        counter_signal.wait_for_target();
        std::cout << "Counter waiter: Target reached! Final count = " << counter_signal.get_count() << "\n";
    });
    
    counter_waiter.join();
    for (auto& t : incrementors) {
        t.join();
    }
    
    // Test 4: Timeout functionality
    std::cout << "\n4. Testing timeout functionality...\n";
    
    AtomicSignal timeout_signal;
    
    auto start_time = std::chrono::steady_clock::now();
    bool result_timeout = timeout_signal.wait_for(std::chrono::milliseconds(200));
    auto end_time = std::chrono::steady_clock::now();
    
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Timeout test: waited for 200ms, elapsed: " << elapsed.count() << "ms, result: " 
              << (result_timeout ? "signaled" : "timed out") << "\n";
    
    // Test 5: Signal with notification
    std::cout << "\n5. Testing signal with early notification...\n";
    
    AtomicSignal early_signal;
    std::thread notifier([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        early_signal.notify_one();
        std::cout << "Notifier: Sent notification\n";
    });
    
    auto early_start = std::chrono::steady_clock::now();
    early_signal.wait();
    auto early_end = std::chrono::steady_clock::now();
    
    auto early_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(early_end - early_start);
    std::cout << "Early signal test: elapsed: " << early_elapsed.count() << "ms\n";
    
    notifier.join();
    
    std::cout << "\nAll tests completed successfully!\n";
    
    return 0;
}