#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <functional>

#include "include/task_scheduler.hpp"

int main() {
    std::cout << "Testing TaskScheduler with Atomic Signaling\n";
    
    btq::TaskScheduler scheduler(4); // 4 worker threads
    
    std::vector<std::future<void>> futures;
    
    // Submit multiple tasks to test the scheduler
    for (int i = 0; i < 10; ++i) {
        auto future = std::async(std::launch::async, [&scheduler, i]() {
            scheduler.enqueue_task([i]() {
                std::cout << "Task " << i << " executing on thread " << std::this_thread::get_id() << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Simulate work
                std::cout << "Task " << i << " completed\n";
            });
        });
        futures.push_back(std::move(future));
    }
    
    // Wait for all tasks to be submitted
    for (auto& f : futures) {
        f.wait();
    }
    
    std::cout << "All tasks submitted, waiting for completion...\n";
    
    // Give some time for tasks to complete
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    std::cout << "Test completed successfully!\n";
    
    return 0;
}