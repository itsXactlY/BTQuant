#include "../include/ui/loading_states.hpp"
#include "../include/task_scheduler.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <future>

int main() {
    std::cout << "Testing Loading States Implementation..." << std::endl;
    
    // Create loading state manager
    btq::ui::LoadingStateManager loading_manager;
    
    // Test 1: Basic loading functionality
    std::cout << "\nTest 1: Basic loading functionality" << std::endl;
    loading_manager.startLoading("Initializing test...", 0.0f);
    
    if (loading_manager.isLoading()) {
        std::cout << "✓ Loading state started correctly" << std::endl;
    } else {
        std::cout << "✗ Failed to start loading state" << std::endl;
    }
    
    // Update progress
    loading_manager.updateProgress(0.5f, "Halfway through test...");
    
    if (loading_manager.getProgress() == 0.5f) {
        std::cout << "✓ Progress updated correctly" << std::endl;
    } else {
        std::cout << "✗ Progress not updated correctly" << std::endl;
    }
    
    // Finish loading
    loading_manager.finishLoading();
    
    if (!loading_manager.isLoading() && loading_manager.getProgress() == 1.0f) {
        std::cout << "✓ Loading finished correctly" << std::endl;
    } else {
        std::cout << "✗ Loading not finished correctly" << std::endl;
    }
    
    // Test 2: Execute with loading
    std::cout << "\nTest 2: Execute with loading functionality" << std::endl;
    
    auto task_func = []() {
        // Simulate a long-running task
        for (int i = 0; i <= 10; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    };
    
    loading_manager.executeWithLoading("Running simulated task...", task_func);
    
    if (!loading_manager.isLoading()) {
        std::cout << "✓ Task executed with loading state correctly" << std::endl;
    } else {
        std::cout << "✗ Task execution with loading failed" << std::endl;
    }
    
    // Test 3: Async execution with loading
    std::cout << "\nTest 3: Async execution with loading" << std::endl;
    
    auto future = loading_manager.executeAsyncWithLoading("Async task...", task_func);
    
    // Wait for completion
    future.wait();
    
    if (future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        std::cout << "✓ Async task executed with loading state correctly" << std::endl;
    } else {
        std::cout << "✗ Async task execution with loading failed" << std::endl;
    }
    
    // Test 4: Integration with task scheduler
    std::cout << "\nTest 4: Integration with task scheduler" << std::endl;
    
    try {
        btq::TaskScheduler scheduler(4); // Use 4 threads
        
        // Test the integration
        std::vector<double> prices = {100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0};
        
        loading_manager.startLoading("Calculating SMA...", 0.0f);
        
        auto future = scheduler.calculate_sma_async(prices, 3);
        
        // Simulate progress updates
        float progress = 0.0f;
        while (progress < 1.0f) {
            progress += 0.1f;
            loading_manager.updateProgress(progress, "Calculating SMA...");
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            
            if (future.wait_for(std::chrono::milliseconds(1)) == std::future_status::ready) {
                break;
            }
        }
        
        auto result = future.get();
        loading_manager.finishLoading();
        
        std::cout << "✓ Task scheduler integration works, SMA result size: " << result.size() << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "✗ Task scheduler integration failed: " << e.what() << std::endl;
    }
    
    std::cout << "\nAll tests completed!" << std::endl;
    
    return 0;
}