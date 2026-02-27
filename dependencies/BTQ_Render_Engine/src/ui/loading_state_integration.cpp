#include "../../include/ui/loading_states.hpp"
#include "../../include/task_scheduler.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>

namespace btq {
namespace ui {

/**
 * @brief Example integration of LoadingStateManager with TaskScheduler
 * 
 * This demonstrates how to use loading states with the existing task scheduler
 * to show progress during long-running operations.
 */
class LoadingStateIntegration {
public:
    /**
     * @brief Execute a calculation with loading state visualization
     * @param scheduler Reference to the task scheduler
     * @param loading_manager Reference to the loading state manager
     */
    static void executeCalculationWithLoading(TaskScheduler& scheduler, LoadingStateManager& loading_manager) {
        // Example: Calculate SMA with loading indicator
        std::vector<double> prices = generateSamplePrices(10000);
        
        loading_manager.startLoading("Calculating SMA...", 0.0f);
        
        // Execute calculation in background with progress updates
        auto future = scheduler.calculate_sma_async(prices, 20);
        
        // Update progress periodically while calculation runs
        float progress = 0.0f;
        while (progress < 1.0f) {
            // Simulate progress (in a real scenario, this would come from the actual calculation)
            progress += 0.01f;
            loading_manager.updateProgress(progress, "Calculating SMA...");
            
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            
            if (future.wait_for(std::chrono::milliseconds(1)) == std::future_status::ready) {
                break; // Calculation finished
            }
        }
        
        // Get result and finish loading
        try {
            auto result = future.get();
            loading_manager.finishLoading();
            std::cout << "SMA calculation completed with " << result.size() << " results" << std::endl;
        } catch (const std::exception& e) {
            loading_manager.stopLoading();
            std::cerr << "Error in SMA calculation: " << e.what() << std::endl;
        }
    }
    
    /**
     * @brief Execute multiple calculations with loading state
     * @param scheduler Reference to the task scheduler
     * @param loading_manager Reference to the loading state manager
     */
    static void executeBatchCalculationsWithLoading(TaskScheduler& scheduler, LoadingStateManager& loading_manager) {
        std::vector<std::pair<std::string, int>> indicators = {
            {"SMA", 10},
            {"EMA", 12},
            {"RSI", 14},
            {"SMA", 20},
            {"EMA", 26}
        };
        
        loading_manager.startLoading("Initializing batch calculations...", 0.0f);
        
        // Calculate total progress steps
        float total_steps = static_cast<float>(indicators.size());
        float current_step = 0.0f;
        
        // Generate sample data
        std::vector<double> prices = generateSamplePrices(5000);
        
        for (const auto& indicator : indicators) {
            std::string operation = "Calculating " + indicator.first + " with period " + std::to_string(indicator.second);
            loading_manager.updateProgress(current_step / total_steps, operation);
            
            // Perform the calculation based on indicator type
            if (indicator.first == "SMA") {
                auto future = scheduler.calculate_sma_async(prices, indicator.second);
                future.wait(); // Wait for completion
                auto result = future.get();
                std::cout << indicator.first << " calculation completed with " << result.size() << " results" << std::endl;
            } else if (indicator.first == "EMA") {
                auto future = scheduler.calculate_ema_async(prices, indicator.second);
                future.wait(); // Wait for completion
                auto result = future.get();
                std::cout << indicator.first << " calculation completed with " << result.size() << " results" << std::endl;
            } else if (indicator.first == "RSI") {
                auto future = scheduler.calculate_rsi_async(prices, indicator.second);
                future.wait(); // Wait for completion
                auto result = future.get();
                std::cout << indicator.first << " calculation completed with " << result.size() << " results" << std::endl;
            }
            
            current_step += 1.0f;
            float progress = current_step / total_steps;
            loading_manager.updateProgress(progress, "Completed: " + indicator.first);
        }
        
        loading_manager.finishLoading();
        std::cout << "All batch calculations completed" << std::endl;
    }
    
    /**
     * @brief Execute async calculation with loading state
     * @param scheduler Reference to the task scheduler
     * @param loading_manager Reference to the loading state manager
     */
    static std::future<void> executeAsyncCalculationWithLoading(TaskScheduler& scheduler, LoadingStateManager& loading_manager) {
        return std::async(std::launch::async, [&scheduler, &loading_manager]() {
            executeCalculationWithLoading(scheduler, loading_manager);
        });
    }

private:
    /**
     * @brief Generate sample price data for testing
     * @param size Number of price points to generate
     * @return Vector of sample prices
     */
    static std::vector<double> generateSamplePrices(size_t size) {
        std::vector<double> prices;
        prices.reserve(size);
        
        double price = 100.0;
        for (size_t i = 0; i < size; ++i) {
            // Simple random walk simulation
            double change = (static_cast<double>(rand()) / RAND_MAX - 0.5) * 0.1;
            price += change;
            prices.push_back(price);
        }
        
        return prices;
    }
};

} // namespace ui
} // namespace btq