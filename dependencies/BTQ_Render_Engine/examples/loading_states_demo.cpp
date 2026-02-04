/**
 * Example usage of LoadingStateManager in a UI context
 * This demonstrates how to use the loading states functionality in a real UI scenario
 */

#include "../include/ui/loading_states.hpp"
#include "../include/task_scheduler.hpp"
#include <imgui.h>
#include <thread>
#include <chrono>

namespace btq {
namespace ui {

class LoadingStateDemo {
public:
    LoadingStateDemo() : demo_operation_("None"), demo_progress_(0.0f) {}
    
    void render() {
        // Create a button to trigger a long-running operation
        if (ImGui::Button("Start Long Operation")) {
            // Start a long-running operation with loading indicator
            loading_manager_.executeAsyncWithLoading("Processing data...", [this]() {
                // Simulate a long-running operation
                for (int i = 0; i <= 100; ++i) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    
                    // Update progress (this would normally happen through a callback mechanism)
                    // In a real application, you would have a way to update progress from the worker thread
                    if (i % 10 == 0) {  // Update every 10%
                        loading_manager_.updateProgress(i / 100.0f, "Processing step " + std::to_string(i));
                    }
                }
            });
        }
        
        // Show current operation and progress
        ImGui::Text("Current operation: %s", loading_manager_.getCurrentOperation().c_str());
        ImGui::Text("Progress: %.1f%%", loading_manager_.getProgress() * 100.0f);
        ImGui::Text("Loading: %s", loading_manager_.isLoading() ? "Yes" : "No");
        
        // Render the loading overlay if loading is in progress
        loading_manager_.renderLoadingOverlay();
    }

private:
    LoadingStateManager loading_manager_;
    std::string demo_operation_;
    float demo_progress_;
};

} // namespace ui
} // namespace btq