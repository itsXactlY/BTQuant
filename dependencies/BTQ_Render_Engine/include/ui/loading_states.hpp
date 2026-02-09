#ifndef BTQ_RENDER_ENGINE_LOADING_STATES_HPP
#define BTQ_RENDER_ENGINE_LOADING_STATES_HPP

#include <string>
#include <functional>
#include <future>
#include <atomic>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <cmath>

#include <imgui.h>
#include "../include/threading/atomic_signal.hpp"

namespace btq {
namespace ui {

/**
 * @brief Manages loading states and progress indicators in the UI
 * * This class provides functionality to show loading states, progress bars,
 * and prevent UI freezing during long operations. It's thread-safe and can
 * be used to wrap long-running operations with loading indicators.
 */
class LoadingStateManager {
public:
    /**
     * @brief Construct a new Loading State Manager object
     */
    LoadingStateManager();

    /**
     * @brief Destroy the Loading State Manager object
     */
    ~LoadingStateManager();

    /**
     * @brief Start a loading operation
     * @param operation_name Name of the operation being performed
     * @param initial_progress Initial progress value (0.0 to 1.0)
     */
    void startLoading(const std::string& operation_name, float initial_progress = 0.0f);

    /**
     * @brief Update the progress of the current loading operation
     * @param new_progress New progress value (0.0 to 1.0)
     * @param status_message Optional status message to display
     */
    void updateProgress(float new_progress, const std::string& status_message = "");

    /**
     * @brief Incrementally update the progress
     * @param increment Amount to add to current progress
     * @param status_message Optional status message to display
     */
    void updateProgressIncremental(float increment, const std::string& status_message = "");

    /**
     * @brief Finish the current loading operation
     */
    void finishLoading();

    /**
     * @brief Stop the current loading operation (without completing it)
     */
    void stopLoading();

    /**
     * @brief Check if a loading operation is currently in progress
     * @return true if loading, false otherwise
     */
    bool isLoading() const;

    /**
     * @brief Get the current progress value
     * @return Current progress (0.0 to 1.0)
     */
    float getProgress() const;

    /**
     * @brief Get the name/message of the current operation
     * @return Current operation name/message
     */
    std::string getCurrentOperation() const;

    /**
     * @brief Get the elapsed time since the operation started
     * @return Elapsed time in seconds
     */
    double getElapsedTime() const;

    /**
     * @brief Wait until the current loading operation completes
     */
    void waitForCompletion();

    /**
     * @brief Render the loading overlay UI
     * Displays a modal window with progress information
     */
    void renderLoadingOverlay();

    /**
     * @brief Render a custom loading spinner
     * @param label Label to display next to the spinner
     * @param radius Radius of the spinner
     * @param segments Number of segments in the spinner
     */
    void renderLoadingSpinner(const char* label = nullptr, float radius = 8.0f, int segments = 8);

    /**
     * @brief Render a progress bar with optimization
     * @param fraction Progress fraction (0.0 to 1.0, negative for indeterminate)
     * @param size_arg Size of the progress bar
     * @param overlay Text to overlay on the progress bar
     */
    void renderProgressBar(float fraction, const ImVec2& size_arg = ImVec2(-FLT_MIN, 0), const char* overlay = nullptr);

    /**
     * @brief Execute a function with loading state management
     * This wraps a long-running operation with loading indicators and prevents UI freezing
     * @param operation_name Name of the operation to display during loading
     * @param task_func Function to execute
     */
    void executeWithLoading(const std::string& operation_name, std::function<void()> task_func);

    /**
     * @brief Execute a function asynchronously with loading state management
     * This returns immediately and executes the task in the background
     * @param operation_name Name of the operation to display during loading
     * @param task_func Function to execute
     * @return Future that can be used to wait for completion
     */
    std::future<void> executeAsyncWithLoading(const std::string& operation_name, std::function<void()> task_func);

private:
    mutable std::mutex mutex_;
    btq::threading::AtomicBooleanSignal loading_complete_signal_;
    std::string current_operation_;            ///< Current operation name<message>
    float progress_;                           ///< Current progress (0.0 to 1.0)
    std::atomic<bool> is_loading_;             ///< Flag indicating if loading is in progress
    std::chrono::steady_clock::time_point operation_start_time_;  ///< Time when operation started
};

} // namespace ui
} // namespace btq

#endif // BTQ_RENDER_ENGINE_LOADING_STATES_HPP