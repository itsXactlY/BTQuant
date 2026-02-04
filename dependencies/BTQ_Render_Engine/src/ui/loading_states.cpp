#include "../../include/ui/loading_states.hpp"
#include <imgui.h>
#include "../../include/threading/lockfree_queue.hpp"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <sstream>
#include <iomanip>

namespace btq {
namespace ui {

// Constructor
LoadingStateManager::LoadingStateManager() 
    : current_operation_("Idle") 
    , progress_(0.0f) 
    , is_loading_(false) 
    , operation_start_time_(std::chrono::steady_clock::now())
{
    // Initialize with default values
}

// Destructor
LoadingStateManager::~LoadingStateManager() {
    // Stop any ongoing operations
    stopLoading();
}

void LoadingStateManager::startLoading(const std::string& operation_name, float initial_progress) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    current_operation_ = operation_name;
    progress_ = initial_progress;
    is_loading_ = true;
    operation_start_time_ = std::chrono::steady_clock::now();
    
    // Notify any waiting threads
    cv_.notify_all();
}

void LoadingStateManager::updateProgress(float new_progress, const std::string& status_message) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    progress_ = std::clamp(new_progress, 0.0f, 1.0f);
    
    if (!status_message.empty()) {
        current_operation_ = status_message;
    }
    
    // Notify any waiting threads
    cv_.notify_all();
}

void LoadingStateManager::updateProgressIncremental(float increment, const std::string& status_message) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    progress_ = std::clamp(progress_ + increment, 0.0f, 1.0f);
    
    if (!status_message.empty()) {
        current_operation_ = status_message;
    }
    
    // Notify any waiting threads
    cv_.notify_all();
}

void LoadingStateManager::finishLoading() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    progress_ = 1.0f;
    is_loading_ = false;
    
    // Notify any waiting threads
    cv_.notify_all();
}

void LoadingStateManager::stopLoading() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    progress_ = 0.0f;
    is_loading_ = false;
    current_operation_ = "Stopped";
    
    // Notify any waiting threads
    cv_.notify_all();
}

bool LoadingStateManager::isLoading() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return is_loading_;
}

float LoadingStateManager::getProgress() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return progress_;
}

std::string LoadingStateManager::getCurrentOperation() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_operation_;
}

double LoadingStateManager::getElapsedTime() const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - operation_start_time_);
    return duration.count() / 1000.0; // Return seconds as double
}

void LoadingStateManager::waitForCompletion() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !is_loading_; });
}

void LoadingStateManager::renderLoadingOverlay() {
    if (!isLoading()) {
        return;
    }
    
    // Create a modal window for the loading overlay
    ImGui::SetNextWindowSize(ImVec2(300, 120), ImGuiCond_Always);
    ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x * 0.5f - 150, 
                                   ImGui::GetIO().DisplaySize.y * 0.5f - 60), 
                           ImGuiCond_Always);
    
    ImGui::Begin("Loading Overlay", nullptr, 
                 ImGuiWindowFlags_NoMove | 
                 ImGuiWindowFlags_NoResize | 
                 ImGuiWindowFlags_NoCollapse |
                 ImGuiWindowFlags_NoTitleBar |
                 ImGuiWindowFlags_NoSavedSettings |
                 ImGuiWindowFlags_NoScrollbar |
                 ImGuiWindowFlags_NoScrollWithMouse);
    
    // Center the content
    ImGui::SetCursorPosX((ImGui::GetWindowSize().x - ImGui::CalcTextSize(getCurrentOperation().c_str()).x) * 0.5f);
    ImGui::Text("%s", getCurrentOperation().c_str());
    
    // Show progress percentage
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1) << (getProgress() * 100.0f) << "%";
    std::string progress_text = ss.str();
    
    ImGui::SetCursorPosX((ImGui::GetWindowSize().x - ImGui::CalcTextSize(progress_text.c_str()).x) * 0.5f);
    ImGui::Text("%s", progress_text.c_str());
    
    // Progress bar
    float progress = getProgress();
    if (progress < 0.0f) {
        // Indeterminate progress bar animation
        float animated_progress = (std::fmod(static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count()) / 1000.0f, 1.0f));
        ImGui::ProgressBar(animated_progress, ImVec2(-1, 0), "Processing...");
    } else {
        // Determinate progress bar
        ImGui::ProgressBar(progress, ImVec2(-1, 0), "");
    }
    
    // Show elapsed time
    double elapsed = getElapsedTime();
    std::stringstream time_ss;
    time_ss << std::fixed << std::setprecision(1) << "Elapsed: " << elapsed << "s";
    ImGui::SetCursorPosX((ImGui::GetWindowSize().x - ImGui::CalcTextSize(time_ss.str().c_str()).x) * 0.5f);
    ImGui::Text("%s", time_ss.str().c_str());
    
    ImGui::End();
}

void LoadingStateManager::renderLoadingSpinner(const char* label, float radius, int segments) {
    // Simple spinner implementation
    auto* draw_list = ImGui::GetWindowDrawList();
    ImVec2 pos = ImGui::GetCursorScreenPos();
    
    // Calculate center position
    ImVec2 center(pos.x + radius, pos.y + radius);
    
    // Get time for animation
    float time = static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count()) / 1000.0f;
    
    // Draw spinner segments
    for (int i = 0; i < segments; i++) {
        float angle = (i * 2.0f * M_PI / segments) + time * 2.0f;
        ImVec2 segment_pos(center.x + cos(angle) * radius, center.y + sin(angle) * radius);
        
        // Fade out segments based on rotation
        float alpha = (sin(time * 5.0f + i * 0.5f) + 1.0f) / 2.0f;
        ImU32 color = ImGui::GetColorU32(ImVec4(0.8f, 0.8f, 0.8f, alpha));
        
        draw_list->AddCircleFilled(segment_pos, 2.0f, color);
    }
    
    // Advance cursor position
    ImGui::Dummy(ImVec2(radius * 2, radius * 2));
    
    if (label) {
        ImGui::SameLine();
        ImGui::Text("%s", label);
    }
}

void LoadingStateManager::renderProgressBar(float fraction, const ImVec2& size_arg, const char* overlay) {
    // Use the regular ImGui progress bar
    ImGui::ProgressBar(fraction, size_arg, overlay);
}

void LoadingStateManager::executeWithLoading(const std::string& operation_name, 
                                           std::function<void()> task_func) {
    startLoading(operation_name, 0.0f);
    
    // Execute the task in a separate thread to prevent UI freezing
    std::thread task_thread([this, task_func]() {
        try {
            task_func();
            finishLoading();
        } catch (...) {
            // On exception, stop loading and rethrow
            stopLoading();
            throw;
        }
    });
    
    // Wait for the task to complete
    waitForCompletion();
    
    // Join the thread
    if (task_thread.joinable()) {
        task_thread.join();
    }
}

std::future<void> LoadingStateManager::executeAsyncWithLoading(const std::string& operation_name,
                                                             std::function<void()> task_func) {
    return std::async(std::launch::async, [this, operation_name, task_func]() {
        executeWithLoading(operation_name, task_func);
    });
}

} // namespace ui
} // namespace btq