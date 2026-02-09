#include "../../include/ui/loading_states.hpp"
#include <iostream>

// --- FIX START: Include internal header for low-level drawing ---
#include <imgui_internal.h>
// --- FIX END ---

// Helper macro for thread safety
#define LOCK_GUARD std::lock_guard<std::mutex> lock(mutex_)

namespace btq {
namespace ui {

LoadingStateManager::LoadingStateManager()
    : progress_(0.0f), is_loading_(false) {
    // Initialize the signal to false (loading not complete)
    loading_complete_signal_.reset();
}

LoadingStateManager::~LoadingStateManager() {
    stopLoading();
}

void LoadingStateManager::startLoading(const std::string& operation_name, float initial_progress) {
    {
        LOCK_GUARD;
        current_operation_ = operation_name;
        progress_ = initial_progress;
        is_loading_ = true;
        operation_start_time_ = std::chrono::steady_clock::now();
    }
    // Reset the completion signal since loading has started again
    loading_complete_signal_.reset();
}

void LoadingStateManager::updateProgress(float new_progress, const std::string& status_message) {
    {
        LOCK_GUARD;
        progress_ = std::clamp(new_progress, 0.0f, 1.0f);
        if (!status_message.empty()) {
            current_operation_ = status_message;
        }
    }
}

void LoadingStateManager::updateProgressIncremental(float increment, const std::string& status_message) {
    {
        LOCK_GUARD;
        progress_ = std::clamp(progress_ + increment, 0.0f, 1.0f);
        if (!status_message.empty()) {
            current_operation_ = status_message;
        }
    }
}

void LoadingStateManager::finishLoading() {
    {
        LOCK_GUARD;
        progress_ = 1.0f;
        is_loading_ = false;
    }
    // Signal that loading is complete
    loading_complete_signal_.signal();
}

void LoadingStateManager::stopLoading() {
    {
        LOCK_GUARD;
        is_loading_ = false;
    }
    // Signal that loading is complete (stopped)
    loading_complete_signal_.signal();
}

bool LoadingStateManager::isLoading() const {
    // Atomic read, no lock needed for simple boolean check
    return is_loading_;
}

float LoadingStateManager::getProgress() const {
    LOCK_GUARD;
    return progress_;
}

std::string LoadingStateManager::getCurrentOperation() const {
    LOCK_GUARD;
    return current_operation_;
}

double LoadingStateManager::getElapsedTime() const {
    LOCK_GUARD;
    if (!is_loading_) return 0.0;
    
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - operation_start_time_);
    return duration.count() / 1000.0;
}

void LoadingStateManager::waitForCompletion() {
    // Optimization: check atomic flag first without lock
    if (!is_loading_) return;

    // Wait for the loading complete signal
    loading_complete_signal_.wait();
}

void LoadingStateManager::renderLoadingOverlay() {
    if (!is_loading_) return;

    // Use a fixed overlay window
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0, 0, 0, 0.5f)); // Semi-transparent black
    
    // Flags: No decorations, no inputs, on top of everything
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | 
                             ImGuiWindowFlags_NoResize | 
                             ImGuiWindowFlags_NoMove | 
                             ImGuiWindowFlags_NoScrollbar | 
                             ImGuiWindowFlags_NoInputs | 
                             ImGuiWindowFlags_NoSavedSettings | 
                             ImGuiWindowFlags_NoFocusOnAppearing | 
                             ImGuiWindowFlags_NoBringToFrontOnFocus;

    if (ImGui::Begin("##LoadingOverlay", nullptr, flags)) {
        ImVec2 center = ImVec2(ImGui::GetIO().DisplaySize.x * 0.5f, ImGui::GetIO().DisplaySize.y * 0.5f);
        
        // Render spinner
        renderLoadingSpinner("##Spinner", 30.0f, 12);
        
        // Render text
        std::string op_name;
        float prog;
        {
            LOCK_GUARD;
            op_name = current_operation_;
            prog = progress_;
        }
        
        // Calculate text size to center it
        ImVec2 text_size = ImGui::CalcTextSize(op_name.c_str());
        ImGui::SetCursorPos(ImVec2(center.x - text_size.x * 0.5f, center.y + 40.0f));
        ImGui::TextUnformatted(op_name.c_str());
        
        // Render progress bar
        float bar_width = 300.0f;
        ImGui::SetCursorPos(ImVec2(center.x - bar_width * 0.5f, center.y + 70.0f));
        renderProgressBar(prog, ImVec2(bar_width, 6.0f));
    }
    ImGui::End();
    ImGui::PopStyleColor();
}

void LoadingStateManager::renderLoadingSpinner(const char* label, float radius, int segments) {
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    if (window->SkipItems)
        return;

    ImGuiContext& g = *GImGui;
    const ImGuiStyle& style = g.Style;
    const ImGuiID id = window->GetID(label);

    ImVec2 pos = window->DC.CursorPos;
    // Center the spinner in the available space if label is hidden
    if (strncmp(label, "##", 2) == 0) {
        ImVec2 avail = ImGui::GetContentRegionAvail();
        pos.x += (avail.x - radius * 2) * 0.5f;
        pos.y += (avail.y - radius * 2) * 0.5f;
    }
    
    ImVec2 size(radius * 2, radius * 2);
    const ImRect bb(pos, ImVec2(pos.x + size.x, pos.y + size.y));
    ImGui::ItemSize(bb, style.FramePadding.y);
    if (!ImGui::ItemAdd(bb, id))
        return;

    // Render
    window->DrawList->PathClear();
    
    int start = (int)(g.Time * 10.0f) % segments; // Animation speed
    const float PI = 3.14159265358979323846f;
    const ImVec2 center = ImVec2(pos.x + radius, pos.y + radius);
    
    for (int i = 0; i < segments; i++) {
        const float a = (i * 2 * PI) / segments;
        const float r_inner = radius * 0.6f;
        const float r_outer = radius * 0.9f;
        
        window->DrawList->PathLineTo(ImVec2(center.x + cos(a) * r_inner, center.y + sin(a) * r_inner));
        window->DrawList->PathLineTo(ImVec2(center.x + cos(a) * r_outer, center.y + sin(a) * r_outer));
        
        ImU32 color = ImGui::GetColorU32(ImGuiCol_Text, 
            1.0f - (float)((i + start) % segments) / (float)segments);
            
        window->DrawList->PathStroke(color, 0, 2.0f); // Thickness
    }
}

void LoadingStateManager::renderProgressBar(float fraction, const ImVec2& size_arg, const char* overlay) {
    ImGui::ProgressBar(fraction, size_arg, overlay);
}

void LoadingStateManager::executeWithLoading(const std::string& operation_name, std::function<void()> task_func) {
    startLoading(operation_name, 0.0f);
    try {
        task_func();
    } catch (...) {
        stopLoading();
        throw;
    }
    finishLoading();
}

std::future<void> LoadingStateManager::executeAsyncWithLoading(const std::string& operation_name, std::function<void()> task_func) {
    startLoading(operation_name, 0.0f);
    
    return std::async(std::launch::async, [this, task_func]() {
        try {
            task_func();
        } catch (...) {
            this->stopLoading();
            throw;
        }
        this->finishLoading();
    });
}

} // namespace ui
} // namespace btq