/**
 * Debug Overlay Implementation
 *
 * Shows performance metrics, frame rate, memory usage, and active features
 */

#include "performance/debug_overlay.hpp"
#include "../include/performance_monitor.hpp"
#include "../include/performance/memory_tracker.hpp"
#include "../src/imgui/imgui.h"
#include <sstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <fstream>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#elif __linux__
#include <sys/sysinfo.h>
#include <unistd.h>
#include <fstream>
#else
#include <mach/mach.h>
#include <mach/host_info.h>
#include <mach/mach_host.h>
#endif

namespace BTQuant {

DebugOverlay::DebugOverlay()
    : visible_(false),
      position_x_(10.0f),
      position_y_(10.0f),
      window_width_(350.0f),
      window_height_(200.0f),
      refresh_rate_(60.0f), // Update 60 times per second
      last_cpu_time_(std::chrono::high_resolution_clock::now()),
      last_cpu_usage_(0.0),
      last_process_time_(0),
      last_system_time_(0) {
    // Initialize CPU usage tracking
    update_cpu_usage();
}

void DebugOverlay::toggle_visibility() {
    visible_ = !visible_;
}

void DebugOverlay::set_visible(bool visible) {
    visible_ = visible;
}

bool DebugOverlay::is_visible() const {
    return visible_;
}

void DebugOverlay::update_position(float x, float y) {
    position_x_ = x;
    position_y_ = y;
}

void DebugOverlay::set_refresh_rate(float hz) {
    refresh_rate_ = hz;
}

float DebugOverlay::get_refresh_rate() const {
    return refresh_rate_;
}

double DebugOverlay::get_cpu_usage() {
    return update_cpu_usage();
}

double DebugOverlay::update_cpu_usage() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_cpu_time_).count();

    if (duration < 100) { // Only update every 100ms to avoid noise
        return last_cpu_usage_;
    }

    #ifdef _WIN32
        FILETIME creation_time, exit_time, kernel_time, user_time;
        GetProcessTimes(GetCurrentProcess(), &creation_time, &exit_time, &kernel_time, &user_time);

        ULARGE_INTEGER kernel_time_int, user_time_int;
        kernel_time_int.LowPart = kernel_time.dwLowDateTime;
        kernel_time_int.HighPart = kernel_time.dwHighDateTime;
        user_time_int.LowPart = user_time.dwLowDateTime;
        user_time_int.HighPart = user_time.dwHighDateTime;

        ULONGLONG total_time = kernel_time_int.QuadPart + user_time_int.QuadPart;

        SYSTEM_INFO sys_info;
        GetSystemInfo(&sys_info);
        int num_cores = sys_info.dwNumberOfProcessors;

        last_cpu_usage_ = double(total_time - last_process_time_) / double(duration * 10000 * num_cores);
        last_process_time_ = total_time;

    #elif __linux__
        std::ifstream stat_stream("/proc/self/stat", std::ios_base::in);
        std::string pid, comm, state, ppid, pgrp, session, tty_nr;
        std::string tpgid, flags, minflt, cminflt, majflt, cmajflt;
        std::string utime, stime, cutime, cstime, priority, nice;
        std::string O, itrealvalue, starttime;
        unsigned long vsize;
        long rss;

        stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
                    >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
                    >> utime >> stime >> cutime >> cstime >> priority >> nice
                    >> O >> itrealvalue >> starttime >> vsize >> rss;

        unsigned long long process_time = std::stoull(utime) + std::stoull(stime);

        // Get system CPU time
        std::ifstream cpu_stat("/proc/stat");
        std::string line;
        std::getline(cpu_stat, line);
        cpu_stat.close();

        std::istringstream iss(line);
        std::string cpu_label;
        unsigned long long user, nice_val, system, idle, iowait, irq, softirq, steal, guest, guest_nice;
        iss >> cpu_label >> user >> nice_val >> system >> idle >> iowait >> irq >> softirq >> steal >> guest >> guest_nice;

        unsigned long long total_time_sys = user + nice_val + system + idle + iowait + irq + softirq + steal;

        if (last_process_time_ != 0) {
            unsigned long long process_delta = process_time - last_process_time_;
            unsigned long long system_delta = total_time_sys - last_system_time_;

            if (system_delta > 0) {
                last_cpu_usage_ = 100.0 * process_delta / system_delta;
            }
        }

        last_process_time_ = process_time;
        last_system_time_ = total_time_sys;

    #else
        // For macOS and other systems, return 0 for now
        last_cpu_usage_ = 0.0;
    #endif

    last_cpu_time_ = now;
    return last_cpu_usage_;
}

double DebugOverlay::get_memory_usage_mb() {
    #ifdef _WIN32
        PROCESS_MEMORY_COUNTERS pmc;
        GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
        return static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
    #elif __linux__
        std::ifstream statm("/proc/self/status");
        std::string line;
        while (std::getline(statm, line)) {
            if (line.substr(0, 6) == "VmRSS:") {
                std::istringstream iss(line);
                std::string key;
                size_t value;
                iss >> key >> value;
                return static_cast<double>(value) / 1024.0; // Value is in KB
            }
        }
        return 0.0;
    #else
        // For macOS and other systems, return 0 for now
        return 0.0;
    #endif
}

void DebugOverlay::render() {
    if (!visible_) {
        return;
    }

    // Set the position for the overlay window
    ImGui::SetNextWindowPos(ImVec2(position_x_, position_y_), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(window_width_, window_height_), ImGuiCond_Always);

    // Create an always-topmost, borderless window for the debug overlay
    ImGui::Begin("Performance Debug Overlay",
                 nullptr,
                 ImGuiWindowFlags_NoTitleBar |
                 ImGuiWindowFlags_NoResize |
                 ImGuiWindowFlags_NoMove |
                 ImGuiWindowFlags_NoScrollbar |
                 ImGuiWindowFlags_NoScrollWithMouse |
                 ImGuiWindowFlags_NoCollapse |
                 ImGuiWindowFlags_AlwaysAutoResize |
                 ImGuiWindowFlags_NoSavedSettings |
                 ImGuiWindowFlags_NoFocusOnAppearing |
                 ImGuiWindowFlags_NoNav);

    // Performance Metrics Section
    ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "=== PERFORMANCE METRICS ===");

    // Frame Rate Information
    double current_fps = g_performance_monitor.get_fps();
    double avg_fps = g_performance_monitor.get_avg_fps(60);
    double min_fps = g_performance_monitor.get_min_fps();
    double max_fps = g_performance_monitor.get_max_fps();

    // Color code FPS based on performance
    ImVec4 fps_color = current_fps > 50.0 ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f) :  // Green for good FPS
                       current_fps > 30.0 ? ImVec4(1.0f, 1.0f, 0.0f, 1.0f) :  // Yellow for moderate FPS
                                            ImVec4(1.0f, 0.0f, 0.0f, 1.0f);    // Red for low FPS

    ImGui::TextColored(fps_color, "FPS: %.1f (Avg: %.1f)", current_fps, avg_fps);
    ImGui::Text("Min: %.1f | Max: %.1f", min_fps, max_fps);

    // Frame Time Information
    double current_frame_time = g_performance_monitor.get_frame_time_ms();
    double avg_frame_time = g_performance_monitor.get_avg_frame_time_ms(60);
    double min_frame_time = g_performance_monitor.get_min_frame_time();
    double max_frame_time = g_performance_monitor.get_max_frame_time();

    // Color code frame time based on performance
    ImVec4 frametime_color = current_frame_time < 16.67 ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f) :  // Green for <60fps equivalent
                             current_frame_time < 33.33 ? ImVec4(1.0f, 1.0f, 0.0f, 1.0f) :  // Yellow for <30fps equivalent
                                                          ImVec4(1.0f, 0.0f, 0.0f, 1.0f);  // Red for <30fps equivalent

    ImGui::TextColored(frametime_color, "Frame Time: %.2f ms (Avg: %.2f ms)", current_frame_time, avg_frame_time);
    ImGui::Text("Min: %.2f ms | Max: %.2f ms", min_frame_time, max_frame_time);

    ImGui::Separator();

    // CPU and Memory Usage Information
    ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "=== SYSTEM RESOURCES ===");

    // CPU Usage
    double cpu_usage = get_cpu_usage();
    ImVec4 cpu_color = cpu_usage < 50.0 ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f) :  // Green for low usage
                       cpu_usage < 80.0 ? ImVec4(1.0f, 1.0f, 0.0f, 1.0f) :  // Yellow for moderate usage
                                          ImVec4(1.0f, 0.0f, 0.0f, 1.0f);    // Red for high usage
    ImGui::TextColored(cpu_color, "CPU: %.1f%%", cpu_usage);

    // Memory Usage Information
    double memory_mb = get_memory_usage_mb();
    size_t used_memory = g_performance_monitor.get_used_memory_bytes();
    size_t total_memory = g_performance_monitor.get_total_memory_bytes();
    double memory_percent = g_performance_monitor.get_memory_usage_percent();

    // Color code memory usage based on percentage
    ImVec4 memory_color = memory_percent < 50.0 ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f) :  // Green for low usage
                          memory_percent < 80.0 ? ImVec4(1.0f, 1.0f, 0.0f, 1.0f) :  // Yellow for moderate usage
                                                  ImVec4(1.0f, 0.0f, 0.0f, 1.0f);  // Red for high usage

    ImGui::TextColored(memory_color, "Memory: %.1f%% (%.1f MB)", memory_percent, memory_mb);
    ImGui::Text("Used: %s", format_bytes(used_memory).c_str());
    ImGui::Text("Total: %s", format_bytes(total_memory).c_str());

    // Get memory usage from memory tracker as well
    auto& mem_tracker = btq::performance::getGlobalMemoryTracker();
    size_t current_mem_usage = mem_tracker.getCurrentMemoryUsage();
    size_t peak_mem_usage = mem_tracker.getPeakMemoryUsage();
    double growth_rate = mem_tracker.getAverageMemoryGrowthRate();

    ImGui::Text("Process Mem: %s", format_bytes(current_mem_usage).c_str());
    ImGui::Text("Peak Usage: %s", format_bytes(peak_mem_usage).c_str());
    ImGui::Text("Growth Rate: %s/s", format_bytes(static_cast<size_t>(growth_rate)).c_str());

    ImGui::Separator();

    // Active Features/Components Information
    ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "=== ACTIVE FEATURES ===");

    // Show active panels/components count
    ImGui::Text("Active Panels: %zu", active_panels_count_);
    ImGui::Text("Active Indicators: %zu", active_indicators_count_);
    ImGui::Text("Active Alerts: %zu", active_alerts_count_);

    // Data processing metrics
    size_t data_processed = g_performance_monitor.get_data_processed_count();
    size_t indicators_calculated = g_performance_monitor.get_indicators_calculated_count();
    double data_processing_time = g_performance_monitor.get_data_processing_time_ms();

    ImGui::Text("Data Processed: %zu", data_processed);
    ImGui::Text("Indicators Calc: %zu", indicators_calculated);
    ImGui::Text("Processing Time: %.2f ms", data_processing_time);

    // Renderer Stats
    ImGui::Text("Frames Rendered: %s", format_large_number(frames_rendered_).c_str());
    ImGui::Text("LOB Updates: %s", format_large_number(lob_updates_).c_str());
    ImGui::Text("Trade Updates: %s", format_large_number(trade_updates_).c_str());
    ImGui::Text("Footprint Cells: %s", format_large_number(footprint_cells_rendered_).c_str());

    // Close the window
    ImGui::End();
}

std::string DebugOverlay::format_large_number(size_t num) const {
    if (num >= 1000000) {
        return std::to_string(num / 1000000) + "M";
    } else if (num >= 1000) {
        return std::to_string(num / 1000) + "K";
    } else {
        return std::to_string(num);
    }
}

void DebugOverlay::set_active_panels_count(size_t count) {
    active_panels_count_ = count;
}

void DebugOverlay::set_active_indicators_count(size_t count) {
    active_indicators_count_ = count;
}

void DebugOverlay::set_active_alerts_count(size_t count) {
    active_alerts_count_ = count;
}

void DebugOverlay::set_renderer_stats(uint32_t frames_rendered, uint32_t lob_updates,
                                   uint32_t trade_updates, uint32_t footprint_cells_rendered) {
    frames_rendered_ = frames_rendered;
    lob_updates_ = lob_updates;
    trade_updates_ = trade_updates;
    footprint_cells_rendered_ = footprint_cells_rendered;
}

std::string DebugOverlay::format_bytes(size_t bytes) const {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_index = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && unit_index < 4) {
        size /= 1024.0;
        unit_index++;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit_index];
    return oss.str();
}

// Global debug overlay instance
DebugOverlay g_debug_overlay;

}  // namespace BTQuant