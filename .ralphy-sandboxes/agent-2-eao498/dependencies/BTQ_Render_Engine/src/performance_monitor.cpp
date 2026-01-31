/**
 * Performance Monitor Implementation
 * 
 * Tracks and reports performance metrics for the trading terminal
 */

#include "performance_monitor.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <iostream>
#include <iomanip>

namespace BTQ::Render {

PerformanceMonitor::PerformanceMonitor()
    : fps_history_(HISTORY_SIZE, 0.0)
    , frame_time_history_(HISTORY_SIZE, 0.0)
    , render_time_history_(HISTORY_SIZE, 0.0)
    , update_time_history_(HISTORY_SIZE, 0.0)
    , data_processing_time_history_(HISTORY_SIZE, 0.0)
    , memory_usage_history_(HISTORY_SIZE, 0.0)
    , gpu_usage_history_(HISTORY_SIZE, 0.0)
    , current_fps_(0.0)
    , current_frame_time_(0.0)
    , current_render_time_(0.0)
    , current_update_time_(0.0)
    , current_data_processing_time_(0.0)
    , current_memory_usage_(0.0)
    , current_gpu_usage_(0.0)
    , frame_count_(0)
    , history_index_(0)
    , last_update_time_(std::chrono::high_resolution_clock::now())
{
}

PerformanceMonitor::~PerformanceMonitor() = default;

void PerformanceMonitor::update_fps(double fps) {
    current_fps_ = fps;
    fps_history_[history_index_] = fps;
}

void PerformanceMonitor::update_frame_time(double frame_time_ms) {
    current_frame_time_ = frame_time_ms;
    frame_time_history_[history_index_] = frame_time_ms;
}

void PerformanceMonitor::update_render_time(double render_time_ms) {
    current_render_time_ = render_time_ms;
    render_time_history_[history_index_] = render_time_ms;
}

void PerformanceMonitor::update_update_time(double update_time_ms) {
    current_update_time_ = update_time_ms;
    update_time_history_[history_index_] = update_time_ms;
}

void PerformanceMonitor::update_data_processing_time(double data_processing_time_ms) {
    current_data_processing_time_ = data_processing_time_ms;
    data_processing_time_history_[history_index_] = data_processing_time_ms;
}

void PerformanceMonitor::update_memory_usage(double memory_usage_mb) {
    current_memory_usage_ = memory_usage_mb;
    memory_usage_history_[history_index_] = memory_usage_mb;
}

void PerformanceMonitor::update_gpu_usage(double gpu_usage_percent) {
    current_gpu_usage_ = gpu_usage_percent;
    gpu_usage_history_[history_index_] = gpu_usage_percent;
}

void PerformanceMonitor::tick() {
    frame_count_++;
    history_index_ = (history_index_ + 1) % HISTORY_SIZE;
    
    // Update metrics every second
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration<double>(now - last_update_time_).count();
    
    if (elapsed >= 1.0) {
        update_aggregated_metrics();
        last_update_time_ = now;
    }
}

PerformanceMetrics PerformanceMonitor::get_metrics() const {
    PerformanceMetrics metrics;
    metrics.fps = current_fps_;
    metrics.frame_time_ms = current_frame_time_;
    metrics.render_time_ms = current_render_time_;
    metrics.update_time_ms = current_update_time_;
    metrics.data_processing_time_ms = current_data_processing_time_;
    metrics.memory_usage_mb = current_memory_usage_;
    metrics.gpu_usage_percent = current_gpu_usage_;
    metrics.frame_count = frame_count_;
    
    return metrics;
}

AggregatedMetrics PerformanceMonitor::get_aggregated_metrics() const {
    return aggregated_metrics_;
}

void PerformanceMonitor::reset() {
    std::fill(fps_history_.begin(), fps_history_.end(), 0.0);
    std::fill(frame_time_history_.begin(), frame_time_history_.end(), 0.0);
    std::fill(render_time_history_.begin(), render_time_history_.end(), 0.0);
    std::fill(update_time_history_.begin(), update_time_history_.end(), 0.0);
    std::fill(data_processing_time_history_.begin(), data_processing_time_history_.end(), 0.0);
    std::fill(memory_usage_history_.begin(), memory_usage_history_.end(), 0.0);
    std::fill(gpu_usage_history_.begin(), gpu_usage_history_.end(), 0.0);
    
    current_fps_ = 0.0;
    current_frame_time_ = 0.0;
    current_render_time_ = 0.0;
    current_update_time_ = 0.0;
    current_data_processing_time_ = 0.0;
    current_memory_usage_ = 0.0;
    current_gpu_usage_ = 0.0;
    
    frame_count_ = 0;
    history_index_ = 0;
    
    aggregated_metrics_ = AggregatedMetrics{};
}

void PerformanceMonitor::print_report() const {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Performance Report" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::cout << "\nCurrent Metrics:" << std::endl;
    std::cout << "  FPS: " << std::fixed << std::setprecision(1) << current_fps_ << std::endl;
    std::cout << "  Frame Time: " << std::fixed << std::setprecision(2) << current_frame_time_ << " ms" << std::endl;
    std::cout << "  Render Time: " << std::fixed << std::setprecision(2) << current_render_time_ << " ms" << std::endl;
    std::cout << "  Update Time: " << std::fixed << std::setprecision(2) << current_update_time_ << " ms" << std::endl;
    std::cout << "  Data Processing: " << std::fixed << std::setprecision(2) << current_data_processing_time_ << " ms" << std::endl;
    std::cout << "  Memory Usage: " << std::fixed << std::setprecision(1) << current_memory_usage_ << " MB" << std::endl;
    std::cout << "  GPU Usage: " << std::fixed << std::setprecision(1) << current_gpu_usage_ << "%" << std::endl;
    
    std::cout << "\nAggregated Metrics (Last " << HISTORY_SIZE << " frames):" << std::endl;
    std::cout << "  Average FPS: " << std::fixed << std::setprecision(1) << aggregated_metrics_.avg_fps << std::endl;
    std::cout << "  Min FPS: " << std::fixed << std::setprecision(1) << aggregated_metrics_.min_fps << std::endl;
    std::cout << "  Max FPS: " << std::fixed << std::setprecision(1) << aggregated_metrics_.max_fps << std::endl;
    std::cout << "  Average Frame Time: " << std::fixed << std::setprecision(2) << aggregated_metrics_.avg_frame_time << " ms" << std::endl;
    std::cout << "  Min Frame Time: " << std::fixed << std::setprecision(2) << aggregated_metrics_.min_frame_time << " ms" << std::endl;
    std::cout << "  Max Frame Time: " << std::fixed << std::setprecision(2) << aggregated_metrics_.max_frame_time << " ms" << std::endl;
    std::cout << "  Frame Time StdDev: " << std::fixed << std::setprecision(2) << aggregated_metrics_.stddev_frame_time << " ms" << std::endl;
    
    std::cout << "\nTotal Frames: " << frame_count_ << std::endl;
    std::cout << "========================================\n" << std::endl;
}

void PerformanceMonitor::update_aggregated_metrics() {
    // Calculate FPS statistics
    double fps_sum = std::accumulate(fps_history_.begin(), fps_history_.end(), 0.0);
    double fps_avg = fps_sum / HISTORY_SIZE;
    double fps_min = *std::min_element(fps_history_.begin(), fps_history_.end());
    double fps_max = *std::max_element(fps_history_.begin(), fps_history_.end());
    
    // Calculate frame time statistics
    double frame_time_sum = std::accumulate(frame_time_history_.begin(), frame_time_history_.end(), 0.0);
    double frame_time_avg = frame_time_sum / HISTORY_SIZE;
    double frame_time_min = *std::min_element(frame_time_history_.begin(), frame_time_history_.end());
    double frame_time_max = *std::max_element(frame_time_history_.begin(), frame_time_history_.end());
    
    // Calculate standard deviation
    double variance = 0.0;
    for (double ft : frame_time_history_) {
        variance += (ft - frame_time_avg) * (ft - frame_time_avg);
    }
    variance /= HISTORY_SIZE;
    double stddev = std::sqrt(variance);
    
    // Calculate render time statistics
    double render_time_sum = std::accumulate(render_time_history_.begin(), render_time_history_.end(), 0.0);
    double render_time_avg = render_time_sum / HISTORY_SIZE;
    
    // Calculate update time statistics
    double update_time_sum = std::accumulate(update_time_history_.begin(), update_time_history_.end(), 0.0);
    double update_time_avg = update_time_sum / HISTORY_SIZE;
    
    // Calculate data processing time statistics
    double data_processing_time_sum = std::accumulate(data_processing_time_history_.begin(), data_processing_time_history_.end(), 0.0);
    double data_processing_time_avg = data_processing_time_sum / HISTORY_SIZE;
    
    // Calculate memory usage statistics
    double memory_usage_sum = std::accumulate(memory_usage_history_.begin(), memory_usage_history_.end(), 0.0);
    double memory_usage_avg = memory_usage_sum / HISTORY_SIZE;
    
    // Calculate GPU usage statistics
    double gpu_usage_sum = std::accumulate(gpu_usage_history_.begin(), gpu_usage_history_.end(), 0.0);
    double gpu_usage_avg = gpu_usage_sum / HISTORY_SIZE;
    
    // Update aggregated metrics
    aggregated_metrics_.avg_fps = fps_avg;
    aggregated_metrics_.min_fps = fps_min;
    aggregated_metrics_.max_fps = fps_max;
    aggregated_metrics_.avg_frame_time = frame_time_avg;
    aggregated_metrics_.min_frame_time = frame_time_min;
    aggregated_metrics_.max_frame_time = frame_time_max;
    aggregated_metrics_.stddev_frame_time = stddev;
    aggregated_metrics_.avg_render_time = render_time_avg;
    aggregated_metrics_.avg_update_time = update_time_avg;
    aggregated_metrics_.avg_data_processing_time = data_processing_time_avg;
    aggregated_metrics_.avg_memory_usage = memory_usage_avg;
    aggregated_metrics_.avg_gpu_usage = gpu_usage_avg;
}

} // namespace BTQ::Render
