#include "regression_detector.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <mutex>
#include <cmath>
#include <iomanip>

namespace BTQuant {

PerformanceRegressionDetector::PerformanceRegressionDetector()
    : regression_callback_(nullptr) {}

void PerformanceRegressionDetector::set_baseline(const std::string& metric_name, 
                                               double baseline_value, 
                                               double tolerance_percentage) {
    std::lock_guard<std::mutex> lock(baselines_mutex_);
    baselines_[metric_name] = PerformanceBaseline(metric_name, baseline_value, tolerance_percentage);
}

bool PerformanceRegressionDetector::check_regression(const std::string& metric_name, 
                                                   double current_value, 
                                                   RegressionAlert* alert) {
    std::lock_guard<std::mutex> lock(baselines_mutex_);
    
    auto it = baselines_.find(metric_name);
    if (it == baselines_.end()) {
        // No baseline exists for this metric
        return false;
    }
    
    const auto& baseline = it->second;
    
    // Calculate deviation percentage
    double deviation_percentage = ((current_value - baseline.baseline_value) / baseline.baseline_value) * 100.0;
    
    // Determine if this is a regression (performance degradation)
    bool is_regression = false;
    if (current_value > baseline.baseline_value) {
        // For metrics where higher values indicate worse performance (e.g., frame time)
        is_regression = (deviation_percentage > baseline.tolerance_percentage);
    } else {
        // For metrics where lower values indicate worse performance (e.g., FPS)
        is_regression = (std::abs(deviation_percentage) > baseline.tolerance_percentage && current_value < baseline.baseline_value);
    }
    
    if (is_regression) {
        RegressionAlert regression_alert(metric_name, current_value, baseline.baseline_value, true);
        
        // Add to recent alerts
        {
            std::lock_guard<std::mutex> lock(alerts_mutex_);
            recent_alerts_.push_back(regression_alert);
            
            // Keep only the most recent 100 alerts
            if (recent_alerts_.size() > 100) {
                recent_alerts_.erase(recent_alerts_.begin(), 
                                   recent_alerts_.begin() + (recent_alerts_.size() - 99));
            }
        }
        
        // Call the regression callback if set
        if (regression_callback_) {
            regression_callback_(regression_alert);
        }
        
        if (alert) {
            *alert = regression_alert;
        }
        
        return true;
    }
    
    return false;
}

bool PerformanceRegressionDetector::load_baselines_from_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open baseline file: " << filepath << std::endl;
        return false;
    }
    
    std::string line;
    std::lock_guard<std::mutex> lock(baselines_mutex_);
    
    // Clear existing baselines
    baselines_.clear();
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;  // Skip empty lines and comments
        
        std::istringstream iss(line);
        std::string metric_name;
        double baseline_value;
        double tolerance_percentage;
        
        if (iss >> metric_name >> baseline_value >> tolerance_percentage) {
            baselines_[metric_name] = PerformanceBaseline(metric_name, baseline_value, tolerance_percentage);
        }
    }
    
    file.close();
    return true;
}

bool PerformanceRegressionDetector::save_baselines_to_file(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to create baseline file: " << filepath << std::endl;
        return false;
    }
    
    std::lock_guard<std::mutex> lock(baselines_mutex_);
    
    // Write header
    file << "# Performance Baseline Configuration\n";
    file << "# Format: metric_name baseline_value tolerance_percentage\n";
    file << "# Generated on: " << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) << "\n\n";
    
    for (const auto& pair : baselines_) {
        const auto& baseline = pair.second;
        file << baseline.metric_name << " " 
             << std::fixed << std::setprecision(6) << baseline.baseline_value << " " 
             << std::fixed << std::setprecision(2) << baseline.tolerance_percentage << "\n";
    }
    
    file.close();
    return true;
}

std::vector<RegressionAlert> PerformanceRegressionDetector::get_recent_alerts() const {
    std::lock_guard<std::mutex> lock(alerts_mutex_);
    return recent_alerts_;
}

void PerformanceRegressionDetector::clear_alerts() {
    std::lock_guard<std::mutex> lock(alerts_mutex_);
    recent_alerts_.clear();
}

void PerformanceRegressionDetector::set_regression_callback(
    std::function<void(const RegressionAlert&)> callback) {
    regression_callback_ = callback;
}

std::unique_ptr<PerformanceBaseline> PerformanceRegressionDetector::get_baseline(
    const std::string& metric_name) const {
    std::lock_guard<std::mutex> lock(baselines_mutex_);
    auto it = baselines_.find(metric_name);
    if (it != baselines_.end()) {
        return std::make_unique<PerformanceBaseline>(it->second);
    }
    return nullptr;
}

// Global instance
PerformanceRegressionDetector g_performance_regression_detector;

} // namespace BTQuant