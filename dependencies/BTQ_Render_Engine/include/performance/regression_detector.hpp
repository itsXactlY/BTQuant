#pragma once

#include <chrono>
#include <map>
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <iostream>
#include <mutex>
#include <iomanip>

namespace BTQuant {

struct PerformanceBaseline {
    std::string metric_name;
    double baseline_value;
    double tolerance_percentage;  // Percentage threshold for regression detection
    std::chrono::system_clock::time_point timestamp;
    
    PerformanceBaseline() = default;  // Default constructor needed for map operations
    PerformanceBaseline(const std::string& name, double value, double tolerance = 5.0)
        : metric_name(name), baseline_value(value), tolerance_percentage(tolerance),
          timestamp(std::chrono::system_clock::now()) {}
};

struct RegressionAlert {
    std::string metric_name;
    double current_value;
    double baseline_value;
    double deviation_percentage;
    std::chrono::system_clock::time_point timestamp;
    bool is_regression;  // True if performance degraded
    
    RegressionAlert(const std::string& name, double curr, double base, bool reg)
        : metric_name(name), current_value(curr), baseline_value(base),
          deviation_percentage(((curr - base) / base) * 100.0),
          timestamp(std::chrono::system_clock::now()), is_regression(reg) {}
};

class PerformanceRegressionDetector {
public:
    PerformanceRegressionDetector();
    ~PerformanceRegressionDetector() = default;

    // Set baseline values for performance metrics
    void set_baseline(const std::string& metric_name, double baseline_value, 
                     double tolerance_percentage = 5.0);
    
    // Check if current metric value indicates a regression compared to baseline
    bool check_regression(const std::string& metric_name, double current_value, 
                         RegressionAlert* alert = nullptr);
    
    // Load baselines from a file
    bool load_baselines_from_file(const std::string& filepath);
    
    // Save baselines to a file
    bool save_baselines_to_file(const std::string& filepath) const;
    
    // Get all current alerts
    std::vector<RegressionAlert> get_recent_alerts() const;
    
    // Clear all alerts
    void clear_alerts();
    
    // Set callback function for when a regression is detected
    void set_regression_callback(std::function<void(const RegressionAlert&)> callback);
    
    // Get current baseline for a metric
    std::unique_ptr<PerformanceBaseline> get_baseline(const std::string& metric_name) const;
    
    // Run a performance test and compare against baseline
    template<typename TestFunction>
    bool run_performance_test(const std::string& test_name, TestFunction test_func, 
                             double tolerance_percentage = 5.0);
    
    // Run a performance test that returns a metric value and compare against baseline
    template<typename TestFunction>
    bool run_performance_test_with_result(const std::string& test_name, TestFunction test_func, 
                                        double tolerance_percentage = 5.0);
    
private:
    std::map<std::string, PerformanceBaseline> baselines_;
    std::vector<RegressionAlert> recent_alerts_;
    std::function<void(const RegressionAlert&)> regression_callback_;
    mutable std::mutex baselines_mutex_;
    mutable std::mutex alerts_mutex_;
};

// Global instance for easy access
extern PerformanceRegressionDetector g_performance_regression_detector;

// Template implementations
template<typename TestFunction>
bool PerformanceRegressionDetector::run_performance_test(const std::string& test_name, TestFunction test_func, 
                             double tolerance_percentage) {
    // Record start time
    auto start = std::chrono::high_resolution_clock::now();
    
    // Run the test function
    test_func();
    
    // Record end time
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double elapsed_ms = static_cast<double>(duration.count()) / 1000.0;
    
    // Set baseline if it doesn't exist
    {
        std::lock_guard<std::mutex> lock(baselines_mutex_);
        if (baselines_.find(test_name) == baselines_.end()) {
            set_baseline(test_name, elapsed_ms, tolerance_percentage);
            return true;  // First run, no regression possible
        }
    }
    
    // Check for regression
    RegressionAlert alert("", 0.0, 0.0, false);  // Initialize with dummy values
    bool regression_detected = check_regression(test_name, elapsed_ms, &alert);
    
    if (regression_detected) {
        std::cout << "PERFORMANCE REGRESSION DETECTED: " << test_name 
                  << " took " << std::fixed << std::setprecision(3) << elapsed_ms 
                  << " ms (baseline: " << alert.baseline_value << " ms, deviation: " 
                  << alert.deviation_percentage << "%)" << std::endl;
    }
    
    return !regression_detected;
}

template<typename TestFunction>
bool PerformanceRegressionDetector::run_performance_test_with_result(const std::string& test_name, TestFunction test_func, 
                                        double tolerance_percentage) {
    // Run the test function and get the result
    double result = test_func();
    
    // Set baseline if it doesn't exist
    {
        std::lock_guard<std::mutex> lock(baselines_mutex_);
        if (baselines_.find(test_name) == baselines_.end()) {
            set_baseline(test_name, result, tolerance_percentage);
            return true;  // First run, no regression possible
        }
    }
    
    // Check for regression
    RegressionAlert alert("", 0.0, 0.0, false);  // Initialize with dummy values
    bool regression_detected = check_regression(test_name, result, &alert);
    
    if (regression_detected) {
        std::cout << "PERFORMANCE REGRESSION DETECTED: " << test_name 
                  << " value: " << std::fixed << std::setprecision(3) << result 
                  << " (baseline: " << alert.baseline_value << ", deviation: " 
                  << alert.deviation_percentage << "%)" << std::endl;
    }
    
    return !regression_detected;
}

} // namespace BTQuant