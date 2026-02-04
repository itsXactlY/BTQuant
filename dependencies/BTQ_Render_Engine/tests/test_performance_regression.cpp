#include <iostream>
#include <thread>
#include <chrono>
#include <random>
#include "../include/performance/regression_detector.hpp"
#include "../include/performance_monitor.hpp"

int main() {
    std::cout << "Testing Performance Regression Detector..." << std::endl;

    // Set up a callback for regression alerts
    BTQuant::g_performance_regression_detector.set_regression_callback([](const BTQuant::RegressionAlert& alert) {
        std::cout << "REGRESSION ALERT: " << alert.metric_name 
                  << " - Current: " << alert.current_value 
                  << ", Baseline: " << alert.baseline_value 
                  << ", Deviation: " << alert.deviation_percentage << "%" << std::endl;
    });

    // Test 1: Basic regression detection with frame time
    std::cout << "\nTest 1: Testing frame time regression detection..." << std::endl;
    
    // Set a baseline for frame time (simulate 16ms per frame ~ 60 FPS)
    BTQuant::g_performance_regression_detector.set_baseline("frame_time_ms", 16.0, 10.0); // 10% tolerance
    
    // Test with acceptable performance (within tolerance)
    bool result1 = BTQuant::g_performance_regression_detector.run_performance_test_with_result(
        "frame_time_ms", []() -> double {
            // Simulate good performance (15ms per frame)
            std::this_thread::sleep_for(std::chrono::milliseconds(15));
            return 15.0; // Return simulated frame time
        }, 10.0);
    
    std::cout << "Test 1a (acceptable performance): " << (result1 ? "PASS" : "FAIL - False positive") << std::endl;
    
    // Test with poor performance (regression - above tolerance)
    bool result2 = BTQuant::g_performance_regression_detector.run_performance_test_with_result(
        "frame_time_ms", []() -> double {
            // Simulate poor performance (25ms per frame - 56% increase, exceeds 10% tolerance)
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
            return 25.0; // Return simulated frame time
        }, 10.0);
    
    std::cout << "Test 1b (poor performance): " << (result2 ? "FAIL - Missed regression" : "PASS - Regression detected") << std::endl;

    // Test 2: FPS regression detection (lower FPS = worse performance)
    std::cout << "\nTest 2: Testing FPS regression detection..." << std::endl;
    
    // Set a baseline for FPS (simulate 60 FPS)
    BTQuant::g_performance_regression_detector.set_baseline("frames_per_second", 60.0, 10.0); // 10% tolerance
    
    // Test with acceptable FPS
    bool result3 = BTQuant::g_performance_regression_detector.run_performance_test_with_result(
        "frames_per_second", []() -> double {
            // Simulate good FPS (58 FPS - within 10% tolerance)
            return 58.0;
        }, 10.0);
    
    std::cout << "Test 2a (acceptable FPS): " << (result3 ? "PASS" : "FAIL - False positive") << std::endl;
    
    // Test with poor FPS (regression - below baseline by more than tolerance)
    bool result4 = BTQuant::g_performance_regression_detector.run_performance_test_with_result(
        "frames_per_second", []() -> double {
            // Simulate poor FPS (45 FPS - 25% decrease, exceeds 10% tolerance)
            return 45.0;
        }, 10.0);
    
    std::cout << "Test 2b (poor FPS): " << (result4 ? "FAIL - Missed regression" : "PASS - Regression detected") << std::endl;

    // Test 3: Time-based performance test
    std::cout << "\nTest 3: Testing time-based performance test..." << std::endl;
    
    bool result5 = BTQuant::g_performance_regression_detector.run_performance_test(
        "data_processing_time", []() {
            // Simulate some data processing that takes ~10ms
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            // Additional computation to simulate work
            volatile double sum = 0;
            for (int i = 0; i < 1000000; ++i) {
                sum += i * 0.001;
            }
        }, 20.0); // 20% tolerance
    
    std::cout << "Test 3a (first run - establishing baseline): " << (result5 ? "PASS" : "UNEXPECTED") << std::endl;
    
    // Run the same test again to check against baseline
    bool result6 = BTQuant::g_performance_regression_detector.run_performance_test(
        "data_processing_time", []() {
            // Same processing but slightly slower (~15ms)
            std::this_thread::sleep_for(std::chrono::milliseconds(15));
            // Additional computation to simulate work
            volatile double sum = 0;
            for (int i = 0; i < 1000000; ++i) {
                sum += i * 0.001;
            }
        }, 20.0); // 20% tolerance
    
    std::cout << "Test 3b (slightly slower - within tolerance): " << (result6 ? "PASS" : "FAIL - False positive") << std::endl;

    // Test 4: File I/O for baselines persistence
    std::cout << "\nTest 4: Testing baseline persistence..." << std::endl;
    
    // Save current baselines
    bool save_success = BTQuant::g_performance_regression_detector.save_baselines_to_file("test_baselines.cfg");
    std::cout << "Save baselines: " << (save_success ? "SUCCESS" : "FAILED") << std::endl;
    
    // Load baselines into a new detector instance to verify
    BTQuant::PerformanceRegressionDetector temp_detector;
    bool load_success = temp_detector.load_baselines_from_file("test_baselines.cfg");
    std::cout << "Load baselines: " << (load_success ? "SUCCESS" : "FAILED") << std::endl;
    
    if (load_success) {
        auto baseline = temp_detector.get_baseline("frames_per_second");
        if (baseline) {
            std::cout << "Loaded FPS baseline: " << baseline->baseline_value << std::endl;
        } else {
            std::cout << "Could not retrieve loaded FPS baseline" << std::endl;
        }
    }

    // Test 5: Multiple metrics tracking
    std::cout << "\nTest 5: Testing multiple metrics tracking..." << std::endl;
    
    // Add more metrics
    BTQuant::g_performance_regression_detector.set_baseline("memory_usage_mb", 100.0, 15.0);
    BTQuant::g_performance_regression_detector.set_baseline("draw_calls", 1000.0, 5.0);
    
    // Simulate checking these metrics
    bool mem_ok = BTQuant::g_performance_regression_detector.check_regression("memory_usage_mb", 110.0); // Within 15% tolerance
    bool draw_ok = BTQuant::g_performance_regression_detector.check_regression("draw_calls", 1020.0); // Within 5% tolerance
    
    std::cout << "Memory usage check: " << (mem_ok ? "REGRESSION DETECTED" : "OK") << std::endl;
    std::cout << "Draw calls check: " << (draw_ok ? "REGRESSION DETECTED" : "OK") << std::endl;
    
    // Check for a regression in draw calls
    bool draw_reg = BTQuant::g_performance_regression_detector.check_regression("draw_calls", 1100.0); // Above 5% tolerance
    std::cout << "Draw calls regression test: " << (draw_reg ? "REGRESSION DETECTED" : "OK - Not detected") << std::endl;

    // Show recent alerts
    auto alerts = BTQuant::g_performance_regression_detector.get_recent_alerts();
    std::cout << "\nRecent regression alerts: " << alerts.size() << std::endl;
    for (const auto& alert : alerts) {
        std::cout << "  - " << alert.metric_name << ": " << alert.current_value 
                  << " vs baseline " << alert.baseline_value 
                  << " (" << alert.deviation_percentage << "%)" << std::endl;
    }

    std::cout << "\nPerformance regression detector tests completed!" << std::endl;

    return 0;
}