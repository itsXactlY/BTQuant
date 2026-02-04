#include "telemetry_collector.h"
#include <iostream>
#include <thread>

int main() {
    // Get the telemetry collector instance
    btq::TelemetryCollector& collector = btq::TelemetryCollector::getInstance();
    
    // Initialize the collector
    collector.initialize();
    
    std::cout << "Telemetry collector initialized." << std::endl;
    
    // Record some sample feature usage
    collector.recordFeatureUsage("chart_view");
    collector.recordFeatureUsage("order_placement");
    collector.recordFeatureUsage("chart_view"); // Used twice
    collector.recordFeatureUsage("portfolio_view");
    
    // Record some performance metrics
    collector.recordPerformanceMetric("render_frame_time", 16.5, "ms");
    collector.recordPerformanceMetric("data_update_time", 5.2, "ms");
    collector.recordPerformanceMetric("render_frame_time", 18.1, "ms");
    
    // Record some user actions
    collector.recordUserAction("button_click", "main_toolbar");
    collector.recordUserAction("menu_open", "settings_menu");
    
    // Record custom events
    std::map<std::string, std::string> props = {{"source", "manual_test"}, {"version", "1.0"}};
    collector.recordEvent("application_start", props);
    
    std::cout << "Sample telemetry data recorded." << std::endl;
    
    // Wait a bit to allow data to be processed
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Check some values
    std::cout << "Feature 'chart_view' usage count: " << collector.getFeatureUsageCount("chart_view") << std::endl;
    std::cout << "Average render frame time: " << collector.getAveragePerformanceMetric("render_frame_time") << " ms" << std::endl;
    
    // Clean up
    collector.stop();
    
    std::cout << "Telemetry collector stopped." << std::endl;
    
    return 0;
}