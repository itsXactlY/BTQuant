/**
 * Test for the enhanced Auto-Quality Controller
 */

#include "rendering/auto_quality.hpp"
#include "rendering/auto_quality_integration.hpp"
#include <iostream>
#include <thread>
#include <chrono>

using namespace RenderEngine;

int main() {
    std::cout << "Testing Enhanced Auto-Quality Controller..." << std::endl;

    // Create configuration
    AutoQualityConfig config;
    config.target_fps = 60;
    config.performance_threshold = 80.0;
    config.variance_threshold = 5.0;
    config.adjustment_cooldown_ms = 1000;  // 1 second cooldown
    config.stability_window_ms = 2000;     // 2 second stability window
    config.enable_logging = true;

    // Create the auto-quality controller
    AutoQualityController controller(config);
    
    // Test the integration helper
    AutoQualityRenderer renderer(config);

    std::cout << "Initial quality index: " << controller.getCurrentQualityIndex() << std::endl;
    std::cout << "Initial performance score: " << controller.getPerformanceScore() << std::endl;

    // Simulate good performance (low frame times)
    std::cout << "\nSimulating good performance..." << std::endl;
    for (int i = 0; i < 60; ++i) {
        controller.recordFrameTime(10.0); // 10ms per frame = 100fps
        renderer.beginFrame();
        // Simulate some work
        std::this_thread::sleep_for(std::chrono::microseconds(5000)); // 5ms of "work"
        renderer.endFrame();
    }

    std::cout << "After good performance:" << std::endl;
    std::cout << "Quality index: " << controller.getCurrentQualityIndex() << std::endl;
    std::cout << "Performance score: " << controller.getPerformanceScore() << std::endl;

    // Simulate poor performance (high frame times)
    std::cout << "\nSimulating poor performance..." << std::endl;
    for (int i = 0; i < 60; ++i) {
        controller.recordFrameTime(50.0); // 50ms per frame = 20fps, well below target
        renderer.beginFrame();
        // Simulate heavy work
        std::this_thread::sleep_for(std::chrono::microseconds(40000)); // 40ms of "work"
        renderer.endFrame();
    }

    std::cout << "After poor performance:" << std::endl;
    std::cout << "Quality index: " << controller.getCurrentQualityIndex() << std::endl;
    std::cout << "Performance score: " << controller.getPerformanceScore() << std::endl;
    
    // Test the new methods
    std::cout << "\nTesting new performance analysis methods:" << std::endl;
    std::cout << "Predicted future performance: " << controller.predictFuturePerformance() << std::endl;
    std::cout << "Degradation rate: " << controller.calculateDegradationRate() << std::endl;
    std::cout << "Memory pressure score: " << controller.calculateMemoryPressureScore() << std::endl;
    std::cout << "Thermal pressure score: " << controller.calculateThermalPressureScore() << std::endl;
    
    // Test getting recommended settings
    QualitySettings settings = controller.getCurrentQualitySettings();
    std::cout << "Current render resolution scale: " << settings.render_resolution_scale << std::endl;
    std::cout << "Current max visible elements: " << settings.max_visible_elements << std::endl;
    
    // Test the integration helper's recommended settings
    const QualitySettings& rendererSettings = renderer.getQualitySettings();
    std::cout << "Renderer's recommended resolution scale: " << rendererSettings.render_resolution_scale << std::endl;
    
    std::cout << "\nEnhanced Auto-Quality Controller test completed successfully!" << std::endl;

    return 0;
}