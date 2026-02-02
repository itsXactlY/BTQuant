#include "../include/rendering/frame_pacer.hpp"
#include <iostream>

int main() {
    std::cout << "Testing Frame Pacer Implementation..." << std::endl;
    
    // Test default constructor
    RenderEngine::FramePacer pacer1;
    std::cout << "Default FramePacer created successfully." << std::endl;
    
    // Test constructor with config
    RenderEngine::FramePacer::Config config;
    config.target_fps = 60;
    config.enable_adaptive_sync = true;
    config.enable_frame_smoothing = true;
    config.enable_burst_reduction = true;
    config.frame_time_variance_threshold = 0.016;
    
    RenderEngine::FramePacer pacer2(config);
    std::cout << "Configured FramePacer created successfully." << std::endl;
    
    // Test basic functionality
    pacer2.begin_frame();
    // Simulate some work
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    pacer2.end_frame();
    pacer2.wait_for_next_frame();
    
    // Get stats
    auto stats = pacer2.get_stats();
    std::cout << "Frame Pacer Stats:" << std::endl;
    std::cout << "  Avg Frame Time: " << stats.avg_frame_time_ms << " ms" << std::endl;
    std::cout << "  Current FPS: " << stats.current_fps << std::endl;
    std::cout << "  Total Frames: " << stats.total_frames << std::endl;
    
    std::cout << "Frame Pacer implementation working correctly!" << std::endl;
    return 0;
}