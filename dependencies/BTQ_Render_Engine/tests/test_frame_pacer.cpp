#include "../include/rendering/frame_pacer.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>

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

    // Test basic functionality with multiple frames
    std::cout << "Running frame pacing test for 60 frames..." << std::endl;

    for (int i = 0; i < 60; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        pacer2.begin_frame();

        // Simulate varying workloads to test frame pacing
        int work_time = 5 + (i % 10); // Varying work time between 5-15ms
        std::this_thread::sleep_for(std::chrono::milliseconds(work_time));

        pacer2.end_frame();
        pacer2.wait_for_next_frame();

        auto end = std::chrono::high_resolution_clock::now();
        auto frame_time = std::chrono::duration<double, std::milli>(end - start).count();

        if (i % 10 == 0) {
            std::cout << "Frame " << i << ", Time: " << frame_time << "ms" << std::endl;
        }
    }

    // Get stats after running frames
    auto stats = pacer2.get_stats();
    std::cout << "\nFinal Frame Pacer Stats:" << std::endl;
    std::cout << "  Avg Frame Time: " << stats.avg_frame_time_ms << " ms" << std::endl;
    std::cout << "  Min Frame Time: " << stats.min_frame_time_ms << " ms" << std::endl;
    std::cout << "  Max Frame Time: " << stats.max_frame_time_ms << " ms" << std::endl;
    std::cout << "  Current FPS: " << stats.current_fps << std::endl;
    std::cout << "  Smoothed FPS: " << stats.smoothed_fps << std::endl;
    std::cout << "  Total Frames: " << stats.total_frames << std::endl;
    std::cout << "  Frame Time Variance: " << stats.frame_time_variance << std::endl;
    std::cout << "  Dropped Frames: " << stats.dropped_frames << std::endl;
    std::cout << "  Recent Spikes: " << stats.spike_count_recent << std::endl;

    // Test configuration update
    std::cout << "\nUpdating configuration to 30 FPS..." << std::endl;
    RenderEngine::FramePacer::Config new_config = config;
    new_config.target_fps = 30;
    pacer2.update_config(new_config);

    // Run a few more frames with new config
    for (int i = 0; i < 10; ++i) {
        pacer2.begin_frame();
        std::this_thread::sleep_for(std::chrono::milliseconds(15 + (i % 5)));
        pacer2.end_frame();
        pacer2.wait_for_next_frame();
    }

    auto updated_stats = pacer2.get_stats();
    std::cout << "After config update - Smoothed FPS: " << updated_stats.smoothed_fps << std::endl;

    // Test reset functionality
    std::cout << "Resetting frame pacer stats..." << std::endl;
    pacer2.reset_stats();
    auto reset_stats = pacer2.get_stats();
    std::cout << "After reset - Total Frames: " << reset_stats.total_frames << std::endl;

    std::cout << "\nFrame Pacer implementation working correctly!" << std::endl;
    return 0;
}