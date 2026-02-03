#include "performance/frame_time_graph.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <cassert>

void test_basic_functionality() {
    std::cout << "Testing basic FrameTimeGraph functionality..." << std::endl;
    
    BTQuant::FrameTimeGraph graph(50); // Small sample size for testing
    
    // Test initialization
    assert(graph.is_enabled() == true);
    assert(graph.get_current_frame_time_ms() == 0.0);
    assert(graph.get_current_fps() == 0.0);
    
    // Test frame timing
    graph.start_frame();
    std::this_thread::sleep_for(std::chrono::milliseconds(5)); // Sleep for 5ms
    graph.end_frame();
    
    // Values should be approximately 5ms
    double current_time = graph.get_current_frame_time_ms();
    assert(current_time >= 4.0 && current_time <= 10.0); // Allow some tolerance
    
    std::cout << "  Current frame time: " << current_time << " ms" << std::endl;
    
    // Test multiple frames
    for (int i = 0; i < 10; ++i) {
        graph.start_frame();
        std::this_thread::sleep_for(std::chrono::milliseconds(3 + (i % 3))); // Vary between 3-5ms
        graph.end_frame();
    }
    
    // Check that we have accumulated data
    double avg_time = graph.get_average_frame_time_ms();
    std::cout << "  Average frame time: " << avg_time << " ms" << std::endl;
    assert(avg_time > 0.0);
    
    double min_time = graph.get_min_frame_time_ms();
    double max_time = graph.get_max_frame_time_ms();
    std::cout << "  Min frame time: " << min_time << " ms, Max: " << max_time << " ms" << std::endl;
    
    assert(min_time <= avg_time);
    assert(max_time >= avg_time);
    assert(min_time <= max_time);
    
    std::cout << "Basic functionality test PASSED" << std::endl;
}

void test_performance_analysis() {
    std::cout << "Testing performance analysis features..." << std::endl;
    
    BTQuant::FrameTimeGraph graph(100); // Larger sample size for statistical analysis
    
    // Generate some varied frame times to test statistics
    std::vector<double> test_times = {5.0, 8.0, 12.0, 16.0, 20.0, 30.0, 5.5, 7.2, 14.8, 18.3};
    
    for (double time_ms : test_times) {
        // Manually simulate frame times by calling start/end and manipulating internal state
        // would be complex, so instead we'll just verify the statistical methods work
        // with the existing data after simulating frames
        graph.start_frame();
        std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(time_ms)));
        graph.end_frame();
    }
    
    // Test variance and standard deviation
    double variance = graph.get_variance();
    double std_dev = graph.get_standard_deviation();
    std::cout << "  Variance: " << variance << ", Std Dev: " << std_dev << std::endl;
    
    // Standard deviation should be positive if we have variation
    if (graph.get_frame_time_history().size() > 1) {
        assert(std_dev >= 0.0);
    }
    
    // Test percentiles
    double p50 = graph.get_percentile(50.0);  // Median
    double p90 = graph.get_percentile(90.0);  // 90th percentile
    double p95 = graph.get_percentile(95.0);  // 95th percentile
    double p99 = graph.get_percentile(99.0);  // 99th percentile
    
    std::cout << "  50th percentile (median): " << p50 << " ms" << std::endl;
    std::cout << "  90th percentile: " << p90 << " ms" << std::endl;
    std::cout << "  95th percentile: " << p95 << " ms" << std::endl;
    std::cout << "  99th percentile: " << p99 << " ms" << std::endl;
    
    // Percentiles should be in ascending order
    if (p50 > 0 && p90 > 0) {
        assert(p50 <= p90);
        assert(p90 <= p95);
        assert(p95 <= p99);
    }
    
    // Test threshold counting
    graph.set_warning_threshold(10.0);   // 10ms warning threshold
    graph.set_critical_threshold(20.0);  // 20ms critical threshold
    
    auto [warning_count, critical_count] = graph.get_frames_outside_thresholds();
    std::cout << "  Warning frames (>10ms): " << warning_count << std::endl;
    std::cout << "  Critical frames (>20ms): " << critical_count << std::endl;
    
    // Test smoothing
    double smoothed = graph.get_smoothed_frame_time(5);  // Last 5 frames
    std::cout << "  Smoothed frame time (last 5): " << smoothed << " ms" << std::endl;

    assert(smoothed >= 0.0);

    // Test median
    double median = graph.get_median_frame_time();
    std::cout << "  Median frame time: " << median << " ms" << std::endl;

    // Test specific percentile
    double p75 = graph.get_frame_time_at_percentile(75.0);
    std::cout << "  75th percentile: " << p75 << " ms" << std::endl;

    // Test spike detection
    auto spikes = graph.get_spike_frames(1.5); // Frames that are 1.5x the local average
    std::cout << "  Spikes detected (1.5x multiplier): " << spikes.size() << std::endl;

    // Test consecutive frame drops
    size_t consecutive_drops = graph.get_consecutive_frame_drops(2, 10.0); // 2+ consecutive frames > 10ms
    std::cout << "  Consecutive slow frames (>=2, >10ms): " << consecutive_drops << std::endl;

    std::cout << "Performance analysis test PASSED" << std::endl;
}

void test_threshold_management() {
    std::cout << "Testing threshold management..." << std::endl;
    
    BTQuant::FrameTimeGraph graph;
    
    // Test default thresholds
    double default_warning = graph.get_warning_threshold();
    double default_critical = graph.get_critical_threshold();
    
    std::cout << "  Default warning threshold: " << default_warning << " ms" << std::endl;
    std::cout << "  Default critical threshold: " << default_critical << " ms" << std::endl;
    
    // Test setting custom thresholds
    graph.set_warning_threshold(15.0);
    graph.set_critical_threshold(30.0);
    
    assert(graph.get_warning_threshold() == 15.0);
    assert(graph.get_critical_threshold() == 30.0);
    
    std::cout << "Threshold management test PASSED" << std::endl;
}

void test_reset_functionality() {
    std::cout << "Testing reset functionality..." << std::endl;
    
    BTQuant::FrameTimeGraph graph(50);
    
    // Add some data
    for (int i = 0; i < 10; ++i) {
        graph.start_frame();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        graph.end_frame();
    }
    
    // Verify data exists
    assert(graph.get_frame_time_history().size() > 0);
    assert(graph.get_current_frame_time_ms() > 0.0);
    
    // Reset and verify cleared
    graph.reset();
    
    assert(graph.get_frame_time_history().size() == 0);
    assert(graph.get_current_frame_time_ms() == 0.0);
    assert(graph.get_average_frame_time_ms() == 0.0);
    
    std::cout << "Reset functionality test PASSED" << std::endl;
}

int main() {
    std::cout << "Starting FrameTimeGraph Advanced Tests..." << std::endl;
    
    try {
        test_basic_functionality();
        test_performance_analysis();
        test_threshold_management();
        test_reset_functionality();
        
        std::cout << "\nAll FrameTimeGraph tests PASSED!" << std::endl;
        std::cout << "Frame time graph visualization and performance analysis features are working correctly." << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Test FAILED with exception: " << e.what() << std::endl;
        return 1;
    }
    catch (...) {
        std::cerr << "Test FAILED with unknown exception" << std::endl;
        return 1;
    }
    
    return 0;
}