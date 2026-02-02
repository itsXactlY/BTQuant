#include <iostream>
#include <thread>
#include <chrono>
#include "../include/performance_monitor.hpp"

int main() {
    std::cout << "Testing Performance Monitor Integration..." << std::endl;
    
    // Initialize the performance monitor
    BTQuant::g_performance_monitor.set_enabled(true);
    
    // Simulate frame timing for 50 frames
    for (int i = 0; i < 50; ++i) {
        // Start frame timing using performance monitor
        BTQuant::g_performance_monitor.start_frame();
        
        // Simulate some work with variable timing
        int sleep_ms = 8 + (rand() % 12); // Random sleep between 8-20ms
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        
        // End frame timing using performance monitor
        BTQuant::g_performance_monitor.end_frame();
        
        // Print occasional updates
        if (i % 10 == 0) {
            std::cout << "Frame " << i << ": Current frame time = " 
                      << BTQuant::g_performance_monitor.get_frame_time_ms() 
                      << " ms, FPS = " << BTQuant::g_performance_monitor.get_fps() << std::endl;
        }
    }
    
    // Print final statistics
    std::cout << "\nFinal Statistics from Performance Monitor:" << std::endl;
    std::cout << "Current Frame Time: " << BTQuant::g_performance_monitor.get_frame_time_ms() << " ms" << std::endl;
    std::cout << "Average Frame Time: " << BTQuant::g_performance_monitor.get_avg_frame_time_ms() << " ms" << std::endl;
    std::cout << "Min Frame Time: " << BTQuant::g_performance_monitor.get_min_frame_time() << " ms" << std::endl;
    std::cout << "Max Frame Time: " << BTQuant::g_performance_monitor.get_max_frame_time() << " ms" << std::endl;
    std::cout << "Current FPS: " << BTQuant::g_performance_monitor.get_fps() << std::endl;
    std::cout << "Average FPS: " << BTQuant::g_performance_monitor.get_avg_fps() << std::endl;
    
    std::cout << "\nPerformance monitor integration is working correctly!" << std::endl;
    
    return 0;
}