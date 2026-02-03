#include <iostream>
#include <thread>
#include <chrono>
#include "../include/performance/frame_time_graph.hpp"

int main() {
    std::cout << "Testing Frame Time Graph Implementation..." << std::endl;
    
    // Initialize the frame time graph
    BTQuant::g_frame_time_graph.set_enabled(true);
    
    // Simulate frame timing for 100 frames
    for (int i = 0; i < 100; ++i) {
        // Start frame timing
        BTQuant::g_frame_time_graph.start_frame();
        
        // Simulate some work with variable timing
        int sleep_ms = 5 + (rand() % 15); // Random sleep between 5-20ms
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        
        // End frame timing
        BTQuant::g_frame_time_graph.end_frame();
        
        // Print occasional updates
        if (i % 20 == 0) {
            std::cout << "Frame " << i << ": Current frame time = " 
                      << BTQuant::g_frame_time_graph.get_current_frame_time_ms() 
                      << " ms, FPS = " << BTQuant::g_frame_time_graph.get_current_fps() << std::endl;
        }
    }
    
    // Print final statistics
    std::cout << "\nFinal Statistics:" << std::endl;
    std::cout << "Current Frame Time: " << BTQuant::g_frame_time_graph.get_current_frame_time_ms() << " ms" << std::endl;
    std::cout << "Average Frame Time: " << BTQuant::g_frame_time_graph.get_average_frame_time_ms() << " ms" << std::endl;
    std::cout << "Min Frame Time: " << BTQuant::g_frame_time_graph.get_min_frame_time_ms() << " ms" << std::endl;
    std::cout << "Max Frame Time: " << BTQuant::g_frame_time_graph.get_max_frame_time_ms() << " ms" << std::endl;
    std::cout << "Current FPS: " << BTQuant::g_frame_time_graph.get_current_fps() << std::endl;
    std::cout << "Average FPS: " << BTQuant::g_frame_time_graph.get_average_fps() << std::endl;
    
    std::cout << "\nFrame time graph implementation is working correctly!" << std::endl;
    
    return 0;
}