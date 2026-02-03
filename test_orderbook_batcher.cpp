#include "dependencies/BTQ_Render_Engine/include/components/orderbook_batcher.hpp"
#include <iostream>
#include <vector>

int main() {
    // Create an instance of the OrderbookBatcher
    BTQuant::OrderbookBatcher batcher;

    // Add some test elements to the batcher
    ImVec2 rect_min(0.0f, 0.0f);
    ImVec2 rect_max(10.0f, 10.0f);
    ImU32 color = 0xFF0000FF; // Blue color
    
    // Add a few rectangles with similar colors
    batcher.addRectFilled(rect_min, rect_max, color);
    batcher.addRectFilled(ImVec2(10.0f, 0.0f), ImVec2(20.0f, 10.0f), color);
    batcher.addRectFilled(ImVec2(20.0f, 0.0f), ImVec2(30.0f, 10.0f), color);
    
    // Add some elements with different colors to test color grouping
    ImU32 red_color = 0xFFFF0000; // Red color
    batcher.addRectFilled(ImVec2(0.0f, 10.0f), ImVec2(10.0f, 20.0f), red_color);
    batcher.addRectFilled(ImVec2(10.0f, 10.0f), ImVec2(20.0f, 20.0f), red_color);
    
    // Test the new optimization methods
    std::cout << "Initial batch count: " << batcher.getBatchCount() << std::endl;

    // Test the new optimization methods
    batcher.advancedBatchOptimization();
    std::cout << "Batch count after advancedBatchOptimization: " << batcher.getBatchCount() << std::endl;
    
    batcher.clear();
    std::cout << "Batch count after clear: " << batcher.getBatchCount() << std::endl;
    
    std::cout << "Orderbook batcher test completed successfully!" << std::endl;
    
    return 0;
}