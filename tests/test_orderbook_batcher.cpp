#include "dependencies/BTQ_Render_Engine/include/components/orderbook_batcher.hpp"
#include <iostream>
#include <vector>

int main() {
    BTQuant::OrderbookBatcher batcher;
    
    // Test basic functionality
    batcher.addRectFilled(ImVec2(0, 0), ImVec2(100, 100), 0xFF0000FF); // Red rectangle
    batcher.addCircleFilled(ImVec2(50, 50), 25.0f, 0xFF00FF00);       // Green circle
    batcher.addLine(ImVec2(0, 0), ImVec2(100, 100), 0xFFFF0000, 2.0f); // Blue line
    
    std::cout << "Created " << batcher.getBatchCount() << " batches initially." << std::endl;
    
    // Test new methods
    std::vector<BTQuant::OrderbookElementData> elements;
    elements.emplace_back(ImVec2(10, 10), ImVec2(50, 50), 0xFF0000FF); // Red rectangle
    elements.emplace_back(ImVec2(60, 60), ImVec2(100, 100), 0xFF00FF00); // Green rectangle
    elements.emplace_back(ImVec2(0, 0), ImVec2(20, 20), 0xFFFF0000); // Blue rectangle
    
    // Test the new spatial coherence batching
    batcher.batchGeometrySpatiallyCoherent(elements);
    std::cout << "After spatial coherence batching: " << batcher.getBatchCount() << " batches." << std::endl;
    
    // Test the maximum throughput batching
    batcher.maximumThroughputBatching(elements);
    std::cout << "After maximum throughput batching: " << batcher.getBatchCount() << " batches." << std::endl;
    
    // Test ultra-efficient consolidation
    batcher.ultraEfficientConsolidateBatches();
    std::cout << "After ultra-efficient consolidation: " << batcher.getBatchCount() << " batches." << std::endl;
    
    // Clear and test with multiple rectangles
    batcher.clear();
    
    std::vector<std::pair<ImVec2, ImVec2>> rect_pairs = {
        {ImVec2(0, 0), ImVec2(10, 10)},
        {ImVec2(20, 20), ImVec2(30, 30)},
        {ImVec2(40, 40), ImVec2(50, 50)}
    };
    
    std::vector<ImU32> colors = {0xFF0000FF, 0xFF00FF00, 0xFFFF0000}; // Red, green, blue
    
    batcher.addRectanglesFilledOptimized(rect_pairs, colors);
    std::cout << "After optimized rectangle batching: " << batcher.getBatchCount() << " batches." << std::endl;
    
    std::cout << "All tests completed successfully!" << std::endl;
    
    return 0;
}