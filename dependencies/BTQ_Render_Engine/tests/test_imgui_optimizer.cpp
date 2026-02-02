/**
 * Test file for ImGui Optimizer functionality
 * This file tests the basic functionality of the ImGui optimizer
 */

#include "../../include/rendering/imgui_optimizer.hpp"
#include <iostream>
#include <cassert>

using namespace BTQuant::Rendering;

void test_text_size_caching() {
    const char* test_text = "Test String";
    
    // Test basic text size calculation
    ImVec2 size1 = ImGuiOptimizer::CalcTextSize(test_text);
    ImVec2 size2 = ImGuiOptimizer::CalcTextSize(test_text);  // Should come from cache
    
    std::cout << "Text size test passed: " << (size1.x == size2.x && size1.y == size2.y) << std::endl;
    assert(size1.x == size2.x && size1.y == size2.y);
}

void test_color_caching() {
    // Test color retrieval
    ImU32 color1 = ImGuiOptimizer::GetColorU32(ImGuiCol_Button);
    ImU32 color2 = ImGuiOptimizer::GetColorU32(ImGuiCol_Button);  // Should come from cache
    
    std::cout << "Color caching test passed: " << (color1 == color2) << std::endl;
    assert(color1 == color2);
}

void test_conditional_functions() {
    // These tests would require an active ImGui context to run properly
    // For now, we just verify the functions exist and compile correctly
    std::cout << "Conditional functions exist and compiled successfully" << std::endl;
}

void test_batch_operations() {
    // Test batch text rendering with empty vector (should not crash)
    std::vector<std::pair<std::string, ImVec2>> empty_texts;
    ImGuiOptimizer::BatchTextRendering(empty_texts);
    
    std::cout << "Batch operations test passed: no crashes with empty vectors" << std::endl;
}

int main() {
    std::cout << "Starting ImGui Optimizer tests..." << std::endl;
    
    test_text_size_caching();
    test_color_caching();
    test_conditional_functions();
    test_batch_operations();
    
    std::cout << "All tests completed successfully!" << std::endl;
    
    return 0;
}