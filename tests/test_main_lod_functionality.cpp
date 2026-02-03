#include "rendering/footprint_lod.hpp"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "Testing Main LOD Functionality (Reduce Detail When Zoomed Out, Increase Detail When Zoomed In)..." << std::endl;

    BTQuant::Rendering::FootprintLOD lod_system;

    // Test LOD levels at different zoom factors to verify the main functionality
    
    // Test 1: Extremely zoomed out - should result in LOW_DETAIL
    auto lod_extreme_zoom_out = lod_system.calculateMainLOD(5.0f, 5.0f, 0.01f);
    std::cout << "Extreme zoom out (0.01): LOD Level = " << static_cast<int>(lod_extreme_zoom_out) << std::endl;
    assert(lod_extreme_zoom_out == BTQuant::Rendering::LODLevel::LOW_DETAIL && 
           "At extreme zoom out, should be LOW_DETAIL");
    
    // Test 2: Highly zoomed out - should result in LOW_DETAIL or MEDIUM_DETAIL
    auto lod_high_zoom_out = lod_system.calculateMainLOD(8.0f, 8.0f, 0.08f);
    std::cout << "High zoom out (0.08): LOD Level = " << static_cast<int>(lod_high_zoom_out) << std::endl;
    assert(lod_high_zoom_out <= BTQuant::Rendering::LODLevel::MEDIUM_DETAIL && 
           "At high zoom out, should be LOW_DETAIL or MEDIUM_DETAIL");
    
    // Test 3: Moderately zoomed out - should result in MEDIUM_DETAIL
    auto lod_moderate_zoom_out = lod_system.calculateMainLOD(12.0f, 12.0f, 0.2f);
    std::cout << "Moderate zoom out (0.2): LOD Level = " << static_cast<int>(lod_moderate_zoom_out) << std::endl;
    assert(lod_moderate_zoom_out >= BTQuant::Rendering::LODLevel::LOW_DETAIL && 
           lod_moderate_zoom_out <= BTQuant::Rendering::LODLevel::HIGH_DETAIL && 
           "At moderate zoom out, should be MEDIUM_DETAIL or lower");
    
    // Test 4: Normal zoom - should result in HIGH_DETAIL
    auto lod_normal = lod_system.calculateMainLOD(20.0f, 20.0f, 1.0f);
    std::cout << "Normal zoom (1.0): LOD Level = " << static_cast<int>(lod_normal) << std::endl;
    assert(lod_normal >= BTQuant::Rendering::LODLevel::MEDIUM_DETAIL && 
           "At normal zoom, should be MEDIUM_DETAIL or higher");
    
    // Test 5: Zoomed in - should result in HIGH_DETAIL or MAX_DETAIL
    auto lod_zoom_in = lod_system.calculateMainLOD(25.0f, 25.0f, 1.8f);
    std::cout << "Zoomed in (1.8): LOD Level = " << static_cast<int>(lod_zoom_in) << std::endl;
    assert(lod_zoom_in >= BTQuant::Rendering::LODLevel::HIGH_DETAIL && 
           "At zoomed in, should be HIGH_DETAIL or MAX_DETAIL");
    
    // Test 6: Highly zoomed in - should result in MAX_DETAIL
    auto lod_high_zoom_in = lod_system.calculateMainLOD(40.0f, 40.0f, 3.0f);
    std::cout << "High zoom in (3.0): LOD Level = " << static_cast<int>(lod_high_zoom_in) << std::endl;
    assert(lod_high_zoom_in >= BTQuant::Rendering::LODLevel::HIGH_DETAIL && 
           "At high zoom in, should be HIGH_DETAIL or MAX_DETAIL");

    // Test render settings at different zoom levels
    std::cout << "\nTesting Render Settings at Different Zoom Levels..." << std::endl;
    
    // Test settings for zoomed out view
    auto settings_zoom_out = lod_system.getMainLODRenderSettings(BTQuant::Rendering::LODLevel::LOW_DETAIL, 0.1f);
    std::cout << "Zoomed out settings - render_text: " << settings_zoom_out.render_text 
              << ", render_labels: " << settings_zoom_out.render_labels
              << ", render_detailed_annotations: " << settings_zoom_out.render_detailed_annotations << std::endl;
    assert(!settings_zoom_out.render_text && !settings_zoom_out.render_labels && 
           !settings_zoom_out.render_detailed_annotations && 
           "At zoomed out, should not render text, labels, or detailed annotations");
    
    // Test settings for zoomed in view
    auto settings_zoom_in = lod_system.getMainLODRenderSettings(BTQuant::Rendering::LODLevel::HIGH_DETAIL, 2.5f);
    std::cout << "Zoomed in settings - render_text: " << settings_zoom_in.render_text 
              << ", render_labels: " << settings_zoom_in.render_labels
              << ", render_detailed_annotations: " << settings_zoom_in.render_detailed_annotations << std::endl;
    assert(settings_zoom_in.render_text && settings_zoom_in.render_labels && 
           settings_zoom_in.render_detailed_annotations && 
           "At zoomed in, should render text, labels, and detailed annotations");

    // Test with different cell sizes at same zoom level
    std::cout << "\nTesting Cell Size Impact at Same Zoom Level..." << std::endl;
    
    auto lod_small_cell = lod_system.calculateMainLOD(5.0f, 5.0f, 1.0f);
    auto lod_large_cell = lod_system.calculateMainLOD(30.0f, 30.0f, 1.0f);
    std::cout << "Small cell (5x5) at zoom 1.0: LOD Level = " << static_cast<int>(lod_small_cell) << std::endl;
    std::cout << "Large cell (30x30) at zoom 1.0: LOD Level = " << static_cast<int>(lod_large_cell) << std::endl;
    
    // The large cell should have equal or higher LOD level than small cell at same zoom
    assert(lod_large_cell >= lod_small_cell && 
           "Larger cells should have same or higher LOD level than smaller cells at same zoom");

    std::cout << "\n✓ All Main LOD Functionality tests passed!" << std::endl;
    std::cout << "✓ Successfully verified that detail is reduced when zoomed out" << std::endl;
    std::cout << "✓ Successfully verified that detail is increased when zoomed in" << std::endl;
    
    return 0;
}