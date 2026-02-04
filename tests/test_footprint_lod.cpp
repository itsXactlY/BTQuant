#include "dependencies/BTQ_Render_Engine/include/rendering/footprint_lod.hpp"
#include <iostream>

int main() {
    BTQuant::Rendering::FootprintLOD lod;
    
    // Test the new LOD methods
    float cell_width = 10.0f;
    float cell_height = 10.0f;
    float zoom_factor_out = 0.2f;  // Zoomed out
    float zoom_factor_in = 3.0f;   // Zoomed in
    
    // Test LOD calculation when zoomed out
    auto lod_level_out = lod.calculateZoomBasedLODDetail(cell_width, cell_height, zoom_factor_out);
    std::cout << "LOD level when zoomed out: " << static_cast<int>(lod_level_out) << std::endl;
    
    // Test LOD calculation when zoomed in
    auto lod_level_in = lod.calculateZoomBasedLODDetail(cell_width, cell_height, zoom_factor_in);
    std::cout << "LOD level when zoomed in: " << static_cast<int>(lod_level_in) << std::endl;
    
    // Test render settings
    auto settings_out = lod.getZoomBasedLODDetailRenderSettings(lod_level_out, zoom_factor_out);
    auto settings_in = lod.getZoomBasedLODDetailRenderSettings(lod_level_in, zoom_factor_in);
    
    std::cout << "Settings when zoomed out - render_text: " << settings_out.render_text 
              << ", render_borders: " << settings_out.render_borders << std::endl;
    std::cout << "Settings when zoomed in - render_text: " << settings_in.render_text 
              << ", render_borders: " << settings_in.render_borders << std::endl;
    
    std::cout << "Footprint LOD functionality test completed successfully!" << std::endl;
    
    return 0;
}