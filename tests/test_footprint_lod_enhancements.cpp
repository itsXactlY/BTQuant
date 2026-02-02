#include "rendering/footprint_lod.hpp"
#include "components/footprint_panel.hpp"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "Testing Enhanced Footprint LOD System..." << std::endl;

    BTQuant::Rendering::FootprintLOD lod_system;

    // Test basic LOD calculation
    auto lod_level = lod_system.calculateLODLevel(10.0f, 10.0f, 1.0f);
    std::cout << "Basic LOD Level: " << static_cast<int>(lod_level) << std::endl;
    
    // Test multi-resolution LOD calculation
    auto multi_res_lod = lod_system.calculateMultiResolutionLOD(10.0f, 10.0f, 1.0f, 1000);
    std::cout << "Multi-resolution LOD Level: " << static_cast<int>(multi_res_lod) << std::endl;
    
    // Test with high cell density
    auto high_density_lod = lod_system.calculateMultiResolutionLOD(5.0f, 5.0f, 0.5f, 10000);
    std::cout << "High density LOD Level: " << static_cast<int>(high_density_lod) << std::endl;
    
    // Test predictive LOD calculation
    auto predictive_lod = lod_system.calculatePredictiveLOD(10.0f, 10.0f, 1.0f, 2.0f);
    std::cout << "Predictive LOD Level: " << static_cast<int>(predictive_lod) << std::endl;
    
    // Test hybrid LOD calculation
    BTQuant::Rendering::PerformanceMetrics metrics;
    metrics.fps = 60.0f;
    metrics.performance_degraded = false;
    
    auto hybrid_lod = lod_system.calculateHybridLOD(10.0f, 10.0f, 1.0f, 1000, metrics);
    std::cout << "Hybrid LOD Level: " << static_cast<int>(hybrid_lod) << std::endl;
    
    // Test with degraded performance
    metrics.fps = 30.0f;  // Below target of 60fps
    metrics.performance_degraded = true;
    auto degraded_hybrid_lod = lod_system.calculateHybridLOD(10.0f, 10.0f, 1.0f, 1000, metrics);
    std::cout << "Degraded performance Hybrid LOD Level: " << static_cast<int>(degraded_hybrid_lod) << std::endl;
    
    // Test rendering settings
    auto settings = lod_system.getMultiResolutionRenderSettings(BTQuant::Rendering::LODLevel::HIGH_DETAIL, 500);
    std::cout << "Multi-resolution render settings - render_text: " << settings.render_text 
              << ", render_labels: " << settings.render_labels << std::endl;
              
    auto high_density_settings = lod_system.getMultiResolutionRenderSettings(BTQuant::Rendering::LODLevel::HIGH_DETAIL, 10000);
    std::cout << "High density render settings - render_text: " << high_density_settings.render_text 
              << ", render_labels: " << high_density_settings.render_labels << std::endl;

    std::cout << "All tests completed successfully!" << std::endl;
    return 0;
}