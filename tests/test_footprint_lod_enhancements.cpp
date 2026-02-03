#include "rendering/footprint_lod.hpp"
#include "components/footprint_panel.hpp"
#include <iostream>
#include <cassert>
#include <vector>

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

    // Test new advanced LOD functionality
    std::cout << "\nTesting Advanced LOD Features:" << std::endl;

    // Test adaptive temporal LOD
    auto temporal_lod_recent = lod_system.calculateAdaptiveTemporalLOD(10.0f, 10.0f, 1.0f, 0.1f);
    auto temporal_lod_old = lod_system.calculateAdaptiveTemporalLOD(10.0f, 10.0f, 1.0f, 10.0f);
    std::cout << "Temporal LOD (recent update): " << static_cast<int>(temporal_lod_recent) << std::endl;
    std::cout << "Temporal LOD (old update): " << static_cast<int>(temporal_lod_old) << std::endl;

    // Test temporal LOD render settings
    auto temporal_settings_recent = lod_system.getTemporalLODRenderSettings(BTQuant::Rendering::LODLevel::HIGH_DETAIL, 0.1f);
    auto temporal_settings_old = lod_system.getTemporalLODRenderSettings(BTQuant::Rendering::LODLevel::HIGH_DETAIL, 5.0f);
    std::cout << "Temporal render settings (recent) - alpha: " << temporal_settings_recent.alpha_multiplier
              << ", detailed annotations: " << temporal_settings_recent.render_detailed_annotations << std::endl;
    std::cout << "Temporal render settings (old) - alpha: " << temporal_settings_old.alpha_multiplier
              << ", detailed annotations: " << temporal_settings_old.render_detailed_annotations << std::endl;

    // Test contextual LOD
    std::vector<BTQuant::FootprintCell> nearby_cells;
    for (int i = 0; i < 5; ++i) {
        nearby_cells.emplace_back(i * 0.1, i * 0.1, 0.01, 0.01, 100.0 + i * 10, 50.0 + i * 5, 10 + i, 100.0);
    }
    auto contextual_lod = lod_system.calculateContextualLOD(10.0f, 10.0f, 1.0f, nearby_cells);
    std::cout << "Contextual LOD Level: " << static_cast<int>(contextual_lod) << std::endl;

    // Test foveated LOD
    ImVec2 cell_center = ImVec2(100.0f, 100.0f);
    ImVec2 focus_point = ImVec2(100.0f, 100.0f);  // Same as cell center
    float focus_radius_inner = 50.0f;
    float focus_radius_outer = 100.0f;
    auto foveated_lod = lod_system.calculateFoveatedLOD(10.0f, 10.0f, 1.0f, cell_center, focus_point,
                                                       focus_radius_inner, focus_radius_outer);
    std::cout << "Foveated LOD Level (at focus): " << static_cast<int>(foveated_lod) << std::endl;

    // Test foveated LOD with distant focus
    ImVec2 distant_focus = ImVec2(1000.0f, 1000.0f);
    auto foveated_lod_distant = lod_system.calculateFoveatedLOD(10.0f, 10.0f, 1.0f, cell_center, distant_focus,
                                                               focus_radius_inner, focus_radius_outer);
    std::cout << "Foveated LOD Level (distant focus): " << static_cast<int>(foveated_lod_distant) << std::endl;

    // Test new zoom-specific LOD functionality
    std::cout << "\nTesting New Zoom-Specific LOD Features:" << std::endl;

    // Test zoom-out LOD calculation
    auto zoom_out_lod = lod_system.calculateZoomOutLOD(10.0f, 10.0f, 0.1f);
    std::cout << "Zoom-out LOD Level (0.1 zoom): " << static_cast<int>(zoom_out_lod) << std::endl;

    // Test zoom-in LOD calculation
    auto zoom_in_lod = lod_system.calculateZoomInLOD(10.0f, 10.0f, 4.0f);
    std::cout << "Zoom-in LOD Level (4.0 zoom): " << static_cast<int>(zoom_in_lod) << std::endl;

    // Test zoom-specific render settings
    auto zoom_out_settings = lod_system.getZoomOutOptimizedRenderSettings(BTQuant::Rendering::LODLevel::HIGH_DETAIL, 0.2f);
    std::cout << "Zoom-out optimized settings - render_text: " << zoom_out_settings.render_text
              << ", render_labels: " << zoom_out_settings.render_labels << std::endl;

    auto zoom_in_settings = lod_system.getZoomInEnhancedRenderSettings(BTQuant::Rendering::LODLevel::MEDIUM_DETAIL, 3.0f);
    std::cout << "Zoom-in enhanced settings - render_text: " << zoom_in_settings.render_text
              << ", render_detailed_annotations: " << zoom_in_settings.render_detailed_annotations
              << ", border_thickness: " << zoom_in_settings.border_thickness << std::endl;

    // Test adaptive zoom LOD calculation
    auto adaptive_lod = lod_system.calculateAdaptiveZoomLOD(10.0f, 10.0f, 0.1f, 800.0f, 600.0f, 10000);
    std::cout << "Adaptive zoom LOD Level (high density): " << static_cast<int>(adaptive_lod) << std::endl;

    auto adaptive_lod_sparse = lod_system.calculateAdaptiveZoomLOD(10.0f, 10.0f, 3.0f, 800.0f, 600.0f, 100);
    std::cout << "Adaptive zoom LOD Level (sparse, zoomed in): " << static_cast<int>(adaptive_lod_sparse) << std::endl;

    // Test adaptive zoom render settings
    auto adaptive_settings = lod_system.getAdaptiveZoomRenderSettings(BTQuant::Rendering::LODLevel::HIGH_DETAIL, 0.1f, 10000);
    std::cout << "Adaptive zoom settings (high density) - render_text: " << adaptive_settings.render_text
              << ", alpha_multiplier: " << adaptive_settings.alpha_multiplier << std::endl;

    auto adaptive_settings_sparse = lod_system.getAdaptiveZoomRenderSettings(BTQuant::Rendering::LODLevel::MEDIUM_DETAIL, 3.0f, 100);
    std::cout << "Adaptive zoom settings (sparse, zoomed in) - render_detailed_annotations: "
              << adaptive_settings_sparse.render_detailed_annotations
              << ", border_thickness: " << adaptive_settings_sparse.border_thickness << std::endl;

    std::cout << "\nAll enhanced LOD tests completed successfully!" << std::endl;
    return 0;
}