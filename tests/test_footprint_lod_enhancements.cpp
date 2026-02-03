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

    std::cout << "\nAll tests completed successfully!" << std::endl;
    return 0;
}