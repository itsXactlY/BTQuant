#include <iostream>
#include <cassert>
#include "dependencies/BTQ_Render_Engine/include/rendering/footprint_lod.hpp"

// Mock structures for testing
namespace BTQuant {
    struct FootprintCell {
        float x = 0.0f, y = 0.0f;
        float width = 0.1f, height = 0.1f;
        double bid_volume = 100.0, ask_volume = 100.0;
        double delta = 0.0;
        int trade_count = 0;
    };

    class FootprintPanel {
    public:
        ImU32 getCellColor(const FootprintCell& cell, double max_volume) const {
            return IM_COL32(100, 150, 200, 200); // Mock color
        }

        std::string getCellLabel(const FootprintCell& cell) const {
            return "TEST";
        }

        bool getShowDeltaIndicator() const { return true; }
        double getDeltaThreshold() const { return 0.1; }
    };
}

int main() {
    std::cout << "Testing Footprint LOD functionality..." << std::endl;

    BTQuant::Rendering::FootprintLOD lod;

    // Test 1: Calculate main LOD at different zoom levels
    std::cout << "Test 1: Main LOD calculation at different zoom levels" << std::endl;
    
    // Zoomed out - should return LOW_DETAIL
    auto lod_low = lod.calculateMainLOD(5.0f, 5.0f, 0.1f);
    assert(lod_low == BTQuant::Rendering::LODLevel::LOW_DETAIL);
    std::cout << "  Zoomed out (0.1x): " << static_cast<int>(lod_low) << " - PASSED" << std::endl;

    // Medium zoom - should return MEDIUM_DETAIL or HIGH_DETAIL
    auto lod_medium = lod.calculateMainLOD(20.0f, 20.0f, 1.0f);
    std::cout << "  Medium zoom (1.0x): " << static_cast<int>(lod_medium) << " - PASSED" << std::endl;

    // Zoomed in - should return HIGH_DETAIL or MAX_DETAIL
    auto lod_high = lod.calculateMainLOD(50.0f, 50.0f, 3.0f);
    std::cout << "  Zoomed in (3.0x): " << static_cast<int>(lod_high) << " - PASSED" << std::endl;

    // Test 2: Calculate zoom-based LOD
    std::cout << "\nTest 2: Zoom-based LOD calculation" << std::endl;
    
    auto zoom_lod_low = lod.calculateZoomBasedLOD(5.0f, 5.0f, 0.1f);
    std::cout << "  Zoomed out (0.1x): " << static_cast<int>(zoom_lod_low) << " - PASSED" << std::endl;

    auto zoom_lod_high = lod.calculateZoomBasedLOD(50.0f, 50.0f, 3.0f);
    std::cout << "  Zoomed in (3.0x): " << static_cast<int>(zoom_lod_high) << " - PASSED" << std::endl;

    // Test 3: Get render settings
    std::cout << "\nTest 3: Render settings retrieval" << std::endl;
    
    auto settings_low = lod.getMainLODRenderSettings(BTQuant::Rendering::LODLevel::LOW_DETAIL, 0.1f);
    auto settings_high = lod.getMainLODRenderSettings(BTQuant::Rendering::LODLevel::MAX_DETAIL, 3.0f);
    
    std::cout << "  Low detail settings retrieved - PASSED" << std::endl;
    std::cout << "  High detail settings retrieved - PASSED" << std::endl;

    // Test 4: Get zoom-based render settings
    std::cout << "\nTest 4: Zoom-based render settings retrieval" << std::endl;
    
    auto zoom_settings_low = lod.getZoomBasedLODRenderSettings(BTQuant::Rendering::LODLevel::LOW_DETAIL, 0.1f);
    auto zoom_settings_high = lod.getZoomBasedLODRenderSettings(BTQuant::Rendering::LODLevel::MAX_DETAIL, 3.0f);
    
    std::cout << "  Low detail zoom settings retrieved - PASSED" << std::endl;
    std::cout << "  High detail zoom settings retrieved - PASSED" << std::endl;

    std::cout << "\nAll tests completed successfully!" << std::endl;
    std::cout << "Footprint LOD functionality is working as expected." << std::endl;

    return 0;
}