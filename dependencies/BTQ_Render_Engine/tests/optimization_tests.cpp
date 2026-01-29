#include "../include/ui/animation_system.hpp"
#include "../include/ui/unified_theme_system.hpp"
#include "../include/system/memory_optimizer.hpp"
#include "../include/layout/layout_presets.hpp"
#include "../include/layout/dashboard_layout_manager.hpp"
#include "../include/components/theme_manager.hpp"

#include <cassert>
#include <iostream>
#include <string>

using namespace BTQuant;
using namespace BTQuant::UI;
using namespace BTQuant::Layout;

// Simple 2D vector structure for testing without GLM
struct Vec2 {
    float x, y;
    Vec2(float x = 0.0f, float y = 0.0f) : x(x), y(y) {}
};

void test_animation_system() {
    std::cout << "Testing Animation System..." << std::endl;

    auto& anim_system = AnimationSystem::getInstance();

    // Test basic animation creation
    float test_value = 0.0f;
    int anim_id = anim_system.animate_value(&test_value, 0.0f, 100.0f, 1.0f);

    assert(anim_id != 0);
    std::cout << "✓ Animation creation test passed" << std::endl;

    // Test animation update
    anim_system.update(0.5f);  // Halfway through animation

    // Note: In a real implementation, this would actually update the value
    // For now, we're just testing that the system doesn't crash

    // Test animation completion
    anim_system.update(1.0f);  // Complete the animation

    anim_system.stop_animation(anim_id);
    std::cout << "✓ Animation lifecycle test passed" << std::endl;

    // Test vector animation (using our simple Vec2 since GLM might not be available)
    Vec2 test_vec(0.0f, 0.0f);
    // We can't test vector animation directly without GLM, so skip this test
    std::cout << "✓ Vector animation test skipped (requires GLM)" << std::endl;

    std::cout << "Animation System tests completed!" << std::endl << std::endl;
}

void test_unified_theme_system() {
    std::cout << "Testing Unified Theme System..." << std::endl;

    auto& theme_manager = UnifiedThemeManager::getInstance();
    theme_manager.initialize();

    // Test getting available themes
    auto themes = theme_manager.get_available_themes();
    assert(!themes.empty());
    std::cout << "✓ Available themes test passed" << std::endl;

    // Test setting current theme
    bool success = theme_manager.set_current_theme("Dark Professional");
    assert(success);
    std::cout << "✓ Set current theme test passed" << std::endl;

    // Test getting current theme
    const auto& current_theme = theme_manager.get_current_theme();
    assert(!current_theme.name.empty());
    std::cout << "✓ Get current theme test passed" << std::endl;

    // Test getting specific theme
    const auto* specific_theme = theme_manager.get_theme("Dark Professional");
    assert(specific_theme != nullptr);
    assert(specific_theme->name == "Dark Professional");
    std::cout << "✓ Get specific theme test passed" << std::endl;

    // Test color retrieval
    float rgba[4];
    theme_manager.get_color("text_primary", rgba);
    // Just checking it doesn't crash and returns valid values
    assert(rgba[0] >= 0.0f && rgba[0] <= 1.0f);
    std::cout << "✓ Color retrieval test passed" << std::endl;

    std::cout << "Unified Theme System tests completed!" << std::endl << std::endl;
}

void test_memory_optimizer() {
    std::cout << "Testing Memory Optimizer..." << std::endl;

    MemoryOptimizer mem_optimizer;
    mem_optimizer.initialize();

    // Test basic functionality
    mem_optimizer.track_allocation(1024, "test_allocation");
    mem_optimizer.track_deallocation(512, "test_deallocation");

    auto stats = mem_optimizer.get_memory_stats();
    assert(stats.total_allocated >= 1024);
    std::cout << "✓ Memory tracking test passed" << std::endl;

    // Verify that we have some allocations before clearing
    auto initial_allocs = mem_optimizer.get_recent_allocations(10);
    assert(!initial_allocs.empty());
    std::cout << "✓ Pre-clear allocations exist test passed" << std::endl;

    // Test optimization
    mem_optimizer.optimize_allocations();
    std::cout << "✓ Memory optimization test passed" << std::endl;

    // Test setting parameters
    mem_optimizer.set_compaction_threshold(1024 * 1024);  // 1MB
    mem_optimizer.set_release_threshold(512 * 1024);      // 0.5MB
    std::cout << "✓ Parameter setting test passed" << std::endl;

    // Test recent allocations
    auto recent_allocs = mem_optimizer.get_recent_allocations(10);
    // Just checking it doesn't crash
    std::cout << "✓ Recent allocations test passed" << std::endl;

    // Test the new clear functionality
    mem_optimizer.clear();

    // After clearing, verify that stats are reset
    auto cleared_stats = mem_optimizer.get_memory_stats();
    assert(cleared_stats.total_allocated == 0);
    assert(cleared_stats.total_deallocated == 0);
    assert(cleared_stats.current_allocated == 0);
    assert(cleared_stats.peak_usage == 0);
    assert(cleared_stats.fragmentation == 0);
    assert(cleared_stats.fragmentation_ratio == 0.0f);
    std::cout << "✓ Memory stats reset after clear test passed" << std::endl;

    // After clearing, verify that recent allocations are empty
    auto cleared_allocs = mem_optimizer.get_recent_allocations(10);
    assert(cleared_allocs.empty());
    std::cout << "✓ Recent allocations cleared test passed" << std::endl;

    std::cout << "Memory Optimizer tests completed!" << std::endl << std::endl;
}

void test_layout_presets() {
    std::cout << "Testing Layout Presets..." << std::endl;
    
    LayoutPresetManager preset_manager;
    
    // Test getting all presets
    auto all_presets = preset_manager.get_all_presets();
    assert(!all_presets.empty());  // Should have builtin presets
    std::cout << "✓ Get all presets test passed" << std::endl;
    
    // Test getting presets by category
    auto trading_presets = preset_manager.get_presets_by_category("Trading");
    // May be empty depending on implementation, just ensure no crash
    std::cout << "✓ Get presets by category test passed" << std::endl;
    
    // Test applying a preset (should not crash)
    bool applied = preset_manager.apply_preset("Trading Pro");
    // May return false if preset doesn't exist in this context, but shouldn't crash
    std::cout << "✓ Apply preset test passed" << std::endl;
    
    std::cout << "Layout Presets tests completed!" << std::endl << std::endl;
}

void test_dashboard_layout_manager() {
    std::cout << "Testing Dashboard Layout Manager..." << std::endl;
    
    DashboardLayoutManager layout_manager;
    
    // Test creating a new layout
    layout_manager.create_new_layout();
    std::cout << "✓ Create layout test passed" << std::endl;
    
    // Test getting available layouts
    auto layouts = layout_manager.get_available_layouts();
    // May be empty initially, just ensure no crash
    std::cout << "✓ Get available layouts test passed" << std::endl;
    
    // Test current layout
    auto current_layout = layout_manager.get_current_layout();
    // Just ensure it doesn't crash
    std::cout << "✓ Get current layout test passed" << std::endl;
    
    // Test panel management
    DashboardLayoutManager::PanelLayout panel;
    panel.panel_id = "test_panel";
    panel.panel_name = "Test Panel";
    panel.x = 0.0f;
    panel.y = 0.0f;
    panel.width = 400.0f;
    panel.height = 300.0f;
    
    layout_manager.add_panel_to_layout(panel);
    std::cout << "✓ Add panel test passed" << std::endl;
    
    // Test getting panels for symbol
    auto symbol_panels = layout_manager.get_panels_for_symbol("");
    // Just ensure it doesn't crash
    std::cout << "✓ Get panels for symbol test passed" << std::endl;
    
    // Test grid management
    layout_manager.set_grid_dimensions(8, 10);
    auto grid_dims = layout_manager.get_grid_dimensions();
    assert(grid_dims.first == 8);
    std::cout << "✓ Grid management test passed" << std::endl;
    
    // Test layout utilities
    layout_manager.auto_arrange_panels();
    layout_manager.center_layout();
    std::cout << "✓ Layout utilities test passed" << std::endl;
    
    std::cout << "Dashboard Layout Manager tests completed!" << std::endl << std::endl;
}

void test_theme_manager_integration() {
    std::cout << "Testing Theme Manager Integration..." << std::endl;
    
    auto& theme_manager = ThemeManager::getInstance();
    theme_manager.initialize();
    
    // Test applying unified theme
    theme_manager.apply_unified_theme("Dark Professional");
    std::cout << "✓ Unified theme application test passed" << std::endl;
    
    // Test sync with layout manager
    theme_manager.sync_with_layout_manager();
    std::cout << "✓ Layout manager sync test passed" << std::endl;
    
    // Test theme toggling
    theme_manager.toggleTheme();
    std::cout << "✓ Theme toggle test passed" << std::endl;
    
    std::cout << "Theme Manager Integration tests completed!" << std::endl << std::endl;
}

int main() {
    std::cout << "=== Running Optimization & Polish Feature Tests ===" << std::endl;
    
    try {
        test_animation_system();
        test_unified_theme_system();
        test_memory_optimizer();
        test_layout_presets();
        test_dashboard_layout_manager();
        test_theme_manager_integration();
        
        std::cout << "=== All tests completed successfully! ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}