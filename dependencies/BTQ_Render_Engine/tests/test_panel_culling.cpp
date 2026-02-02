#include "../include/rendering/panel_culler.hpp"
#include "../include/components/panel_base.hpp"
#include <iostream>
#include <cassert>

// Mock panel implementation for testing
class MockPanel : public BTQuant::PanelBase {
public:
    MockPanel(const BTQuant::PanelConfig& config) : BTQuant::PanelBase(config) {}
    
    void render() override {
        // Mock implementation
    }
};

int main() {
    std::cout << "Testing Panel Culling Functionality...\n";
    
    BTQuant::RenderEngine::PanelCuller culler;
    
    // Test 1: Visible panel within viewport should render
    {
        BTQuant::PanelConfig config;
        config.visible = true;
        config.position = ImVec2(100, 100);
        config.size = ImVec2(200, 150);
        
        MockPanel panel(config);
        
        // Set viewport bounds
        culler.set_viewport_bounds(ImVec2(0, 0), ImVec2(1920, 1080));
        
        bool should_render = culler.should_render_panel(panel);
        assert(should_render == true);
        std::cout << "✓ Test 1 PASSED: Visible panel within viewport should render\n";
    }
    
    // Test 2: Hidden/minimized panel should not render
    {
        BTQuant::PanelConfig config;
        config.visible = false;  // Panel is hidden/minimized
        config.position = ImVec2(100, 100);
        config.size = ImVec2(200, 150);
        
        MockPanel panel(config);
        
        culler.set_viewport_bounds(ImVec2(0, 0), ImVec2(1920, 1080));
        
        bool should_render = culler.should_render_panel(panel);
        assert(should_render == false);
        std::cout << "✓ Test 2 PASSED: Hidden/minimized panel should not render\n";
    }
    
    // Test 3: Off-screen panel (too far right) should not render
    {
        BTQuant::PanelConfig config;
        config.visible = true;
        config.position = ImVec2(2000, 100);  // Beyond right edge of 1920px screen
        config.size = ImVec2(200, 150);
        
        MockPanel panel(config);
        
        culler.set_viewport_bounds(ImVec2(0, 0), ImVec2(1920, 1080));
        
        bool should_render = culler.should_render_panel(panel);
        assert(should_render == false);
        std::cout << "✓ Test 3 PASSED: Off-screen panel (too far right) should not render\n";
    }
    
    // Test 4: Off-screen panel (too far down) should not render
    {
        BTQuant::PanelConfig config;
        config.visible = true;
        config.position = ImVec2(100, 1200);  // Below bottom edge of 1080px screen
        config.size = ImVec2(200, 150);
        
        MockPanel panel(config);
        
        culler.set_viewport_bounds(ImVec2(0, 0), ImVec2(1920, 1080));
        
        bool should_render = culler.should_render_panel(panel);
        assert(should_render == false);
        std::cout << "✓ Test 4 PASSED: Off-screen panel (too far down) should not render\n";
    }
    
    // Test 5: Off-screen panel (too far left) should not render
    {
        BTQuant::PanelConfig config;
        config.visible = true;
        config.position = ImVec2(-300, 100);  // Too far left
        config.size = ImVec2(200, 150);
        
        MockPanel panel(config);
        
        culler.set_viewport_bounds(ImVec2(0, 0), ImVec2(1920, 1080));
        
        bool should_render = culler.should_render_panel(panel);
        assert(should_render == false);
        std::cout << "✓ Test 5 PASSED: Off-screen panel (too far left) should not render\n";
    }
    
    // Test 6: Off-screen panel (too far up) should not render
    {
        BTQuant::PanelConfig config;
        config.visible = true;
        config.position = ImVec2(100, -200);  // Too far up
        config.size = ImVec2(200, 150);
        
        MockPanel panel(config);
        
        culler.set_viewport_bounds(ImVec2(0, 0), ImVec2(1920, 1080));
        
        bool should_render = culler.should_render_panel(panel);
        assert(should_render == false);
        std::cout << "✓ Test 6 PASSED: Off-screen panel (too far up) should not render\n";
    }
    
    // Test 7: Minimized panel should not render (regardless of size)
    {
        BTQuant::PanelConfig config;
        config.visible = true;
        config.minimized = true;  // Panel is explicitly minimized
        config.position = ImVec2(100, 100);
        config.size = ImVec2(200, 150);  // Normal size

        MockPanel panel(config);

        culler.set_viewport_bounds(ImVec2(0, 0), ImVec2(1920, 1080));

        bool should_render = culler.should_render_panel(panel);
        assert(should_render == false);
        std::cout << "✓ Test 7 PASSED: Minimized panel should not render\n";
    }

    // Test 7b: Small but not minimized panel should render if in viewport
    {
        BTQuant::PanelConfig config;
        config.visible = true;
        config.minimized = false;  // Panel is not minimized despite being small
        config.position = ImVec2(100, 100);
        config.size = ImVec2(50, 20);  // Very small but not minimized

        MockPanel panel(config);

        culler.set_viewport_bounds(ImVec2(0, 0), ImVec2(1920, 1080));

        bool should_render = culler.should_render_panel(panel);
        assert(should_render == true);  // Should render since it's in viewport and not minimized
        std::cout << "✓ Test 7b PASSED: Small but not minimized panel should render if in viewport\n";
    }
    
    // Test 8: Zero-size panel should not render
    {
        BTQuant::PanelConfig config;
        config.visible = true;
        config.position = ImVec2(100, 100);
        config.size = ImVec2(0, 0);  // Zero size

        MockPanel panel(config);

        culler.set_viewport_bounds(ImVec2(0, 0), ImVec2(1920, 1080));

        bool should_render = culler.should_render_panel(panel);
        assert(should_render == false);
        std::cout << "✓ Test 8 PASSED: Zero-size panel should not render\n";
    }

    // Test 9: Configurable visibility thresholds
    {
        BTQuant::PanelConfig config;
        config.visible = true;
        config.position = ImVec2(100, 100);
        config.size = ImVec2(50, 50);  // Area = 2500px^2

        MockPanel panel(config);

        culler.set_viewport_bounds(ImVec2(0, 0), ImVec2(1920, 1080));

        // Set custom thresholds
        culler.set_visibility_thresholds(1000.0f, 0.1f);  // At least 1000px^2 OR 10% visibility

        bool should_render = culler.should_render_panel(panel);
        assert(should_render == true);  // Panel has 2500px^2 which is > 1000px^2 threshold
        std::cout << "✓ Test 9 PASSED: Configurable visibility thresholds work\n";
    }

    // Test 10: Small visible area with strict thresholds should not render
    {
        BTQuant::PanelConfig config;
        config.visible = true;
        config.position = ImVec2(-45, 100);  // Mostly off-screen to the left
        config.size = ImVec2(50, 50);       // Area = 2500px^2

        MockPanel panel(config);

        culler.set_viewport_bounds(ImVec2(0, 0), ImVec2(1920, 1080));

        // Set very strict thresholds to test small visible area logic
        culler.set_visibility_thresholds(100.0f, 0.20f);  // Need at least 100px^2 AND 20% visibility

        bool should_render = culler.should_render_panel(panel);
        // Panel intersects viewport: width = 5px (from x=0 to x=5), height = 50px
        // Visible area = 5 * 50 = 250px^2, which is > 100px^2
        // Percentage = 250/2500 = 0.1 = 10%, which is < 20%
        // So it should NOT render (fails percentage check)
        assert(should_render == false);
        std::cout << "✓ Test 10 PASSED: Small visible area with strict thresholds correctly culled\n";
    }

    std::cout << "\nAll tests passed! Panel culling functionality is working correctly.\n";

    return 0;
}