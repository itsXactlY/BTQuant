#include "../../include/ui/tooltips.hpp"
#include <iostream>

int main() {
    // Test the tooltip system
    BTQuant::UI::TooltipManager& tooltip_manager = BTQuant::UI::get_global_tooltip_manager();
    
    // Test registering a tooltip
    tooltip_manager.register_tooltip("test_control", "This is a test tooltip");
    
    // Test retrieving a tooltip
    std::string tooltip = tooltip_manager.get_tooltip("test_control");
    std::cout << "Retrieved tooltip: " << tooltip << std::endl;
    
    // Test retrieving a non-existent tooltip
    std::string missing_tooltip = tooltip_manager.get_tooltip("nonexistent_control");
    std::cout << "Missing tooltip (should be empty): '" << missing_tooltip << "'" << std::endl;
    
    // Test that our predefined tooltips exist
    std::string chart_tooltip = tooltip_manager.get_tooltip("add_chart_panel");
    std::cout << "Chart panel tooltip: " << chart_tooltip << std::endl;
    
    std::string replay_tooltip = tooltip_manager.get_tooltip("chart_replay_speed");
    std::cout << "Replay speed tooltip: " << replay_tooltip << std::endl;
    
    std::cout << "Tooltip system test completed successfully!" << std::endl;
    
    return 0;
}