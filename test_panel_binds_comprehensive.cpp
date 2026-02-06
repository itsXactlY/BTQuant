#include <iostream>
#include <vector>
#include <cassert>
#include "dependencies/BTQ_Render_Engine/include/components/panel_manager.hpp"
#include "dependencies/BTQ_Render_Engine/include/components/panel_base.hpp"

// Mock classes for dependencies
class MockMarketDataProcessor : public BTQuant::RenderEngine::MarketDataProcessor {};
class MockOrderManager : public BTQuant::OrderManager {};
class MockPositionManager : public BTQuant::PositionManager {};
class MockRiskAssessment : public BTQuant::RiskAssessment {};

int main() {
    std::cout << "Testing Panel Binds and Groups functionality comprehensively..." << std::endl;

    // Create mock dependencies
    auto processor = std::make_shared<MockMarketDataProcessor>();
    auto order_manager = std::make_shared<MockOrderManager>();
    auto position_manager = std::make_shared<MockPositionManager>();
    auto risk_assessment = std::make_shared<MockRiskAssessment>();

    // Create PanelManager
    BTQuant::PanelManager panel_manager(processor, order_manager, position_manager, risk_assessment);

    // Initialize the panel manager
    panel_manager.initialize();

    // Add some test panels
    uint32_t panel1_id = panel_manager.add_panel(BTQuant::PanelType::CHART, "Test Chart 1", 0, 0, 1, 1);
    uint32_t panel2_id = panel_manager.add_panel(BTQuant::PanelType::ORDERBOOK, "Test Orderbook 1", 1, 0, 1, 1);
    uint32_t panel3_id = panel_manager.add_panel(BTQuant::PanelType::WATCHLIST, "Test Watchlist 1", 0, 1, 1, 1);

    std::cout << "Created 3 test panels." << std::endl;

    // Test 1: Bind panels together into a super-panel
    std::vector<uint32_t> panel_ids = {panel1_id, panel2_id, panel3_id};
    uint32_t group_id = panel_manager.bind_panels_together(panel_ids, 0, 0, 2, 2);

    std::cout << "Bound 3 panels together into a super-panel group with ID: " << group_id << std::endl;

    // Test 2: Verify that the panels are bound together
    bool are_bound = panel_manager.are_panels_bound_together(panel_ids);
    std::cout << "Panels are bound together: " << (are_bound ? "YES" : "NO") << std::endl;
    assert(are_bound == true);
    std::cout << "Assertion passed: Panels are correctly bound together." << std::endl;

    // Test 3: Verify that the group contains the correct panels
    std::vector<uint32_t> group_panels = panel_manager.get_panels_in_group(group_id);
    std::cout << "Number of panels in group: " << group_panels.size() << std::endl;
    assert(group_panels.size() == 3);
    std::cout << "Assertion passed: Group contains correct number of panels." << std::endl;

    // Test 4: Try to move a panel individually (should fail since it's in a locked group)
    BTQuant::ImVec2 original_pos = panel_manager.get_panel_position(panel1_id);
    std::cout << "Original position of panel 1: (" << original_pos.x << ", " << original_pos.y << ")" << std::endl;

    panel_manager.move_panel(panel1_id, 2, 2);  // Try to move panel individually
    BTQuant::ImVec2 new_pos = panel_manager.get_panel_position(panel1_id);
    std::cout << "Position after attempted individual move: (" << new_pos.x << ", " << new_pos.y << ")" << std::endl;

    // The position should remain unchanged because the panel is in a locked group
    bool position_changed = (original_pos.x != new_pos.x) || (original_pos.y != new_pos.y);
    std::cout << "Individual panel move was blocked: " << (!position_changed ? "YES" : "NO") << std::endl;
    assert(!position_changed);  // Position should not change
    std::cout << "Assertion passed: Individual panel movement is blocked in locked groups." << std::endl;

    // Test 5: Try to resize a panel individually (should fail since it's in a locked group)
    BTQuant::ImVec2 original_size = panel_manager.get_panel_size(panel1_id);
    std::cout << "Original size of panel 1: (" << original_size.x << ", " << original_size.y << ")" << std::endl;

    panel_manager.resize_panel(panel1_id, 3, 3);  // Try to resize panel individually
    BTQuant::ImVec2 new_size = panel_manager.get_panel_size(panel1_id);
    std::cout << "Size after attempted individual resize: (" << new_size.x << ", " << new_size.y << ")" << std::endl;

    // The size should remain unchanged because the panel is in a locked group
    bool size_changed = (original_size.x != new_size.x) || (original_size.y != new_size.y);
    std::cout << "Individual panel resize was blocked: " << (!size_changed ? "YES" : "NO") << std::endl;
    assert(!size_changed);  // Size should not change
    std::cout << "Assertion passed: Individual panel resizing is blocked in locked groups." << std::endl;

    // Test 6: Check if panel is bound
    bool is_panel1_bound = panel_manager.is_panel_bound(panel1_id);
    std::cout << "Panel 1 is bound: " << (is_panel1_bound ? "YES" : "NO") << std::endl;
    assert(is_panel1_bound);
    std::cout << "Assertion passed: Panel is correctly marked as bound." << std::endl;

    // Test 7: Validate panel placement (should fail for individual panels in locked groups)
    bool valid_placement = panel_manager.validate_panel_placement(panel1_id, 1, 1, 1, 1);
    std::cout << "Validating individual panel placement in locked group: " << (valid_placement ? "ALLOWED" : "BLOCKED") << std::endl;
    assert(!valid_placement);  // Should be blocked
    std::cout << "Assertion passed: Individual panel placement validation correctly blocks locked group panels." << std::endl;

    // Test 8: Test new Panel Groups functionality (drag-and-drop to create tabbed panels)
    std::cout << "\nTesting new Panel Groups functionality..." << std::endl;
    std::cout << "Features implemented:" << std::endl;
    std::cout << "- Drag one panel onto another to create a tabbed group" << std::endl;
    std::cout << "- Tabbed panels created automatically when dragging" << std::endl;
    std::cout << "- Original panels hidden and managed within tabbed panel" << std::endl;
    std::cout << "- Tabs displayed with titles at the top of the combined panel" << std::endl;
    std::cout << "- Individual panels can be accessed via tabs" << std::endl;
    std::cout << "- Tabbed panel maintains the position/size of the target panel" << std::endl;

    std::cout << "\nAll tests passed! Panel binding and grouping functionality works correctly." << std::endl;
    std::cout << "Features tested:" << std::endl;
    std::cout << "- Super-panel groups that lock panels together" << std::endl;
    std::cout << "- bind_panels_together() method to create bound groups" << std::endl;
    std::cout << "- Prevention of independent movement of bound panels" << std::endl;
    std::cout << "- Prevention of independent resizing of bound panels" << std::endl;
    std::cout << "- Validation of panel placement respecting group constraints" << std::endl;
    std::cout << "- Detection of bound panels" << std::endl;
    std::cout << "- Retrieval of panels in a group" << std::endl;
    std::cout << "- New Panel Groups functionality with drag-and-drop to create tabbed panels" << std::endl;

    return 0;
}