#ifndef PANEL_MANAGER_HPP
#define PANEL_MANAGER_HPP

#include <memory>
#include <vector>
#include <functional>
#include <string>

// Include the render engine panel manager to leverage existing functionality
#include "../../dependencies/BTQ_Render_Engine/include/components/panel_manager.hpp"

namespace ui {

class PanelManager {
public:
    PanelManager(
        std::shared_ptr<BTQuant::HotSpineDataBridge> bridge,
        std::shared_ptr<BTQuant::RenderEngine::MarketDataProcessor> processor,
        std::shared_ptr<BTQuant::OrderManager> order_manager,
        std::shared_ptr<BTQuant::PositionManager> position_manager,
        std::shared_ptr<BTQuant::RiskAssessment> risk_assessment,
        BTQuant::RenderEngine::MarketMicrostructureRenderer* micro_renderer = nullptr
    );
    ~PanelManager();

    // Public interface methods for panel management
    void initialize();
    void update(float dt);
    void render();

    // Panel management
    uint32_t add_panel(BTQuant::PanelType type, const std::string& title = "", int grid_x = -1,
                       int grid_y = -1, int width = 1, int height = 1);
    void remove_panel(uint32_t panel_id);
    void move_panel(uint32_t panel_id, int new_grid_x, int new_grid_y);
    void resize_panel(uint32_t panel_id, int new_width, int new_height);
    
    // Tabbed group functionality - allowing panels to be combined into tabs
    uint32_t create_tabbed_group(uint32_t target_panel_id);
    bool add_panel_to_tabbed_group(uint32_t tabbed_group_id, uint32_t panel_to_add_id);
    bool remove_panel_from_tabbed_group(uint32_t tabbed_group_id, uint32_t panel_to_remove_id);
    bool is_panel_in_tabbed_group(uint32_t panel_id) const;
    uint32_t get_containing_tabbed_group_id(uint32_t panel_id) const;
    bool can_drag_panel_to_target(uint32_t source_panel_id, uint32_t target_panel_id) const;

    // Drag and drop for tabbed groups
    void handle_panel_drag_drop();

    // Layout management
    void save_layout(const std::string& filename);
    void load_layout(const std::string& filename);
    void auto_arrange_panels();

    // Panel binding functionality - "Super-panel" grid locking
    uint32_t create_panel_group(const std::vector<uint32_t>& panel_ids);
    bool add_panel_to_group(uint32_t group_id, uint32_t panel_id);
    bool remove_panel_from_group(uint32_t group_id, uint32_t panel_id);
    bool destroy_panel_group(uint32_t group_id);
    bool is_panel_in_group(uint32_t panel_id) const;
    uint32_t get_panel_group_id(uint32_t panel_id) const;
    uint32_t create_super_panel_from_adjacent(uint32_t panel1_id, uint32_t panel2_id);
    uint32_t create_super_panel_from_rectangular_region(int start_x, int start_y, int width, int height);
    void lock_panel_group(uint32_t group_id, bool locked = true);
    bool is_panel_group_locked(uint32_t group_id) const;
    void set_prevent_overlap_for_group(uint32_t group_id, bool prevent = true);
    bool does_group_prevent_overlap(uint32_t group_id) const;

    // Access to underlying render engine panel manager
    BTQuant::PanelManager* get_render_engine_panel_manager() { return render_engine_panel_manager_.get(); }

private:
    std::unique_ptr<BTQuant::PanelManager> render_engine_panel_manager_;
};

} // namespace ui

#endif // PANEL_MANAGER_HPP