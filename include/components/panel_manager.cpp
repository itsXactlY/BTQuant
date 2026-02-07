#include "panel_manager.hpp"
#include <iostream>
#include <vector>

// Include required headers for the render engine panel manager
#include "../../dependencies/BTQ_Render_Engine/include/hotspine_data_bridge.hpp"
#include "../../dependencies/BTQ_Render_Engine/include/market_data_processor.hpp"
#include "../../dependencies/BTQ_Render_Engine/include/trading/order_manager.hpp"
#include "../../dependencies/BTQ_Render_Engine/include/trading/position_manager.hpp"
#include "../../dependencies/BTQ_Render_Engine/include/trading/risk_assessment.hpp"
#include "../../dependencies/BTQ_Render_Engine/include/components/MarketMicrostructureRenderer.h"

namespace ui {

PanelManager::PanelManager(
    std::shared_ptr<BTQuant::HotSpineDataBridge> bridge,
    std::shared_ptr<BTQuant::RenderEngine::MarketDataProcessor> processor,
    std::shared_ptr<BTQuant::OrderManager> order_manager,
    std::shared_ptr<BTQuant::PositionManager> position_manager,
    std::shared_ptr<BTQuant::RiskAssessment> risk_assessment,
    BTQuant::RenderEngine::MarketMicrostructureRenderer* micro_renderer
) {
    render_engine_panel_manager_ = std::make_unique<BTQuant::PanelManager>(
        bridge,
        processor,
        order_manager,
        position_manager,
        risk_assessment,
        micro_renderer
    );
    std::cout << "PanelManager initialized with render engine integration" << std::endl;
}

PanelManager::~PanelManager() {
    std::cout << "PanelManager destroyed" << std::endl;
}

void PanelManager::initialize() {
    if (render_engine_panel_manager_) {
        render_engine_panel_manager_->initialize();
    }
}

void PanelManager::update(float dt) {
    if (render_engine_panel_manager_) {
        render_engine_panel_manager_->update(dt);
    }
}

void PanelManager::render() {
    if (render_engine_panel_manager_) {
        render_engine_panel_manager_->render();
    }
}

uint32_t PanelManager::add_panel(BTQuant::PanelType type, const std::string& title, int grid_x,
                                 int grid_y, int width, int height) {
    if (render_engine_panel_manager_) {
        return render_engine_panel_manager_->add_panel(type, title, grid_x, grid_y, width, height);
    }
    return 0;
}

void PanelManager::remove_panel(uint32_t panel_id) {
    if (render_engine_panel_manager_) {
        render_engine_panel_manager_->remove_panel(panel_id);
    }
}

void PanelManager::move_panel(uint32_t panel_id, int new_grid_x, int new_grid_y) {
    if (render_engine_panel_manager_) {
        render_engine_panel_manager_->move_panel(panel_id, new_grid_x, new_grid_y);
    }
}

void PanelManager::resize_panel(uint32_t panel_id, int new_width, int new_height) {
    if (render_engine_panel_manager_) {
        render_engine_panel_manager_->resize_panel(panel_id, new_width, new_height);
    }
}


uint32_t PanelManager::create_tabbed_group(uint32_t target_panel_id) {
    if (render_engine_panel_manager_) {
        return render_engine_panel_manager_->create_tabbed_group(target_panel_id);
    }
    return 0;
}

bool PanelManager::add_panel_to_tabbed_group(uint32_t tabbed_group_id, uint32_t panel_to_add_id) {
    if (render_engine_panel_manager_) {
        return render_engine_panel_manager_->add_panel_to_tabbed_group(tabbed_group_id, panel_to_add_id);
    }
    return false;
}

bool PanelManager::remove_panel_from_tabbed_group(uint32_t tabbed_group_id, uint32_t panel_to_remove_id) {
    if (render_engine_panel_manager_) {
        return render_engine_panel_manager_->remove_panel_from_tabbed_group(tabbed_group_id, panel_to_remove_id);
    }
    return false;
}

bool PanelManager::is_panel_in_tabbed_group(uint32_t panel_id) const {
    if (render_engine_panel_manager_) {
        return render_engine_panel_manager_->is_panel_in_tabbed_group(panel_id);
    }
    return false;
}

uint32_t PanelManager::get_containing_tabbed_group_id(uint32_t panel_id) const {
    if (render_engine_panel_manager_) {
        return render_engine_panel_manager_->get_containing_tabbed_group_id(panel_id);
    }
    return 0;
}

bool PanelManager::can_drag_panel_to_target(uint32_t source_panel_id, uint32_t target_panel_id) const {
    if (render_engine_panel_manager_) {
        return render_engine_panel_manager_->can_drag_panel_to_target(source_panel_id, target_panel_id);
    }
    return false;
}

void PanelManager::handle_panel_drag_drop() {
    if (render_engine_panel_manager_) {
        render_engine_panel_manager_->handle_panel_drag_drop();
    }
}

void PanelManager::save_layout(const std::string& filename) {
    if (render_engine_panel_manager_) {
        render_engine_panel_manager_->save_layout(filename);
    }
}

void PanelManager::load_layout(const std::string& filename) {
    if (render_engine_panel_manager_) {
        render_engine_panel_manager_->load_layout(filename);
    }
}

void PanelManager::auto_arrange_panels() {
    if (render_engine_panel_manager_) {
        render_engine_panel_manager_->auto_arrange_panels();
    }
}

uint32_t PanelManager::create_panel_group(const std::vector<uint32_t>& panel_ids) {
    if (render_engine_panel_manager_) {
        return render_engine_panel_manager_->create_panel_group(panel_ids);
    }
    return 0;
}

bool PanelManager::add_panel_to_group(uint32_t group_id, uint32_t panel_id) {
    if (render_engine_panel_manager_) {
        return render_engine_panel_manager_->add_panel_to_group(group_id, panel_id);
    }
    return false;
}

bool PanelManager::remove_panel_from_group(uint32_t group_id, uint32_t panel_id) {
    if (render_engine_panel_manager_) {
        return render_engine_panel_manager_->remove_panel_from_group(group_id, panel_id);
    }
    return false;
}

bool PanelManager::destroy_panel_group(uint32_t group_id) {
    if (render_engine_panel_manager_) {
        return render_engine_panel_manager_->destroy_panel_group(group_id);
    }
    return false;
}

bool PanelManager::is_panel_in_group(uint32_t panel_id) const {
    if (render_engine_panel_manager_) {
        return render_engine_panel_manager_->is_panel_in_group(panel_id);
    }
    return false;
}

uint32_t PanelManager::get_panel_group_id(uint32_t panel_id) const {
    if (render_engine_panel_manager_) {
        return render_engine_panel_manager_->get_panel_group_id(panel_id);
    }
    return 0;
}

uint32_t PanelManager::create_super_panel_from_adjacent(uint32_t panel1_id, uint32_t panel2_id) {
    if (render_engine_panel_manager_) {
        return render_engine_panel_manager_->create_super_panel_from_adjacent(panel1_id, panel2_id);
    }
    return 0;
}

uint32_t PanelManager::create_super_panel_from_rectangular_region(int start_x, int start_y, int width, int height) {
    if (render_engine_panel_manager_) {
        return render_engine_panel_manager_->create_super_panel_from_rectangular_region(start_x, start_y, width, height);
    }
    return 0;
}

void PanelManager::lock_panel_group(uint32_t group_id, bool locked) {
    if (render_engine_panel_manager_) {
        render_engine_panel_manager_->lock_panel_group(group_id, locked);
    }
}

bool PanelManager::is_panel_group_locked(uint32_t group_id) const {
    if (render_engine_panel_manager_) {
        return render_engine_panel_manager_->is_panel_group_locked(group_id);
    }
    return false;
}

void PanelManager::set_prevent_overlap_for_group(uint32_t group_id, bool prevent) {
    if (render_engine_panel_manager_) {
        render_engine_panel_manager_->set_prevent_overlap_for_group(group_id, prevent);
    }
}

bool PanelManager::does_group_prevent_overlap(uint32_t group_id) const {
    if (render_engine_panel_manager_) {
        return render_engine_panel_manager_->does_group_prevent_overlap(group_id);
    }
    return false;
}

} // namespace ui