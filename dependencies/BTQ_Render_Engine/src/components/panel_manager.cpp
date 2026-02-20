#include "components/panel_manager.hpp"

#include "components/chart_panel.hpp"
namespace BTQuant {
PanelManager::PanelManager(std::shared_ptr<MarketDataProcessor> processor) : processor_(processor) {
  chart_manager_ = std::make_unique<ChartManager>(processor_);
}
PanelManager::~PanelManager() = default;

void PanelManager::initialize() {}

void PanelManager::update(float delta_time) {
  for (auto& pair : panels_) pair.second->update(delta_time);
  if (chart_manager_) chart_manager_->update();
}

void PanelManager::render() {
  for (auto& pair : panels_) pair.second->render();
}

uint32_t PanelManager::add_panel(PanelType type, const std::string& title, int /*grid_x*/,
                                 int /*grid_y*/, int /*width*/, int /*height*/) {
  uint32_t current_id = next_panel_id_++;
  PanelConfig config;
  config.id = current_id;
  config.title = title.empty() ? "Vulkan Chart" : title;
  config.type = type;
  panels_[current_id] =
      std::make_unique<ChartPanel>(config, processor_, chart_manager_.get(), this);
  return current_id;
}

uint32_t PanelManager::add_panel_with_symbol(PanelType type, const std::string& title,
                                             const std::string& symbol, int grid_x, int grid_y,
                                             int width, int height) {
  uint32_t id = add_panel(type, title, grid_x, grid_y, width, height);
  set_panel_symbol(id, symbol);
  return id;
}

void PanelManager::remove_panel(uint32_t id) { panels_.erase(id); }
void PanelManager::clear_panels() { panels_.clear(); }
void PanelManager::move_panel(uint32_t, int, int) {}
void PanelManager::resize_panel(uint32_t, int, int) {}
void PanelManager::set_panel_visible(uint32_t, bool) {}

void PanelManager::set_panel_symbol(uint32_t id, const std::string& symbol) {
  auto it = panels_.find(id);
  if (it != panels_.end()) it->second->set_symbol(symbol);
}

void PanelManager::set_active_symbol(uint32_t, const std::string&) {}

void PanelManager::auto_arrange_panels() {}
ImVec2 PanelManager::get_panel_position(uint32_t) const { return ImVec2(0, 0); }
ImVec2 PanelManager::get_panel_size(uint32_t) const { return ImVec2(400, 300); }
void PanelManager::save_layout(const std::string&) {}
void PanelManager::load_layout(const std::string&) {}
void PanelManager::apply_layout_preset(LayoutPreset) {}

void PanelManager::register_panel_added_callback(PanelAddedCallback) {}
void PanelManager::register_panel_removed_callback(PanelRemovedCallback) {}
void PanelManager::set_vulkan_core(VulkanCore* core) { vulkan_core_ = core; }

size_t PanelManager::get_panel_count() const { return panels_.size(); }
std::vector<uint32_t> PanelManager::get_all_panel_ids() const {
  std::vector<uint32_t> ids;
  for (const auto& pair : panels_) ids.push_back(pair.first);
  return ids;
}

PanelConfig PanelManager::get_panel_config(uint32_t id) const {
  auto it = panels_.find(id);
  if (it != panels_.end()) return it->second->get_config();
  return PanelConfig{};
}

void PanelManager::update_panel_config(uint32_t, const PanelConfig&) {}
uint32_t PanelManager::find_panel_by_type(PanelType) const { return 0; }
PanelBase* PanelManager::get_panel_by_id(uint32_t id) const {
  auto it = panels_.find(id);
  if (it != panels_.end()) return it->second.get();
  return nullptr;
}

std::string PanelManager::serialize_layout() const { return ""; }
void PanelManager::deserialize_layout(const std::string&) {}
void PanelManager::save_all_panel_configs(const std::string&) const {}
void PanelManager::load_all_panel_configs(const std::string&) {}

}  // namespace BTQuant
