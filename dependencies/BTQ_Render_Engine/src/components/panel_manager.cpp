#include "components/panel_manager.hpp"

#include <filesystem>  // Added for std::filesystem::current_path()
#include <iostream>

#include "components/chart_panel.hpp"
#include "components/dom_surface_panel.hpp"
#include "components/footprint_panel.hpp"
#include "components/tape_panel.hpp"
#include "components/volume_profile_panel.hpp"
#include "data/cluster_engine.hpp"
#include "rendering/vulkan_chart_pipeline.hpp"
#include "vulkan_base_types.hpp"
namespace BTQuant {
PanelManager::PanelManager(std::shared_ptr<MarketDataProcessor> processor) : processor_(processor) {
  chart_manager_ = std::make_unique<ChartManager>(processor_);
  cluster_engine_ = std::make_unique<ClusterEngine>();  // 1-min candles by default
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
  config.type = type;

  switch (type) {
    case PanelType::TAPE:
      config.title = title.empty() ? "Tape" : title;
      panels_[current_id] = std::make_unique<TapePanel>(config, processor_);
      break;
    case PanelType::DOM_SURFACE:
      config.title = title.empty() ? "DOM" : title;
      panels_[current_id] = std::make_unique<DomSurfacePanel>(config, processor_);
      break;
    case PanelType::FOOTPRINT_CHART: {
      config.title = title.empty() ? "Footprint" : title;
      panels_[current_id] = std::make_unique<FootprintPanel>(config, cluster_engine_.get());
      break;
    }
    case PanelType::VOLUME_PROFILE: {
      config.title = title.empty() ? "Volume Profile" : title;
      auto vp = std::make_unique<VolumeProfilePanel>(config, processor_);
      vp->set_cluster_engine(cluster_engine_.get());
      panels_[current_id] = std::move(vp);
      break;
    }
    default:
      config.title = title.empty() ? "Vulkan Chart" : title;
      panels_[current_id] =
          std::make_unique<ChartPanel>(config, processor_, chart_manager_.get(), this);
      // Wire Vulkan pipeline to chart panel if available
      if (chart_pipeline_) {
        auto* chart = dynamic_cast<ChartPanel*>(panels_[current_id].get());
        if (chart) {
          chart->set_vulkan_pipeline(chart_pipeline_.get(), VK_NULL_HANDLE);
        }
      }
      break;
  }

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
void PanelManager::apply_layout_preset(LayoutPreset preset) {
  clear_panels();

  switch (preset) {
    case LayoutPreset::MODERN_TRADING:
    default:
      // Main chart panel
      add_panel(PanelType::CHART, "Chart");
      // Tape panel
      add_panel(PanelType::TAPE, "Tape");
      // DOM panel
      add_panel(PanelType::DOM_SURFACE, "DOM");
      // Footprint panel
      add_panel(PanelType::FOOTPRINT_CHART, "Footprint");
      // Volume Profile panel
      add_panel(PanelType::VOLUME_PROFILE, "Volume Profile");
      break;
  }
}

void PanelManager::register_panel_added_callback(PanelAddedCallback) {}
void PanelManager::register_panel_removed_callback(PanelRemovedCallback) {}
void PanelManager::set_vulkan_core(VulkanCore* core) {
  vulkan_core_ = core;

  // Create the chart pipeline now that we have Vulkan context
  if (core && !chart_pipeline_) {
    std::cerr << "[PanelManager] CWD: " << std::filesystem::current_path() << "\n";
    try {
      chart_pipeline_ = std::make_unique<Rendering::VulkanChartPipeline>(
          core->get_device(), core->get_physical_device(), core->get_render_pass());
      std::cerr << "[PanelManager] VulkanChartPipeline created successfully\n";

      // Wire pipeline to all existing chart panels
      for (auto& [id, panel] : panels_) {
        auto* chart = dynamic_cast<ChartPanel*>(panel.get());
        if (chart) {
          chart->set_vulkan_pipeline(chart_pipeline_.get(), VK_NULL_HANDLE);
        }
      }
    } catch (const std::exception& e) {
      std::cerr << "[PanelManager] Failed to create VulkanChartPipeline: " << e.what() << "\n";
    }
  }
}

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
