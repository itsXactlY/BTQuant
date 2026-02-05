#pragma once

#include <imgui.h>

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "../trading/order_manager.hpp"
#include "../trading/position_manager.hpp"
#include "../trading/risk_assessment.hpp"
#include "MarketMicrostructureRenderer.h"
#include "chart_manager.hpp"
#include "panel_base.hpp"
#include "../ui/context_menus.hpp"

// Forward declaration for AlertsPanel
namespace BTQuant {
    class AlertsPanel;
}

namespace BTQuant {

struct GridLayout {
  int columns = 3;
  int rows = 2;
  float cell_padding = 8.0f;
  float panel_spacing = 4.0f;
};

class PanelManager {
 public:
  using PanelAddedCallback = std::function<void(uint32_t panel_id, PanelType type)>;
  using PanelRemovedCallback = std::function<void(uint32_t panel_id)>;

  PanelManager(std::shared_ptr<HotSpineDataBridge> bridge,
               std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
               std::shared_ptr<OrderManager> order_manager,
               std::shared_ptr<PositionManager> position_manager,
               std::shared_ptr<RiskAssessment> risk_assessment,
               RenderEngine::MarketMicrostructureRenderer* micro_renderer = nullptr);

  ~PanelManager();

  void initialize();
  void update(float dt);
  void render();

  // Panel management
  uint32_t add_panel(PanelType type, const std::string& title = "", int grid_x = -1,
                     int grid_y = -1, int width = 1, int height = 1);
  uint32_t add_panel_with_symbol(PanelType type, const std::string& title,
                                 const std::string& symbol, int grid_x, int grid_y, int width,
                                 int height);
  void remove_panel(uint32_t panel_id);
  void clear_panels();
  void move_panel(uint32_t panel_id, int new_grid_x, int new_grid_y);
  void resize_panel(uint32_t panel_id, int new_width, int new_height);
  void set_panel_visible(uint32_t panel_id, bool visible);
  void set_panel_symbol(uint32_t panel_id, const std::string& symbol);

  // Layout management
  void set_grid_layout(int columns, int rows);
  void auto_arrange_panels();
  ImVec2 get_panel_position(uint32_t panel_id) const;
  ImVec2 get_panel_size(uint32_t panel_id) const;
  void save_layout(const std::string& filename);
  void load_layout(const std::string& filename);

  // Symbol propagation
  void set_active_symbol(uint32_t symbol_id, const std::string& symbol_name);

  // Callback management
  void register_panel_added_callback(PanelAddedCallback callback);
  void register_panel_removed_callback(PanelRemovedCallback callback);

  // Accessors
  ChartManager* get_chart_manager() const { return chart_manager_.get(); }

  // Helper methods
  size_t get_panel_count() const;
  std::vector<uint32_t> get_all_panel_ids() const;
  PanelConfig get_panel_config(uint32_t panel_id) const;
  void update_panel_config(uint32_t panel_id, const PanelConfig& config);

  // Find panel by type
  uint32_t find_panel_by_type(PanelType type) const;

  // Get panel by ID
  PanelBase* get_panel_by_id(uint32_t panel_id) const;


  // Serialization
  std::string serialize_layout() const;
  void deserialize_layout(const std::string& layout_json);

  // Config management
  void save_all_panel_configs(const std::string& config_file) const;
  void load_all_panel_configs(const std::string& config_file);

  // Layout name tracking
  std::string get_current_layout_name() const { return current_layout_name_; }
  void set_current_layout_name(const std::string& name) { current_layout_name_ = name; }

 private:
  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
  std::shared_ptr<OrderManager> order_manager_;
  std::shared_ptr<PositionManager> position_manager_;
  std::shared_ptr<RiskAssessment> risk_assessment_;
  RenderEngine::MarketMicrostructureRenderer* micro_renderer_;

  std::unique_ptr<ChartManager> chart_manager_;
  std::unique_ptr<ContextMenuManager> context_menu_manager_;
  std::unique_ptr<RenderEngine::StrategyBuilder> strategy_builder_;
  GridLayout grid_layout_;
  std::unordered_map<uint32_t, std::unique_ptr<PanelBase>> panels_;
  uint32_t next_panel_id_ = 1;

  ImVec2 dashboard_size_ = ImVec2(1920, 1080);

  // Active symbol tracking for cross-panel propagation
  uint32_t active_symbol_id_ = 0;
  std::string active_symbol_name_;

  // Current layout tracking
  std::string current_layout_name_ = "default";

  // Callbacks
  std::vector<PanelAddedCallback> panel_added_callbacks_;
  std::vector<PanelRemovedCallback> panel_removed_callbacks_;

  PanelConfig create_panel_config(PanelType type, const std::string& title, int grid_x, int grid_y,
                                  int width, int height);
  PanelConfig create_panel_config_with_symbol(PanelType type, const std::string& title,
                                              const std::string& symbol, int grid_x, int grid_y,
                                              int width, int height);
  ImVec2 calculate_panel_position(int grid_x, int grid_y) const;
  ImVec2 calculate_panel_size(int width, int height) const;
  std::string get_default_panel_title(PanelType type);
};

}  // namespace BTQuant