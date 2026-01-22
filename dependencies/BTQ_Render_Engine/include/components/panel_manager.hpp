#pragma once

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "../trading/order_manager.hpp"
#include "../trading/position_manager.hpp"
#include "../trading/risk_assessment.hpp"
#include "chart_manager.hpp"
#include "panel_base.hpp"
#include <imgui.h>
#include <memory>
#include <unordered_map>
#include <vector>

namespace BTQuant {

struct GridLayout {
  int columns = 3;
  int rows = 2;
  float cell_padding = 8.0f;
  float panel_spacing = 4.0f;
};

class PanelManager {
public:
  PanelManager(std::shared_ptr<HotSpineDataBridge> bridge,
               std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
               std::shared_ptr<OrderManager> order_manager,
               std::shared_ptr<PositionManager> position_manager,
               std::shared_ptr<RiskAssessment> risk_assessment);

  ~PanelManager();

  void initialize();
  void update(float dt);
  void render();

  // Panel management
  uint32_t add_panel(PanelType type, const std::string &title = "",
                     int grid_x = -1, int grid_y = -1, int width = 1,
                     int height = 1);
  void remove_panel(uint32_t panel_id);
  void move_panel(uint32_t panel_id, int new_grid_x, int new_grid_y);
  void resize_panel(uint32_t panel_id, int new_width, int new_height);
  void set_panel_visible(uint32_t panel_id, bool visible);

  // Layout management
  void set_grid_layout(int columns, int rows);
  void auto_arrange_panels();
  ImVec2 get_panel_position(uint32_t panel_id) const;
  ImVec2 get_panel_size(uint32_t panel_id) const;
  void save_layout(const std::string &filename);
  void load_layout(const std::string &filename);

  // Symbol propagation
  void set_active_symbol(uint32_t symbol_id, const std::string &symbol_name);

  // Accessors
  ChartManager *get_chart_manager() const { return chart_manager_.get(); }

  // Serialization
  std::string serialize_layout() const;
  void deserialize_layout(const std::string &layout_json);

private:
  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
  std::shared_ptr<OrderManager> order_manager_;
  std::shared_ptr<PositionManager> position_manager_;
  std::shared_ptr<RiskAssessment> risk_assessment_;

  std::unique_ptr<ChartManager> chart_manager_;
  GridLayout grid_layout_;
  std::unordered_map<uint32_t, std::unique_ptr<PanelBase>> panels_;
  uint32_t next_panel_id_ = 1;

  ImVec2 dashboard_size_ = ImVec2(1920, 1080);

  PanelConfig create_panel_config(PanelType type, const std::string &title,
                                  int grid_x, int grid_y, int width,
                                  int height);
  ImVec2 calculate_panel_position(int grid_x, int grid_y) const;
  ImVec2 calculate_panel_size(int width, int height) const;
  std::string get_default_panel_title(PanelType type);
};

} // namespace BTQuant