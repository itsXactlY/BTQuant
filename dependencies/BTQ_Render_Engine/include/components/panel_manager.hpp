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
#include "strategy_builder.hpp"
#include "tabbed_panel.hpp"
#include "../ui/context_menus.hpp"

// Forward declarations
namespace BTQuant {
    class AlertsPanel;
}

namespace BTQuant {

enum class LayoutPreset { DEFAULT, MODERN_TRADING, PRO_QUANT, SCALPER_DOM, ANALYTICS_FOCUS };

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
  
  // Auto-dock functionality - finds empty edges of existing panels
  std::pair<int, int> find_auto_dock_position(int width, int height) const;
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

  // Active panel management
  uint32_t get_active_panel_id() const { return active_panel_id_; }
  void set_active_panel_id(uint32_t panel_id) { active_panel_id_ = panel_id; }
  PanelBase* get_active_panel() const { return active_panel_id_ != 0 ? get_panel_by_id(active_panel_id_) : nullptr; }


  // Serialization
  std::string serialize_layout() const;
  void deserialize_layout(const std::string& layout_json);

  // Config management
  void save_all_panel_configs(const std::string& config_file) const;
  void load_all_panel_configs(const std::string& config_file);

  // Layout name tracking
  std::string get_current_layout_name() const { return current_layout_name_; }
  void set_current_layout_name(const std::string& name) { current_layout_name_ = name; }

  // Layout presets
  void apply_layout_preset(LayoutPreset preset);

  // Panel binding functionality - "Super-panel" grid locking
  struct PanelGroup {
    uint32_t group_id;
    std::vector<uint32_t> panel_ids;  // IDs of panels in this group
    int min_grid_x = 0;               // Top-left corner of the group in grid coordinates
    int min_grid_y = 0;
    int total_width = 0;              // Total width of the group in grid units
    int total_height = 0;             // Total height of the group in grid units
    bool locked = true;               // Whether the group is locked (non-resizable as a unit)
    bool prevent_overlap = true;      // Whether the group prevents overlapping with other panels/groups

    PanelGroup(uint32_t id) : group_id(id) {}
  };
  
  // Helper method to check if there's space for a panel group at a specific location
  bool can_place_group_at(uint32_t group_id, int grid_x, int grid_y) const;
  
  // Panel grouping methods
  uint32_t create_panel_group(const std::vector<uint32_t>& panel_ids);
  bool add_panel_to_group(uint32_t group_id, uint32_t panel_id);
  bool remove_panel_from_group(uint32_t group_id, uint32_t panel_id);
  bool destroy_panel_group(uint32_t group_id);
  bool is_panel_in_group(uint32_t panel_id) const;
  uint32_t get_panel_group_id(uint32_t panel_id) const;
  PanelGroup* get_panel_group(uint32_t group_id);
  const PanelGroup* get_panel_group(uint32_t group_id) const;

  // Super-panel creation methods - create a "super-panel" from adjacent panels
  uint32_t create_super_panel_from_adjacent(uint32_t panel1_id, uint32_t panel2_id);
  uint32_t create_super_panel_from_rectangular_region(int start_x, int start_y, int width, int height);
  
  // Lock/unlock panel groups to control whether they behave as a single unit
  void lock_panel_group(uint32_t group_id, bool locked = true);
  bool is_panel_group_locked(uint32_t group_id) const;

  // Control overlap prevention for panel groups
  void set_prevent_overlap_for_group(uint32_t group_id, bool prevent = true);
  bool does_group_prevent_overlap(uint32_t group_id) const;

  // Tabbed group functionality - allowing panels to be combined into tabs
  uint32_t create_tabbed_group(uint32_t target_panel_id);
  bool add_panel_to_tabbed_group(uint32_t tabbed_group_id, uint32_t panel_to_add_id);
  bool remove_panel_from_tabbed_group(uint32_t tabbed_group_id, uint32_t panel_to_remove_id);
  bool is_panel_in_tabbed_group(uint32_t panel_id) const;
  uint32_t get_containing_tabbed_group_id(uint32_t panel_id) const;
  bool can_drag_panel_to_target(uint32_t source_panel_id, uint32_t target_panel_id) const;

  // Drag and drop for tabbed groups
  void handle_panel_drag_drop();

  // Symbol linking functionality
  uint32_t create_symbol_link_group(SymbolLinkGroupColor color);
  bool add_panel_to_symbol_link_group(uint32_t group_id, uint32_t panel_id);
  bool remove_panel_from_symbol_link_group(uint32_t group_id, uint32_t panel_id);
  bool destroy_symbol_link_group(uint32_t group_id);
  bool is_panel_in_symbol_link_group(uint32_t panel_id) const;
  uint32_t get_panel_symbol_link_group_id(uint32_t panel_id) const;
  SymbolLinkGroup* get_symbol_link_group(uint32_t group_id);
  const SymbolLinkGroup* get_symbol_link_group(uint32_t group_id) const;
  void update_symbol_link_group_symbol(uint32_t group_id, const std::string& symbol);
  void propagate_symbol_to_linked_panels(uint32_t source_panel_id, const std::string& symbol);

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

  // Active panel tracking for context menu and focus
  uint32_t active_panel_id_ = 0;

  // Current layout tracking
  std::string current_layout_name_ = "default";

  // Panel grouping data
  std::unordered_map<uint32_t, std::unique_ptr<PanelGroup>> panel_groups_;
  std::unordered_map<uint32_t, uint32_t> panel_to_group_map_;  // Maps panel ID to group ID
  uint32_t next_group_id_ = 1;

  // Drag and drop state for tabbed groups
  uint32_t dragged_panel_id_ = 0;
  uint32_t drag_target_panel_id_ = 0;
  bool is_dragging_ = false;

  // Symbol linking groups - Red(0), Green(1), Blue(2)
  enum class SymbolLinkGroupColor { RED = 0, GREEN = 1, BLUE = 2, NONE = 3 };
  
  struct SymbolLinkGroup {
    SymbolLinkGroupColor color;
    std::vector<uint32_t> panel_ids;  // IDs of panels in this link group
    std::string linked_symbol;        // The symbol that all panels in this group share
    
    SymbolLinkGroup(SymbolLinkGroupColor c) : color(c), linked_symbol("") {}
  };

  // Symbol linking functionality
  std::unordered_map<uint32_t, std::unique_ptr<SymbolLinkGroup>> symbol_link_groups_;  // Maps group ID to link group
  std::unordered_map<uint32_t, uint32_t> panel_to_symbol_link_group_map_;  // Maps panel ID to link group ID
  uint32_t next_symbol_link_group_id_ = 1000;  // Start from 1000 to avoid conflicts with regular groups

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