#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace BTQuant {
namespace Layout {

// ============================================================================
// Dashboard Layout Manager
// ============================================================================

enum class PanelType {
  CHART,
  ORDERBOOK,
  HEATMAP,
  TAPE,
  WATCHLIST,
  VOLUME_PROFILE,
  FOOTPRINT_CHART,
  TPO_PROFILE,
  PERFORMANCE_MONITOR,
  TRADING_ORDERS,
  TRADING_POSITIONS,
  STATUS_BAR,
  ALERTS,
  DEPTH_CHART,
  SCATTER_PLOT,
  TIME_SERIES,
  RISK_METRICS,
  HISTOGRAM,
  SCREENER,
  LOG_PANEL,
  CUSTOM
};

class LayoutPresetManager;  // Forward declaration

class DashboardLayoutManager {
 public:
  struct PanelLayout {
    std::string panel_id;
    std::string panel_name;
    PanelType type = PanelType::CUSTOM;
    float x = 0.0f;
    float y = 0.0f;
    float width = 400.0f;
    float height = 300.0f;
    bool is_docked = false;
    int dock_node_id = 0;
    std::string symbol;         // Associated symbol for the panel
    std::string timeframe;      // Associated timeframe for the panel
    bool is_collapsed = false;  // Whether the panel is collapsed
    bool is_focused = false;    // Whether the panel is currently focused
  };

  struct DashboardLayout {
    std::string layout_name;
    std::vector<PanelLayout> panels;
    std::string theme;
    std::string created_time;
    int grid_columns = 6;       // Default grid columns
    int grid_rows = 9;          // Default grid rows
    float min_width = 200.0f;   // Minimum panel width
    float min_height = 150.0f;  // Minimum panel height
  };

  DashboardLayoutManager();
  ~DashboardLayoutManager();

  // Layout management
  void create_new_layout();
  void load_layout(const std::string& layout_name);
  void save_layout(const std::string& layout_name);
  void reset_layout();
  void delete_layout(const std::string& layout_name);

  // Layout queries
  std::vector<std::string> get_available_layouts() const;
  DashboardLayout get_current_layout() const;
  bool has_layout(const std::string& layout_name) const;

  // Panel management
  void add_panel_to_layout(const PanelLayout& panel);
  void remove_panel_from_layout(const std::string& panel_id);
  void update_panel_layout(const std::string& panel_id, const PanelLayout& layout);
  std::vector<PanelLayout> get_panels_for_symbol(const std::string& symbol) const;
  void update_symbol_for_all_panels(const std::string& old_symbol, const std::string& new_symbol);

  // Grid management
  void set_grid_dimensions(int columns, int rows);
  std::pair<int, int> get_grid_dimensions() const;

  // Layout utilities
  void auto_arrange_panels();
  void center_layout();
  void maximize_panel(const std::string& panel_id);
  void restore_panel_sizes();

  // Current layout modification
  void update_current_layout(const DashboardLayout& new_layout);

  // Integration with layout presets
  void set_preset_manager(LayoutPresetManager* preset_manager) { preset_manager_ = preset_manager; }
  LayoutPresetManager* get_preset_manager() const { return preset_manager_; }

 private:
  std::unordered_map<std::string, DashboardLayout> layouts_;
  std::string current_layout_name_;
  std::string layouts_directory_;
  LayoutPresetManager* preset_manager_ = nullptr;  // Reference to preset manager

  void initialize_layouts_directory();
  void load_all_layouts();
  void save_layout_to_file(const DashboardLayout& layout);
  DashboardLayout load_layout_from_file(const std::string& layout_name);
  std::string get_layout_file_path(const std::string& layout_name) const;
};

}  // namespace Layout
}  // namespace BTQuant
