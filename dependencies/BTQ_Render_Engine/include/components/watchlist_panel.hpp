#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "panel_base.hpp"
#include "watchlist_alerts.hpp"

namespace BTQuant {

struct WatchlistEntry {
  uint32_t symbol_id;
  std::string symbol;
  std::string exchange;
  double price = 0.0;
  double change_pct = 0.0;      // Change %
  double change_dollar = 0.0;   // Change $
  double volume_24h = 0.0;
  double vwap = 0.0;
  double high_24h = 0.0;
  double low_24h = 0.0;
  double open_24h = 0.0;        // Opening price
  uint64_t last_update_ts = 0;
  bool is_active = true;

  // Animation state for price changes
  double previous_price = 0.0;
  double previous_vwap = 0.0;  // Previous VWAP for animation
  double previous_volume = 0.0;  // Previous volume for change calculation
  float animation_timer = 0.0f;
  static constexpr float ANIMATION_DURATION = 0.8f; // Animation duration in seconds - optimized for smooth visual feedback
};

class WatchlistPanel : public PanelBase {
 public:
  using SymbolSelectedCallback = std::function<void(uint32_t symbol_id, const std::string& symbol)>;

  WatchlistPanel(const PanelConfig& config, std::shared_ptr<HotSpineDataBridge> bridge,
                 std::shared_ptr<RenderEngine::MarketDataProcessor> processor);
  ~WatchlistPanel();

  void update(float dt) override;
  void render() override;

  // Watchlist management
  void add_symbol(uint32_t symbol_id, const std::string& symbol, const std::string& exchange);
  void remove_symbol(uint32_t symbol_id);
  void clear_watchlist();

  // Watchlist group management
  void add_symbol_to_group(const std::string& group_name, uint32_t symbol_id, const std::string& symbol, const std::string& exchange);
  void remove_symbol_from_group(const std::string& group_name, uint32_t symbol_id);
  void clear_group(const std::string& group_name);
  void create_group(const std::string& group_name);
  void delete_group(const std::string& group_name);
  void rename_group(const std::string& old_name, const std::string& new_name);
  void switch_to_group(const std::string& group_name);

  // Symbol selection callback (e.g., to open chart when clicked)
  void set_symbol_selected_callback(SymbolSelectedCallback cb) {
    on_symbol_selected_ = std::move(cb);
  }

  // Configuration methods for saving/loading watchlist order
  void save_watchlist_order_to_config(const std::string& config_file) const;
  void load_watchlist_order_from_config(const std::string& config_file);

  // Method to set the config file path
  void set_config_file_path(const std::string& path) { config_file_path_ = path; }
  const std::string& get_config_file_path() const { return config_file_path_; }

  // Method to get the currently selected symbol
  uint32_t get_selected_symbol_id() const { return selected_symbol_id_; }
  std::string get_selected_symbol() const {
    auto it = get_current_watchlist().find(selected_symbol_id_);
    return (it != get_current_watchlist().end()) ? it->second.symbol : "";
  }

  // Get current group name
  const std::string& get_current_group_name() const { return current_group_name_; }

  // Alert management methods
  void set_alerts_panel(std::shared_ptr<AlertsPanel> alerts_panel);
  void add_price_alert(uint32_t symbol_id, const std::string& symbol_name,
                      double target_price, WatchlistPriceAlert::Direction direction);
  void remove_alerts_for_symbol(uint32_t symbol_id);
  void ensure_default_groups_order();

 private:
  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
  std::shared_ptr<WatchlistAlertManager> alert_manager_;

  // Single watchlist for backward compatibility
  std::map<uint32_t, WatchlistEntry> watchlist_;

  // Multiple watchlist groups
  std::map<std::string, std::map<uint32_t, WatchlistEntry>> watchlist_groups_;
  std::vector<std::string> group_names_;
  std::string current_group_name_ = "Default";

  // Display orders for each group
  std::map<std::string, std::vector<uint32_t>> group_display_orders_;

  // UI state
  int sort_column_ = 0;  // 0=symbol, 1=exchange, 2=last price, 3=change%, 4=change$, 5=volume, 6=high, 7=low, 8=open, 9=vwap
  bool sort_ascending_ = true;
  char filter_buffer_[256] = {0};
  char new_symbol_buffer_[128] = {0};  // Buffer for new symbol input
  uint32_t selected_symbol_id_ = 0;
  SymbolSelectedCallback on_symbol_selected_;

  // Configuration
  std::string config_file_path_ = "watchlist_config.ini";  // Path to the config file

  // Delete confirmation state
  uint32_t symbol_to_delete_ = 0;
  bool show_delete_confirmation_ = false;
  bool show_clear_all_confirmation_ = false;

  // Real-time subscription ID
  uint64_t subscription_id_ = 0;

  // Map to store subscription IDs for individual symbols
  std::unordered_map<uint32_t, uint64_t> symbol_subscriptions_;

  // Performance - timer removed since we now use real-time updates
  // float update_timer_ = 0.0f;
  // static constexpr float UPDATE_INTERVAL = 0.1f;  // 10 FPS updates

  // Column customization data structures
  struct ColumnInfo {
    std::string name;
    bool visible;
    float width;
    int order;  // Position in the table

    ColumnInfo(const std::string& n, bool v, float w, int o)
      : name(n), visible(v), width(w), order(o) {}
  };

  // Get reference to current watchlist based on selected group
  std::map<uint32_t, WatchlistEntry>& get_current_watchlist();
  const std::map<uint32_t, WatchlistEntry>& get_current_watchlist() const;
  std::vector<uint32_t>& get_current_display_order();
  const std::vector<uint32_t>& get_current_display_order() const;

  void update_watchlist_data();
  void render_table_header();
  void render_table_row(const WatchlistEntry& entry);
  void render_filter_input();
  void render_group_tabs();
  void sort_watchlist();
  std::vector<uint32_t> get_filtered_symbols() const;

  // Column customization methods
  void initialize_column_settings();
  void render_column_context_menu();
  void toggle_column_visibility(int column_index);
  void save_column_settings_to_config(const std::string& config_file) const;
  void load_column_settings_from_config(const std::string& config_file);
  void swap_column_positions(int index1, int index2);
  void reorder_columns(int source_index, int target_index);
  void render_draggable_header(int column_index, const char* label);

  // Drag and drop helpers
  void handle_drag_drop_reordering();
  void cleanup_drag_resources();

  // Helpers
  double calculate_24h_change(const RenderEngine::OHLCVCandle& current,
                              const RenderEngine::OHLCVCandle& old) const;

  static const char* get_sort_column_name(int column);
  const char* get_column_name_by_index(int column_index);

  // Real-time update handler
  void on_market_data_update(uint32_t symbol_id, RenderEngine::NotificationType type);

  // Helper methods for managing subscriptions
  void subscribe_to_symbol(uint32_t symbol_id);
  void unsubscribe_from_symbol(uint32_t symbol_id);
  void verify_subscriptions();
  void refresh_all_subscriptions();
  void ensure_all_symbols_subscribed();
  void process_pending_updates();
  void subscribe_to_all_watchlist_symbols();

 private:
  // Column customization members
  std::vector<ColumnInfo> column_info_;
  bool column_context_menu_open_ = false;
  int clicked_column_index_ = -1;
};

}  // namespace BTQuant