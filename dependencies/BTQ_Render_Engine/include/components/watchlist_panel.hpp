#pragma once

#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "panel_base.hpp"

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

 private:
  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;

  std::map<uint32_t, WatchlistEntry> watchlist_;
  std::vector<uint32_t> display_order_;  // For custom ordering

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

  void update_watchlist_data();
  void render_table_header();
  void render_table_row(const WatchlistEntry& entry);
  void render_filter_input();
  void sort_watchlist();
  std::vector<uint32_t> get_filtered_symbols() const;

  // Drag and drop helpers
  void handle_drag_drop_reordering();
  void cleanup_drag_resources();

  // Helpers
  double calculate_24h_change(const RenderEngine::OHLCVCandle& current,
                              const RenderEngine::OHLCVCandle& old) const;

  static const char* get_sort_column_name(int column);

  // Real-time update handler
  void on_market_data_update(uint32_t symbol_id, RenderEngine::NotificationType type);

  // Helper methods for managing subscriptions
  void subscribe_to_symbol(uint32_t symbol_id);
  void unsubscribe_from_symbol(uint32_t symbol_id);
  void verify_subscriptions();
  void refresh_all_subscriptions();
};

}  // namespace BTQuant