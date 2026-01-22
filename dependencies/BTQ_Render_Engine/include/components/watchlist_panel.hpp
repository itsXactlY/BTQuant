#pragma once

#include "../hotspine_data_bridge.hpp"
#include "../market_data_processor.hpp"
#include "panel_base.hpp"
#include <map>
#include <string>
#include <vector>

namespace BTQuant {

struct WatchlistEntry {
  uint32_t symbol_id;
  std::string symbol;
  std::string exchange;
  double price = 0.0;
  double change_24h = 0.0;
  double volume_24h = 0.0;
  double vwap = 0.0;
  uint64_t last_update_ts = 0;
  bool is_active = true;
};

class WatchlistPanel : public PanelBase {
public:
  WatchlistPanel(const PanelConfig &config,
                 std::shared_ptr<HotSpineDataBridge> bridge,
                 std::shared_ptr<RenderEngine::MarketDataProcessor> processor);

  void update(float dt) override;
  void render() override;

  // Watchlist management
  void add_symbol(uint32_t symbol_id, const std::string &symbol,
                  const std::string &exchange);
  void remove_symbol(uint32_t symbol_id);
  void clear_watchlist();

private:
  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;

  std::map<uint32_t, WatchlistEntry> watchlist_;
  std::vector<uint32_t> display_order_; // For custom ordering

  // UI state
  int sort_column_ = 0; // 0=symbol, 1=price, 2=change, 3=volume
  bool sort_ascending_ = true;
  char filter_buffer_[256] = {0};

  // Performance
  float update_timer_ = 0.0f;
  static constexpr float UPDATE_INTERVAL = 0.1f; // 10 FPS updates

  void update_watchlist_data();
  void render_table_header();
  void render_table_row(const WatchlistEntry &entry);
  void render_filter_input();
  void sort_watchlist();
  std::vector<uint32_t> get_filtered_symbols() const;

  static const char *get_sort_column_name(int column);
};

} // namespace BTQuant