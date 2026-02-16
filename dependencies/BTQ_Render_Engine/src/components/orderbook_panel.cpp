#include "../../include/components/orderbook_panel.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "imgui.h"
#include "implot.h"

#include "../../include/components/orderbook_batcher.hpp"

namespace BTQuant {

// Destructor to clean up the lock-free cache
OrderbookPanel::~OrderbookPanel() {
    OrderbookCache* cache = latest_orderbook_cache_.load(std::memory_order_acquire);
    if (cache) {
        delete cache;
    }
}

// Helper function to round to nearest multiple
double roundToNearest(double value, double multiple) {
    if (multiple == 0.0) return value;
    return std::round(value / multiple) * multiple;
}

OrderbookPanel::OrderbookPanel(const PanelConfig& config,
                               std::shared_ptr<HotSpineDataBridge> bridge,
                               std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(config), bridge_(bridge), processor_(processor), selected_levels_count_(20),
      aggregation_mode_(OrderbookAggregationMode::NONE), custom_aggregation_value_(1.0),
      volume_delta_period_us_(5000000) { // Initialize to 5 seconds (5,000,000 microseconds)
      
    // Initialize the lock-free cache with a default empty object
    latest_orderbook_cache_.store(new OrderbookCache(), std::memory_order_relaxed);
}

double OrderbookPanel::getAggregationValue(double price) const {
    switch (aggregation_mode_) {
        case OrderbookAggregationMode::TICK_SIZE:
            // For tick size aggregation, we need to get the tick size for the symbol
            // Since we don't have direct access to tick size, we'll use a default of 0.01 for now
            // In a real implementation, this would come from market data
            return roundToNearest(price, 0.01);
        case OrderbookAggregationMode::PERCENT_0_1:
            // Group by 0.1% of the price
            return roundToNearest(price, price * 0.001);
        case OrderbookAggregationMode::PERCENT_0_5:
            // Group by 0.5% of the price
            return roundToNearest(price, price * 0.005);
        case OrderbookAggregationMode::PERCENT_1:
            // Group by 1% of the price
            return roundToNearest(price, price * 0.01);
        case OrderbookAggregationMode::CUSTOM_VALUE:
            // Group by custom value
            return roundToNearest(price, custom_aggregation_value_);
        case OrderbookAggregationMode::NONE:
        default:
            // No aggregation, return the original price
            return price;
    }
}

std::vector<PriceLevel> OrderbookPanel::aggregateOrderbookLevels(
    const std::vector<PriceLevel>& levels) const {

    if (aggregation_mode_ == OrderbookAggregationMode::NONE) {
        return levels; // Return original levels if no aggregation
    }

    std::map<double, PriceLevel> aggregated_levels;

    for (const auto& level : levels) {
        double aggregated_price = getAggregationValue(level.price);

        auto it = aggregated_levels.find(aggregated_price);
        if (it != aggregated_levels.end()) {
            // Aggregate with existing level
            it->second.size += level.size;
        } else {
            // Create new aggregated level
            PriceLevel new_level;
            new_level.price = aggregated_price;
            new_level.size = level.size;
            aggregated_levels[aggregated_price] = new_level;
        }
    }

    // Convert map back to vector
    std::vector<PriceLevel> result;
    result.reserve(aggregated_levels.size());

    for (const auto& pair : aggregated_levels) {
        result.push_back(pair.second);
    }

    // Sort by price (ascending for asks, descending for bids in the UI)
    std::sort(result.begin(), result.end(), [](const PriceLevel& a, const PriceLevel& b) {
        return a.price < b.price;
    });

    return result;
}

void OrderbookPanel::set_symbol(uint32_t symbol_id, const std::string& symbol_name) {
  symbol_id_ = symbol_id;
  symbol_name_ = symbol_name;
  config_.title = symbol_name + " Orderbook";
}

void OrderbookPanel::update(float /*dt*/) {
  // Request data update from data bridge
  bridge_->sync();

  if (bridge_) {
    // Process trades to update volume profile and order flow
    auto trades = bridge_->getTradeBuffer();
    // Simple linear scan. In production, use monotonic index or similar.
    for (const auto& trade : trades) {
      // Skip potential empty slots
      if (trade.ts_local == 0) continue;

      if (trade.ts_local > last_processed_trade_ts_ && trade.symbol_id == symbol_id_) {
        auto& vol = volume_profile_[trade.price];
        if (trade.side == 0)
          vol.bought += trade.size;  // Buy
        else
          vol.sold += trade.size;  // Sell

        // Track execution event for order flow
        auto& activity = order_flow_activity_[trade.price];
        activity.executions++;
        activity.last_activity_ts = trade.ts_local;

        if (trade.ts_local > last_processed_trade_ts_) {
          last_processed_trade_ts_ = trade.ts_local;
        }
      }
    }

    // Process orderbook snapshots to track additions and cancellations
    auto books = bridge_->getBookBuffer();
    for (const auto& book : books) {
      if (book.ts_local == 0) continue;

      if (book.ts_local > last_order_flow_update_ts_ && book.symbol_id == symbol_id_) {
        // Get previous snapshot for comparison
        auto prev_it = previous_snapshots_.find(book.symbol_id);
        if (prev_it != previous_snapshots_.end()) {
          // Compare current snapshot with previous to detect order flow events
          detectOrderFlowEvents(book, prev_it->second);
        }

        // Store current snapshot as previous for next comparison
        previous_snapshots_[book.symbol_id] = book;

        // Track volume changes for delta calculation
        trackVolumeChanges(book, book.ts_local);

        if (book.ts_local > last_order_flow_update_ts_) {
          last_order_flow_update_ts_ = book.ts_local;
        }
      }
    }

    // Clean up old order flow activity data periodically
    uint64_t current_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    if (current_time - last_order_flow_update_ts_ > order_flow_reset_interval_) {
      // Decay the activity counts over time
      for (auto& [price, activity] : order_flow_activity_) {
        if (current_time - activity.last_activity_ts > order_flow_reset_interval_) {
          activity.reset();
        } else {
          // Apply decay to the activity counts
          activity.additions = static_cast<int>(activity.additions * order_flow_decay_factor_);
          activity.cancellations = static_cast<int>(activity.cancellations * order_flow_decay_factor_);
          activity.executions = static_cast<int>(activity.executions * order_flow_decay_factor_);
        }
      }
    }
  }
  
  // Update the lock-free cache with the latest orderbook data
  updateOrderbookCache();
}

void OrderbookPanel::updateOrderbookCache() {
  auto orderbook_opt = processor_->getOrderbookData(symbol_id_);
  
  if (orderbook_opt.has_value()) {
    const auto& orderbook = orderbook_opt.value();
    
    // Create a new cache object with the latest data
    OrderbookCache* new_cache = new OrderbookCache(orderbook);
    
    // Atomically swap the old cache with the new one
    OrderbookCache* old_cache = latest_orderbook_cache_.exchange(new_cache, std::memory_order_acq_rel);
    
    // Clean up the old cache
    if (old_cache) {
      delete old_cache;
    }
  }
}

void OrderbookPanel::detectOrderFlowEvents(const HotOrderbookSnapshot& current_snapshot, const HotOrderbookSnapshot& previous_snapshot) {
  // Map previous prices to sizes for quick lookup
  std::map<double, double> prev_bid_prices;
  std::map<double, double> prev_ask_prices;

  // Populate previous snapshot maps
  for (int i = 0; i < previous_snapshot.bids_count && i < 200; ++i) {
    if (previous_snapshot.bids[i].size > 0) {
      prev_bid_prices[previous_snapshot.bids[i].price] = previous_snapshot.bids[i].size;
    }
  }

  for (int i = 0; i < previous_snapshot.asks_count && i < 200; ++i) {
    if (previous_snapshot.asks[i].size > 0) {
      prev_ask_prices[previous_snapshot.asks[i].price] = previous_snapshot.asks[i].size;
    }
  }

  // Process current bids to detect additions and cancellations
  for (int i = 0; i < current_snapshot.bids_count && i < 200; ++i) {
    if (current_snapshot.bids[i].size > 0) {
      auto prev_it = prev_bid_prices.find(current_snapshot.bids[i].price);

      if (prev_it == prev_bid_prices.end()) {
        // New price level - this is an addition
        auto& activity = order_flow_activity_[current_snapshot.bids[i].price];
        activity.additions++;
        activity.last_activity_ts = current_snapshot.ts_local;
      } else if (current_snapshot.bids[i].size > prev_it->second) {
        // Size increased - this indicates new orders added at this level
        auto& activity = order_flow_activity_[current_snapshot.bids[i].price];
        activity.additions++;
        activity.last_activity_ts = current_snapshot.ts_local;
      } else if (current_snapshot.bids[i].size < prev_it->second) {
        // Size decreased - this indicates orders cancelled at this level
        auto& activity = order_flow_activity_[current_snapshot.bids[i].price];
        activity.cancellations++;
        activity.last_activity_ts = current_snapshot.ts_local;
      }
    }
  }

  // Process previous bids to detect cancellations (prices that disappeared)
  for (const auto& [price, size] : prev_bid_prices) {
    bool found_in_current = false;
    for (int i = 0; i < current_snapshot.bids_count && i < 200; ++i) {
      if (current_snapshot.bids[i].price == price) {
        found_in_current = true;
        break;
      }
    }

    if (!found_in_current) {
      // Price level disappeared - this is a cancellation
      auto& activity = order_flow_activity_[price];
      activity.cancellations++;
      activity.last_activity_ts = current_snapshot.ts_local;
    }
  }

  // Process current asks to detect additions and cancellations
  for (int i = 0; i < current_snapshot.asks_count && i < 200; ++i) {
    if (current_snapshot.asks[i].size > 0) {
      auto prev_it = prev_ask_prices.find(current_snapshot.asks[i].price);

      if (prev_it == prev_ask_prices.end()) {
        // New price level - this is an addition
        auto& activity = order_flow_activity_[current_snapshot.asks[i].price];
        activity.additions++;
        activity.last_activity_ts = current_snapshot.ts_local;
      } else if (current_snapshot.asks[i].size > prev_it->second) {
        // Size increased - this indicates new orders added at this level
        auto& activity = order_flow_activity_[current_snapshot.asks[i].price];
        activity.additions++;
        activity.last_activity_ts = current_snapshot.ts_local;
      } else if (current_snapshot.asks[i].size < prev_it->second) {
        // Size decreased - this indicates orders cancelled at this level
        auto& activity = order_flow_activity_[current_snapshot.asks[i].price];
        activity.cancellations++;
        activity.last_activity_ts = current_snapshot.ts_local;
      }
    }
  }

  // Process previous asks to detect cancellations (prices that disappeared)
  for (const auto& [price, size] : prev_ask_prices) {
    bool found_in_current = false;
    for (int i = 0; i < current_snapshot.asks_count && i < 200; ++i) {
      if (current_snapshot.asks[i].price == price) {
        found_in_current = true;
        break;
      }
    }

    if (!found_in_current) {
      // Price level disappeared - this is a cancellation
      auto& activity = order_flow_activity_[price];
      activity.cancellations++;
      activity.last_activity_ts = current_snapshot.ts_local;
    }
  }
}

void OrderbookPanel::trackVolumeChanges(const HotOrderbookSnapshot& snapshot, uint64_t timestamp) {
  // Process bids
  for (int i = 0; i < snapshot.bids_count && i < 200; ++i) {
    if (snapshot.bids[i].size > 0) {
      volume_level_history_[snapshot.bids[i].price].addBidPoint(timestamp, snapshot.bids[i].size);
    }
  }

  // Process asks
  for (int i = 0; i < snapshot.asks_count && i < 200; ++i) {
    if (snapshot.asks[i].size > 0) {
      volume_level_history_[snapshot.asks[i].price].addAskPoint(timestamp, snapshot.asks[i].size);
    }
  }
}

void OrderbookPanel::render_content() {
  begin_panel_window();

  // If panel is hidden via X button, we still need to call end
  if (!is_visible()) {
    end_panel_window();
    return;
  }

  // Symbol selector for this orderbook panel
  auto active_symbols = processor_->getActiveSymbols();

  // Debug Info
  static int frame_count = 0;
  if (frame_count++ % 300 == 0) {
    std::cout << "[OrderbookPanel] Rendering. SymID=" << symbol_id_
              << " ActiveSyms=" << active_symbols.size() << std::endl;
  }

  if (!active_symbols.empty()) {
    // Check if current symbol has orderbook data, re-select if not
    auto current_ob = processor_->getOrderbookData(symbol_id_);
    bool need_reselect = (symbol_id_ == 0) || !current_ob.has_value();

    if (need_reselect) {
      for (uint32_t sym_id : active_symbols) {
        auto ob_opt = processor_->getOrderbookData(sym_id);
        if (ob_opt.has_value()) {
          if (symbol_id_ != sym_id) {
            symbol_id_ = sym_id;
            symbol_name_ = bridge_->getSymbolName(symbol_id_);
            config_.title = symbol_name_ + " Orderbook";
            std::cout << "[OrderbookPanel] Auto-selected: " << symbol_name_ << " (ID=" << symbol_id_
                      << ")" << std::endl;
          }
          break;
        }
      }
      // If still 0, just pick first to show "waiting"
      if (symbol_id_ == 0 && !active_symbols.empty()) {
        symbol_id_ = active_symbols[0];
        symbol_name_ = bridge_->getSymbolName(symbol_id_);
      }
    }

    // Build symbol names for combo
    ImGui::PushID(this);
    if (ImGui::BeginCombo("Select Symbol", symbol_name_.c_str())) {
      for (size_t i = 0; i < active_symbols.size(); ++i) {
        uint32_t sym_id = active_symbols[i];
        ImGui::PushID(static_cast<int>(sym_id));

        std::string sym_name = bridge_->getSymbolName(sym_id);
        std::string exchange = bridge_->getExchangeName(sym_id);
        std::string display_name = "[" + exchange + "] " + sym_name;

        bool is_selected = (sym_id == symbol_id_);
        if (ImGui::Selectable(display_name.c_str(), is_selected)) {
          symbol_id_ = sym_id;
          symbol_name_ = sym_name;
          config_.title = symbol_name_ + " Orderbook";
          std::cout << "[OrderbookPanel] Title updated to: " << config_.title << std::endl;
        }
        if (is_selected) ImGui::SetItemDefaultFocus();
        ImGui::PopID();
      }
      ImGui::EndCombo();
    }
    ImGui::PopID();

    // Level count selector
    ImGui::SameLine();
    ImGui::Text("Levels:");
    ImGui::SameLine();
    ImGui::PushItemWidth(100);
    if (ImGui::BeginCombo("##LevelCount", LEVEL_OPTION_NAMES[get_level_option_index()])) {
      for (int i = 0; i < 6; ++i) {
        bool is_selected = (LEVEL_OPTIONS[i] == selected_levels_count_);
        if (ImGui::Selectable(LEVEL_OPTION_NAMES[i], is_selected)) {
          selected_levels_count_ = LEVEL_OPTIONS[i];
        }
        if (is_selected) ImGui::SetItemDefaultFocus();
      }
      ImGui::EndCombo();
    }
    ImGui::PopItemWidth();

    // Aggregation mode selector
    ImGui::SameLine();
    ImGui::Text("Aggregation:");
    ImGui::SameLine();
    ImGui::PushItemWidth(120);

    const char* aggregation_modes[] = {
        "None", "Tick Size", "0.1%", "0.5%", "1%", "Custom"
    };

    int current_aggregation_mode = static_cast<int>(aggregation_mode_);
    if (ImGui::BeginCombo("##AggregationMode", aggregation_modes[current_aggregation_mode])) {
        for (int i = 0; i < 6; ++i) {
            bool is_selected = (current_aggregation_mode == i);
            if (ImGui::Selectable(aggregation_modes[i], is_selected)) {
                aggregation_mode_ = static_cast<OrderbookAggregationMode>(i);
            }
            if (is_selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    ImGui::PopItemWidth();

    // Show custom value input if custom aggregation mode is selected
    if (aggregation_mode_ == OrderbookAggregationMode::CUSTOM_VALUE) {
        ImGui::SameLine();
        ImGui::Text("Value:");
        ImGui::SameLine();
        ImGui::PushItemWidth(80);
        ImGui::InputDouble("##CustomAggValue", &custom_aggregation_value_, 0.01f, 1.0f, "%.4f");
        ImGui::PopItemWidth();
    }
  } else {
    const auto& colors = ThemeManager::getInstance().getColors();
    ImGui::TextColored(colors.accent_red, "No active symbols detected in SHM!");
  }

  // Add heatmap intensity slider to the panel header
  ImGui::Separator();
  ImGui::Text("Heatmap Intensity:");
  ImGui::SameLine();
  ImGui::PushItemWidth(200);
  ImGui::SliderFloat("##HeatmapIntensity", &heatmap_intensity_, 0.1f, 5.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
  ImGui::PopItemWidth();
  ImGui::SameLine();
  if (ImGui::Button("Reset##HeatmapIntensity")) {
    heatmap_intensity_ = 1.0f;
  }
  ImGui::Separator();

  // Get orderbook data from the lock-free cache
  OrderbookCache* cache = latest_orderbook_cache_.load(std::memory_order_acquire);
  
  if (!cache || cache->bids.empty() || cache->asks.empty()) {
    ImGui::Text("Waiting for Orderbook: %s", symbol_name_.c_str());
    ImGui::Text("ID: %u", symbol_id_);
    ImGui::ProgressBar(((frame_count % 100) / 100.0f), ImVec2(-1, 0), "Polling Data Processor...");
    end_panel_window();
    return;
  }

  // Use the cached data for rendering
  RenderEngine::OrderbookData orderbook;
  orderbook.bids = cache->bids;
  orderbook.asks = cache->asks;
  orderbook.spread = cache->spread;
  orderbook.imbalance = cache->imbalance;
  orderbook.timestamp = cache->timestamp;
  orderbook.symbol_id = symbol_id_;
  orderbook.symbol = symbol_name_;

  // Calculate bid/ask ratio
  double total_bid_volume = 0.0;
  double total_ask_volume = 0.0;

  for (const auto& bid : orderbook.bids) {
    total_bid_volume += bid.size;
  }
  for (const auto& ask : orderbook.asks) {
    total_ask_volume += ask.size;
  }

  double bid_ask_ratio = (total_ask_volume > 0) ? total_bid_volume / total_ask_volume : 0.0;

  // Statistics Header
  ImGui::Columns(3, "Stats", false);
  ImGui::Text("Spread: %.4f", orderbook.spread);
  ImGui::NextColumn();
  ImGui::Text("Imbalance: %.2f", orderbook.imbalance);
  ImGui::NextColumn();

  // Display bid/ask ratio with colored arrow
  if (bid_ask_ratio > 1.0) {
    // Bid heavy - green arrow pointing up
    ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "▲ Ratio: %.2f", bid_ask_ratio);
  } else if (bid_ask_ratio < 1.0) {
    // Ask heavy - red arrow pointing down
    ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "▼ Ratio: %.2f", bid_ask_ratio);
  } else {
    // Balanced - white arrow
    ImGui::Text("■ Ratio: %.2f", bid_ask_ratio);
  }

  ImGui::Columns(1);
  ImGui::Separator();

  // Order Flow Legend
  ImGui::Text("Order Flow:");
  ImGui::SameLine();
  ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "●"); // Green dot for additions
  ImGui::SameLine();
  ImGui::Text("Additions ");
  ImGui::SameLine();
  ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "●"); // Red dot for cancellations
  ImGui::SameLine();
  ImGui::Text("Cancellations ");
  ImGui::SameLine();
  ImGui::TextColored(ImVec4(0.0f, 0.0f, 1.0f, 1.0f), "●"); // Blue dot for executions
  ImGui::SameLine();
  ImGui::Text("Executions");

  // Render Orderbook Ladder
  render_orderbook_ladder(orderbook);

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Text("Market Depth (Cumulative)");
  render_market_depth_chart(orderbook);

  end_panel_window();
}

int OrderbookPanel::get_level_option_index() {
  for (int i = 0; i < 6; ++i) {
    if (LEVEL_OPTIONS[i] == selected_levels_count_) {
      return i;
    }
  }
  return 1; // Default to 20 if not found
}

void OrderbookPanel::render_orderbook_ladder(const RenderEngine::OrderbookData& orderbook) {
  // Apply aggregation to bids and asks if needed
  std::vector<PriceLevel> aggregated_bids = aggregateOrderbookLevels(orderbook.bids);
  std::vector<PriceLevel> aggregated_asks = aggregateOrderbookLevels(orderbook.asks);

  // Calculate average order size for large order detection
  size_t total_levels = aggregated_bids.size() + aggregated_asks.size();
  if (total_levels > 0) {
    double total_size = 0.0;
    for (const auto& level : aggregated_bids) total_size += level.size;
    for (const auto& level : aggregated_asks) total_size += level.size;
    average_order_size_ = total_size / total_levels;
  } else {
    average_order_size_ = 0.0;
  }

  // Calculate max volume for relative scaling
  double max_vol = 1.0;
  for (const auto& level : aggregated_bids) max_vol = std::max(max_vol, level.size);
  for (const auto& level : aggregated_asks) max_vol = std::max(max_vol, level.size);
  if (max_vol < 1.0) max_vol = 1.0;

  // Calculate cumulative volumes for liquidity bars
  std::vector<double> cumulative_bids(aggregated_bids.size());
  std::vector<double> cumulative_asks(aggregated_asks.size());

  // Calculate cumulative bid volumes (from best bid outward)
  double bid_sum = 0.0;
  for (size_t i = 0; i < aggregated_bids.size(); ++i) {
    bid_sum += aggregated_bids[i].size;
    cumulative_bids[i] = bid_sum;
  }

  // Calculate cumulative ask volumes (from best ask outward)
  double ask_sum = 0.0;
  for (size_t i = 0; i < aggregated_asks.size(); ++i) {
    ask_sum += aggregated_asks[i].size;
    cumulative_asks[i] = ask_sum;
  }

  // Find max cumulative volume for scaling
  double max_cumulative_vol = max_vol;
  if (!cumulative_bids.empty()) max_cumulative_vol = std::max(max_cumulative_vol, cumulative_bids.back());
  if (!cumulative_asks.empty()) max_cumulative_vol = std::max(max_cumulative_vol, cumulative_asks.back());

  const auto& colors = ThemeManager::getInstance().getColors();

  // Determine how many levels to show based on selected_levels_count_
  int max_levels_to_show = selected_levels_count_ == -1 ?
                           std::max(aggregated_asks.size(), aggregated_bids.size()) :
                           selected_levels_count_;

  int ask_count = std::min((int)aggregated_asks.size(), max_levels_to_show);
  int bid_count = std::min((int)aggregated_bids.size(), max_levels_to_show);

  // Calculate mid-price
  double mid_price = 0.0;
  if (!aggregated_bids.empty() && !aggregated_asks.empty()) {
    mid_price = (aggregated_bids[0].price + aggregated_asks[0].price) / 2.0;
  }

  // DOM Hardware Instancing: Collect all liquidity bar rectangles for batched rendering
  struct LiquidityBar {
      ImVec2 min;
      ImVec2 max;
      ImU32 color;
  };
  std::vector<LiquidityBar> bid_liquidity_bars;
  std::vector<LiquidityBar> ask_liquidity_bars;
  std::vector<std::pair<ImVec2, ImVec2>> bid_bar_pairs;
  std::vector<std::pair<ImVec2, ImVec2>> ask_bar_pairs;

  ImDrawList* draw_list = ImGui::GetWindowDrawList();
  const float row_height = ImGui::GetTextLineHeightWithSpacing();

  // Split panel 50/50: Asks (top half) and Bids (bottom half)
  float available_height = ImGui::GetContentRegionAvail().y - 40.0f; // Reserve 40px for mid-price block
  float half_height = available_height / 2.0f;

  // ========== ASKS PANEL (TOP 50%) - Descending order (worst ask to best ask) ==========
  if (ImGui::BeginTable("AsksTable", 8,
                        ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame)) {
    ImGui::TableSetupColumn("Bid", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Sold", ImGuiTableColumnFlags_WidthFixed, 40);
    ImGui::TableSetupColumn("Price", ImGuiTableColumnFlags_WidthFixed, 80);
    ImGui::TableSetupColumn("Bought", ImGuiTableColumnFlags_WidthFixed, 40);
    ImGui::TableSetupColumn("Ask", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Delta", ImGuiTableColumnFlags_WidthFixed, 40);
    ImGui::TableSetupColumn("Δ Last 5s", ImGuiTableColumnFlags_WidthFixed, 60);
    ImGui::TableSetupColumn("Vol", ImGuiTableColumnFlags_WidthFixed, 40);

    // Collect ask liquidity bars
    for (int i = ask_count - 1; i >= 0; --i) {
      ImGui::TableNextRow();
      const auto& level = aggregated_asks[i];
      ImVec2 row_pos = ImGui::GetCursorScreenPos();

      // Size bar (column 4)
      ImGui::TableSetColumnIndex(4);
      float size_col_width = ImGui::GetContentRegionAvail().x;
      float size_bar_width = size_col_width * (float)(level.size / max_vol);
      ImVec2 size_bar_min = ImVec2(row_pos.x, row_pos.y);
      ImVec2 size_bar_max = ImVec2(size_bar_min.x + size_bar_width, row_pos.y + row_height);
      ImU32 size_bar_color = ImGui::GetColorU32(ImVec4(colors.accent_red.x, colors.accent_red.y, colors.accent_red.z, 0.2f));
      ask_liquidity_bars.push_back({size_bar_min, size_bar_max, size_bar_color});

      // Cumulative volume bar (column 2)
      ImGui::TableSetColumnIndex(2);
      float cum_col_width = ImGui::GetContentRegionAvail().x;
      float cum_bar_width = cum_col_width * (float)(cumulative_asks[i] / max_cumulative_vol) * 0.7f;
      ImVec2 cum_bar_min = ImVec2(row_pos.x, row_pos.y);
      ImVec2 cum_bar_max = ImVec2(cum_bar_min.x + cum_bar_width, row_pos.y + row_height);
      ImU32 cum_bar_color = ImGui::GetColorU32(ImVec4(colors.accent_red.x * 0.6f, colors.accent_red.y * 0.6f, colors.accent_red.z * 0.6f, 0.3f));
      ask_liquidity_bars.push_back({cum_bar_min, cum_bar_max, cum_bar_color});
    }

    // Collect ask liquidity bar pairs
    for (const auto& bar : ask_liquidity_bars) {
      ask_bar_pairs.push_back({bar.min, bar.max});
    }

    // Render Asks content (descending: worst ask at top, best ask at bottom)
    for (int i = ask_count - 1; i >= 0; --i) {
      const auto& level = aggregated_asks[i];
      ImGui::TableNextRow();
      ImGui::PushID(i);

      bool is_large_order = average_order_size_ > 0 &&
                           (level.size / average_order_size_) * 100.0 >= large_order_threshold_percentage_;

      // Column 0: Bid (Empty)
      ImGui::TableSetColumnIndex(0);

      // Column 1: Sold + Order Flow
      ImGui::TableSetColumnIndex(1);
      if (volume_profile_.contains(level.price)) {
        double sold = volume_profile_[level.price].sold;
        if (sold > 0) {
          if (is_large_order) {
            ImVec2 text_pos = ImGui::GetCursorScreenPos();
            ImVec2 text_size = ImGui::CalcTextSize(std::format("%.0f", sold).c_str());
            ImGui::GetWindowDrawList()->AddRectFilled(text_pos,
                ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
                ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f)));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
            ImGui::TextColored(ImVec4(1, 0.5f, 0.5f, 1), "%.0f", sold);
            ImGui::PopStyleColor();
          } else {
            ImGui::TextColored(ImVec4(1, 0.5f, 0.5f, 1), "%.0f", sold);
          }
        }
      }
      if (order_flow_activity_.contains(level.price)) {
        const auto& activity = order_flow_activity_[level.price];
        int total_activity = activity.additions + activity.cancellations + activity.executions;
        if (total_activity > 0) {
          ImVec2 pos = ImGui::GetCursorScreenPos();
          float dot_size = std::min(4.0f + (total_activity / 10.0f), 8.0f);
          if (activity.additions > 0) {
            float intensity = std::min(activity.additions / 10.0f, 1.0f);
            ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(pos.x + 2, pos.y + 2),
                dot_size * 0.5f, ImGui::GetColorU32(ImVec4(0.0f, 1.0f, 0.0f, intensity)));
          }
          if (activity.cancellations > 0) {
            float intensity = std::min(activity.cancellations / 10.0f, 1.0f);
            ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(pos.x + 2, pos.y + 8),
                dot_size * 0.5f, ImGui::GetColorU32(ImVec4(1.0f, 0.0f, 0.0f, intensity)));
          }
          if (activity.executions > 0) {
            float intensity = std::min(activity.executions / 10.0f, 1.0f);
            ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(pos.x + 2, pos.y + 14),
                dot_size * 0.5f, ImGui::GetColorU32(ImVec4(0.0f, 0.0f, 1.0f, intensity)));
          }
        }
      }

      // Column 2: Price
      ImGui::TableSetColumnIndex(2);
      float cursor_check = ImGui::GetCursorPosX() + (ImGui::GetContentRegionAvail().x -
          ImGui::CalcTextSize(std::to_string(level.price).c_str()).x) * 0.5f;
      ImGui::SetCursorPosX(cursor_check);
      if (is_large_order) {
        ImVec2 cell_pos = ImGui::GetCursorScreenPos();
        ImVec2 cell_size = ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetTextLineHeightWithSpacing());
        ImGui::GetWindowDrawList()->AddRectFilled(cell_pos,
            ImVec2(cell_pos.x + cell_size.x, cell_pos.y + cell_size.y),
            ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.3f)));
      }
      ImGui::Selectable(std::format("{:.2f}", level.price).c_str(), false,
                        ImGuiSelectableFlags_SpanAllColumns);
      if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
        ImGui::SetDragDropPayload("PRICE_LEVEL", &level.price, sizeof(double));
        ImGui::Text("Price: %.2f", level.price);
        ImGui::EndDragDropSource();
      }
      ImGui::SameLine();
      if (is_large_order) {
        ImVec2 text_pos = ImGui::GetCursorScreenPos();
        ImVec2 text_size = ImGui::CalcTextSize(std::format("%.2f", level.price).c_str());
        ImGui::GetWindowDrawList()->AddRectFilled(text_pos,
            ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
            ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f)));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
        ImGui::TextColored(colors.accent_red, "%.2f", level.price);
        ImGui::PopStyleColor();
      } else {
        ImGui::TextColored(colors.accent_red, "%.2f", level.price);
      }

      // Column 3: Bought
      ImGui::TableSetColumnIndex(3);
      if (volume_profile_.contains(level.price)) {
        double bought = volume_profile_[level.price].bought;
        if (bought > 0) {
          if (is_large_order) {
            ImVec2 text_pos = ImGui::GetCursorScreenPos();
            ImVec2 text_size = ImGui::CalcTextSize(std::format("%.0f", bought).c_str());
            ImGui::GetWindowDrawList()->AddRectFilled(text_pos,
                ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
                ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f)));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
            ImGui::TextColored(ImVec4(0.5f, 1, 0.5f, 1), "%.0f", bought);
            ImGui::PopStyleColor();
          } else {
            ImGui::TextColored(ImVec4(0.5f, 1, 0.5f, 1), "%.0f", bought);
          }
        }
      }

      // Column 4: Ask Size
      ImGui::TableSetColumnIndex(4);
      if (is_large_order) {
        ImVec2 text_pos = ImGui::GetCursorScreenPos();
        ImVec2 text_size = ImGui::CalcTextSize(std::format("%.4f", level.size).c_str());
        ImGui::GetWindowDrawList()->AddRectFilled(text_pos,
            ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
            ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f)));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
        ImGui::Text("%.4f", level.size);
        ImGui::PopStyleColor();
      } else {
        ImGui::Text("%.4f", level.size);
      }

      // Column 5: Delta
      ImGui::TableSetColumnIndex(5);
      if (volume_profile_.contains(level.price)) {
        const auto& vol = volume_profile_[level.price];
        double delta = vol.bought - vol.sold;
        if (delta != 0) {
          ImVec4 color = delta > 0 ? ImVec4(0.5f, 1, 0.5f, 1) : ImVec4(1, 0.5f, 0.5f, 1);
          if (is_large_order) {
            ImVec2 text_pos = ImGui::GetCursorScreenPos();
            ImVec2 text_size = ImGui::CalcTextSize(std::format("%+.0f", delta).c_str());
            ImGui::GetWindowDrawList()->AddRectFilled(text_pos,
                ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
                ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f)));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
            ImGui::TextColored(color, "%+.0f", delta);
            ImGui::PopStyleColor();
          } else {
            ImGui::TextColored(color, "%+.0f", delta);
          }
        }
      }

      // Column 6: Volume Delta Last 5s
      ImGui::TableSetColumnIndex(6);
      {
        uint64_t current_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        auto hist_it = volume_level_history_.find(level.price);
        if (hist_it != volume_level_history_.end()) {
          double volume_delta = hist_it->second.getAskDeltaOverPeriod(current_time, volume_delta_period_us_);
          if (volume_delta != 0) {
            ImVec4 color = volume_delta > 0 ? ImVec4(0.5f, 1, 0.5f, 1) : ImVec4(1, 0.5f, 0.5f, 1);
            if (is_large_order) {
              ImVec2 text_pos = ImGui::GetCursorScreenPos();
              ImVec2 text_size = ImGui::CalcTextSize(std::format("%+.2f", volume_delta).c_str());
              ImGui::GetWindowDrawList()->AddRectFilled(text_pos,
                  ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
                  ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f)));
              ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
              ImGui::TextColored(color, "%+.2f", volume_delta);
              ImGui::PopStyleColor();
            } else {
              ImGui::TextColored(color, "%+.2f", volume_delta);
            }
          }
        }
      }

      // Column 7: Volume
      ImGui::TableSetColumnIndex(7);
      if (volume_profile_.contains(level.price)) {
        const auto& vol = volume_profile_[level.price];
        double total = vol.bought + vol.sold;
        if (total > 0) {
          if (is_large_order) {
            ImVec2 text_pos = ImGui::GetCursorScreenPos();
            ImVec2 text_size = ImGui::CalcTextSize(std::format("%.0f", total).c_str());
            ImGui::GetWindowDrawList()->AddRectFilled(text_pos,
                ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
                ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f)));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
            ImGui::Text("%.0f", total);
            ImGui::PopStyleColor();
          } else {
            ImGui::Text("%.0f", total);
          }
        }
      }

      ImGui::PopID();
    }

    ImGui::EndTable();
  }

  // ========== MID-PRICE BLOCK (40px distinct) ==========
  ImGui::Separator();
  ImVec2 mid_price_pos = ImGui::GetCursorScreenPos();
  ImVec2 mid_price_size = ImVec2(ImGui::GetContentRegionAvail().x, 40.0f);

  // Draw distinct mid-price background
  draw_list->AddRectFilled(mid_price_pos,
      ImVec2(mid_price_pos.x + mid_price_size.x, mid_price_pos.y + mid_price_size.y),
      ImGui::GetColorU32(ImVec4(0.3f, 0.3f, 0.5f, 0.4f)));

  // Draw border
  draw_list->AddRect(mid_price_pos,
      ImVec2(mid_price_pos.x + mid_price_size.x, mid_price_pos.y + mid_price_size.y),
      ImGui::GetColorU32(ImVec4(0.6f, 0.6f, 0.8f, 0.8f)), 0.0f, 0, 2.0f);

  // Center mid-price text
  std::string mid_price_text = std::format("MID: {:.2f}", mid_price);
  ImVec2 text_size = ImGui::CalcTextSize(mid_price_text.c_str());
  ImVec2 text_pos = ImVec2(
      mid_price_pos.x + (mid_price_size.x - text_size.x) * 0.5f,
      mid_price_pos.y + (mid_price_size.y - text_size.y) * 0.5f);

  draw_list->AddText(ImVec2(text_pos.x, text_pos.y),
      ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 1.0f, 1.0f)), mid_price_text.c_str());

  // Draw spread info
  std::string spread_text = std::format("Spread: {:.4f}", orderbook.spread);
  ImVec2 spread_text_size = ImGui::CalcTextSize(spread_text.c_str());
  ImVec2 spread_text_pos = ImVec2(
      mid_price_pos.x + (mid_price_size.x - spread_text_size.x) * 0.5f,
      text_pos.y + text_size.y + 4.0f);

  draw_list->AddText(ImVec2(spread_text_pos.x, spread_text_pos.y),
      ImGui::GetColorU32(ImVec4(0.8f, 0.8f, 0.8f, 0.8f)), spread_text.c_str());

  ImGui::Dummy(mid_price_size);
  ImGui::Separator();

  // ========== BIDS PANEL (BOTTOM 50%) - Ascending order (worst bid to best bid) ==========
  bid_liquidity_bars.clear();
  if (ImGui::BeginTable("BidsTable", 8,
                        ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame)) {
    ImGui::TableSetupColumn("Bid", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Sold", ImGuiTableColumnFlags_WidthFixed, 40);
    ImGui::TableSetupColumn("Price", ImGuiTableColumnFlags_WidthFixed, 80);
    ImGui::TableSetupColumn("Bought", ImGuiTableColumnFlags_WidthFixed, 40);
    ImGui::TableSetupColumn("Ask", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Delta", ImGuiTableColumnFlags_WidthFixed, 40);
    ImGui::TableSetupColumn("Δ Last 5s", ImGuiTableColumnFlags_WidthFixed, 60);
    ImGui::TableSetupColumn("Vol", ImGuiTableColumnFlags_WidthFixed, 40);

    // Collect bid liquidity bars (ascending: worst bid at top, best bid at bottom)
    for (int i = bid_count - 1; i >= 0; --i) {
      ImGui::TableNextRow();
      const auto& level = aggregated_bids[i];
      ImVec2 row_pos = ImGui::GetCursorScreenPos();

      // Size bar (column 0)
      ImGui::TableSetColumnIndex(0);
      float size_col_width = ImGui::GetContentRegionAvail().x;
      float size_bar_width = size_col_width * (float)(level.size / max_vol);
      ImVec2 size_bar_min = ImVec2(row_pos.x + size_col_width - size_bar_width, row_pos.y);
      ImVec2 size_bar_max = ImVec2(row_pos.x + size_col_width, row_pos.y + row_height);
      ImU32 size_bar_color = ImGui::GetColorU32(ImVec4(colors.accent_green.x, colors.accent_green.y, colors.accent_green.z, 0.2f));
      bid_liquidity_bars.push_back({size_bar_min, size_bar_max, size_bar_color});

      // Cumulative volume bar (column 2)
      ImGui::TableSetColumnIndex(2);
      float cum_col_width = ImGui::GetContentRegionAvail().x;
      float cum_bar_width = cum_col_width * (float)(cumulative_bids[i] / max_cumulative_vol) * 0.7f;
      ImVec2 cum_bar_min = ImVec2(row_pos.x + cum_col_width - cum_bar_width, row_pos.y);
      ImVec2 cum_bar_max = ImVec2(row_pos.x + cum_col_width, row_pos.y + row_height);
      ImU32 cum_bar_color = ImGui::GetColorU32(ImVec4(colors.accent_green.x * 0.6f, colors.accent_green.y * 0.6f, colors.accent_green.z * 0.6f, 0.3f));
      bid_liquidity_bars.push_back({cum_bar_min, cum_bar_max, cum_bar_color});
    }

    // Collect bid liquidity bar pairs
    for (const auto& bar : bid_liquidity_bars) {
      bid_bar_pairs.push_back({bar.min, bar.max});
    }

    // Render Bids content (ascending: worst bid at top, best bid at bottom)
    for (int i = bid_count - 1; i >= 0; --i) {
      const auto& level = aggregated_bids[i];
      ImGui::TableNextRow();
      ImGui::PushID(i + 1000);

      bool is_large_order = average_order_size_ > 0 &&
                           (level.size / average_order_size_) * 100.0 >= large_order_threshold_percentage_;

      // Column 0: Bid Size
      ImGui::TableSetColumnIndex(0);
      auto text = std::format("{:.4f}", level.size);
      float text_width = ImGui::CalcTextSize(text.c_str()).x;
      float col_width = ImGui::GetContentRegionAvail().x;
      ImGui::SetCursorPosX(ImGui::GetCursorPosX() + col_width - text_width);
      if (is_large_order) {
        ImVec2 text_pos = ImGui::GetCursorScreenPos();
        ImVec2 text_size = ImGui::CalcTextSize(text.c_str());
        ImGui::GetWindowDrawList()->AddRectFilled(text_pos,
            ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
            ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f)));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
        ImGui::TextUnformatted(text.c_str());
        ImGui::PopStyleColor();
      } else {
        ImGui::TextUnformatted(text.c_str());
      }

      // Column 1: Sold + Order Flow
      ImGui::TableSetColumnIndex(1);
      if (volume_profile_.contains(level.price)) {
        double sold = volume_profile_[level.price].sold;
        if (sold > 0) {
          if (is_large_order) {
            ImVec2 text_pos = ImGui::GetCursorScreenPos();
            ImVec2 text_size = ImGui::CalcTextSize(std::format("%.0f", sold).c_str());
            ImGui::GetWindowDrawList()->AddRectFilled(text_pos,
                ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
                ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f)));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
            ImGui::TextColored(ImVec4(1, 0.5f, 0.5f, 1), "%.0f", sold);
            ImGui::PopStyleColor();
          } else {
            ImGui::TextColored(ImVec4(1, 0.5f, 0.5f, 1), "%.0f", sold);
          }
        }
      }
      if (order_flow_activity_.contains(level.price)) {
        const auto& activity = order_flow_activity_[level.price];
        int total_activity = activity.additions + activity.cancellations + activity.executions;
        if (total_activity > 0) {
          ImVec2 pos = ImGui::GetCursorScreenPos();
          float dot_size = std::min(4.0f + (total_activity / 10.0f), 8.0f);
          if (activity.additions > 0) {
            float intensity = std::min(activity.additions / 10.0f, 1.0f);
            ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(pos.x + 2, pos.y + 2),
                dot_size * 0.5f, ImGui::GetColorU32(ImVec4(0.0f, 1.0f, 0.0f, intensity)));
          }
          if (activity.cancellations > 0) {
            float intensity = std::min(activity.cancellations / 10.0f, 1.0f);
            ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(pos.x + 2, pos.y + 8),
                dot_size * 0.5f, ImGui::GetColorU32(ImVec4(1.0f, 0.0f, 0.0f, intensity)));
          }
          if (activity.executions > 0) {
            float intensity = std::min(activity.executions / 10.0f, 1.0f);
            ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(pos.x + 2, pos.y + 14),
                dot_size * 0.5f, ImGui::GetColorU32(ImVec4(0.0f, 0.0f, 1.0f, intensity)));
          }
        }
      }

      // Column 2: Price
      ImGui::TableSetColumnIndex(2);
      float cursor_check = ImGui::GetCursorPosX() + (ImGui::GetContentRegionAvail().x -
          ImGui::CalcTextSize(std::to_string(level.price).c_str()).x) * 0.5f;
      ImGui::SetCursorPosX(cursor_check);
      if (is_large_order) {
        ImVec2 cell_pos = ImGui::GetCursorScreenPos();
        ImVec2 cell_size = ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetTextLineHeightWithSpacing());
        ImGui::GetWindowDrawList()->AddRectFilled(cell_pos,
            ImVec2(cell_pos.x + cell_size.x, cell_pos.y + cell_size.y),
            ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.3f)));
      }
      ImGui::Selectable(std::format("{:.2f}", level.price).c_str(), false,
                        ImGuiSelectableFlags_SpanAllColumns);
      if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
        ImGui::SetDragDropPayload("PRICE_LEVEL", &level.price, sizeof(double));
        ImGui::Text("Price: %.2f", level.price);
        ImGui::EndDragDropSource();
      }
      ImGui::SameLine();
      if (is_large_order) {
        ImVec2 text_pos = ImGui::GetCursorScreenPos();
        ImVec2 text_size = ImGui::CalcTextSize(std::format("%.2f", level.price).c_str());
        ImGui::GetWindowDrawList()->AddRectFilled(text_pos,
            ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
            ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f)));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
        ImGui::TextColored(colors.accent_green, "%.2f", level.price);
        ImGui::PopStyleColor();
      } else {
        ImGui::TextColored(colors.accent_green, "%.2f", level.price);
      }

      // Column 3: Bought
      ImGui::TableSetColumnIndex(3);
      if (volume_profile_.contains(level.price)) {
        double bought = volume_profile_[level.price].bought;
        if (bought > 0) {
          if (is_large_order) {
            ImVec2 text_pos = ImGui::GetCursorScreenPos();
            ImVec2 text_size = ImGui::CalcTextSize(std::format("%.0f", bought).c_str());
            ImGui::GetWindowDrawList()->AddRectFilled(text_pos,
                ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
                ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f)));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
            ImGui::TextColored(ImVec4(0.5f, 1, 0.5f, 1), "%.0f", bought);
            ImGui::PopStyleColor();
          } else {
            ImGui::TextColored(ImVec4(0.5f, 1, 0.5f, 1), "%.0f", bought);
          }
        }
      }

      // Column 4: Ask (Empty)
      ImGui::TableSetColumnIndex(4);

      // Column 5: Delta
      ImGui::TableSetColumnIndex(5);
      if (volume_profile_.contains(level.price)) {
        const auto& vol = volume_profile_[level.price];
        double delta = vol.bought - vol.sold;
        if (delta != 0) {
          ImVec4 color = delta > 0 ? ImVec4(0.5f, 1, 0.5f, 1) : ImVec4(1, 0.5f, 0.5f, 1);
          if (is_large_order) {
            ImVec2 text_pos = ImGui::GetCursorScreenPos();
            ImVec2 text_size = ImGui::CalcTextSize(std::format("%+.0f", delta).c_str());
            ImGui::GetWindowDrawList()->AddRectFilled(text_pos,
                ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
                ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f)));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
            ImGui::TextColored(color, "%+.0f", delta);
            ImGui::PopStyleColor();
          } else {
            ImGui::TextColored(color, "%+.0f", delta);
          }
        }
      }

      // Column 6: Volume Delta Last 5s
      ImGui::TableSetColumnIndex(6);
      {
        uint64_t current_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        auto hist_it = volume_level_history_.find(level.price);
        if (hist_it != volume_level_history_.end()) {
          double volume_delta = hist_it->second.getBidDeltaOverPeriod(current_time, volume_delta_period_us_);
          if (volume_delta != 0) {
            ImVec4 color = volume_delta > 0 ? ImVec4(0.5f, 1, 0.5f, 1) : ImVec4(1, 0.5f, 0.5f, 1);
            if (is_large_order) {
              ImVec2 text_pos = ImGui::GetCursorScreenPos();
              ImVec2 text_size = ImGui::CalcTextSize(std::format("%+.2f", volume_delta).c_str());
              ImGui::GetWindowDrawList()->AddRectFilled(text_pos,
                  ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
                  ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f)));
              ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
              ImGui::TextColored(color, "%+.2f", volume_delta);
              ImGui::PopStyleColor();
            } else {
              ImGui::TextColored(color, "%+.2f", volume_delta);
            }
          }
        }
      }

      // Column 7: Volume
      ImGui::TableSetColumnIndex(7);
      if (volume_profile_.contains(level.price)) {
        const auto& vol = volume_profile_[level.price];
        double total = vol.bought + vol.sold;
        if (total > 0) {
          if (is_large_order) {
            ImVec2 text_pos = ImGui::GetCursorScreenPos();
            ImVec2 text_size = ImGui::CalcTextSize(std::format("%.0f", total).c_str());
            ImGui::GetWindowDrawList()->AddRectFilled(text_pos,
                ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
                ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f)));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f));
            ImGui::Text("%.0f", total);
            ImGui::PopStyleColor();
          } else {
            ImGui::Text("%.0f", total);
          }
        }
      }

      ImGui::PopID();
    }

    ImGui::EndTable();
  }

  // Submit all liquidity bars via a single draw command (DOM Hardware Instancing)
  OrderbookBatcher batcher;
  ImU32 bid_color = ImGui::GetColorU32(ImVec4(colors.accent_green.x, colors.accent_green.y, colors.accent_green.z, 0.2f));
  ImU32 ask_color = ImGui::GetColorU32(ImVec4(colors.accent_red.x, colors.accent_red.y, colors.accent_red.z, 0.2f));
  batcher.renderLiquidityBars(draw_list, bid_bar_pairs, ask_bar_pairs, bid_color, ask_color);
}

void OrderbookPanel::render_market_depth_chart(const RenderEngine::OrderbookData& orderbook) {
  // Apply aggregation to bids and asks if needed
  std::vector<PriceLevel> aggregated_bids = aggregateOrderbookLevels(orderbook.bids);
  std::vector<PriceLevel> aggregated_asks = aggregateOrderbookLevels(orderbook.asks);

  if (aggregated_bids.empty() || aggregated_asks.empty()) return;

  if (ImPlot::BeginPlot("##Depth", ImVec2(-1, 150), ImPlotFlags_CanvasOnly)) {
    ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);

    // Handle Bids - Cumulative depth from best bid down
    std::vector<double> bx, by;
    if (!aggregated_bids.empty()) {
      double cumulative_depth = 0.0;

      // Add points from worst bid to best bid for increasing X-axis
      for (auto it = aggregated_bids.rbegin(); it != aggregated_bids.rend(); ++it) {
        cumulative_depth += it->size;
        bx.push_back(it->price);
        by.push_back(cumulative_depth);
      }

      // Extend to left for visual completeness
      const auto& worst_bid = aggregated_bids.back();
      bx.insert(bx.begin(), worst_bid.price * 0.995);
      by.insert(by.begin(), cumulative_depth);

      // Add point at best bid with 0 depth for shading
      const auto& best_bid = aggregated_bids[0];
      bx.push_back(best_bid.price);
      by.push_back(0.0);
    }

    const auto& colors = ThemeManager::getInstance().getColors();
    // ImPlot::SetNextFillStyle(colors.accent_green);
    ImPlot::PlotShaded("Bids", bx.data(), by.data(), (int)bx.size(), 0);

    // Handle Asks - Cumulative depth from best ask up
    std::vector<double> ax, ay;
    if (!aggregated_asks.empty()) {
      double cumulative_depth = 0.0;

      // Add points from best ask to worst ask
      for (const auto& ask : aggregated_asks) {
        cumulative_depth += ask.size;
        ax.push_back(ask.price);
        ay.push_back(cumulative_depth);
      }

      // Extend to right for visual completeness
      const auto& worst_ask = aggregated_asks.back();
      ax.push_back(worst_ask.price * 1.005);
      ay.push_back(cumulative_depth);

      // Add point at best ask with 0 depth for shading
      const auto& best_ask = aggregated_asks[0];
      ax.insert(ax.begin(), best_ask.price);
      ay.insert(ay.begin(), 0.0);
    }

    // ImPlot::SetNextFillStyle(colors.accent_red);
    ImPlot::PlotShaded("Asks", ax.data(), ay.data(), (int)ax.size(), 0);

    ImPlot::EndPlot();
  }
}

void OrderbookPanel::center_price() {
  // This method would center the view on the current mid-price
  // For now, we'll just log that the action was triggered
  OrderbookCache* cache = latest_orderbook_cache_.load(std::memory_order_acquire);

  if (cache && !cache->bids.empty() && !cache->asks.empty()) {
    // Calculate mid price (average of best bid and best ask)
    double mid_price = (cache->bids[0].price + cache->asks[0].price) / 2.0;

    // In a real implementation, this would adjust the viewport to center on mid_price
    // For now, we'll just log the action
    std::cout << "[OrderbookPanel] Centering view on mid-price: " << mid_price << std::endl;
  }
}

void OrderbookPanel::reset_depth() {
  // This method would reset the depth chart view to default zoom/position
  // For now, we'll just log that the action was triggered

  std::cout << "[OrderbookPanel] Resetting depth chart view" << std::endl;

  // In a real implementation, this would reset any zoom/pan state of the depth chart
  // For now, we'll just log the action
}

void OrderbookPanel::render_panel_header() {
  // Call parent implementation to render the default header
  PanelBase::render_panel_header();

  // Add heatmap intensity slider to the panel header
  ImGui::Separator();
  ImGui::Text("Heatmap Intensity:");
  ImGui::SameLine();
  ImGui::PushItemWidth(200);
  ImGui::SliderFloat("##HeatmapIntensity", &heatmap_intensity_, 0.1f, 5.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
  ImGui::PopItemWidth();
  ImGui::SameLine();
  if (ImGui::Button("Reset##HeatmapIntensity")) {
    heatmap_intensity_ = 1.0f;
  }
  ImGui::Separator();
}

}  // namespace BTQuant
