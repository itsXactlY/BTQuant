#include "../../include/components/orderbook_panel.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "imgui.h"
#include "implot.h"

namespace BTQuant {

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
      volume_delta_period_us_(5000000) {} // Initialize to 5 seconds (5,000,000 microseconds)

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

void OrderbookPanel::render() {
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

  ImGui::Separator();

  // Get orderbook data
  auto orderbook_opt = processor_->getOrderbookData(symbol_id_);

  if (!orderbook_opt.has_value()) {
    ImGui::Text("Waiting for Orderbook: %s", symbol_name_.c_str());
    ImGui::Text("ID: %u", symbol_id_);
    ImGui::ProgressBar(((frame_count % 100) / 100.0f), ImVec2(-1, 0), "Polling Data Processor...");
    end_panel_window();
    return;
  }

  const auto& orderbook = orderbook_opt.value();

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
  double max_cumulative_vol = max_vol; // fallback to individual max if no cumulative data
  if (!cumulative_bids.empty()) max_cumulative_vol = std::max(max_cumulative_vol, cumulative_bids.back());
  if (!cumulative_asks.empty()) max_cumulative_vol = std::max(max_cumulative_vol, cumulative_asks.back());

  // Use Table instead of Columns for modern layout (C++26 style UI)
  if (ImGui::BeginTable("OrderbookTable", 8,
                        ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchSame)) {
    // Setup Columns
    ImGui::TableSetupColumn("Bid", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Sold", ImGuiTableColumnFlags_WidthFixed, 40);
    ImGui::TableSetupColumn("Price", ImGuiTableColumnFlags_WidthFixed, 80);
    ImGui::TableSetupColumn("Bought", ImGuiTableColumnFlags_WidthFixed, 40);
    ImGui::TableSetupColumn("Ask", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Delta", ImGuiTableColumnFlags_WidthFixed, 40);
    ImGui::TableSetupColumn("Δ Last 5s", ImGuiTableColumnFlags_WidthFixed, 60); // New column for volume delta over last 5 seconds
    ImGui::TableSetupColumn("Vol", ImGuiTableColumnFlags_WidthFixed, 40);
    ImGui::TableHeadersRow();

    const auto& colors = ThemeManager::getInstance().getColors();

    // Determine how many levels to show based on selected_levels_count_
    int max_levels_to_show = selected_levels_count_ == -1 ?
                             std::max(aggregated_asks.size(), aggregated_bids.size()) :
                             selected_levels_count_;

    // Use channel splitting to draw backgrounds before text content
    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    draw_list->ChannelsSplit(2); // Split into 2 channels: 0 for backgrounds, 1 for text (default)

    // Switch to background channel (0) to draw heatmap backgrounds first
    draw_list->ChannelsSetCurrent(0);

    // First, we need to render the table structure to establish row positions,
    // then we can draw the backgrounds in the correct positions

    // Render Asks (Sell) - Top down, but only to calculate positions
    int ask_count = std::min((int)aggregated_asks.size(), max_levels_to_show);
    for (int i = ask_count - 1; i >= 0; --i) {
      const auto& level = aggregated_asks[i];
      ImGui::TableNextRow();

      // Calculate heatmap intensity for this level
      float intensity = std::clamp((float)(level.size / max_vol), 0.0f, 1.0f);
      if (intensity > 0.05f) {
        // Calculate position for the entire row background
        ImVec2 row_pos = ImGui::GetCursorScreenPos();
        float row_height = ImGui::GetTextLineHeightWithSpacing();

        // Get the width of the table row
        float table_width = ImGui::GetContentRegionAvail().x + ImGui::GetCursorPosX();

        // Calculate the background rectangle for the entire row
        ImVec2 pos_min = row_pos;
        ImVec2 pos_max = ImVec2(row_pos.x + table_width, row_pos.y + row_height);

        // Red heatmap for asks
        ImU32 bg_color = ImGui::GetColorU32(ImVec4(1.0f, 0.5f, 0.0f, intensity * 0.3f));

        // Draw the rectangle in the background channel
        draw_list->AddRectFilled(pos_min, pos_max, bg_color);
      }
    }

    // Spread Row - also need to account for this in positioning
    ImGui::TableNextRow();

    // Render Bids (Buy) - but only to calculate positions
    int bid_count = std::min((int)aggregated_bids.size(), max_levels_to_show);
    for (int i = 0; i < bid_count; ++i) {
      const auto& level = aggregated_bids[i];
      ImGui::TableNextRow();

      // Calculate heatmap intensity for this level
      float intensity = std::clamp((float)(level.size / max_vol), 0.0f, 1.0f);
      if (intensity > 0.05f) {
        // Calculate position for the entire row background
        ImVec2 row_pos = ImGui::GetCursorScreenPos();
        float row_height = ImGui::GetTextLineHeightWithSpacing();

        // Get the width of the table row
        float table_width = ImGui::GetContentRegionAvail().x + ImGui::GetCursorPosX();

        // Calculate the background rectangle for the entire row
        ImVec2 pos_min = row_pos;
        ImVec2 pos_max = ImVec2(row_pos.x + table_width, row_pos.y + row_height);

        // Blue heatmap for bids
        ImU32 bg_color = ImGui::GetColorU32(ImVec4(0.0f, 0.6f, 1.0f, intensity * 0.3f));

        // Draw the rectangle in the background channel
        draw_list->AddRectFilled(pos_min, pos_max, bg_color);
      }
    }

    // Switch back to the default channel (1) for text content
    draw_list->ChannelsSetCurrent(1);

    // Now render the actual content in the default channel
    // Render Asks (Sell) - Top down

    // Render Asks (Sell) - Top down
    for (int i = ask_count - 1; i >= 0; --i) {
      const auto& level = aggregated_asks[i];
      ImGui::TableNextRow();
      ImGui::PushID(i);  // Unique ID for this row/side

      // Check if this is a large order
      bool is_large_order = average_order_size_ > 0 &&
                           (level.size / average_order_size_) * 100.0 >= large_order_threshold_percentage_;

      // 1. Bid (Empty)
      ImGui::TableSetColumnIndex(0);

      // 2. Sold (Accumulated) + Order Flow Activity Indicators
      ImGui::TableSetColumnIndex(1);
      if (volume_profile_.contains(level.price)) {
        double sold = volume_profile_[level.price].sold;
        if (sold > 0) {
          if (is_large_order) {
            // Draw yellow background for large orders
            ImVec2 text_pos = ImGui::GetCursorScreenPos();
            ImVec2 text_size = ImGui::CalcTextSize(std::format("%.0f", sold).c_str());
            ImGui::GetWindowDrawList()->AddRectFilled(
                text_pos,
                ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
                ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f))); // Yellow background

            // Draw text with increased weight effect
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f)); // Yellow text for large orders
            ImGui::TextColored(ImVec4(1, 0.5f, 0.5f, 1), "%.0f", sold);
            ImGui::PopStyleColor();
          } else {
            ImGui::TextColored(ImVec4(1, 0.5f, 0.5f, 1), "%.0f", sold);
          }
        }
      }

      // Draw order flow activity indicators
      if (order_flow_activity_.contains(level.price)) {
        const auto& activity = order_flow_activity_[level.price];
        int total_activity = activity.additions + activity.cancellations + activity.executions;

        if (total_activity > 0) {
          ImVec2 pos = ImGui::GetCursorScreenPos();

          // Draw small activity indicator dots
          float dot_size = std::min(4.0f + (total_activity / 10.0f), 8.0f); // Scale dot size with activity

          // Addition activity (green)
          if (activity.additions > 0) {
            float intensity = std::min(activity.additions / 10.0f, 1.0f); // Normalize intensity
            ImGui::GetWindowDrawList()->AddCircleFilled(
                ImVec2(pos.x + 2, pos.y + 2),
                dot_size * 0.5f,
                ImGui::GetColorU32(ImVec4(0.0f, 1.0f, 0.0f, intensity))
            );
          }

          // Cancellation activity (red)
          if (activity.cancellations > 0) {
            float intensity = std::min(activity.cancellations / 10.0f, 1.0f); // Normalize intensity
            ImGui::GetWindowDrawList()->AddCircleFilled(
                ImVec2(pos.x + 2, pos.y + 8),
                dot_size * 0.5f,
                ImGui::GetColorU32(ImVec4(1.0f, 0.0f, 0.0f, intensity))
            );
          }

          // Execution activity (blue)
          if (activity.executions > 0) {
            float intensity = std::min(activity.executions / 10.0f, 1.0f); // Normalize intensity
            ImGui::GetWindowDrawList()->AddCircleFilled(
                ImVec2(pos.x + 2, pos.y + 14),
                dot_size * 0.5f,
                ImGui::GetColorU32(ImVec4(0.0f, 0.0f, 1.0f, intensity))
            );
          }
        }
      }

      // 3. Price
      ImGui::TableSetColumnIndex(2);
      // Center Price text
      float cursor_check =
          ImGui::GetCursorPosX() + (ImGui::GetContentRegionAvail().x -
                                    ImGui::CalcTextSize(std::to_string(level.price).c_str()).x) *
                                       0.5f;
      ImGui::SetCursorPosX(cursor_check);

      if (is_large_order) {
        // Draw yellow background for the entire price cell
        ImVec2 cell_pos = ImGui::GetCursorScreenPos();
        ImVec2 cell_size = ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetTextLineHeightWithSpacing());
        ImGui::GetWindowDrawList()->AddRectFilled(
            cell_pos,
            ImVec2(cell_pos.x + cell_size.x, cell_pos.y + cell_size.y),
            ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.3f))); // Semi-transparent yellow background
      }

      ImGui::Selectable(std::format("{:.2f}", level.price).c_str(), false,
                        ImGuiSelectableFlags_SpanAllColumns);
      if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
        ImGui::SetDragDropPayload("PRICE_LEVEL", &level.price, sizeof(double));
        ImGui::Text("Price: %.2f", level.price);
        ImGui::EndDragDropSource();
      }

      // Draw cumulative volume bar extending from price column to the right
      if (i < static_cast<int>(cumulative_asks.size())) {
          float width = ImGui::GetContentRegionAvail().x;
          float bar_width = width * (float)(cumulative_asks[i] / max_cumulative_vol) * 0.7f; // Scale to fit in column
          ImVec2 pos = ImGui::GetCursorScreenPos();

          // Position the bar to start from the left edge of the price column and extend right
          ImGui::GetWindowDrawList()->AddRectFilled(
              ImVec2(pos.x, pos.y),
              ImVec2(pos.x + bar_width, pos.y + ImGui::GetTextLineHeightWithSpacing()),
              ImGui::GetColorU32(
                  ImVec4(colors.accent_red.x * 0.6f, colors.accent_red.y * 0.6f, colors.accent_red.z * 0.6f, 0.3f)));
      }

      ImGui::SameLine();
      if (is_large_order) {
        // Draw yellow background for large orders
        ImVec2 text_pos = ImGui::GetCursorScreenPos();
        ImVec2 text_size = ImGui::CalcTextSize(std::format("%.2f", level.price).c_str());
        ImGui::GetWindowDrawList()->AddRectFilled(
            text_pos,
            ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
            ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f))); // Yellow background

        // Draw text with increased weight effect
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f)); // Yellow text for large orders
        ImGui::TextColored(colors.accent_red, "%.2f", level.price);
        ImGui::PopStyleColor();
      } else {
        ImGui::TextColored(colors.accent_red, "%.2f", level.price);
      }

      // 4. Bought (Accumulated)
      ImGui::TableSetColumnIndex(3);
      if (volume_profile_.contains(level.price)) {
        double bought = volume_profile_[level.price].bought;
        if (bought > 0) {
          if (is_large_order) {
            // Draw yellow background for large orders
            ImVec2 text_pos = ImGui::GetCursorScreenPos();
            ImVec2 text_size = ImGui::CalcTextSize(std::format("%.0f", bought).c_str());
            ImGui::GetWindowDrawList()->AddRectFilled(
                text_pos,
                ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
                ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f))); // Yellow background

            // Draw text with increased weight effect
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f)); // Yellow text for large orders
            ImGui::TextColored(ImVec4(0.5f, 1, 0.5f, 1), "%.0f", bought);
            ImGui::PopStyleColor();
          } else {
            ImGui::TextColored(ImVec4(0.5f, 1, 0.5f, 1), "%.0f", bought);
          }
        }
      }

      // 5. Ask Size (with Bar)
      ImGui::TableSetColumnIndex(4);
      {
        float width = ImGui::GetContentRegionAvail().x;
        float bar_width = width * (float)(level.size / max_vol);
        ImVec2 pos = ImGui::GetCursorScreenPos();

        ImGui::GetWindowDrawList()->AddRectFilled(
            pos, ImVec2(pos.x + bar_width, pos.y + ImGui::GetTextLineHeightWithSpacing()),
            ImGui::GetColorU32(
                ImVec4(colors.accent_red.x, colors.accent_red.y, colors.accent_red.z, 0.2f)));

        if (is_large_order) {
          // Draw yellow background for large orders
          ImVec2 text_pos = ImGui::GetCursorScreenPos();
          ImVec2 text_size = ImGui::CalcTextSize(std::format("%.4f", level.size).c_str());
          ImGui::GetWindowDrawList()->AddRectFilled(
              text_pos,
              ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
              ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f))); // Yellow background

          // Draw text with increased weight effect by drawing it multiple times slightly offset
          // ImVec4 original_col = ImGui::GetStyle().Colors[ImGuiCol_Text];  // Unused variable
          ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f)); // Yellow text for large orders
          ImGui::Text("%.4f", level.size);
          ImGui::PopStyleColor();
        } else {
          ImGui::Text("%.4f", level.size);
        }
      }

      // 6. Delta (Accumulated)
      ImGui::TableSetColumnIndex(5);
      if (volume_profile_.contains(level.price)) {
        const auto& vol = volume_profile_[level.price];
        double delta = vol.bought - vol.sold;
        if (delta != 0) {
          ImVec4 color = delta > 0 ? ImVec4(0.5f, 1, 0.5f, 1) : ImVec4(1, 0.5f, 0.5f, 1);
          if (is_large_order) {
            // Draw yellow background for large orders
            ImVec2 text_pos = ImGui::GetCursorScreenPos();
            ImVec2 text_size = ImGui::CalcTextSize(std::format("%+.0f", delta).c_str());
            ImGui::GetWindowDrawList()->AddRectFilled(
                text_pos,
                ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
                ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f))); // Yellow background

            // Draw text with increased weight effect
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f)); // Yellow text for large orders
            ImGui::TextColored(color, "%+.0f", delta);
            ImGui::PopStyleColor();
          } else {
            ImGui::TextColored(color, "%+.0f", delta);
          }
        }
      }

      // 7. Volume Delta Over Last N Seconds
      ImGui::TableSetColumnIndex(6);
      {
        // Get the current time for delta calculation
        uint64_t current_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();

        // Look up the volume delta for this price level
        auto hist_it = volume_level_history_.find(level.price);
        if (hist_it != volume_level_history_.end()) {
          // For asks, we want the ask delta
          double volume_delta = hist_it->second.getAskDeltaOverPeriod(current_time, volume_delta_period_us_);

          if (volume_delta != 0) {
            ImVec4 color = volume_delta > 0 ? ImVec4(0.5f, 1, 0.5f, 1) : ImVec4(1, 0.5f, 0.5f, 1);

            if (is_large_order) {
              // Draw yellow background for large orders
              ImVec2 text_pos = ImGui::GetCursorScreenPos();
              ImVec2 text_size = ImGui::CalcTextSize(std::format("%+.2f", volume_delta).c_str());
              ImGui::GetWindowDrawList()->AddRectFilled(
                  text_pos,
                  ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
                  ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f))); // Yellow background

              // Draw text with increased weight effect
              ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f)); // Yellow text for large orders
              ImGui::TextColored(color, "%+.2f", volume_delta);
              ImGui::PopStyleColor();
            } else {
              ImGui::TextColored(color, "%+.2f", volume_delta);
            }
          }
        }
      }

      // 8. Volume (Accumulated)
      ImGui::TableSetColumnIndex(7);
      if (volume_profile_.contains(level.price)) {
        const auto& vol = volume_profile_[level.price];
        double total = vol.bought + vol.sold;
        if (total > 0) {
          if (is_large_order) {
            // Draw yellow background for large orders
            ImVec2 text_pos = ImGui::GetCursorScreenPos();
            ImVec2 text_size = ImGui::CalcTextSize(std::format("%.0f", total).c_str());
            ImGui::GetWindowDrawList()->AddRectFilled(
                text_pos,
                ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
                ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f))); // Yellow background

            // Draw text with increased weight effect
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f)); // Yellow text for large orders
            ImGui::Text("%.0f", total);
            ImGui::PopStyleColor();
          } else {
            ImGui::Text("%.0f", total);
          }
        }
      }

      ImGui::PopID();
    }

    // Spread Row
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(2);
    ImGui::TextColored(ImVec4(1, 1, 1, 0.5f), "--- %.1f ---", orderbook.spread);

    // Render Bids (Buy)
    for (int i = 0; i < bid_count; ++i) {
      const auto& level = aggregated_bids[i];
      ImGui::TableNextRow();
      ImGui::PushID(i + 1000);  // Offset to ensure uniqueness from Asks

      // Check if this is a large order
      bool is_large_order = average_order_size_ > 0 &&
                           (level.size / average_order_size_) * 100.0 >= large_order_threshold_percentage_;

      // 1. Bid Size (with Bar)
      ImGui::TableSetColumnIndex(0);
      {
        // Draw bar from right to left? Standard is Left or Right aligned.
        // Image 1 implies Right aligned for Bid? No, standard is bars grow from
        // center spine (Price). But here Columns are separated. Let's do
        // Standard Left-to-Right for now, or Right-to-Left if it looks better
        // next to Price. Let's do Right-to-Left for Bid to "point" to Price.
        float width = ImGui::GetContentRegionAvail().x;
        float bar_width = width * (float)(level.size / max_vol);
        ImVec2 pos = ImGui::GetCursorScreenPos();

        ImGui::GetWindowDrawList()->AddRectFilled(
            ImVec2(pos.x + width - bar_width, pos.y),
            ImVec2(pos.x + width, pos.y + ImGui::GetTextLineHeightWithSpacing()),
            ImGui::GetColorU32(
                ImVec4(colors.accent_green.x, colors.accent_green.y, colors.accent_green.z, 0.2f)));

        // Text Right Aligned
        auto text = std::format("{:.4f}", level.size);
        float text_width = ImGui::CalcTextSize(text.c_str()).x;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + width - text_width);

        if (is_large_order) {
          // Draw yellow background for large orders
          ImVec2 text_pos = ImGui::GetCursorScreenPos();
          ImVec2 text_size = ImGui::CalcTextSize(text.c_str());
          ImGui::GetWindowDrawList()->AddRectFilled(
              text_pos,
              ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
              ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f))); // Yellow background

          // Draw text with increased weight effect
          ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f)); // Yellow text for large orders
          ImGui::TextUnformatted(text.c_str());
          ImGui::PopStyleColor();
        } else {
          ImGui::TextUnformatted(text.c_str());
        }
      }

      // 2. Sold + Order Flow Activity Indicators
      ImGui::TableSetColumnIndex(1);
      if (volume_profile_.contains(level.price)) {
        double sold = volume_profile_[level.price].sold;
        if (sold > 0) {
          if (is_large_order) {
            // Draw yellow background for large orders
            ImVec2 text_pos = ImGui::GetCursorScreenPos();
            ImVec2 text_size = ImGui::CalcTextSize(std::format("%.0f", sold).c_str());
            ImGui::GetWindowDrawList()->AddRectFilled(
                text_pos,
                ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
                ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f))); // Yellow background

            // Draw text with increased weight effect
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f)); // Yellow text for large orders
            ImGui::TextColored(ImVec4(1, 0.5f, 0.5f, 1), "%.0f", sold);
            ImGui::PopStyleColor();
          } else {
            ImGui::TextColored(ImVec4(1, 0.5f, 0.5f, 1), "%.0f", sold);
          }
        }
      }

      // Draw order flow activity indicators
      if (order_flow_activity_.contains(level.price)) {
        const auto& activity = order_flow_activity_[level.price];
        int total_activity = activity.additions + activity.cancellations + activity.executions;

        if (total_activity > 0) {
          ImVec2 pos = ImGui::GetCursorScreenPos();

          // Draw small activity indicator dots
          float dot_size = std::min(4.0f + (total_activity / 10.0f), 8.0f); // Scale dot size with activity

          // Addition activity (green)
          if (activity.additions > 0) {
            float intensity = std::min(activity.additions / 10.0f, 1.0f); // Normalize intensity
            ImGui::GetWindowDrawList()->AddCircleFilled(
                ImVec2(pos.x + 2, pos.y + 2),
                dot_size * 0.5f,
                ImGui::GetColorU32(ImVec4(0.0f, 1.0f, 0.0f, intensity))
            );
          }

          // Cancellation activity (red)
          if (activity.cancellations > 0) {
            float intensity = std::min(activity.cancellations / 10.0f, 1.0f); // Normalize intensity
            ImGui::GetWindowDrawList()->AddCircleFilled(
                ImVec2(pos.x + 2, pos.y + 8),
                dot_size * 0.5f,
                ImGui::GetColorU32(ImVec4(1.0f, 0.0f, 0.0f, intensity))
            );
          }

          // Execution activity (blue)
          if (activity.executions > 0) {
            float intensity = std::min(activity.executions / 10.0f, 1.0f); // Normalize intensity
            ImGui::GetWindowDrawList()->AddCircleFilled(
                ImVec2(pos.x + 2, pos.y + 14),
                dot_size * 0.5f,
                ImGui::GetColorU32(ImVec4(0.0f, 0.0f, 1.0f, intensity))
            );
          }
        }
      }

      // 3. Price
      ImGui::TableSetColumnIndex(2);
      float cursor_check =
          ImGui::GetCursorPosX() + (ImGui::GetContentRegionAvail().x -
                                    ImGui::CalcTextSize(std::to_string(level.price).c_str()).x) *
                                       0.5f;
      ImGui::SetCursorPosX(cursor_check);

      if (is_large_order) {
        // Draw yellow background for the entire price cell
        ImVec2 cell_pos = ImGui::GetCursorScreenPos();
        ImVec2 cell_size = ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetTextLineHeightWithSpacing());
        ImGui::GetWindowDrawList()->AddRectFilled(
            cell_pos,
            ImVec2(cell_pos.x + cell_size.x, cell_pos.y + cell_size.y),
            ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.3f))); // Semi-transparent yellow background
      }

      ImGui::Selectable(std::format("{:.2f}", level.price).c_str(), false,
                        ImGuiSelectableFlags_SpanAllColumns);
      if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
        ImGui::SetDragDropPayload("PRICE_LEVEL", &level.price, sizeof(double));
        ImGui::Text("Price: %.2f", level.price);
        ImGui::EndDragDropSource();
      }

      // Draw cumulative volume bar extending from price column to the left
      if (i < static_cast<int>(cumulative_bids.size())) {
          float width = ImGui::GetContentRegionAvail().x;
          float bar_width = width * (float)(cumulative_bids[i] / max_cumulative_vol) * 0.7f; // Scale to fit in column
          ImVec2 pos = ImGui::GetCursorScreenPos();

          // Position the bar to start from the right edge of the price column and extend left
          ImGui::GetWindowDrawList()->AddRectFilled(
              ImVec2(pos.x + width - bar_width, pos.y),
              ImVec2(pos.x + width, pos.y + ImGui::GetTextLineHeightWithSpacing()),
              ImGui::GetColorU32(
                  ImVec4(colors.accent_green.x * 0.6f, colors.accent_green.y * 0.6f, colors.accent_green.z * 0.6f, 0.3f)));
      }

      ImGui::SameLine();
      if (is_large_order) {
        // Draw yellow background for large orders
        ImVec2 text_pos = ImGui::GetCursorScreenPos();
        ImVec2 text_size = ImGui::CalcTextSize(std::format("%.2f", level.price).c_str());
        ImGui::GetWindowDrawList()->AddRectFilled(
            text_pos,
            ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
            ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f))); // Yellow background

        // Draw text with increased weight effect
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f)); // Yellow text for large orders
        ImGui::TextColored(colors.accent_green, "%.2f",
                           level.price);  // Green for Bid Price
        ImGui::PopStyleColor();
      } else {
        ImGui::TextColored(colors.accent_green, "%.2f",
                           level.price);  // Green for Bid Price
      }

      // 4. Bought
      ImGui::TableSetColumnIndex(3);
      if (volume_profile_.contains(level.price)) {
        double bought = volume_profile_[level.price].bought;
        if (bought > 0) {
          if (is_large_order) {
            // Draw yellow background for large orders
            ImVec2 text_pos = ImGui::GetCursorScreenPos();
            ImVec2 text_size = ImGui::CalcTextSize(std::format("%.0f", bought).c_str());
            ImGui::GetWindowDrawList()->AddRectFilled(
                text_pos,
                ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
                ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f))); // Yellow background

            // Draw text with increased weight effect
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f)); // Yellow text for large orders
            ImGui::TextColored(ImVec4(0.5f, 1, 0.5f, 1), "%.0f", bought);
            ImGui::PopStyleColor();
          } else {
            ImGui::TextColored(ImVec4(0.5f, 1, 0.5f, 1), "%.0f", bought);
          }
        }
      }

      // 5. Ask (Empty)
      ImGui::TableSetColumnIndex(4);

      // 6. Delta
      ImGui::TableSetColumnIndex(5);
      if (volume_profile_.contains(level.price)) {
        const auto& vol = volume_profile_[level.price];
        double delta = vol.bought - vol.sold;
        if (delta != 0) {
          ImVec4 color = delta > 0 ? ImVec4(0.5f, 1, 0.5f, 1) : ImVec4(1, 0.5f, 0.5f, 1);
          if (is_large_order) {
            // Draw yellow background for large orders
            ImVec2 text_pos = ImGui::GetCursorScreenPos();
            ImVec2 text_size = ImGui::CalcTextSize(std::format("%+.0f", delta).c_str());
            ImGui::GetWindowDrawList()->AddRectFilled(
                text_pos,
                ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
                ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f))); // Yellow background

            // Draw text with increased weight effect
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f)); // Yellow text for large orders
            ImGui::TextColored(color, "%+.0f", delta);
            ImGui::PopStyleColor();
          } else {
            ImGui::TextColored(color, "%+.0f", delta);
          }
        }
      }

      // 7. Volume Delta Over Last N Seconds
      ImGui::TableSetColumnIndex(6);
      {
        // Get the current time for delta calculation
        uint64_t current_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();

        // Look up the volume delta for this price level
        auto hist_it = volume_level_history_.find(level.price);
        if (hist_it != volume_level_history_.end()) {
          // For bids, we want the bid delta
          double volume_delta = hist_it->second.getBidDeltaOverPeriod(current_time, volume_delta_period_us_);

          if (volume_delta != 0) {
            ImVec4 color = volume_delta > 0 ? ImVec4(0.5f, 1, 0.5f, 1) : ImVec4(1, 0.5f, 0.5f, 1);

            if (is_large_order) {
              // Draw yellow background for large orders
              ImVec2 text_pos = ImGui::GetCursorScreenPos();
              ImVec2 text_size = ImGui::CalcTextSize(std::format("%+.2f", volume_delta).c_str());
              ImGui::GetWindowDrawList()->AddRectFilled(
                  text_pos,
                  ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
                  ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f))); // Yellow background

              // Draw text with increased weight effect
              ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f)); // Yellow text for large orders
              ImGui::TextColored(color, "%+.2f", volume_delta);
              ImGui::PopStyleColor();
            } else {
              ImGui::TextColored(color, "%+.2f", volume_delta);
            }
          }
        }
      }

      // 8. Vol
      ImGui::TableSetColumnIndex(7);
      if (volume_profile_.contains(level.price)) {
        const auto& vol = volume_profile_[level.price];
        double total = vol.bought + vol.sold;
        if (total > 0) {
          if (is_large_order) {
            // Draw yellow background for large orders
            ImVec2 text_pos = ImGui::GetCursorScreenPos();
            ImVec2 text_size = ImGui::CalcTextSize(std::format("%.0f", total).c_str());
            ImGui::GetWindowDrawList()->AddRectFilled(
                text_pos,
                ImVec2(text_pos.x + text_size.x, text_pos.y + text_size.y),
                ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 0.5f))); // Yellow background

            // Draw text with increased weight effect
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.0f, 1.0f)); // Yellow text for large orders
            ImGui::Text("%.0f", total);
            ImGui::PopStyleColor();
          } else {
            ImGui::Text("%.0f", total);
          }
        }
      }

      ImGui::PopID();
    }

    // Merge the channels back together
    draw_list->ChannelsMerge();

    ImGui::EndTable();
  }
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
    ImPlot::SetNextFillStyle(colors.accent_green);
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

    ImPlot::SetNextFillStyle(colors.accent_red);
    ImPlot::PlotShaded("Asks", ax.data(), ay.data(), (int)ax.size(), 0);

    ImPlot::EndPlot();
  }
}

}  // namespace BTQuant
