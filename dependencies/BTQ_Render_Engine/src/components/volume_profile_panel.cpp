#include "../../include/components/volume_profile_panel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

#include "imgui.h"
#include "implot.h"

namespace BTQuant {

VolumeProfilePanel::VolumeProfilePanel(const PanelConfig& config,
                                       std::shared_ptr<HotSpineDataBridge> bridge,
                                       std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(config), bridge_(bridge), processor_(processor) {
  volume_profile_.reserve(NUM_PRICE_LEVELS);

  // C++26: Subscribe to push notifications instead of polling
  subscribe_to_updates();
}

VolumeProfilePanel::~VolumeProfilePanel() {
  // C++26: Clean unsubscription on destruction
  if (processor_ && subscription_id_ != 0) {
    processor_->unsubscribe(subscription_id_);
  }
}

void VolumeProfilePanel::subscribe_to_updates() {
  if (!processor_ || symbol_id_ == 0) return;

  // Unsubscribe from previous symbol if any
  if (subscription_id_ != 0) {
    processor_->unsubscribe(subscription_id_);
  }

  // Subscribe to TRADE notifications for this symbol
  // Callback sets dirty flag - will be processed in next render()
  subscription_id_ = processor_->subscribe(
      symbol_id_, RenderEngine::NotificationType::TRADE,
      [this](uint32_t /*symbol_id*/, RenderEngine::NotificationType /*type*/) {
        // Thread-safe: atomic flag set from worker thread
        this->markDirty();
      });
}

void VolumeProfilePanel::render() {
  begin_panel_window();

  if (!is_visible()) {
    end_panel_window();
    return;
  }

  render_panel_header();
  render_controls();
  ImGui::Separator();

  // C++26 Reactive: Only rebuild when new data arrives or first load
  if (processor_ && symbol_id_ != 0) {
    // consumeDirty() returns true initially or when notified
    if (consumeDirty() || volume_profile_.empty()) {
      build_volume_profile();
    }
  }

  render_volume_bars();

  end_panel_window();
}

void VolumeProfilePanel::set_symbol(uint32_t symbol_id, const std::string& symbol_name) {
  symbol_id_ = symbol_id;
  symbol_name_ = symbol_name;
  volume_profile_.clear();
  max_volume_ = 0.0;
  poc_price_ = 0.0;

  // Re-subscribe to new symbol
  subscribe_to_updates();
  markDirty();  // Force immediate build
}

void VolumeProfilePanel::build_volume_profile() {
  auto analytics = processor_->getSymbolAnalytics(symbol_id_);
  auto trades = analytics.recent_trades;  // Copy to potentially filter

  // Filter trades based on custom time range if enabled and in Custom Profile mode
  if (use_custom_time_range_ && profile_mode_ == ProfileMode::Custom && !trades.empty()) {
    std::vector<RenderEngine::TradeData> filtered_trades;

    for (const auto& trade : trades) {
      if (trade.timestamp >= custom_start_time_ && trade.timestamp <= custom_end_time_) {
        filtered_trades.push_back(trade);
      }
    }

    trades = std::move(filtered_trades);
  }

  if (trades.empty()) return;

  // Check if we have an existing volume profile and if the new trades fit in the current range
  bool needs_rebuild = false;
  double min_price = std::numeric_limits<double>::max();
  double max_price = std::numeric_limits<double>::lowest();

  // Find the range of new trades
  for (const auto& trade : trades) {
    min_price = std::min(min_price, trade.price);
    max_price = std::max(max_price, trade.price);
  }

  // If we have an existing profile, check if new trades fall outside the current range
  if (!volume_profile_.empty()) {
    double current_min_price = volume_profile_.front().price - (price_bucket_size_ / 2.0);
    double current_max_price = volume_profile_.back().price + (price_bucket_size_ / 2.0);

    // Check if any new trade falls outside the current range
    if (min_price < current_min_price || max_price > current_max_price) {
      needs_rebuild = true;
    } else {
      // Trades fit in existing range, update incrementally
      for (const auto& trade : trades) {
        // Calculate which bucket this trade belongs to based on existing bucket size
        size_t bucket_index =
            static_cast<size_t>((trade.price - current_min_price) / price_bucket_size_);

        if (bucket_index < volume_profile_.size()) {
          if (trade.is_buy) {
            volume_profile_[bucket_index].buy_volume += trade.size;
          } else {
            volume_profile_[bucket_index].sell_volume += trade.size;
          }
          volume_profile_[bucket_index].total_volume += trade.size;

          // Update max volume if needed
          double max_vol_in_bucket = std::max(volume_profile_[bucket_index].buy_volume,
                                              volume_profile_[bucket_index].sell_volume);
          if (max_vol_in_bucket > max_volume_) {
            max_volume_ = max_vol_in_bucket;
          }
        }
      }

      // Update POC
      double poc_volume = 0;
      for (const auto& level : volume_profile_) {
        double total = level.buy_volume + level.sell_volume;
        if (total > poc_volume) {
          poc_volume = total;
          poc_price_ = level.price;
        }
      }

      // Recalculate Value Area after incremental update
      calculate_value_area();

      return;  // Exit early since we've updated incrementally
    }
  } else {
    needs_rebuild = true;  // First time building, so we need to rebuild
  }

  if (needs_rebuild || volume_profile_.empty()) {
    if (max_price <= min_price) return;

    // Compute bucket size
    double range = max_price - min_price;
    price_bucket_size_ = range / NUM_PRICE_LEVELS;
    if (price_bucket_size_ <= 0) price_bucket_size_ = 1.0;

    // Reset profile
    volume_profile_.clear();
    volume_profile_.resize(NUM_PRICE_LEVELS);

    for (size_t i = 0; i < NUM_PRICE_LEVELS; ++i) {
      volume_profile_[i].price = min_price + (i + 0.5) * price_bucket_size_;
      volume_profile_[i].buy_volume = 0;
      volume_profile_[i].sell_volume = 0;
      volume_profile_[i].total_volume = 0;
    }

    // Aggregate trades into buckets
    for (const auto& trade : trades) {
      size_t bucket = static_cast<size_t>((trade.price - min_price) / price_bucket_size_);
      bucket = std::min(bucket, NUM_PRICE_LEVELS - 1);

      if (trade.is_buy) {
        volume_profile_[bucket].buy_volume += trade.size;
      } else {
        volume_profile_[bucket].sell_volume += trade.size;
      }
      volume_profile_[bucket].total_volume += trade.size;
    }

    // Find POC and max volume
    max_volume_ = 0;
    poc_price_ = volume_profile_[0].price;
    double poc_volume = 0;

    for (const auto& level : volume_profile_) {
      double total = level.buy_volume + level.sell_volume;
      max_volume_ = std::max(max_volume_, std::max(level.buy_volume, level.sell_volume));
      if (total > poc_volume) {
        poc_volume = total;
        poc_price_ = level.price;
      }
    }

    // Calculate Value Area
    calculate_value_area();
  }
}

void VolumeProfilePanel::render_controls() {
  if (ImGui::Button("Reset View")) {
    ImPlot::SetNextAxesToFit();
  }
  ImGui::SameLine();
  ImGui::Text("Symbol: %s", symbol_name_.c_str());
  ImGui::SameLine();
  ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "| POC: %.4f", poc_price_);

  // Add level count control
  ImGui::SameLine();
  ImGui::Text("| Levels: %zu", volume_profile_.size());

  // Add profile mode selection
  ImGui::SameLine();
  const char* profile_modes[] = {"Step", "Right", "Left", "Custom"};
  ImGui::Combo("Profile Mode", reinterpret_cast<int*>(&profile_mode_), profile_modes, 4);

  // Add VA% control
  ImGui::SameLine();
  ImGui::SliderInt("VA%", &profile_settings_.vaPercent, 50, 99);

  // Add controls for Custom Profile range when Custom mode is selected
  if (profile_mode_ == ProfileMode::Custom) {
    ImGui::SameLine();
    ImGui::Checkbox("Use Time Range", &use_custom_time_range_);

    if (use_custom_time_range_) {
      ImGui::Separator();

      // Initialize start and end times if not already set
      if (custom_start_time_ == 0.0 && custom_end_time_ == 0.0) {
        // Get current time range from available data
        auto analytics = processor_->getSymbolAnalytics(symbol_id_);
        const auto& trades = analytics.recent_trades;

        if (!trades.empty()) {
          // Find min and max timestamps
          double min_time = trades[0].timestamp;
          double max_time = trades[0].timestamp;

          for (const auto& trade : trades) {
            if (trade.timestamp < min_time) min_time = trade.timestamp;
            if (trade.timestamp > max_time) max_time = trade.timestamp;
          }

          custom_start_time_ = min_time;
          custom_end_time_ = max_time;
        }
      }

      // Show controls for start and end times
      ImGui::Text("Time Range:");
      ImGui::SameLine();
      ImGui::Text("Start: %.2f", custom_start_time_);
      ImGui::SameLine();
      ImGui::Text("End: %.2f", custom_end_time_);

      // Buttons to reset to full range
      if (ImGui::Button("Reset Time Range")) {
        auto analytics = processor_->getSymbolAnalytics(symbol_id_);
        const auto& trades = analytics.recent_trades;

        if (!trades.empty()) {
          // Find min and max timestamps
          double min_time = trades[0].timestamp;
          double max_time = trades[0].timestamp;

          for (const auto& trade : trades) {
            if (trade.timestamp < min_time) min_time = trade.timestamp;
            if (trade.timestamp > max_time) max_time = trade.timestamp;
          }

          custom_start_time_ = min_time;
          custom_end_time_ = max_time;
        }
      }
    }
  }

  // Display VAH and VAL if available
  ImGui::SameLine();
  ImGui::Text("| VAH: %.4f", vah_price_);
  ImGui::SameLine();
  ImGui::Text("| VAL: %.4f", val_price_);

  // Add profile statistics panel
  if (ImGui::CollapsingHeader("Profile Statistics")) {
    ImGui::Indent();

    // Display POC price
    ImGui::Text("POC Price: %.4f", poc_price_);

    // Display VAH price
    ImGui::Text("VAH Price: %.4f", vah_price_);

    // Display VAL price
    ImGui::Text("VAL Price: %.4f", val_price_);

    // Calculate and display total volume in value area
    double total_volume_in_value_area = 0.0;
    double total_volume_above_poc = 0.0;
    double total_volume_below_poc = 0.0;

    if (!volume_profile_.empty()) {
      for (const auto& level : volume_profile_) {
        // Check if this price level is within the value area
        if (level.price >= val_price_ && level.price <= vah_price_) {
          total_volume_in_value_area += level.total_volume;
        }

        // Calculate volume above and below POC
        if (level.price > poc_price_) {
          total_volume_above_poc += level.total_volume;
        } else if (level.price < poc_price_) {
          total_volume_below_poc += level.total_volume;
        }
      }
    }

    ImGui::Text("Total Volume in Value Area: %.2f", total_volume_in_value_area);

    // Calculate and display percentage of volume above POC
    double total_volume = total_volume_above_poc + total_volume_below_poc;
    if (total_volume > 0) {
      double percentage_above_poc = (total_volume_above_poc / total_volume) * 100.0;
      ImGui::Text("Percentage of Volume Above POC: %.2f%%", percentage_above_poc);
    } else {
      ImGui::Text("Percentage of Volume Above POC: N/A");
    }

    ImGui::Unindent();
  }
}

void VolumeProfilePanel::render_volume_bars() {
  if (volume_profile_.empty() || max_volume_ <= 0) {
    ImGui::Text("No volume data available");
    return;
  }

  ImVec2 region = ImGui::GetContentRegionAvail();
  if (region.x < 100 || region.y < 100) return;

  // Prepare data for ImPlot horizontal bars
  std::vector<double> prices;
  std::vector<double> buy_volumes;
  std::vector<double> sell_volumes;

  prices.reserve(volume_profile_.size());
  buy_volumes.reserve(volume_profile_.size());
  sell_volumes.reserve(volume_profile_.size());

  for (const auto& level : volume_profile_) {
    prices.push_back(level.price);
    buy_volumes.push_back(level.buy_volume);
    sell_volumes.push_back(-level.sell_volume);  // Negative for left side
  }

  // Unique plot ID per panel instance to avoid ImGui ID conflicts
  char plot_id[64];
  snprintf(plot_id, sizeof(plot_id), "##VolumeProfile_%s", config_.title.c_str());

  if (ImPlot::BeginPlot(plot_id, region, ImPlotFlags_NoTitle | ImPlotFlags_NoLegend)) {
    ImPlot::SetupAxes("Volume", "Price", ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_Invert,
                      ImPlotAxisFlags_AutoFit);

    // Bar height based on price bucket size
    double bar_height = price_bucket_size_ * 0.8;

    // For Right and Left profiles, recalculate value area based on visible range during rendering
    double local_vah_price = vah_price_;
    double local_val_price = val_price_;

    if (profile_mode_ == ProfileMode::Right || profile_mode_ == ProfileMode::Left) {
      // Get the plot limits to determine what's currently visible
      ImPlotRect plot_limits = ImPlot::GetPlotLimits();

      // Create a temporary vector of visible levels with their volumes
      std::vector<std::pair<double, double>> visible_levels; // {price, total_volume}
      double total_volume = 0.0;

      for (const auto& level : volume_profile_) {
        if (level.price >= plot_limits.Y.Min && level.price <= plot_limits.Y.Max) {
          visible_levels.push_back({level.price, level.total_volume});
          total_volume += level.total_volume;
        }
      }

      if (!visible_levels.empty() && total_volume > 0) {
        // Target volume for value area (based on profile_settings_.vaPercent % of total volume)
        double target_volume = (static_cast<double>(profile_settings_.vaPercent) / 100.0) * total_volume;

        // Sort visible levels by volume in descending order to find POC
        std::vector<std::pair<double, double>> sorted_levels = visible_levels;
        std::sort(sorted_levels.begin(), sorted_levels.end(),
                  [](const std::pair<double, double>& a, const std::pair<double, double>& b) {
                    return a.second > b.second; // Sort by volume descending
                  });

        // Find the POC (Point of Control) - the price level with highest volume among visible levels
        double poc_price = sorted_levels[0].first;

        // Sort visible levels by price to make expansion easier
        std::sort(visible_levels.begin(), visible_levels.end());

        // Find the index of POC in the price-sorted array
        size_t poc_idx_sorted = 0;
        for (size_t i = 0; i < visible_levels.size(); ++i) {
          if (std::abs(visible_levels[i].first - poc_price) < 0.000001) { // Use epsilon comparison for floating point
            poc_idx_sorted = i;
            break;
          }
        }

        // Expand from POC outward to capture the required volume
        size_t start_idx = poc_idx_sorted;
        size_t end_idx = poc_idx_sorted;
        double current_volume = visible_levels[poc_idx_sorted].second;

        // Expand upward (higher prices) and downward (lower prices) alternately
        // until we reach the target volume
        while (current_volume < target_volume) {
          // Decide whether to expand up or down
          bool expand_up = false;
          bool expand_down = false;

          // Check if we can expand in each direction
          if (start_idx > 0) expand_down = true;
          if (end_idx < visible_levels.size() - 1) expand_up = true;

          // If we can't expand in either direction, break
          if (!expand_up && !expand_down) break;

          // If we can only expand in one direction, do that
          if (!expand_up && expand_down) {
            start_idx--;
            current_volume += visible_levels[start_idx].second;
          } else if (expand_up && !expand_down) {
            end_idx++;
            current_volume += visible_levels[end_idx].second;
          } else {
            // We can expand in both directions - choose the direction with higher volume
            double vol_up = visible_levels[end_idx + 1].second;
            double vol_down = visible_levels[start_idx - 1].second;

            if (vol_up >= vol_down) {
              end_idx++;
              current_volume += visible_levels[end_idx].second;
            } else {
              start_idx--;
              current_volume += visible_levels[start_idx].second;
            }
          }

          // If we've captured enough volume, break
          if (current_volume >= target_volume) break;
        }

        // Set the local VAH and VAL prices for this render pass
        local_vah_price = visible_levels[end_idx].first;
        local_val_price = visible_levels[start_idx].first;
      }
    }

    // Draw Value Area overlay if we have valid VAH and VAL
    if (local_vah_price > 0 && local_val_price > 0 && local_vah_price >= local_val_price) {
      ImDrawList* draw_list = ImPlot::GetPlotDrawList();

      // Convert value area prices to pixel coordinates
      ImVec2 val_pixel = ImPlot::PlotToPixels(0, local_val_price);  // Left side of plot at VAL price
      ImVec2 vah_pixel = ImPlot::PlotToPixels(0, local_vah_price);  // Left side of plot at VAH price

      // Get the plot area dimensions
      ImVec2 plot_size = ImPlot::GetPlotSize();
      ImVec2 plot_pos = ImPlot::GetPlotPos();

      // Draw semi-transparent rectangle for value area
      // Note: In ImPlot coordinate system, y increases downward, so VAL should have a higher y
      // value than VAH
      ImVec2 area_top_left = ImVec2(plot_pos.x, vah_pixel.y);
      ImVec2 area_bottom_right = ImVec2(plot_pos.x + plot_size.x, val_pixel.y);

      // Draw the value area overlay with semi-transparent color
      draw_list->AddRectFilled(area_top_left, area_bottom_right,
                               IM_COL32(138, 43, 226, 80));  // Semi-transparent purple
    }

    // Render based on profile mode
    switch (profile_mode_) {
      case ProfileMode::Step:
        render_step_profile(prices.data(), buy_volumes.data(), sell_volumes.data(),
                            static_cast<int>(prices.size()), bar_height);
        break;
      case ProfileMode::Right: {
        // Right Profile: Aggregate all visible trades into single histogram anchored to right edge with horizontal bars extending left
        // Get the plot limits to determine what's currently visible
        ImPlotRect plot_limits = ImPlot::GetPlotLimits();  // This gets the current visible range

        // Calculate aggregated volumes for visible trades only
        double aggregated_buy_volume = 0.0;
        double aggregated_sell_volume = 0.0;

        // Aggregate visible trades
        for (size_t i = 0; i < buy_volumes.size(); ++i) {
          if (prices[i] >= plot_limits.Y.Min && prices[i] <= plot_limits.Y.Max) {
            if (buy_volumes[i] > 0) {
              aggregated_buy_volume += buy_volumes[i];
            }
            if (sell_volumes[i] < 0) {
              aggregated_sell_volume += std::abs(sell_volumes[i]);  // Store as positive for aggregation
            }
          }
        }

        // Calculate the maximum volume for scaling - use the aggregated max to ensure proper display
        double max_vol = std::max(aggregated_buy_volume, aggregated_sell_volume);
        if (max_vol <= 0) max_vol = max_volume_;  // Fallback to global max if aggregated volumes are zero
        if (max_vol <= 0) max_vol = 1.0;  // Ultimate fallback to 1.0 if no volume data

        // Define positions for the aggregated bars - anchored to the right edge of the plot
        double visible_center_price = (plot_limits.Y.Min + plot_limits.Y.Max) / 2.0;
        float bar_height_total = (plot_limits.Y.Max - plot_limits.Y.Min) * 0.25f;  // Use 25% of visible height for each bar
        float bar_spacing = (plot_limits.Y.Max - plot_limits.Y.Min) * 0.05f;      // Small spacing between bars

        // Calculate the rightmost x-coordinate in plot space (this will be our anchor)
        // In horizontal bar charts, volume is on X-axis and price is on Y-axis
        // So we want to anchor to the maximum X value (right edge of chart)
        double right_anchor = max_vol;  // Anchor to the maximum volume value

        // Draw aggregated buy bar (green) extending left from right edge
        if (aggregated_buy_volume > 0) {
          // Position at upper portion of visible range
          double buy_bar_price = visible_center_price - (bar_height_total + bar_spacing) * 0.5;  // Offset to separate from sell bar

          // Calculate the left extent of the bar based on the aggregated volume
          double buy_bar_left_extent = right_anchor - (aggregated_buy_volume / max_vol) * max_vol;

          // Draw the aggregated buy bar extending left from the right edge
          ImDrawList* draw_list = ImPlot::GetPlotDrawList();

          // Convert plot coordinates to pixel coordinates for the bar
          ImVec2 right_edge_px = ImPlot::PlotToPixels(right_anchor, buy_bar_price);
          ImVec2 left_edge_px = ImPlot::PlotToPixels(buy_bar_left_extent, buy_bar_price);

          float bar_top = right_edge_px.y - bar_height_total / 2.0f;
          float bar_bottom = right_edge_px.y + bar_height_total / 2.0f;
          float bar_right = right_edge_px.x;  // Right edge of chart
          float bar_left = left_edge_px.x;    // Left extent of the bar

          // Draw the aggregated buy bar
          draw_list->AddRectFilled(ImVec2(bar_left, bar_top), ImVec2(bar_right, bar_bottom),
                                   IM_COL32(26, 204, 26, 179));  // Green with transparency

          // Add border for better visibility
          draw_list->AddRect(ImVec2(bar_left, bar_top), ImVec2(bar_right, bar_bottom),
                             IM_COL32(0, 0, 0, 100), 0.0f, 0, 1.0f);
        }

        // Draw aggregated sell bar (red) extending left from right edge
        if (aggregated_sell_volume > 0) {
          // Position at lower portion of visible range
          double sell_bar_price = visible_center_price + (bar_height_total + bar_spacing) * 0.5;  // Offset to separate from buy bar

          // Calculate the left extent of the bar based on the aggregated volume
          double sell_bar_left_extent = right_anchor - (aggregated_sell_volume / max_vol) * max_vol;

          // Draw the aggregated sell bar extending left from the right edge
          ImDrawList* draw_list = ImPlot::GetPlotDrawList();

          // Convert plot coordinates to pixel coordinates for the bar
          ImVec2 right_edge_px = ImPlot::PlotToPixels(right_anchor, sell_bar_price);
          ImVec2 left_edge_px = ImPlot::PlotToPixels(sell_bar_left_extent, sell_bar_price);

          float bar_top = right_edge_px.y - bar_height_total / 2.0f;
          float bar_bottom = right_edge_px.y + bar_height_total / 2.0f;
          float bar_right = right_edge_px.x;  // Right edge of chart
          float bar_left = left_edge_px.x;    // Left extent of the bar

          // Draw the aggregated sell bar
          draw_list->AddRectFilled(ImVec2(bar_left, bar_top), ImVec2(bar_right, bar_bottom),
                                   IM_COL32(204, 26, 26, 179));  // Red with transparency

          // Add border for better visibility
          draw_list->AddRect(ImVec2(bar_left, bar_top), ImVec2(bar_right, bar_bottom),
                             IM_COL32(0, 0, 0, 100), 0.0f, 0, 1.0f);
        }

        // Draw labels for the aggregated bars
        if (aggregated_buy_volume > 0 || aggregated_sell_volume > 0) {
          ImDrawList* draw_list = ImPlot::GetPlotDrawList();

          // Draw text labels
          char buy_label[64];
          char sell_label[64];
          snprintf(buy_label, sizeof(buy_label), "B: %.2f", aggregated_buy_volume);
          snprintf(sell_label, sizeof(sell_label), "S: %.2f", aggregated_sell_volume);

          // Position labels appropriately
          if (aggregated_buy_volume > 0) {
            ImVec2 center_pos = ImPlot::PlotToPixels(right_anchor - ((aggregated_buy_volume / max_vol) * max_vol)/2,
                                                    visible_center_price - (bar_height_total + bar_spacing) * 0.5);
            draw_list->AddText(ImVec2(center_pos.x, center_pos.y - 8), IM_COL32(255, 255, 255, 255), buy_label);
          }

          if (aggregated_sell_volume > 0) {
            ImVec2 center_pos = ImPlot::PlotToPixels(right_anchor - ((aggregated_sell_volume / max_vol) * max_vol)/2,
                                                    visible_center_price + (bar_height_total + bar_spacing) * 0.5);
            draw_list->AddText(ImVec2(center_pos.x, center_pos.y - 8), IM_COL32(255, 255, 255, 255), sell_label);
          }
        }
        break;
      }
      case ProfileMode::Left: {
        // Left Profile: Aggregate all visible trades into single histogram anchored to left edge with horizontal bars extending right
        // Get the plot limits to determine what's currently visible
        ImPlotRect plot_limits = ImPlot::GetPlotLimits();  // This gets the current visible range

        // Calculate aggregated volumes for visible trades only
        double aggregated_buy_volume = 0.0;
        double aggregated_sell_volume = 0.0;

        // Aggregate visible trades
        for (size_t i = 0; i < buy_volumes.size(); ++i) {
          if (prices[i] >= plot_limits.Y.Min && prices[i] <= plot_limits.Y.Max) {
            if (buy_volumes[i] > 0) {
              aggregated_buy_volume += buy_volumes[i];
            }
            if (sell_volumes[i] < 0) {
              aggregated_sell_volume += std::abs(sell_volumes[i]);  // Store as positive for aggregation
            }
          }
        }

        // Calculate the maximum volume for scaling - use the aggregated max to ensure proper display
        double max_vol = std::max(aggregated_buy_volume, aggregated_sell_volume);
        if (max_vol <= 0) max_vol = max_volume_;  // Fallback to global max if aggregated volumes are zero
        if (max_vol <= 0) max_vol = 1.0;  // Ultimate fallback to 1.0 if no volume data

        // Define positions for the aggregated bars - anchored to the left edge of the plot
        double visible_center_price = (plot_limits.Y.Min + plot_limits.Y.Max) / 2.0;
        float bar_height_total = (plot_limits.Y.Max - plot_limits.Y.Min) * 0.25f;  // Use 25% of visible height for each bar
        float bar_spacing = (plot_limits.Y.Max - plot_limits.Y.Min) * 0.05f;      // Small spacing between bars

        // Calculate the leftmost x-coordinate in plot space (this will be our anchor)
        // In horizontal bar charts, volume is on X-axis and price is on Y-axis
        // So we want to anchor to the minimum X value (left edge of chart)
        double left_anchor = 0.0;  // Anchor to the left edge (0)

        // Draw aggregated buy bar (green) extending right from left edge
        if (aggregated_buy_volume > 0) {
          // Position at upper portion of visible range
          double buy_bar_price = visible_center_price - (bar_height_total + bar_spacing) * 0.5;  // Offset to separate from sell bar

          // Calculate the right extent of the bar based on the aggregated volume
          double buy_bar_right_extent = left_anchor + (aggregated_buy_volume / max_vol) * max_vol;

          // Draw the aggregated buy bar extending right from the left edge
          ImDrawList* draw_list = ImPlot::GetPlotDrawList();

          // Convert plot coordinates to pixel coordinates for the bar
          ImVec2 left_edge_px = ImPlot::PlotToPixels(left_anchor, buy_bar_price);
          ImVec2 right_edge_px = ImPlot::PlotToPixels(buy_bar_right_extent, buy_bar_price);

          float bar_top = left_edge_px.y - bar_height_total / 2.0f;
          float bar_bottom = left_edge_px.y + bar_height_total / 2.0f;
          float bar_left = left_edge_px.x;   // Left edge of chart
          float bar_right = right_edge_px.x; // Right extent of the bar

          // Draw the aggregated buy bar
          draw_list->AddRectFilled(ImVec2(bar_left, bar_top), ImVec2(bar_right, bar_bottom),
                                   IM_COL32(26, 204, 26, 179));  // Green with transparency

          // Add border for better visibility
          draw_list->AddRect(ImVec2(bar_left, bar_top), ImVec2(bar_right, bar_bottom),
                             IM_COL32(0, 0, 0, 100), 0.0f, 0, 1.0f);
        }

        // Draw aggregated sell bar (red) extending right from left edge
        if (aggregated_sell_volume > 0) {
          // Position at lower portion of visible range
          double sell_bar_price = visible_center_price + (bar_height_total + bar_spacing) * 0.5;  // Offset to separate from buy bar

          // Calculate the right extent of the bar based on the aggregated volume
          double sell_bar_right_extent = left_anchor + (aggregated_sell_volume / max_vol) * max_vol;

          // Draw the aggregated sell bar extending right from the left edge
          ImDrawList* draw_list = ImPlot::GetPlotDrawList();

          // Convert plot coordinates to pixel coordinates for the bar
          ImVec2 left_edge_px = ImPlot::PlotToPixels(left_anchor, sell_bar_price);
          ImVec2 right_edge_px = ImPlot::PlotToPixels(sell_bar_right_extent, sell_bar_price);

          float bar_top = left_edge_px.y - bar_height_total / 2.0f;
          float bar_bottom = left_edge_px.y + bar_height_total / 2.0f;
          float bar_left = left_edge_px.x;   // Left edge of chart
          float bar_right = right_edge_px.x; // Right extent of the bar

          // Draw the aggregated sell bar
          draw_list->AddRectFilled(ImVec2(bar_left, bar_top), ImVec2(bar_right, bar_bottom),
                                   IM_COL32(204, 26, 26, 179));  // Red with transparency

          // Add border for better visibility
          draw_list->AddRect(ImVec2(bar_left, bar_top), ImVec2(bar_right, bar_bottom),
                             IM_COL32(0, 0, 0, 100), 0.0f, 0, 1.0f);
        }

        // Draw labels for the aggregated bars
        if (aggregated_buy_volume > 0 || aggregated_sell_volume > 0) {
          ImDrawList* draw_list = ImPlot::GetPlotDrawList();

          // Draw text labels
          char buy_label[64];
          char sell_label[64];
          snprintf(buy_label, sizeof(buy_label), "B: %.2f", aggregated_buy_volume);
          snprintf(sell_label, sizeof(sell_label), "S: %.2f", aggregated_sell_volume);

          // Position labels appropriately
          if (aggregated_buy_volume > 0) {
            ImVec2 center_pos = ImPlot::PlotToPixels(((aggregated_buy_volume / max_vol) * max_vol)/2,
                                                    visible_center_price - (bar_height_total + bar_spacing) * 0.5);
            draw_list->AddText(ImVec2(center_pos.x, center_pos.y - 8), IM_COL32(255, 255, 255, 255), buy_label);
          }

          if (aggregated_sell_volume > 0) {
            ImVec2 center_pos = ImPlot::PlotToPixels(((aggregated_sell_volume / max_vol) * max_vol)/2,
                                                    visible_center_price + (bar_height_total + bar_spacing) * 0.5);
            draw_list->AddText(ImVec2(center_pos.x, center_pos.y - 8), IM_COL32(255, 255, 255, 255), sell_label);
          }
        }
        break;
      }
      case ProfileMode::Custom:
      default:
        // Custom Profile Mode: Split bars with buy volume on left (green) and sell volume on right (red)
        // Each bar is centered at zero with buy volume extending left (negative) and sell volume extending right (positive)
        render_split_profile(prices.data(), buy_volumes.data(), sell_volumes.data(),
                             static_cast<int>(prices.size()), bar_height);
        break;
    }

    // POC line - draw differently based on profile mode for consistency
    // For Step Profile mode, POC line is drawn in render_step_profile function
    if (poc_price_ > 0) {
      if (profile_mode_ != ProfileMode::Step) {
        // For other modes, use ImPlot's PlotLine
        double poc_line_x[2] = {-max_volume_, max_volume_};
        double poc_line_y[2] = {poc_price_, poc_price_};
        ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.8f, 0.0f, 1.0f));
        ImPlot::PlotLine("POC", poc_line_x, poc_line_y, 2);
        ImPlot::PopStyleColor();
      }
    }

    // VAH and VAL lines - use local values for Right/Left profiles
    if (local_vah_price > 0) {
      double vah_line_x[2] = {-max_volume_, max_volume_};
      double vah_line_y[2] = {local_vah_price, local_vah_price};
      ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.0f, 1.0f, 1.0f, 0.7f));  // Light blue
      ImPlot::PlotLine("VAH", vah_line_x, vah_line_y, 2);
      ImPlot::PopStyleColor();

      // Add label for VAH line in Right/Left profile modes
      if (profile_mode_ == ProfileMode::Right || profile_mode_ == ProfileMode::Left) {
        ImDrawList* draw_list = ImPlot::GetPlotDrawList();
        ImVec2 plot_size = ImPlot::GetPlotSize();
        ImVec2 plot_pos = ImPlot::GetPlotPos();

        // Position label appropriately based on profile mode to avoid overlap with bars
        ImVec2 vah_pos;
        if (profile_mode_ == ProfileMode::Right) {
          // For Right profile, place label on the left side to avoid overlapping with right-anchored bars
          vah_pos = ImVec2(plot_pos.x + 10, ImPlot::PlotToPixels(0, local_vah_price).y - 10);
        } else {
          // For Left profile, place label on the right side to avoid overlapping with left-anchored bars
          vah_pos = ImVec2(plot_pos.x + plot_size.x - 80, ImPlot::PlotToPixels(0, local_vah_price).y - 10);
        }
        char vah_label[32];
        snprintf(vah_label, sizeof(vah_label), "VAH: %.4f", local_vah_price);
        draw_list->AddText(vah_pos, IM_COL32(0, 255, 255, 255), vah_label);
      }
    }

    if (local_val_price > 0) {
      double val_line_x[2] = {-max_volume_, max_volume_};
      double val_line_y[2] = {local_val_price, local_val_price};
      ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.0f, 1.0f, 1.0f, 0.7f));  // Light blue
      ImPlot::PlotLine("VAL", val_line_x, val_line_y, 2);
      ImPlot::PopStyleColor();

      // Add label for VAL line in Right/Left profile modes
      if (profile_mode_ == ProfileMode::Right || profile_mode_ == ProfileMode::Left) {
        ImDrawList* draw_list = ImPlot::GetPlotDrawList();
        ImVec2 plot_size = ImPlot::GetPlotSize();
        ImVec2 plot_pos = ImPlot::GetPlotPos();

        // Position label appropriately based on profile mode to avoid overlap with bars
        ImVec2 val_pos;
        if (profile_mode_ == ProfileMode::Right) {
          // For Right profile, place label on the left side to avoid overlapping with right-anchored bars
          val_pos = ImVec2(plot_pos.x + 10, ImPlot::PlotToPixels(0, local_val_price).y - 10);
        } else {
          // For Left profile, place label on the right side to avoid overlapping with left-anchored bars
          val_pos = ImVec2(plot_pos.x + plot_size.x - 80, ImPlot::PlotToPixels(0, local_val_price).y - 10);
        }
        char val_label[32];
        snprintf(val_label, sizeof(val_label), "VAL: %.4f", local_val_price);
        draw_list->AddText(val_pos, IM_COL32(0, 255, 255, 255), val_label);
      }
    }

    // Add VWAP line if available
    auto analytics = processor_->getSymbolAnalytics(symbol_id_);
    if (analytics.vwap > 0) {
      double vwap_line_x[2] = {-max_volume_, max_volume_};
      double vwap_line_y[2] = {analytics.vwap, analytics.vwap};
      ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.0f, 0.5f, 1.0f, 1.0f));
      ImPlot::PlotLine("VWAP", vwap_line_x, vwap_line_y, 2);
      ImPlot::PopStyleColor();
    }

    // Add profile anchor markers for Custom Profile mode
    if (profile_mode_ == ProfileMode::Custom && use_custom_time_range_) {
      ImDrawList* draw_list = ImPlot::GetPlotDrawList();

      // Get plot limits to determine the Y range for vertical lines
      ImPlotRect plot_limits = ImPlot::GetPlotLimits();

      // Draw start time vertical line (green)
      if (custom_start_time_ > 0) {
        double start_line_x[2] = {custom_start_time_, custom_start_time_};
        double start_line_y[2] = {plot_limits.Y.Min, plot_limits.Y.Max};

        // Draw the vertical line
        ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.0f, 1.0f, 0.0f, 0.8f)); // Green
        ImPlot::PlotLine("Start Time", start_line_x, start_line_y, 2);
        ImPlot::PopStyleColor();

        // Draw a draggable handle at the top of the line
        ImVec2 handle_pos = ImPlot::PlotToPixels(custom_start_time_, plot_limits.Y.Max - (plot_limits.Y.Max - plot_limits.Y.Min) * 0.1);
        ImVec2 handle_size = ImVec2(10.0f, 20.0f);
        ImVec2 handle_tl = ImVec2(handle_pos.x - handle_size.x/2, handle_pos.y - handle_size.y/2);
        ImVec2 handle_br = ImVec2(handle_pos.x + handle_size.x/2, handle_pos.y + handle_size.y/2);

        // Draw the handle
        draw_list->AddRectFilled(handle_tl, handle_br, IM_COL32(0, 255, 0, 200)); // Green handle
        draw_list->AddRect(handle_tl, handle_br, IM_COL32(255, 255, 255, 255)); // White border

        // Handle dragging for start time
        ImGui::SetCursorScreenPos(handle_tl);
        ImGui::InvisibleButton("start_handle", handle_size);
        if (ImGui::IsItemActive()) {
            start_time_drag_active_ = true;
            double new_time = ImPlot::GetPlotMousePos().x;
            // Constrain to valid range
            if (new_time < custom_end_time_) {
                custom_start_time_ = new_time;
            }
        } else if (start_time_drag_active_ && !ImGui::IsMouseDown(0)) {
            start_time_drag_active_ = false;
        }
      }

      // Draw end time vertical line (red)
      if (custom_end_time_ > 0) {
        double end_line_x[2] = {custom_end_time_, custom_end_time_};
        double end_line_y[2] = {plot_limits.Y.Min, plot_limits.Y.Max};

        // Draw the vertical line
        ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.0f, 0.0f, 0.8f)); // Red
        ImPlot::PlotLine("End Time", end_line_x, end_line_y, 2);
        ImPlot::PopStyleColor();

        // Draw a draggable handle at the top of the line
        ImVec2 handle_pos = ImPlot::PlotToPixels(custom_end_time_, plot_limits.Y.Max - (plot_limits.Y.Max - plot_limits.Y.Min) * 0.1);
        ImVec2 handle_size = ImVec2(10.0f, 20.0f);
        ImVec2 handle_tl = ImVec2(handle_pos.x - handle_size.x/2, handle_pos.y - handle_size.y/2);
        ImVec2 handle_br = ImVec2(handle_pos.x + handle_size.x/2, handle_pos.y + handle_size.y/2);

        // Draw the handle
        draw_list->AddRectFilled(handle_tl, handle_br, IM_COL32(255, 0, 0, 200)); // Red handle
        draw_list->AddRect(handle_tl, handle_br, IM_COL32(255, 255, 255, 255)); // White border

        // Handle dragging for end time
        ImGui::SetCursorScreenPos(handle_tl);
        ImGui::InvisibleButton("end_handle", handle_size);
        if (ImGui::IsItemActive()) {
            end_time_drag_active_ = true;
            double new_time = ImPlot::GetPlotMousePos().x;
            // Constrain to valid range
            if (new_time > custom_start_time_) {
                custom_end_time_ = new_time;
            }
        } else if (end_time_drag_active_ && !ImGui::IsMouseDown(0)) {
            end_time_drag_active_ = false;
        }
      }
    }

    ImPlot::EndPlot();
  }
}

void VolumeProfilePanel::calculate_value_area() {
  if (volume_profile_.empty()) {
    vah_price_ = 0.0;
    val_price_ = 0.0;
    return;
  }

  // Calculate total volume in the profile for all modes
  double total_volume = 0.0;
  for (const auto& level : volume_profile_) {
    total_volume += level.total_volume;
  }

  if (total_volume <= 0) {
    vah_price_ = 0.0;
    val_price_ = 0.0;
    return;
  }

  // Target volume for value area (based on profile_settings_.vaPercent % of total volume)
  double target_volume = (static_cast<double>(profile_settings_.vaPercent) / 100.0) * total_volume;

  // Find the POC index
  size_t poc_index = 0;
  double max_total_volume = 0.0;
  for (size_t i = 0; i < volume_profile_.size(); ++i) {
    double total = volume_profile_[i].buy_volume + volume_profile_[i].sell_volume;
    if (total > max_total_volume) {
      max_total_volume = total;
      poc_index = i;
    }
  }

  // Expand from POC outward to capture the required volume
  size_t start_idx = poc_index;
  size_t end_idx = poc_index;
  double current_volume = volume_profile_[poc_index].total_volume;

  // Expand upward (higher prices) and downward (lower prices) alternately
  // until we reach the target volume
  while (current_volume < target_volume) {
    // Decide whether to expand up or down
    bool expand_up = false;
    bool expand_down = false;

    // Check if we can expand in each direction
    if (start_idx > 0) expand_down = true;
    if (end_idx < volume_profile_.size() - 1) expand_up = true;

    // If we can't expand in either direction, break
    if (!expand_up && !expand_down) break;

    // If we can only expand in one direction, do that
    if (!expand_up && expand_down) {
      start_idx--;
      current_volume += volume_profile_[start_idx].total_volume;
    } else if (expand_up && !expand_down) {
      end_idx++;
      current_volume += volume_profile_[end_idx].total_volume;
    } else {
      // We can expand in both directions - choose the direction with higher volume
      double vol_up = volume_profile_[end_idx + 1].total_volume;
      double vol_down = volume_profile_[start_idx - 1].total_volume;

      if (vol_up >= vol_down) {
        end_idx++;
        current_volume += volume_profile_[end_idx].total_volume;
      } else {
        start_idx--;
        current_volume += volume_profile_[start_idx].total_volume;
      }
    }

    // If we've captured enough volume, break
    if (current_volume >= target_volume) break;
  }

  // Set the VAH and VAL prices
  vah_price_ = volume_profile_[end_idx].price;
  val_price_ = volume_profile_[start_idx].price;
}

void VolumeProfilePanel::render_step_profile(const double* xs, const double* ys,
                                             const double* neg_ys, int count, double height) {
  if (count <= 0) return;
  ImDrawList* draw_list = ImPlot::GetPlotDrawList();
  const ImU32 col_pos = IM_COL32(0, 255, 0, 170);    // Green for positive
  const ImU32 col_neg = IM_COL32(255, 0, 0, 170);    // Red for negative
  const ImU32 col_poc = IM_COL32(255, 255, 0, 255);  // Yellow for POC (Point of Control)

  // Find the POC (Point of Control) - the price level with highest total volume
  int poc_index = -1;
  double max_total_volume = 0.0;

  for (int i = 0; i < count; ++i) {
    double total_volume = std::abs(ys[i]) + std::abs(neg_ys[i]);
    if (total_volume > max_total_volume) {
      max_total_volume = total_volume;
      poc_index = i;
    }
  }

  for (int i = 0; i < count; ++i) {
    // Determine if this is the POC bar
    bool is_poc_bar = (i == poc_index && max_total_volume > 0);
    ImU32 current_col_pos = is_poc_bar ? col_poc : col_pos;
    ImU32 current_col_neg = is_poc_bar ? col_poc : col_neg;

    if (ys[i] != 0) {
      ImVec2 p1 = ImPlot::PlotToPixels(0, xs[i]);  // x-axis is volume, y-axis is price
      ImVec2 p2 = ImPlot::PlotToPixels(ys[i], xs[i]);

      // Draw step-style bar
      ImVec2 bar_tl = ImVec2(std::min(p1.x, p2.x), p1.y - height / 2);
      ImVec2 bar_br = ImVec2(std::max(p1.x, p2.x), p1.y + height / 2);

      draw_list->AddRectFilled(bar_tl, bar_br, current_col_pos);
    }

    if (neg_ys[i] != 0) {
      ImVec2 p1 = ImPlot::PlotToPixels(0, xs[i]);
      ImVec2 p2 = ImPlot::PlotToPixels(neg_ys[i], xs[i]);

      // Draw step-style bar for negative values
      ImVec2 bar_tl = ImVec2(std::min(p1.x, p2.x), p1.y - height / 2);
      ImVec2 bar_br = ImVec2(std::max(p1.x, p2.x), p1.y + height / 2);

      draw_list->AddRectFilled(bar_tl, bar_br, current_col_neg);
    }
  }

  // Draw horizontal yellow POC line at the price level with highest volume
  // Use the locally calculated POC for consistency with highlighted bar
  if (poc_index >= 0 && max_total_volume > 0) {
    // Calculate appropriate min/max x values for the line
    // Find the actual min/max volumes in the dataset to determine line length
    double min_vol = 0.0, max_vol = 0.0;
    for (int i = 0; i < count; ++i) {
      min_vol = std::min(min_vol, std::min(ys[i], neg_ys[i]));
      max_vol = std::max(max_vol, std::max(ys[i], neg_ys[i]));
    }

    // Ensure we have valid min/max values for the line
    if (max_vol <= min_vol) {
      max_vol = max_volume_;
      min_vol = -max_volume_;
    }

    // Use the locally calculated POC price (xs[poc_index]) instead of global poc_price_
    ImVec2 poc_start = ImPlot::PlotToPixels(min_vol, xs[poc_index]);
    ImVec2 poc_end = ImPlot::PlotToPixels(max_vol, xs[poc_index]);

    // Draw the horizontal POC line - make it more prominent with consistent styling
    // Use the same color as in other profile modes for consistency
    draw_list->AddLine(poc_start, poc_end, IM_COL32(255, 204, 0, 255), 2.0f);  // Yellow with consistent thickness

    // Also update the global POC price to reflect the current calculation for display purposes
    poc_price_ = xs[poc_index];
  }

  // Ensure the POC line is always visible and properly calculated for Step Profile mode
  // This ensures that even if the local calculation didn't find a POC, we use the global one
  if (poc_index < 0 && poc_price_ > 0) {
    // Calculate min/max x values for the line based on global data
    double min_vol = 0.0, max_vol = 0.0;
    for (int i = 0; i < count; ++i) {
      min_vol = std::min(min_vol, std::min(ys[i], neg_ys[i]));
      max_vol = std::max(max_vol, std::max(ys[i], neg_ys[i]));
    }

    // Ensure we have valid min/max values for the line
    if (max_vol <= min_vol) {
      max_vol = max_volume_;
      min_vol = -max_volume_;
    }

    // Draw POC line using global POC price
    ImVec2 poc_start = ImPlot::PlotToPixels(min_vol, poc_price_);
    ImVec2 poc_end = ImPlot::PlotToPixels(max_vol, poc_price_);

    // Draw the horizontal POC line - make it more prominent with consistent styling
    draw_list->AddLine(poc_start, poc_end, IM_COL32(255, 204, 0, 255), 2.0f);  // Yellow with consistent thickness
  }
}

// Method to render mini histogram overlays on candlestick charts
void VolumeProfilePanel::render_mini_histograms_on_candles(
    ImDrawList* draw_list, const std::vector<RenderEngine::OHLCVCandle>& candles,
    const std::vector<double>& x_coords, const std::vector<double>& y_coords_high,
    const std::vector<double>& y_coords_low) {
  if (candles.empty() || x_coords.size() != candles.size() ||
      y_coords_high.size() != candles.size() || y_coords_low.size() != candles.size()) {
    return;
  }

  // Get theme colors for consistent styling
  const auto& colors = ThemeManager::getInstance().getColors();

  // Iterate through each candle to draw mini volume profile histogram
  for (size_t i = 0; i < candles.size(); ++i) {
    const auto& candle = candles[i];

    // Skip if candle has no volume
    if (candle.volume <= 0) continue;

    // Get the candle's price range (high - low)
    double price_range = candle.high - candle.low;
    if (price_range <= 0) continue;

    // Determine number of price buckets for this candle's range
    int num_buckets = 8;  // Fixed number of buckets for mini histogram
    double bucket_size = price_range / num_buckets;

    // Get recent trades for this symbol to populate the histogram
    auto analytics = processor_->getSymbolAnalytics(symbol_id_);
    const auto& trades = analytics.recent_trades;

    // Create temporary buckets for this candle's price range
    std::vector<double> bucket_volumes(num_buckets, 0.0);
    std::vector<int> bucket_counts(num_buckets, 0);

    // Aggregate trades into buckets based on price within this candle's range
    for (const auto& trade : trades) {
      if (trade.price >= candle.low && trade.price <= candle.high) {
        int bucket_idx = static_cast<int>((trade.price - candle.low) / bucket_size);
        // Ensure we don't exceed bounds
        bucket_idx = std::max(0, std::min(bucket_idx, num_buckets - 1));

        bucket_volumes[bucket_idx] += trade.size;
        bucket_counts[bucket_idx]++;
      }
    }

    // Find max volume in this candle's histogram for scaling
    double max_vol_in_candle = 0.0;
    int poc_bucket_idx = 0;  // Index of the bucket with highest volume (POC)

    for (int j = 0; j < num_buckets; ++j) {
      if (bucket_volumes[j] > max_vol_in_candle) {
        max_vol_in_candle = bucket_volumes[j];
        poc_bucket_idx = j;
      }
    }

    if (max_vol_in_candle <= 0) continue;

    // Calculate screen coordinates for the mini histogram
    float x_center = static_cast<float>(x_coords[i]);
    float y_high = static_cast<float>(y_coords_high[i]);
    float y_low = static_cast<float>(y_coords_low[i]);

    // Calculate height of each bucket in screen coordinates
    float total_height = y_low - y_high;  // Height of the candle in screen space
    float bucket_height = total_height / num_buckets;

    // Draw mini histogram inside the candle
    for (int j = 0; j < num_buckets; ++j) {
      if (bucket_volumes[j] > 0) {
        // Calculate the fill percentage of this bucket
        float fill_percentage =
            static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

        // Calculate the top-left and bottom-right coordinates for this bucket
        float y_top = y_high + j * bucket_height;
        float y_bottom = y_high + (j + 1) * bucket_height;

        // Calculate width of the bar based on fill percentage
        float bar_width = (y_bottom - y_top) * 0.8f;  // Use 80% of height as width for visibility
        float filled_width = bar_width * fill_percentage;

        // Calculate the x positions for the bar
        float x_left = x_center - bar_width / 2.0f;
        float x_right = x_left + filled_width;

        // Choose color based on whether this is the POC bucket or not
        ImU32 color;
        if (j == poc_bucket_idx) {
          // Highlight POC bucket with yellow
          color = IM_COL32(255, 255, 0, 220);  // Brighter yellow for POC
        } else {
          // Use gradient colors based on volume intensity and position in the candle
          float volume_ratio = static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);
          float position_ratio = static_cast<float>(j) / static_cast<float>(num_buckets - 1);

          // Create a color gradient based on volume intensity and position
          if (position_ratio < 0.5f) {
            // Lower half - red for selling pressure, with intensity based on volume
            color = IM_COL32(255, static_cast<int>(100 * volume_ratio), static_cast<int>(100 * volume_ratio), 150);
          } else {
            // Upper half - green for buying pressure, with intensity based on volume
            color = IM_COL32(static_cast<int>(100 * volume_ratio), 255, static_cast<int>(100 * volume_ratio), 150);
          }
        }

        // Draw the mini histogram bar with a slight outline for better visibility
        draw_list->AddRectFilled(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), color);

        // Add a subtle border to make individual bars more distinguishable
        draw_list->AddRect(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), IM_COL32(0, 0, 0, 50));
      }
    }

    // Draw POC (Point of Control) line - horizontal yellow line at the price level with highest
    // volume Calculate the y-coordinate for the POC line
    float poc_y = y_high + (poc_bucket_idx + 0.5f) * bucket_height;  // Center of the POC bucket

    // Draw horizontal yellow line across the candle width
    float poc_line_half_width = (y_low - y_high) * 0.4f;  // Same width as candle
    float poc_x_left = x_center - poc_line_half_width;
    float poc_x_right = x_center + poc_line_half_width;

    // Draw the POC line as a horizontal yellow line
    draw_list->AddLine(ImVec2(poc_x_left, poc_y), ImVec2(poc_x_right, poc_y),
                       IM_COL32(255, 255, 0, 255),  // Yellow color for POC
                       2.0f                         // Line thickness
    );
  }
}

void VolumeProfilePanel::render_step_profile_histograms(ImDrawList* draw_list,
                                                       const std::vector<RenderEngine::OHLCVCandle>& candles,
                                                       const std::vector<double>& x_coords,
                                                       const std::vector<double>& y_coords_high,
                                                       const std::vector<double>& y_coords_low,
                                                       bool show_poc_line,
                                                       int num_buckets) {
  if (candles.empty() || x_coords.size() != candles.size() ||
      y_coords_high.size() != candles.size() || y_coords_low.size() != candles.size()) {
    return;
  }

  // Get theme colors for consistent styling
  const auto& colors = ThemeManager::getInstance().getColors();

  // Get recent trades for this symbol to populate the histograms
  // Only fetch once for all candles to improve efficiency
  auto analytics = processor_->getSymbolAnalytics(symbol_id_);
  const auto& trades = analytics.recent_trades;

  // Iterate through each candle to draw step profile histogram
  for (size_t i = 0; i < candles.size(); ++i) {
    const auto& candle = candles[i];

    // Skip if candle has no volume
    if (candle.volume <= 0) continue;

    // Get the candle's price range (high - low)
    double price_range = candle.high - candle.low;
    if (price_range <= 0) continue;

    // Calculate bucket size for this candle's range
    double bucket_size = price_range / num_buckets;

    // Create temporary buckets for this candle's price range
    std::vector<double> bucket_volumes(num_buckets, 0.0);
    std::vector<int> bucket_counts(num_buckets, 0);

    // Aggregate trades into buckets based on price within this candle's range
    for (const auto& trade : trades) {
      if (trade.price >= candle.low && trade.price <= candle.high) {
        int bucket_idx = static_cast<int>((trade.price - candle.low) / bucket_size);
        // Ensure we don't exceed bounds
        bucket_idx = std::max(0, std::min(bucket_idx, num_buckets - 1));

        bucket_volumes[bucket_idx] += trade.size;
        bucket_counts[bucket_idx]++;
      }
    }

    // Find max volume in this candle's histogram for scaling
    double max_vol_in_candle = 0.0;
    int poc_bucket_idx = 0;  // Index of the bucket with highest volume (POC)

    for (int j = 0; j < num_buckets; ++j) {
      if (bucket_volumes[j] > max_vol_in_candle) {
        max_vol_in_candle = bucket_volumes[j];
        poc_bucket_idx = j;
      }
    }

    if (max_vol_in_candle <= 0) continue;

    // Calculate screen coordinates for the step profile histogram
    float x_center = static_cast<float>(x_coords[i]);
    float y_high = static_cast<float>(y_coords_high[i]);
    float y_low = static_cast<float>(y_coords_low[i]);

    // Calculate height of each bucket in screen coordinates
    float total_height = y_low - y_high;  // Height of the candle in screen space
    float bucket_height = total_height / num_buckets;

    // Draw step profile histogram inside the candle
    for (int j = 0; j < num_buckets; ++j) {
      if (bucket_volumes[j] > 0) {
        // Calculate the fill percentage of this bucket
        float fill_percentage =
            static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

        // Calculate the top-left and bottom-right coordinates for this bucket
        float y_top = y_high + j * bucket_height;
        float y_bottom = y_high + (j + 1) * bucket_height;

        // Calculate width of the bar based on fill percentage
        float bar_width = (y_bottom - y_top) * 0.8f;  // Use 80% of height as width for visibility
        float filled_width = bar_width * fill_percentage;

        // Calculate the x positions for the bar
        float x_left = x_center - bar_width / 2.0f;
        float x_right = x_left + filled_width;

        // Choose color based on whether this is the POC bucket or not
        ImU32 color;
        if (j == poc_bucket_idx) {
          // Highlight POC bucket with yellow
          color = IM_COL32(255, 255, 0, 220);  // Brighter yellow for POC
        } else {
          // Use gradient colors based on volume intensity and position in the candle
          float volume_ratio = static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);
          float position_ratio = static_cast<float>(j) / static_cast<float>(num_buckets - 1);

          // Create a color gradient based on volume intensity and position
          if (position_ratio < 0.5f) {
            // Lower half - red for selling pressure, with intensity based on volume
            color = IM_COL32(255, static_cast<int>(100 * volume_ratio), static_cast<int>(100 * volume_ratio), 150);
          } else {
            // Upper half - green for buying pressure, with intensity based on volume
            color = IM_COL32(static_cast<int>(100 * volume_ratio), 255, static_cast<int>(100 * volume_ratio), 150);
          }
        }

        // Draw the step profile histogram bar with a slight outline for better visibility
        draw_list->AddRectFilled(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), color);

        // Add a subtle border to make individual bars more distinguishable
        draw_list->AddRect(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), IM_COL32(0, 0, 0, 50));
      }
    }

    // Optionally draw POC (Point of Control) line - horizontal yellow line at the price level with highest volume
    if (show_poc_line) {
      // Calculate the y-coordinate for the POC line
      float poc_y = y_high + (poc_bucket_idx + 0.5f) * bucket_height;  // Center of the POC bucket

      // Draw horizontal yellow line across the candle width
      float poc_line_half_width = (y_low - y_high) * 0.4f;  // Same width as candle
      float poc_x_left = x_center - poc_line_half_width;
      float poc_x_right = x_center + poc_line_half_width;

      // Draw the POC line as a horizontal yellow line
      draw_list->AddLine(ImVec2(poc_x_left, poc_y), ImVec2(poc_x_right, poc_y),
                         IM_COL32(255, 255, 0, 255),  // Yellow color for POC
                         2.0f                         // Line thickness
      );
    }
  }
}

void VolumeProfilePanel::render_candle_volume_distribution(ImDrawList* draw_list,
                                                        const std::vector<RenderEngine::OHLCVCandle>& candles,
                                                        const std::vector<double>& x_coords,
                                                        const std::vector<double>& y_coords_high,
                                                        const std::vector<double>& y_coords_low,
                                                        bool show_poc_line,
                                                        int num_buckets) {
  // This method implements the Step Profile rendering: draw mini histogram overlay
  // on each candlestick bar showing volume distribution for that bar's price range
  render_step_profile_histograms(draw_list, candles, x_coords, y_coords_high, y_coords_low,
                                show_poc_line, num_buckets);
}

void VolumeProfilePanel::render_step_profile_on_candles(ImDrawList* draw_list,
                                                      const std::vector<RenderEngine::OHLCVCandle>& candles,
                                                      const std::vector<double>& x_coords,
                                                      const std::vector<double>& y_coords_high,
                                                      const std::vector<double>& y_coords_low,
                                                      bool show_poc_line,
                                                      int num_buckets_per_candle) {
  // Enhanced Step Profile rendering: draw mini histogram overlay on each candlestick bar
  // showing volume distribution for that bar's price range
  if (candles.empty() || x_coords.size() != candles.size() ||
      y_coords_high.size() != candles.size() || y_coords_low.size() != candles.size()) {
    return;
  }

  // Get recent trades for this symbol to populate the histograms
  auto analytics = processor_->getSymbolAnalytics(symbol_id_);
  const auto& trades = analytics.recent_trades;

  // Iterate through each candle to draw step profile histogram
  for (size_t i = 0; i < candles.size(); ++i) {
    const auto& candle = candles[i];

    // Skip if candle has no meaningful price range
    if (candle.high <= candle.low || candle.volume <= 0) continue;

    // Get the candle's price range (high - low)
    double price_range = candle.high - candle.low;
    if (price_range <= 0) continue;

    // Calculate bucket size for this candle's range
    double bucket_size = price_range / num_buckets_per_candle;

    // Create temporary buckets for this candle's price range
    std::vector<double> bucket_volumes(num_buckets_per_candle, 0.0);
    std::vector<int> bucket_counts(num_buckets_per_candle, 0);

    // Aggregate trades into buckets based on price within this candle's range
    for (const auto& trade : trades) {
      // Only consider trades within this candle's price range
      if (trade.price >= candle.low && trade.price <= candle.high) {
        int bucket_idx = static_cast<int>((trade.price - candle.low) / bucket_size);
        // Ensure we don't exceed bounds
        bucket_idx = std::max(0, std::min(bucket_idx, num_buckets_per_candle - 1));

        bucket_volumes[bucket_idx] += trade.size;
        bucket_counts[bucket_idx]++;
      }
    }

    // Find max volume in this candle's histogram for scaling
    double max_vol_in_candle = 0.0;
    int poc_bucket_idx = -1;  // Index of the bucket with highest volume (POC)

    for (int j = 0; j < num_buckets_per_candle; ++j) {
      if (bucket_volumes[j] > max_vol_in_candle) {
        max_vol_in_candle = bucket_volumes[j];
        poc_bucket_idx = j;
      }
    }

    // If no trades found in this candle's range, skip
    if (max_vol_in_candle <= 0 || poc_bucket_idx < 0) continue;

    // Calculate screen coordinates for the step profile histogram
    float x_center = static_cast<float>(x_coords[i]);
    float y_high = static_cast<float>(y_coords_high[i]);
    float y_low = static_cast<float>(y_coords_low[i]);

    // Calculate height of each bucket in screen coordinates
    float total_height = y_low - y_high;  // Height of the candle in screen space
    float bucket_height = total_height / num_buckets_per_candle;

    // Draw step profile histogram inside the candle
    for (int j = 0; j < num_buckets_per_candle; ++j) {
      if (bucket_volumes[j] > 0) {
        // Calculate the fill percentage of this bucket
        float fill_percentage =
            static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

        // Calculate the top-left and bottom-right coordinates for this bucket
        float y_top = y_high + j * bucket_height;
        float y_bottom = y_high + (j + 1) * bucket_height;

        // Calculate width of the bar based on fill percentage
        float bar_width = (y_bottom - y_top) * 0.6f;  // Use 60% of height as width for visibility
        float filled_width = bar_width * fill_percentage;

        // Calculate the x positions for the bar
        float x_left = x_center - bar_width / 2.0f;
        float x_right = x_left + filled_width;

        // Choose color based on whether this is the POC bucket or not
        ImU32 color;
        if (j == poc_bucket_idx) {
          // Highlight POC bucket with bright yellow
          color = IM_COL32(255, 255, 0, 240);  // Bright yellow for POC with high opacity
        } else {
          // Use gradient colors based on volume intensity and position in the candle
          float volume_ratio = static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

          // Determine if this bucket is in the upper or lower half of the candle
          float position_ratio = static_cast<float>(j) / static_cast<float>(num_buckets_per_candle - 1);

          // Create a color gradient based on volume intensity and position
          if (position_ratio < 0.5f) {
            // Lower half - red for selling pressure, with intensity based on volume
            int red_intensity = static_cast<int>(255 * volume_ratio);
            color = IM_COL32(red_intensity, static_cast<int>(50 * volume_ratio),
                            static_cast<int>(50 * volume_ratio), 180);
          } else {
            // Upper half - green for buying pressure, with intensity based on volume
            int green_intensity = static_cast<int>(255 * volume_ratio);
            color = IM_COL32(static_cast<int>(50 * volume_ratio), green_intensity,
                            static_cast<int>(50 * volume_ratio), 180);
          }
        }

        // Draw the step profile histogram bar with a slight outline for better visibility
        draw_list->AddRectFilled(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), color);

        // Add a subtle border to make individual bars more distinguishable
        draw_list->AddRect(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), IM_COL32(0, 0, 0, 80), 0.0f, 0, 1.0f);
      }
    }

    // Optionally draw POC (Point of Control) line - horizontal yellow line at the price level with highest volume
    if (show_poc_line && poc_bucket_idx >= 0) {
      // Calculate the y-coordinate for the POC line
      float poc_y = y_high + (poc_bucket_idx + 0.5f) * bucket_height;  // Center of the POC bucket

      // Draw horizontal yellow line across the candle width
      float poc_line_half_width = (total_height) * 0.3f;  // Reduced width to avoid overlapping with candle wicks
      float poc_x_left = x_center - poc_line_half_width;
      float poc_x_right = x_center + poc_line_half_width;

      // Draw the POC line as a horizontal yellow line
      draw_list->AddLine(ImVec2(poc_x_left, poc_y), ImVec2(poc_x_right, poc_y),
                         IM_COL32(255, 255, 0, 255),  // Bright yellow color for POC
                         2.0f);                        // Line thickness
    }
  }
}

void VolumeProfilePanel::render_mini_histograms_direct(ImDrawList* draw_list,
                                                     const std::vector<RenderEngine::OHLCVCandle>& candles,
                                                     const std::vector<double>& x_coords,
                                                     const std::vector<double>& y_coords_high,
                                                     const std::vector<double>& y_coords_low,
                                                     const std::vector<RenderEngine::TradeData>& trades,
                                                     bool show_poc_line,
                                                     int num_buckets) {
  if (candles.empty() || x_coords.size() != candles.size() ||
      y_coords_high.size() != candles.size() || y_coords_low.size() != candles.size()) {
    return;
  }

  // Get theme colors for consistent styling
  const auto& colors = ThemeManager::getInstance().getColors();

  // Iterate through each candle to draw step profile histogram
  for (size_t i = 0; i < candles.size(); ++i) {
    const auto& candle = candles[i];

    // Skip if candle has no volume
    if (candle.volume <= 0) continue;

    // Get the candle's price range (high - low)
    double price_range = candle.high - candle.low;
    if (price_range <= 0) continue;

    // Calculate bucket size for this candle's range
    double bucket_size = price_range / num_buckets;

    // Create temporary buckets for this candle's price range
    std::vector<double> bucket_volumes(num_buckets, 0.0);
    std::vector<int> bucket_counts(num_buckets, 0);

    // Aggregate trades into buckets based on price within this candle's range
    for (const auto& trade : trades) {
      if (trade.price >= candle.low && trade.price <= candle.high) {
        int bucket_idx = static_cast<int>((trade.price - candle.low) / bucket_size);
        // Ensure we don't exceed bounds
        bucket_idx = std::max(0, std::min(bucket_idx, num_buckets - 1));

        bucket_volumes[bucket_idx] += trade.size;
        bucket_counts[bucket_idx]++;
      }
    }

    // Find max volume in this candle's histogram for scaling
    double max_vol_in_candle = 0.0;
    int poc_bucket_idx = 0;  // Index of the bucket with highest volume (POC)

    for (int j = 0; j < num_buckets; ++j) {
      if (bucket_volumes[j] > max_vol_in_candle) {
        max_vol_in_candle = bucket_volumes[j];
        poc_bucket_idx = j;
      }
    }

    if (max_vol_in_candle <= 0) continue;

    // Calculate screen coordinates for the step profile histogram
    float x_center = static_cast<float>(x_coords[i]);
    float y_high = static_cast<float>(y_coords_high[i]);
    float y_low = static_cast<float>(y_coords_low[i]);

    // Calculate height of each bucket in screen coordinates
    float total_height = y_low - y_high;  // Height of the candle in screen space
    float bucket_height = total_height / num_buckets;

    // Draw step profile histogram inside the candle
    for (int j = 0; j < num_buckets; ++j) {
      if (bucket_volumes[j] > 0) {
        // Calculate the fill percentage of this bucket
        float fill_percentage =
            static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

        // Calculate the top-left and bottom-right coordinates for this bucket
        float y_top = y_high + j * bucket_height;
        float y_bottom = y_high + (j + 1) * bucket_height;

        // Calculate width of the bar based on fill percentage
        float bar_width = (y_bottom - y_top) * 0.8f;  // Use 80% of height as width for visibility
        float filled_width = bar_width * fill_percentage;

        // Calculate the x positions for the bar
        float x_left = x_center - bar_width / 2.0f;
        float x_right = x_left + filled_width;

        // Choose color based on whether this is the POC bucket or not
        ImU32 color;
        if (j == poc_bucket_idx) {
          // Highlight POC bucket with yellow
          color = IM_COL32(255, 255, 0, 220);  // Brighter yellow for POC
        } else {
          // Use gradient colors based on volume intensity and position in the candle
          float volume_ratio = static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);
          float position_ratio = static_cast<float>(j) / static_cast<float>(num_buckets - 1);

          // Create a color gradient based on volume intensity and position
          if (position_ratio < 0.5f) {
            // Lower half - red for selling pressure, with intensity based on volume
            color = IM_COL32(255, static_cast<int>(100 * volume_ratio), static_cast<int>(100 * volume_ratio), 150);
          } else {
            // Upper half - green for buying pressure, with intensity based on volume
            color = IM_COL32(static_cast<int>(100 * volume_ratio), 255, static_cast<int>(100 * volume_ratio), 150);
          }
        }

        // Draw the step profile histogram bar with a slight outline for better visibility
        draw_list->AddRectFilled(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), color);

        // Add a subtle border to make individual bars more distinguishable
        draw_list->AddRect(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), IM_COL32(0, 0, 0, 50));
      }
    }

    // Optionally draw POC (Point of Control) line - horizontal yellow line at the price level with highest volume
    if (show_poc_line) {
      // Calculate the y-coordinate for the POC line
      float poc_y = y_high + (poc_bucket_idx + 0.5f) * bucket_height;  // Center of the POC bucket

      // Draw horizontal yellow line across the candle width
      float poc_line_half_width = (y_low - y_high) * 0.4f;  // Same width as candle
      float poc_x_left = x_center - poc_line_half_width;
      float poc_x_right = x_center + poc_line_half_width;

      // Draw the POC line as a horizontal yellow line
      draw_list->AddLine(ImVec2(poc_x_left, poc_y), ImVec2(poc_x_right, poc_y),
                         IM_COL32(255, 255, 0, 255),  // Yellow color for POC
                         2.0f                         // Line thickness
      );
    }
  }
}

// Static method to render step profile directly on candles with improved visualization
void VolumeProfilePanel::render_step_profile_on_candles_static(
    ImDrawList* draw_list,
    const std::vector<RenderEngine::OHLCVCandle>& candles,
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords_high,
    const std::vector<double>& y_coords_low,
    const std::vector<RenderEngine::TradeData>& trades,
    bool show_poc_line,
    int num_buckets_per_candle) {

  if (candles.empty() || x_coords.size() != candles.size() ||
      y_coords_high.size() != candles.size() || y_coords_low.size() != candles.size()) {
    return;
  }

  // Iterate through each candle to draw step profile histogram
  for (size_t i = 0; i < candles.size(); ++i) {
    const auto& candle = candles[i];

    // Skip if candle has no meaningful price range
    if (candle.high <= candle.low || candle.volume <= 0) continue;

    // Get the candle's price range (high - low)
    double price_range = candle.high - candle.low;
    if (price_range <= 0) continue;

    // Calculate bucket size for this candle's range
    double bucket_size = price_range / num_buckets_per_candle;

    // Create temporary buckets for this candle's price range
    std::vector<double> bucket_volumes(num_buckets_per_candle, 0.0);
    std::vector<int> bucket_counts(num_buckets_per_candle, 0);

    // Aggregate trades into buckets based on price within this candle's range
    for (const auto& trade : trades) {
      // Only consider trades within this candle's price range
      if (trade.price >= candle.low && trade.price <= candle.high) {
        int bucket_idx = static_cast<int>((trade.price - candle.low) / bucket_size);
        // Ensure we don't exceed bounds
        bucket_idx = std::max(0, std::min(bucket_idx, num_buckets_per_candle - 1));

        bucket_volumes[bucket_idx] += trade.size;
        bucket_counts[bucket_idx]++;
      }
    }

    // Find max volume in this candle's histogram for scaling
    double max_vol_in_candle = 0.0;
    int poc_bucket_idx = -1;  // Index of the bucket with highest volume (POC)

    for (int j = 0; j < num_buckets_per_candle; ++j) {
      if (bucket_volumes[j] > max_vol_in_candle) {
        max_vol_in_candle = bucket_volumes[j];
        poc_bucket_idx = j;
      }
    }

    // If no trades found in this candle's range, skip
    if (max_vol_in_candle <= 0 || poc_bucket_idx < 0) continue;

    // Calculate screen coordinates for the step profile histogram
    float x_center = static_cast<float>(x_coords[i]);
    float y_high = static_cast<float>(y_coords_high[i]);
    float y_low = static_cast<float>(y_coords_low[i]);

    // Calculate height of each bucket in screen coordinates
    float total_height = y_low - y_high;  // Height of the candle in screen space
    float bucket_height = total_height / num_buckets_per_candle;

    // Draw step profile histogram inside the candle
    for (int j = 0; j < num_buckets_per_candle; ++j) {
      if (bucket_volumes[j] > 0) {
        // Calculate the fill percentage of this bucket
        float fill_percentage =
            static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

        // Calculate the top-left and bottom-right coordinates for this bucket
        float y_top = y_high + j * bucket_height;
        float y_bottom = y_high + (j + 1) * bucket_height;

        // Calculate width of the bar based on fill percentage
        float bar_width = (y_bottom - y_top) * 0.6f;  // Use 60% of height as width for visibility
        float filled_width = bar_width * fill_percentage;

        // Calculate the x positions for the bar
        float x_left = x_center - bar_width / 2.0f;
        float x_right = x_left + filled_width;

        // Choose color based on whether this is the POC bucket or not
        ImU32 color;
        if (j == poc_bucket_idx) {
          // Highlight POC bucket with bright yellow
          color = IM_COL32(255, 255, 0, 240);  // Bright yellow for POC with high opacity
        } else {
          // Use gradient colors based on volume intensity and position in the candle
          float volume_ratio = static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

          // Determine if this bucket is in the upper or lower half of the candle
          float position_ratio = static_cast<float>(j) / static_cast<float>(num_buckets_per_candle - 1);

          // Create a color gradient based on volume intensity and position
          if (position_ratio < 0.5f) {
            // Lower half - red for selling pressure, with intensity based on volume
            int red_intensity = static_cast<int>(255 * volume_ratio);
            color = IM_COL32(red_intensity, static_cast<int>(50 * volume_ratio),
                            static_cast<int>(50 * volume_ratio), 180);
          } else {
            // Upper half - green for buying pressure, with intensity based on volume
            int green_intensity = static_cast<int>(255 * volume_ratio);
            color = IM_COL32(static_cast<int>(50 * volume_ratio), green_intensity,
                            static_cast<int>(50 * volume_ratio), 180);
          }
        }

        // Draw the step profile histogram bar with a slight outline for better visibility
        draw_list->AddRectFilled(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), color);

        // Add a subtle border to make individual bars more distinguishable
        draw_list->AddRect(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), IM_COL32(0, 0, 0, 80), 0.0f, 0, 1.0f);
      }
    }

    // Optionally draw POC (Point of Control) line - horizontal yellow line at the price level with highest volume
    if (show_poc_line && poc_bucket_idx >= 0) {
      // Calculate the y-coordinate for the POC line
      float poc_y = y_high + (poc_bucket_idx + 0.5f) * bucket_height;  // Center of the POC bucket

      // Draw horizontal yellow line across the candle width
      float poc_line_half_width = (total_height) * 0.3f;  // Reduced width to avoid overlapping with candle wicks
      float poc_x_left = x_center - poc_line_half_width;
      float poc_x_right = x_center + poc_line_half_width;

      // Draw the POC line as a horizontal yellow line
      draw_list->AddLine(ImVec2(poc_x_left, poc_y), ImVec2(poc_x_right, poc_y),
                         IM_COL32(255, 255, 0, 255),  // Bright yellow color for POC
                         2.0f);                        // Line thickness
    }
  }
}

void VolumeProfilePanel::render_split_profile(const double* xs, const double* buy_vols,
                                             const double* sell_vols, int count, double height) {
  if (count <= 0) return;

  ImDrawList* draw_list = ImPlot::GetPlotDrawList();
  const ImU32 col_buy = IM_COL32(0, 255, 0, 170);    // Green for buy volume
  const ImU32 col_sell = IM_COL32(255, 0, 0, 170);  // Red for sell volume
  const ImU32 col_center = IM_COL32(255, 255, 255, 200);  // White for center line

  // Find max volume to scale properly
  double max_vol = 0.0;
  for (int i = 0; i < count; ++i) {
    max_vol = std::max(max_vol, std::max(buy_vols[i], sell_vols[i]));
  }
  if (max_vol <= 0) max_vol = 1.0;

  for (int i = 0; i < count; ++i) {
    // Get pixel coordinates for the center of the bar (price level)
    ImVec2 center_point = ImPlot::PlotToPixels(0, xs[i]);

    // Calculate the extents of buy and sell volumes
    double scaled_buy_vol = (buy_vols[i] / max_vol) * max_volume_;
    double scaled_sell_vol = (sell_vols[i] / max_vol) * max_volume_;

    ImVec2 buy_point = ImPlot::PlotToPixels(-scaled_buy_vol, xs[i]);
    ImVec2 sell_point = ImPlot::PlotToPixels(scaled_sell_vol, xs[i]);

    // Calculate bar dimensions
    float bar_top = center_point.y - height / 2;
    float bar_bottom = center_point.y + height / 2;

    // Draw buy volume bar (left side, green)
    if (buy_vols[i] > 0) {
      ImVec2 bar_tl = ImVec2(buy_point.x, bar_top);
      ImVec2 bar_br = ImVec2(center_point.x, bar_bottom);

      draw_list->AddRectFilled(bar_tl, bar_br, col_buy);
    }

    // Draw sell volume bar (right side, red)
    if (sell_vols[i] > 0) {
      ImVec2 bar_tl = ImVec2(center_point.x, bar_top);
      ImVec2 bar_br = ImVec2(sell_point.x, bar_bottom);

      draw_list->AddRectFilled(bar_tl, bar_br, col_sell);
    }

    // Draw center vertical line to separate buy and sell volumes
    draw_list->AddLine(ImVec2(center_point.x, bar_top), ImVec2(center_point.x, bar_bottom),
                       col_center, 1.0f);
  }
}

// Enhanced method to render step profile histograms on candlesticks with additional features
void VolumeProfilePanel::render_enhanced_step_profile_on_candles(
    ImDrawList* draw_list,
    const std::vector<RenderEngine::OHLCVCandle>& candles,
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords_high,
    const std::vector<double>& y_coords_low,
    const std::vector<RenderEngine::TradeData>& trades,
    bool show_poc_line,
    int num_buckets_per_candle,
    float opacity_factor) {

  if (candles.empty() || x_coords.size() != candles.size() ||
      y_coords_high.size() != candles.size() || y_coords_low.size() != candles.size()) {
    return;
  }

  // Iterate through each candle to draw enhanced step profile histogram
  for (size_t i = 0; i < candles.size(); ++i) {
    const auto& candle = candles[i];

    // Skip if candle has no meaningful price range
    if (candle.high <= candle.low || candle.volume <= 0) continue;

    // Get the candle's price range (high - low)
    double price_range = candle.high - candle.low;
    if (price_range <= 0) continue;

    // Calculate bucket size for this candle's range
    double bucket_size = price_range / num_buckets_per_candle;

    // Create temporary buckets for this candle's price range
    std::vector<double> bucket_volumes(num_buckets_per_candle, 0.0);
    std::vector<int> bucket_counts(num_buckets_per_candle, 0);

    // Aggregate trades into buckets based on price within this candle's range
    for (const auto& trade : trades) {
      // Only consider trades within this candle's price range
      if (trade.price >= candle.low && trade.price <= candle.high) {
        int bucket_idx = static_cast<int>((trade.price - candle.low) / bucket_size);
        // Ensure we don't exceed bounds
        bucket_idx = std::max(0, std::min(bucket_idx, num_buckets_per_candle - 1));

        bucket_volumes[bucket_idx] += trade.size;
        bucket_counts[bucket_idx]++;
      }
    }

    // Find max volume in this candle's histogram for scaling
    double max_vol_in_candle = 0.0;
    int poc_bucket_idx = -1;  // Index of the bucket with highest volume (POC)

    for (int j = 0; j < num_buckets_per_candle; ++j) {
      if (bucket_volumes[j] > max_vol_in_candle) {
        max_vol_in_candle = bucket_volumes[j];
        poc_bucket_idx = j;
      }
    }

    // If no trades found in this candle's range, skip
    if (max_vol_in_candle <= 0 || poc_bucket_idx < 0) continue;

    // Calculate screen coordinates for the step profile histogram
    float x_center = static_cast<float>(x_coords[i]);
    float y_high = static_cast<float>(y_coords_high[i]);
    float y_low = static_cast<float>(y_coords_low[i]);

    // Calculate height of each bucket in screen coordinates
    float total_height = y_low - y_high;  // Height of the candle in screen space
    float bucket_height = total_height / num_buckets_per_candle;

    // Draw enhanced step profile histogram inside the candle
    for (int j = 0; j < num_buckets_per_candle; ++j) {
      if (bucket_volumes[j] > 0) {
        // Calculate the fill percentage of this bucket
        float fill_percentage =
            static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

        // Calculate the top-left and bottom-right coordinates for this bucket
        float y_top = y_high + j * bucket_height;
        float y_bottom = y_high + (j + 1) * bucket_height;

        // Calculate width of the bar based on fill percentage
        float bar_width = (y_bottom - y_top) * 0.7f;  // Slightly wider for better visibility
        float filled_width = bar_width * fill_percentage;

        // Calculate the x positions for the bar
        float x_left = x_center - bar_width / 2.0f;
        float x_right = x_left + filled_width;

        // Choose color based on whether this is the POC bucket or not
        ImU32 color;
        if (j == poc_bucket_idx) {
          // Highlight POC bucket with bright yellow
          color = IM_COL32(255, 255, 0, static_cast<int>(240 * opacity_factor));  // Bright yellow for POC with adjustable opacity
        } else {
          // Use gradient colors based on volume intensity and position in the candle
          float volume_ratio = static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

          // Determine if this bucket is in the upper or lower half of the candle
          float position_ratio = static_cast<float>(j) / static_cast<float>(num_buckets_per_candle - 1);

          // Create a color gradient based on volume intensity and position
          if (position_ratio < 0.5f) {
            // Lower half - red for selling pressure, with intensity based on volume
            int red_intensity = static_cast<int>(255 * volume_ratio * opacity_factor);
            int alpha = static_cast<int>(180 * opacity_factor);
            color = IM_COL32(red_intensity, static_cast<int>(50 * volume_ratio * opacity_factor),
                            static_cast<int>(50 * volume_ratio * opacity_factor), alpha);
          } else {
            // Upper half - green for buying pressure, with intensity based on volume
            int green_intensity = static_cast<int>(255 * volume_ratio * opacity_factor);
            int alpha = static_cast<int>(180 * opacity_factor);
            color = IM_COL32(static_cast<int>(50 * volume_ratio * opacity_factor), green_intensity,
                            static_cast<int>(50 * volume_ratio * opacity_factor), alpha);
          }
        }

        // Draw the enhanced step profile histogram bar with a slight outline for better visibility
        draw_list->AddRectFilled(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), color);

        // Add a subtle border to make individual bars more distinguishable
        if (opacity_factor > 0.3f) {  // Only add border if not too transparent
          draw_list->AddRect(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), IM_COL32(0, 0, 0, static_cast<int>(80 * opacity_factor)), 0.0f, 0, 1.0f);
        }
      }
    }

    // Optionally draw POC (Point of Control) line - horizontal yellow line at the price level with highest volume
    if (show_poc_line && poc_bucket_idx >= 0) {
      // Calculate the y-coordinate for the POC line
      float poc_y = y_high + (poc_bucket_idx + 0.5f) * bucket_height;  // Center of the POC bucket

      // Draw horizontal yellow line across the candle width
      float poc_line_half_width = (total_height) * 0.35f;  // Slightly wider for better visibility
      float poc_x_left = x_center - poc_line_half_width;
      float poc_x_right = x_center + poc_line_half_width;

      // Draw the POC line as a horizontal yellow line with adjustable opacity
      draw_list->AddLine(ImVec2(poc_x_left, poc_y), ImVec2(poc_x_right, poc_y),
                         IM_COL32(255, 255, 0, static_cast<int>(255 * opacity_factor)),  // Bright yellow color for POC with adjustable opacity
                         2.0f);                        // Line thickness
    }
  }
}

// Main method to implement Step Profile rendering: draw mini histogram overlay on each candlestick bar
// showing volume distribution for that bar's price range
void VolumeProfilePanel::render_step_profile_on_candles_with_volume_distribution(
    ImDrawList* draw_list,
    const std::vector<RenderEngine::OHLCVCandle>& candles,
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords_high,
    const std::vector<double>& y_coords_low,
    bool show_poc_line,
    int num_buckets_per_candle) {

  // This method implements the core requirement: draw mini histogram overlay on each candlestick bar
  // showing volume distribution for that bar's price range
  if (candles.empty() || x_coords.size() != candles.size() ||
      y_coords_high.size() != candles.size() || y_coords_low.size() != candles.size()) {
    return;
  }

  // Get recent trades for this symbol to populate the histograms
  auto analytics = processor_->getSymbolAnalytics(symbol_id_);
  const auto& trades = analytics.recent_trades;

  // Iterate through each candle to draw step profile histogram
  for (size_t i = 0; i < candles.size(); ++i) {
    const auto& candle = candles[i];

    // Skip if candle has no meaningful price range
    if (candle.high <= candle.low || candle.volume <= 0) continue;

    // Get the candle's price range (high - low)
    double price_range = candle.high - candle.low;
    if (price_range <= 0) continue;

    // Calculate bucket size for this candle's range
    double bucket_size = price_range / num_buckets_per_candle;

    // Create temporary buckets for this candle's price range
    std::vector<double> bucket_volumes(num_buckets_per_candle, 0.0);
    std::vector<int> bucket_trade_counts(num_buckets_per_candle, 0);

    // Aggregate trades into buckets based on price within this candle's range
    for (const auto& trade : trades) {
      // Only consider trades within this candle's price range
      if (trade.price >= candle.low && trade.price <= candle.high) {
        int bucket_idx = static_cast<int>((trade.price - candle.low) / bucket_size);
        // Ensure we don't exceed bounds
        bucket_idx = std::max(0, std::min(bucket_idx, num_buckets_per_candle - 1));

        bucket_volumes[bucket_idx] += trade.size;
        bucket_trade_counts[bucket_idx]++;
      }
    }

    // Find max volume in this candle's histogram for scaling
    double max_vol_in_candle = 0.0;
    int poc_bucket_idx = -1;  // Index of the bucket with highest volume (POC)

    for (int j = 0; j < num_buckets_per_candle; ++j) {
      if (bucket_volumes[j] > max_vol_in_candle) {
        max_vol_in_candle = bucket_volumes[j];
        poc_bucket_idx = j;
      }
    }

    // If no trades found in this candle's range, skip
    if (max_vol_in_candle <= 0 || poc_bucket_idx < 0) continue;

    // Calculate screen coordinates for the step profile histogram
    float x_center = static_cast<float>(x_coords[i]);
    float y_high = static_cast<float>(y_coords_high[i]);
    float y_low = static_cast<float>(y_coords_low[i]);

    // Calculate height of each bucket in screen coordinates
    float total_height = y_low - y_high;  // Height of the candle in screen space
    float bucket_height = total_height / num_buckets_per_candle;

    // Draw step profile histogram inside the candle
    for (int j = 0; j < num_buckets_per_candle; ++j) {
      if (bucket_volumes[j] > 0) {
        // Calculate the fill percentage of this bucket
        float fill_percentage =
            static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

        // Calculate the top-left and bottom-right coordinates for this bucket
        float y_top = y_high + j * bucket_height;
        float y_bottom = y_high + (j + 1) * bucket_height;

        // Calculate width of the bar based on fill percentage
        float bar_width = (y_bottom - y_top) * 0.6f;  // Use 60% of height as width for visibility
        float filled_width = bar_width * fill_percentage;

        // Calculate the x positions for the bar
        float x_left = x_center - bar_width / 2.0f;
        float x_right = x_left + filled_width;

        // Choose color based on whether this is the POC bucket or not
        ImU32 color;
        if (j == poc_bucket_idx) {
          // Highlight POC bucket with bright yellow
          color = IM_COL32(255, 255, 0, 240);  // Bright yellow for POC with high opacity
        } else {
          // Use gradient colors based on volume intensity and position in the candle
          float volume_ratio = static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

          // Determine if this bucket is in the upper or lower half of the candle
          float position_ratio = static_cast<float>(j) / static_cast<float>(num_buckets_per_candle - 1);

          // Create a color gradient based on volume intensity and position
          if (position_ratio < 0.5f) {
            // Lower half - red for selling pressure, with intensity based on volume
            int red_intensity = static_cast<int>(255 * volume_ratio);
            color = IM_COL32(red_intensity, static_cast<int>(50 * volume_ratio),
                            static_cast<int>(50 * volume_ratio), 180);
          } else {
            // Upper half - green for buying pressure, with intensity based on volume
            int green_intensity = static_cast<int>(255 * volume_ratio);
            color = IM_COL32(static_cast<int>(50 * volume_ratio), green_intensity,
                            static_cast<int>(50 * volume_ratio), 180);
          }
        }

        // Draw the step profile histogram bar with a slight outline for better visibility
        draw_list->AddRectFilled(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), color);

        // Add a subtle border to make individual bars more distinguishable
        draw_list->AddRect(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), IM_COL32(0, 0, 0, 80), 0.0f, 0, 1.0f);
      }
    }

    // Optionally draw POC (Point of Control) line - horizontal yellow line at the price level with highest volume
    if (show_poc_line && poc_bucket_idx >= 0) {
      // Calculate the y-coordinate for the POC line
      float poc_y = y_high + (poc_bucket_idx + 0.5f) * bucket_height;  // Center of the POC bucket

      // Draw horizontal yellow line across the candle width
      float poc_line_half_width = (total_height) * 0.3f;  // Reduced width to avoid overlapping with candle wicks
      float poc_x_left = x_center - poc_line_half_width;
      float poc_x_right = x_center + poc_line_half_width;

      // Draw the POC line as a horizontal yellow line
      draw_list->AddLine(ImVec2(poc_x_left, poc_y), ImVec2(poc_x_right, poc_y),
                         IM_COL32(255, 255, 0, 255),  // Bright yellow color for POC
                         2.0f);                        // Line thickness
    }
  }
}

// Additional method to render step profile with enhanced visualization options
void VolumeProfilePanel::render_enhanced_step_profile_with_volume_distribution(
    ImDrawList* draw_list,
    const std::vector<RenderEngine::OHLCVCandle>& candles,
    const std::vector<double>& x_coords,
    const std::vector<double>& y_coords_high,
    const std::vector<double>& y_coords_low,
    bool show_poc_line,
    int num_buckets_per_candle,
    float bar_opacity,
    bool use_transparent_background) {

  // This method implements an enhanced version of the Step Profile rendering with additional customization options
  if (candles.empty() || x_coords.size() != candles.size() ||
      y_coords_high.size() != candles.size() || y_coords_low.size() != candles.size()) {
    return;
  }

  // Get recent trades for this symbol to populate the histograms
  auto analytics = processor_->getSymbolAnalytics(symbol_id_);
  const auto& trades = analytics.recent_trades;

  // Iterate through each candle to draw enhanced step profile histogram
  for (size_t i = 0; i < candles.size(); ++i) {
    const auto& candle = candles[i];

    // Skip if candle has no meaningful price range
    if (candle.high <= candle.low || candle.volume <= 0) continue;

    // Get the candle's price range (high - low)
    double price_range = candle.high - candle.low;
    if (price_range <= 0) continue;

    // Calculate bucket size for this candle's range
    double bucket_size = price_range / num_buckets_per_candle;

    // Create temporary buckets for this candle's price range
    std::vector<double> bucket_volumes(num_buckets_per_candle, 0.0);
    std::vector<int> bucket_trade_counts(num_buckets_per_candle, 0);

    // Aggregate trades into buckets based on price within this candle's range
    for (const auto& trade : trades) {
      // Only consider trades within this candle's price range
      if (trade.price >= candle.low && trade.price <= candle.high) {
        int bucket_idx = static_cast<int>((trade.price - candle.low) / bucket_size);
        // Ensure we don't exceed bounds
        bucket_idx = std::max(0, std::min(bucket_idx, num_buckets_per_candle - 1));

        bucket_volumes[bucket_idx] += trade.size;
        bucket_trade_counts[bucket_idx]++;
      }
    }

    // Find max volume in this candle's histogram for scaling
    double max_vol_in_candle = 0.0;
    int poc_bucket_idx = -1;  // Index of the bucket with highest volume (POC)

    for (int j = 0; j < num_buckets_per_candle; ++j) {
      if (bucket_volumes[j] > max_vol_in_candle) {
        max_vol_in_candle = bucket_volumes[j];
        poc_bucket_idx = j;
      }
    }

    // If no trades found in this candle's range, skip
    if (max_vol_in_candle <= 0 || poc_bucket_idx < 0) continue;

    // Calculate screen coordinates for the step profile histogram
    float x_center = static_cast<float>(x_coords[i]);
    float y_high = static_cast<float>(y_coords_high[i]);
    float y_low = static_cast<float>(y_coords_low[i]);

    // Calculate height of each bucket in screen coordinates
    float total_height = y_low - y_high;  // Height of the candle in screen space
    float bucket_height = total_height / num_buckets_per_candle;

    // Optionally draw a subtle background for the entire candle to highlight the histogram area
    if (use_transparent_background) {
        draw_list->AddRectFilled(ImVec2(x_center - total_height * 0.4f, y_high),
                                ImVec2(x_center + total_height * 0.1f, y_low),
                                IM_COL32(0, 0, 0, 30));  // Very subtle dark background
    }

    // Draw step profile histogram inside the candle
    for (int j = 0; j < num_buckets_per_candle; ++j) {
      if (bucket_volumes[j] > 0) {
        // Calculate the fill percentage of this bucket
        float fill_percentage =
            static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

        // Calculate the top-left and bottom-right coordinates for this bucket
        float y_top = y_high + j * bucket_height;
        float y_bottom = y_high + (j + 1) * bucket_height;

        // Calculate width of the bar based on fill percentage
        float bar_width = (y_bottom - y_top) * 0.7f;  // Slightly wider for better visibility
        float filled_width = bar_width * fill_percentage;

        // Calculate the x positions for the bar
        float x_left = x_center - bar_width / 2.0f;
        float x_right = x_left + filled_width;

        // Choose color based on whether this is the POC bucket or not
        ImU32 color;
        if (j == poc_bucket_idx) {
          // Highlight POC bucket with bright yellow
          color = IM_COL32(255, 255, 0, static_cast<int>(255 * bar_opacity));  // Bright yellow for POC with adjustable opacity
        } else {
          // Use gradient colors based on volume intensity and position in the candle
          float volume_ratio = static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

          // Determine if this bucket is in the upper or lower half of the candle
          float position_ratio = static_cast<float>(j) / static_cast<float>(num_buckets_per_candle - 1);

          // Create a color gradient based on volume intensity and position
          if (position_ratio < 0.5f) {
            // Lower half - red for selling pressure, with intensity based on volume
            int red_intensity = static_cast<int>(255 * volume_ratio * bar_opacity);
            int alpha = static_cast<int>(180 * bar_opacity);
            color = IM_COL32(red_intensity, static_cast<int>(50 * volume_ratio * bar_opacity),
                            static_cast<int>(50 * volume_ratio * bar_opacity), alpha);
          } else {
            // Upper half - green for buying pressure, with intensity based on volume
            int green_intensity = static_cast<int>(255 * volume_ratio * bar_opacity);
            int alpha = static_cast<int>(180 * bar_opacity);
            color = IM_COL32(static_cast<int>(50 * volume_ratio * bar_opacity), green_intensity,
                            static_cast<int>(50 * volume_ratio * bar_opacity), alpha);
          }
        }

        // Draw the enhanced step profile histogram bar with a slight outline for better visibility
        draw_list->AddRectFilled(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), color);

        // Add a subtle border to make individual bars more distinguishable
        if (bar_opacity > 0.3f) {  // Only add border if not too transparent
          draw_list->AddRect(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom),
                            IM_COL32(0, 0, 0, static_cast<int>(80 * bar_opacity)), 0.0f, 0, 1.0f);
        }
      }
    }

    // Optionally draw POC (Point of Control) line - horizontal yellow line at the price level with highest volume
    if (show_poc_line && poc_bucket_idx >= 0) {
      // Calculate the y-coordinate for the POC line
      float poc_y = y_high + (poc_bucket_idx + 0.5f) * bucket_height;  // Center of the POC bucket

      // Draw horizontal yellow line across the candle width
      float poc_line_half_width = (total_height) * 0.35f;  // Slightly wider for better visibility
      float poc_x_left = x_center - poc_line_half_width;
      float poc_x_right = x_center + poc_line_half_width;

      // Draw the POC line as a horizontal yellow line with adjustable opacity
      draw_list->AddLine(ImVec2(poc_x_left, poc_y), ImVec2(poc_x_right, poc_y),
                         IM_COL32(255, 255, 0, static_cast<int>(255 * bar_opacity)),  // Bright yellow color for POC with adjustable opacity
                         2.0f);                        // Line thickness
    }
  }
}

}  // namespace BTQuant
