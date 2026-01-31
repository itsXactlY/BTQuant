#include "../../include/components/volume_profile_panel.hpp"
#include "imgui.h"
#include "implot.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace BTQuant {

VolumeProfilePanel::VolumeProfilePanel(
    const PanelConfig &config, std::shared_ptr<HotSpineDataBridge> bridge,
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
  if (!processor_ || symbol_id_ == 0)
    return;

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

void VolumeProfilePanel::set_symbol(uint32_t symbol_id,
                                    const std::string &symbol_name) {
  symbol_id_ = symbol_id;
  symbol_name_ = symbol_name;
  volume_profile_.clear();
  max_volume_ = 0.0;
  poc_price_ = 0.0;

  // Re-subscribe to new symbol
  subscribe_to_updates();
  markDirty(); // Force immediate build
}

void VolumeProfilePanel::build_volume_profile() {
  auto analytics = processor_->getSymbolAnalytics(symbol_id_);
  const auto &trades = analytics.recent_trades;

  if (trades.empty())
    return;

  // Check if we have an existing volume profile and if the new trades fit in the current range
  bool needs_rebuild = false;
  double min_price = std::numeric_limits<double>::max();
  double max_price = std::numeric_limits<double>::lowest();

  // Find the range of new trades
  for (const auto &trade : trades) {
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
      for (const auto &trade : trades) {
        // Calculate which bucket this trade belongs to based on existing bucket size
        size_t bucket_index = static_cast<size_t>((trade.price - current_min_price) / price_bucket_size_);

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
      for (const auto &level : volume_profile_) {
        double total = level.buy_volume + level.sell_volume;
        if (total > poc_volume) {
          poc_volume = total;
          poc_price_ = level.price;
        }
      }

      return; // Exit early since we've updated incrementally
    }
  } else {
    needs_rebuild = true; // First time building, so we need to rebuild
  }

  if (needs_rebuild || volume_profile_.empty()) {
    if (max_price <= min_price)
      return;

    // Compute bucket size
    double range = max_price - min_price;
    price_bucket_size_ = range / NUM_PRICE_LEVELS;
    if (price_bucket_size_ <= 0)
      price_bucket_size_ = 1.0;

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
    for (const auto &trade : trades) {
      size_t bucket =
          static_cast<size_t>((trade.price - min_price) / price_bucket_size_);
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

    for (const auto &level : volume_profile_) {
      double total = level.buy_volume + level.sell_volume;
      max_volume_ =
          std::max(max_volume_, std::max(level.buy_volume, level.sell_volume));
      if (total > poc_volume) {
        poc_volume = total;
        poc_price_ = level.price;
      }
    }
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
  ImGui::SliderInt("VA%", &va_percent_, 50, 99);
}

void VolumeProfilePanel::render_volume_bars() {
  if (volume_profile_.empty() || max_volume_ <= 0) {
    ImGui::Text("No volume data available");
    return;
  }

  ImVec2 region = ImGui::GetContentRegionAvail();
  if (region.x < 100 || region.y < 100)
    return;

  // Prepare data for ImPlot horizontal bars
  std::vector<double> prices;
  std::vector<double> buy_volumes;
  std::vector<double> sell_volumes;

  prices.reserve(volume_profile_.size());
  buy_volumes.reserve(volume_profile_.size());
  sell_volumes.reserve(volume_profile_.size());

  for (const auto &level : volume_profile_) {
    prices.push_back(level.price);
    buy_volumes.push_back(level.buy_volume);
    sell_volumes.push_back(-level.sell_volume); // Negative for left side
  }

  // Unique plot ID per panel instance to avoid ImGui ID conflicts
  char plot_id[64];
  snprintf(plot_id, sizeof(plot_id), "##VolumeProfile_%s",
           config_.title.c_str());

  if (ImPlot::BeginPlot(plot_id, region,
                        ImPlotFlags_NoTitle | ImPlotFlags_NoLegend)) {

    ImPlot::SetupAxes("Volume", "Price",
                      ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_Invert,
                      ImPlotAxisFlags_AutoFit);

    // Bar height based on price bucket size
    double bar_height = price_bucket_size_ * 0.8;

    // Render based on profile mode
    switch (profile_mode_) {
      case ProfileMode::Step:
        render_step_profile(prices.data(), buy_volumes.data(), sell_volumes.data(),
                           static_cast<int>(prices.size()), bar_height);
        break;
      case ProfileMode::Right:
        // Right Profile: Anchor to right edge with bars extending left
        // Calculate the maximum volume to determine the right edge position
        double max_vol = max_volume_;

        // Draw buy bars (green) extending left from right edge
        // For right profile, we want bars that start from the right edge and extend left
        // We'll use a custom approach with ImDrawList to draw bars anchored to the right
        ImDrawList* draw_list = ImPlot::GetPlotDrawList();

        for (size_t i = 0; i < buy_volumes.size(); ++i) {
            if (buy_volumes[i] > 0) {
                // Convert plot coordinates to pixel coordinates
                ImVec2 right_edge = ImPlot::PlotToPixels(max_vol, prices[i]);
                ImVec2 left_edge = ImPlot::PlotToPixels(max_vol - buy_volumes[i], prices[i]);

                // Calculate bar dimensions
                float bar_top = right_edge.y - bar_height / 2.0;
                float bar_bottom = right_edge.y + bar_height / 2.0;
                float bar_right = right_edge.x;  // Right edge of chart
                float bar_left = left_edge.x;    // Left extent of the bar

                // Draw the bar
                draw_list->AddRectFilled(ImVec2(bar_left, bar_top), ImVec2(bar_right, bar_bottom),
                                        IM_COL32(26, 204, 26, 179)); // Green with transparency
            }
        }

        // Draw sell bars (red) extending left from right edge
        for (size_t i = 0; i < sell_volumes.size(); ++i) {
            if (sell_volumes[i] < 0) {  // Remember sell volumes are stored as negative
                // Convert plot coordinates to pixel coordinates
                ImVec2 right_edge = ImPlot::PlotToPixels(max_vol, prices[i]);
                ImVec2 left_edge = ImPlot::PlotToPixels(max_vol - std::abs(sell_volumes[i]), prices[i]);

                // Calculate bar dimensions
                float bar_top = right_edge.y - bar_height / 2.0;
                float bar_bottom = right_edge.y + bar_height / 2.0;
                float bar_right = right_edge.x;  // Right edge of chart
                float bar_left = left_edge.x;    // Left extent of the bar

                // Draw the bar
                draw_list->AddRectFilled(ImVec2(bar_left, bar_top), ImVec2(bar_right, bar_bottom),
                                        IM_COL32(204, 26, 26, 179)); // Red with transparency
            }
        }
        break;
      case ProfileMode::Left:
        // Sell bars (red, negative X) - mirrored
        ImPlot::SetNextFillStyle(ImVec4(0.8f, 0.1f, 0.1f, 0.7f));
        ImPlot::PlotBars("Sell", sell_volumes.data(), prices.data(),
                         static_cast<int>(prices.size()), bar_height,
                         ImPlotBarsFlags_Horizontal);

        // Buy bars (green, positive X)
        ImPlot::SetNextFillStyle(ImVec4(0.1f, 0.8f, 0.1f, 0.7f));
        ImPlot::PlotBars("Buy", buy_volumes.data(), prices.data(),
                         static_cast<int>(prices.size()), bar_height,
                         ImPlotBarsFlags_Horizontal);
        break;
      case ProfileMode::Custom:
      default:
        // Default behavior - both buy and sell volumes
        ImPlot::SetNextFillStyle(ImVec4(0.1f, 0.8f, 0.1f, 0.7f));
        ImPlot::PlotBars("Buy", buy_volumes.data(), prices.data(),
                         static_cast<int>(prices.size()), bar_height,
                         ImPlotBarsFlags_Horizontal);

        ImPlot::SetNextFillStyle(ImVec4(0.8f, 0.1f, 0.1f, 0.7f));
        ImPlot::PlotBars("Sell", sell_volumes.data(), prices.data(),
                         static_cast<int>(prices.size()), bar_height,
                         ImPlotBarsFlags_Horizontal);
        break;
    }

    // POC line
    if (poc_price_ > 0) {
      double poc_line_x[2] = {-max_volume_, max_volume_};
      double poc_line_y[2] = {poc_price_, poc_price_};
      ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.8f, 0.0f, 1.0f));
      ImPlot::PlotLine("POC", poc_line_x, poc_line_y, 2);
      ImPlot::PopStyleColor();
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

    ImPlot::EndPlot();
  }
}

void VolumeProfilePanel::render_step_profile(const double* xs, const double* ys,
                                            const double* neg_ys, int count,
                                            double height) {
  if (count <= 0) return;

  ImDrawList* draw_list = ImPlot::GetPlotDrawList();
  const ImU32 col_pos = IM_COL32(0, 255, 0, 170);  // Green for positive
  const ImU32 col_neg = IM_COL32(255, 0, 0, 170);  // Red for negative
  const ImU32 col_poc = IM_COL32(255, 255, 0, 255); // Yellow for POC (Point of Control)

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
}

// Method to render mini histogram overlays on candlestick charts
void VolumeProfilePanel::render_mini_histograms_on_candles(ImDrawList* draw_list,
                                                          const std::vector<RenderEngine::OHLCVCandle>& candles,
                                                          const std::vector<double>& x_coords,
                                                          const std::vector<double>& y_coords_high,
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
    int num_buckets = 8; // Fixed number of buckets for mini histogram
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
    int poc_bucket_idx = 0; // Index of the bucket with highest volume (POC)

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
    float total_height = y_low - y_high; // Height of the candle in screen space
    float bucket_height = total_height / num_buckets;

    // Draw mini histogram inside the candle
    for (int j = 0; j < num_buckets; ++j) {
      if (bucket_volumes[j] > 0) {
        // Calculate the fill percentage of this bucket
        float fill_percentage = static_cast<float>(bucket_volumes[j]) / static_cast<float>(max_vol_in_candle);

        // Calculate the top-left and bottom-right coordinates for this bucket
        float y_top = y_high + j * bucket_height;
        float y_bottom = y_high + (j + 1) * bucket_height;

        // Calculate width of the bar based on fill percentage
        float bar_width = (y_bottom - y_top) * 0.8f; // Use 80% of height as width for visibility
        float filled_width = bar_width * fill_percentage;

        // Calculate the x positions for the bar
        float x_left = x_center - bar_width / 2.0f;
        float x_right = x_left + filled_width;

        // Choose color based on whether this is in the upper or lower half of the candle
        // (to represent buy/sell pressure)
        ImU32 color;
        if (j < num_buckets / 2) {
          // Lower half - potentially more selling pressure
          color = IM_COL32(255, 100, 100, 150); // Reddish for sells
        } else {
          // Upper half - potentially more buying pressure
          color = IM_COL32(100, 255, 100, 150); // Greenish for buys
        }

        // Draw the mini histogram bar
        draw_list->AddRectFilled(ImVec2(x_left, y_top), ImVec2(x_right, y_bottom), color);
      }
    }

    // Draw POC (Point of Control) line - horizontal yellow line at the price level with highest volume
    // Calculate the y-coordinate for the POC line
    float poc_y = y_high + (poc_bucket_idx + 0.5f) * bucket_height; // Center of the POC bucket

    // Draw horizontal yellow line across the candle width
    float poc_line_half_width = (y_low - y_high) * 0.4f; // Same width as candle
    float poc_x_left = x_center - poc_line_half_width;
    float poc_x_right = x_center + poc_line_half_width;

    // Draw the POC line as a horizontal yellow line
    draw_list->AddLine(
        ImVec2(poc_x_left, poc_y),
        ImVec2(poc_x_right, poc_y),
        IM_COL32(255, 255, 0, 255), // Yellow color for POC
        2.0f // Line thickness
    );
  }
}

} // namespace BTQuant
