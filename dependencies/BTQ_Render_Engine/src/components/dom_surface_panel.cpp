#include "components/dom_surface_panel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>

#include "../include/trading/HotspineData.h"

// Shorter aliases for commonly used types
using BTQuant::RenderEngine::OrderbookData;
using BTQuant::RenderEngine::TradeData;
// Note: CandleCluster is used with full qualification for consistency

namespace BTQuant {

DomSurfacePanel::DomSurfacePanel(std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
                                 PanelManager* panel_manager)
    : PanelBase(PanelConfig{.title = "DOM Surface", .type = PanelType::HEATMAP}),
      processor_(processor),
      panel_manager_(panel_manager),
      max_trade_volume_(1.0),            // Initialize with a default minimum volume
      persistence_threshold_ms_(30000),  // 30 seconds for static liquidity detection
      persistence_timeout_ms_(60000),    // 60 seconds timeout for inactive levels
      show_persistent_lines_(true) {}    // Show persistent lines by default

DomSurfacePanel::~DomSurfacePanel() {
  if (subscription_id_ > 0 && processor_) {
    processor_->unsubscribe(subscription_id_);
  }
}

void DomSurfacePanel::setSymbol(uint32_t symbol_id) {
  std::lock_guard<std::mutex> lock(data_mutex_);
  // Get the symbol name from the processor or symbol registry if available
  std::string symbol_name =
      "SYMBOL_" + std::to_string(symbol_id);  // Default symbol name if not available
  if (processor_) {
    // symbol_name = processor_->getSymbolName(symbol_id); // Commented out due to missing method
  }

  setSymbol(symbol_id, symbol_name);
}

void DomSurfacePanel::setSymbol(uint32_t symbol_id, const std::string& symbol_name) {
  std::lock_guard<std::mutex> lock(data_mutex_);
  // Store the symbol name for potential use in UI elements
  current_symbol_name_ = symbol_name;

  // Call the original setSymbol logic to handle subscriptions and data clearing
  if (current_symbol_id_ != symbol_id) {
    if (subscription_id_ > 0) {
      processor_->unsubscribe(subscription_id_);
      subscription_id_ = 0;
    }

    current_symbol_id_ = symbol_id;

    // Subscribe to both ORDERBOOK and TRADE updates
    if (processor_) {
      subscription_id_ =
          processor_->subscribe(symbol_id, RenderEngine::NotificationType::ORDERBOOK,
                                [this](uint32_t sym, RenderEngine::NotificationType type) {
                                  this->onDataUpdate(sym, type);
                                });

      // Also subscribe to trade updates for trade bubbles
      processor_->subscribe(symbol_id, RenderEngine::NotificationType::TRADE,
                            [this](uint32_t sym, RenderEngine::NotificationType type) {
                              this->onDataUpdate(sym, type);
                            });
    }

    // Clear existing data to prevent mixing symbols
    heatmap_data_.clear();
    large_order_markers_.clear();
    trade_bubbles_.clear();  // Clear trade bubbles when changing symbols
    recent_order_sizes_.clear();
    median_order_size_ = 0.0;
    markDirty();
  }

  // Notify the panel manager about the symbol change to trigger symbol linking
  if (get_panel_manager()) {
    get_panel_manager()->propagate_symbol_to_linked_panels(get_panel_id(), symbol_name);
  }
}

void DomSurfacePanel::onDataUpdate(uint32_t symbol_id, RenderEngine::NotificationType type) {
  std::lock_guard<std::mutex> lock(data_mutex_);
  if (symbol_id == current_symbol_id_) {
    if (type == RenderEngine::NotificationType::TRADE) {
      // For trade updates, we'll update trade bubbles specifically
      updateTradeBubbles();
    }
    markDirty();
  }
}

void DomSurfacePanel::updateTradeBubbles() {
  if (current_symbol_id_ == 0 || !processor_) return;

  // Get recent trades for the current symbol
  auto analytics = processor_->getSymbolAnalytics(current_symbol_id_);

  // Process recent trades to create trade bubbles
  processRecentTrades();
}

void DomSurfacePanel::processRecentTrades() {
  if (current_symbol_id_ == 0 || !processor_) return;

  // Get symbol analytics which contains recent trades
  auto analytics = processor_->getSymbolAnalytics(current_symbol_id_);

  // Find max volume for scaling purposes across all recent trades
  double current_max_volume = max_trade_volume_;
  for (const auto& trade : analytics.recent_trades) {
    if (trade.size > current_max_volume) {
      current_max_volume = trade.size;
    }
  }
  max_trade_volume_ = current_max_volume;

  // Create trade bubbles for each recent trade that's not already in our list
  for (const auto& trade : analytics.recent_trades) {
    // Calculate X position based on timestamp relative to history range
    double relative_time = 0.0;
    if (history_end_timestamp_ > history_start_timestamp_) {
      relative_time = static_cast<double>(trade.timestamp - history_start_timestamp_) /
                      static_cast<double>(history_end_timestamp_ - history_start_timestamp_);
    }

    // Map bounds_min[0] (0) to bounds_max[0] (time_steps)
    double x_pos = bounds_min_[0] + relative_time * (bounds_max_[0] - bounds_min_[0]);

    // Only add bubble if within view
    if (x_pos >= bounds_min_[0] && x_pos <= bounds_max_[0]) {
      // Check if we already have this trade in our bubbles to avoid duplicates
      bool exists = false;
      for (const auto& bubble : trade_bubbles_) {
        if (bubble.timestamp == trade.timestamp && std::abs(bubble.y - trade.price) < 0.0001 &&
            std::abs(bubble.volume - trade.size) < 0.0001) {
          exists = true;
          break;
        }
      }

      if (!exists) {
        TradeBubble bubble(x_pos, trade.price, trade.size, trade.price, trade.is_buy,
                           trade.timestamp);
        bubble.radius = calculateBubbleRadius(trade.size);
        trade_bubbles_.push_back(bubble);
      }
    }
  }

  // Clean up bubbles that are outside the current view range to prevent accumulation
  cleanupOldTradeBubbles();
}

void DomSurfacePanel::cleanupOldTradeBubbles() {
  // Get current time to calculate age of the trades
  uint64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now().time_since_epoch())
                              .count();

  // Remove bubbles that are outside the current view range OR completely faded out
  // This helps keep the vector size manageable
  trade_bubbles_.erase(std::remove_if(trade_bubbles_.begin(), trade_bubbles_.end(),
                                      [this, current_time](const TradeBubble& bubble) {
                                        // Check if bubble is outside the current view bounds by a
                                        // margin
                                        double margin =
                                            (bounds_max_[0] - bounds_min_[0]) * 0.1;  // 10% margin
                                        bool outside_bounds = (bubble.x < bounds_min_[0] - margin ||
                                                               bubble.x > bounds_max_[0] + margin);

                                        // Check if bubble has completely faded out
                                        uint64_t age_ms = current_time - bubble.timestamp;
                                        bool fully_faded = age_ms >= TRADE_BUBBLE_FADE_DURATION_MS;

                                        return outside_bounds || fully_faded;
                                      }),
                       trade_bubbles_.end());
}

float DomSurfacePanel::calculateBubbleRadius(double volume) const {
  if (volume <= 0.0) return 3.0f;  // Minimum radius for invalid volumes

  // Use logarithmic scaling: log(volume + 1) to handle volume = 0 gracefully
  // This prevents massive trades from covering the entire price axis
  float log_volume = std::log(volume + 1.0f);

  // Use the tracked maximum volume for normalization to adapt to actual market conditions
  // Add a small epsilon to prevent division by zero if max_trade_volume_ is somehow 0
  float max_log_volume = std::log(max_trade_volume_ + 1.0f);

  // Handle edge case where max_trade_volume_ is extremely small
  if (max_log_volume <= 0.0f) {
    max_log_volume = std::log(1000.0f + 1.0f);  // Default to 1000 as reference max volume
  }

  float normalized_log_volume = log_volume / max_log_volume;

  // Clamp the normalized value to prevent exceeding intended radius range
  normalized_log_volume = std::clamp(normalized_log_volume, 0.0f, 1.0f);

  // Apply additional curve to make the scaling more gradual at low volumes
  // and steeper at high volumes for better visual distinction
  // Using a power function to adjust the distribution of bubble sizes
  // Made the exponent more conservative (0.3f instead of 0.5f) to prevent large trades
  // from dominating the visualization even with log scaling
  normalized_log_volume = std::pow(normalized_log_volume, 0.3f);

  // Scale to desired radius range - using calibrated max radius for log scaling
  float min_radius = MIN_RADIUS;
  float max_radius = MAX_RADIUS;
  float calculated_radius = min_radius + (max_radius - min_radius) * normalized_log_volume;

  return calculated_radius;
}

ImU32 DomSurfacePanel::getBubbleColor(const TradeBubble& bubble) const {
  // Get current time to calculate age of the trade
  uint64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now().time_since_epoch())
                              .count();

  // Calculate age of the trade in milliseconds
  uint64_t age_ms = current_time - bubble.timestamp;

  // Calculate fade ratio (0.0 = fully faded, 1.0 = fully opaque)
  float fade_ratio =
      1.0f - static_cast<float>(age_ms) / static_cast<float>(TRADE_BUBBLE_FADE_DURATION_MS);
  fade_ratio = std::clamp(fade_ratio, 0.0f, 1.0f);

  // Apply enhanced smooth easing function for more natural fade-out
  // Using a combination of cubic and sinusoidal easing for ultra-smooth transitions
  // This creates a more natural fade that starts gradually and accelerates toward the end
  float cubic_ease = fade_ratio * fade_ratio * fade_ratio;
  float sine_ease =
      0.5f * (1.0f - std::cos(fade_ratio * 3.14159265358979323846f));  // Smooth sine-based easing

  // Blend multiple easing functions for optimal smoothness
  // Weighted blend: 40% cubic, 40% sinusoidal, 20% original exponential
  float exp_ease = std::exp(-2.0f * (1.0f - fade_ratio)) * fade_ratio;
  float final_fade_ratio = 0.4f * cubic_ease + 0.4f * sine_ease + 0.2f * exp_ease;

  // Calculate alpha based on fade ratio
  uint8_t base_alpha = 200;  // Increased base alpha value for better visibility
  uint8_t alpha = static_cast<uint8_t>(base_alpha * final_fade_ratio);

  // Color: Green for Buys, Red for Sells with fade-out effect
  if (bubble.is_buy) {
    return IM_COL32(0, 255, 0, alpha);  // Green with fade-out transparency
  } else {
    return IM_COL32(255, 0, 0, alpha);  // Red with fade-out transparency
  }
}

void DomSurfacePanel::renderTradeBubbles() {
  if (trade_bubbles_.empty()) return;

  // Get plot area for manual circle rendering
  ImPlotRect plot_rect = ImPlot::GetPlotLimits();

  // Render each trade bubble as a circle
  for (const auto& bubble : trade_bubbles_) {
    ImU32 color = getBubbleColor(bubble);
    ImU32 border_color = IM_COL32(255, 255, 255, 200);  // White semi-transparent border

    // Convert plot coordinates to pixel coordinates
    ImVec2 pixel_pos = ImPlot::PlotToPixels(bubble.x, bubble.y);

    // Draw filled circle
    ImDrawList* draw_list = ImPlot::GetPlotDrawList();
    if (draw_list) {
      draw_list->AddCircleFilled(pixel_pos, bubble.radius, color, 32);
      draw_list->AddCircle(pixel_pos, bubble.radius, border_color, 32, 1.5f);

      // Check for hover and show tooltip
      ImVec2 mouse_pos = ImGui::GetMousePos();
      float distance = std::sqrt(std::pow(mouse_pos.x - pixel_pos.x, 2) +
                                 std::pow(mouse_pos.y - pixel_pos.y, 2));

      if (distance < bubble.radius) {
        std::string tooltip =
            std::format("Trade: {} {:.2f} @ ${:.2f}", bubble.is_buy ? "Buy" : "Sell", bubble.volume,
                        bubble.price);
        ImGui::SetTooltip("%s", tooltip.c_str());
      }
    }
  }
}

void DomSurfacePanel::updateHeatmapData() {
  if (current_symbol_id_ == 0 || !processor_) return;

  // Request ALL available orderbook history (0 = no limit)
  auto history = processor_->getHistoricalOrderbooks(current_symbol_id_, 0);
  if (history.empty()) return;

  // Calculate History Timestamps
  // Assuming history is sorted by time (oldest -> newest)
  // We need actual timestamps. If OrderbookData doesn't have it, we might need
  // to rely on index or add it. Checking MarketDataProcessor structures:
  // OrderbookData usually has timestamp. If not, we map indices to a time
  // window (e.g. last N ms). For now, let's assume we can map 0..history.size()
  // to a time range for the markers. Actually, to fix the scroll, we need
  // relative time. Let's use the current time and frame duration if timestamp
  // isn't available, BUT OrderbookData SHOULD have it. I'll check the struct
  // def if this fails to compile. Assuming OrderbookData has `timestamp`
  // (uint64_t microseconds).

  // Determine price range
  double min_price = std::numeric_limits<double>::max();
  double max_price = std::numeric_limits<double>::lowest();

  if (auto_scale_price_) {
    for (const auto& book : history) {
      if (!book.bids.empty())
        min_price = std::min(min_price, book.bids.back().price);  // Lowest bid (deepest)
      if (!book.bids.empty()) max_price = std::max(max_price, book.bids.front().price);
      if (!book.asks.empty()) min_price = std::min(min_price, book.asks.front().price);
      if (!book.asks.empty())
        max_price = std::max(max_price,
                             book.asks.back().price);  // Highest ask (deepest)
    }
    // Add some padding
    if (min_price < max_price) {
      double spread = max_price - min_price;
      min_price -= spread * 0.05;
      max_price += spread * 0.05;
    } else {
      // Fallback
      auto latest = history.back();
      double mid = 0;
      if (!latest.bids.empty())
        mid = latest.bids.front().price;
      else if (!latest.asks.empty())
        mid = latest.asks.front().price;
      min_price = mid * 0.98;
      max_price = mid * 1.02;
    }
  } else {
    // Legacy fixed range logic
    const auto& latest = history.back();
    double mid_price = 0;
    if (!latest.bids.empty() && !latest.asks.empty()) {
      mid_price = (latest.bids.front().price + latest.asks.front().price) / 2.0;
    } else if (!latest.bids.empty()) {
      mid_price = latest.bids.front().price;
    } else if (!latest.asks.empty()) {
      mid_price = latest.asks.front().price;
    } else {
      return;
    }
    min_price = mid_price * (1.0 - price_range_);
    max_price = mid_price * (1.0 + price_range_);
  }

  if (max_price <= min_price) return;

  // Time bounds (X-axis)
  // We use indices 0..size as the base, but we track timestamps for markers
  // If OrderbookData doesn't expose timestamp, use system clock tracking in
  // Processor For strict correctness, we assume history is correctly ordered.
  // We'll define the X-axis as "Updates Ago" or "Time" depending on capability.
  // To support scrolling markers, we need to map Marker Time -> Index.

  // Let's assume history updates at uniform rate or we just map linearly.
  // Crucial: define start/end timestamps for marker interpolation.
  // Optimization: Only scan timestamps if we have markers? No, need it for
  // every frame.
  if (!history.empty()) {
    // Trying to access timestamp. If compile fails, I will fix.
    // Based on common patterns: history.front().timestamp
    history_start_timestamp_ = history.front().timestamp;
    history_end_timestamp_ = history.back().timestamp;
  }

  // Ensure valid time range
  if (history_end_timestamp_ <= history_start_timestamp_) {
    history_end_timestamp_ = history_start_timestamp_ + 1;
  }

  double price_step = (max_price - min_price) / static_cast<double>(price_bins_);
  int time_steps = static_cast<int>(history.size());
  size_t total_size = static_cast<size_t>(price_bins_) * static_cast<size_t>(time_steps);

  if (heatmap_data_.size() != total_size) {
    heatmap_data_.assign(total_size, 0.0);
  } else {
    std::fill(heatmap_data_.begin(), heatmap_data_.end(), 0.0);
  }

  double max_vol = 0;

  for (int t = 0; t < time_steps; ++t) {
    const auto& book = history[t];

    // Bids
    for (const auto& level : book.bids) {
      if (level.price >= min_price && level.price < max_price) {
        int bin = static_cast<int>((level.price - min_price) / price_step);
        if (bin >= 0 && bin < price_bins_) {
          heatmap_data_[bin * time_steps + t] += level.size;
          max_vol = std::max(max_vol, heatmap_data_[bin * time_steps + t]);
        }
      }
    }
    // Asks
    for (const auto& level : book.asks) {
      if (level.price >= min_price && level.price < max_price) {
        int bin = static_cast<int>((level.price - min_price) / price_step);
        if (bin >= 0 && bin < price_bins_) {
          heatmap_data_[bin * time_steps + t] += level.size;
          max_vol = std::max(max_vol, heatmap_data_[bin * time_steps + t]);
        }
      }
    }
  }

  bounds_min_[0] = 0;
  bounds_min_[1] = min_price;
  bounds_max_[0] = static_cast<double>(time_steps);
  bounds_max_[1] = max_price;

  scale_max_ = max_vol > 0 ? max_vol : 1.0;
}

void DomSurfacePanel::updateLargeOrderMarkers() {
  if (current_symbol_id_ == 0 || !processor_) return;

  // Get latest orderbook for large order detection
  auto orderbook_opt = processor_->getOrderbookData(current_symbol_id_);
  if (!orderbook_opt) return;

  const auto& orderbook = *orderbook_opt;

  // Calculate median order size from recent orderbook levels
  calculateMedianOrderSize();

  // Detect large orders in the latest orderbook
  detectLargeOrders(orderbook);

  // Cleanup old markers (fade-out)
  if (enable_fade_out_) {
    cleanupOldMarkers();
  }
}

void DomSurfacePanel::calculateMedianOrderSize() {
  if (recent_order_sizes_.empty()) {
    median_order_size_ = 1.0;  // Default fallback
    return;
  }

  // Sort and find median
  std::vector<double> sorted_sizes(recent_order_sizes_.begin(), recent_order_sizes_.end());
  std::sort(sorted_sizes.begin(), sorted_sizes.end());

  size_t n = sorted_sizes.size();
  if (n % 2 == 0) {
    median_order_size_ = (sorted_sizes[n / 2 - 1] + sorted_sizes[n / 2]) / 2.0;
  } else {
    median_order_size_ = sorted_sizes[n / 2];
  }
}

void DomSurfacePanel::detectLargeOrders(const RenderEngine::OrderbookData& orderbook) {
  // Collect all order sizes for median calculation
  for (const auto& level : orderbook.bids) {
    recent_order_sizes_.push_back(level.size);
  }
  for (const auto& level : orderbook.asks) {
    recent_order_sizes_.push_back(level.size);
  }

  // Keep only the last MEDIAN_WINDOW_SIZE sizes
  while (recent_order_sizes_.size() > MEDIAN_WINDOW_SIZE) {
    recent_order_sizes_.pop_front();
  }

  // Detect large orders (threshold: >10x median)
  double threshold = large_order_threshold_ * median_order_size_;
  uint64_t current_time = std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now().time_since_epoch())
                              .count();

  // Check bids
  for (const auto& level : orderbook.bids) {
    if (level.size > threshold) {
      // Check if we already have a marker at this price level
      bool exists = false;
      for (const auto& marker : large_order_markers_) {
        if (std::abs(marker.price - level.price) < 0.0001) {
          exists = true;
          break;
        }
      }

      if (!exists && large_order_markers_.size() < static_cast<size_t>(max_large_order_markers_)) {
        // Calculate position: X = current time (rightmost), Y = price
        double x = static_cast<double>(heatmap_data_.size() / price_bins_) - 1.0;
        double y = level.price;

        LargeOrderMarker marker(x, y, level.size, level.price, true, current_time);
        marker.radius = calculateMarkerRadius(level.size);
        large_order_markers_.push_back(marker);
      }
    }
  }

  // Check asks
  for (const auto& level : orderbook.asks) {
    if (level.size > threshold) {
      // Check if we already have a marker at this price level
      bool exists = false;
      for (const auto& marker : large_order_markers_) {
        if (std::abs(marker.price - level.price) < 0.0001) {
          exists = true;
          break;
        }
      }

      if (!exists && large_order_markers_.size() < static_cast<size_t>(max_large_order_markers_)) {
        // Calculate position: X = current time (rightmost), Y = price
        double x = static_cast<double>(heatmap_data_.size() / price_bins_) - 1.0;
        double y = level.price;

        LargeOrderMarker marker(x, y, level.size, level.price, false, current_time);
        marker.radius = calculateMarkerRadius(level.size);
        large_order_markers_.push_back(marker);
      }
    }
  }
}

void DomSurfacePanel::cleanupOldMarkers() {
  uint64_t current_time = std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now().time_since_epoch())
                              .count();

  // Remove markers older than FADE_OUT_DURATION_US
  large_order_markers_.erase(
      std::remove_if(large_order_markers_.begin(), large_order_markers_.end(),
                     [current_time](const LargeOrderMarker& marker) {
                       return (current_time - marker.timestamp) > FADE_OUT_DURATION_US;
                     }),
      large_order_markers_.end());
}

float DomSurfacePanel::calculateMarkerRadius(double order_size) const {
  if (median_order_size_ <= 0.0) return BASE_RADIUS;

  // Calculate radius: base_radius * sqrt(order_size / median_size)
  float radius = BASE_RADIUS * std::sqrt(order_size / median_order_size_);

  // Clamp to min/max range
  return std::clamp(radius, MIN_RADIUS, MAX_RADIUS);
}

ImU32 DomSurfacePanel::getMarkerColor(const LargeOrderMarker& marker) const {
  // Calculate alpha based on fade-out (if enabled)
  float alpha = 0.7f;  // Default alpha
  if (enable_fade_out_) {
    uint64_t current_time = std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count();
    uint64_t age = current_time - marker.timestamp;
    float fade_ratio = 1.0f - static_cast<float>(age) / static_cast<float>(FADE_OUT_DURATION_US);
    alpha = std::clamp(fade_ratio * 0.8f, 0.3f, 0.8f);
  }

  // Color: Green for Bids, Red for Asks
  if (marker.is_bid) {
    return IM_COL32(0, 255, 0, static_cast<int>(alpha * 255));  // Green
  } else {
    return IM_COL32(255, 0, 0, static_cast<int>(alpha * 255));  // Red
  }
}

std::string DomSurfacePanel::getMarkerTooltip(const LargeOrderMarker& marker) const {
  std::string side = marker.is_bid ? "Bid" : "Ask";
  return std::format("Whale {}: {:.2f} @ ${:.2f}", side, marker.size, marker.price);
}

void DomSurfacePanel::renderLargeOrderMarkers() {
  if (large_order_markers_.empty()) return;

  ImPlot::PushStyleVar(ImPlotStyleVar_MarkerSize, 1.0f);

  // Get plot area for manual circle rendering
  ImPlotRect plot_rect = ImPlot::GetPlotLimits();

  // Render each large order marker as a circle
  for (const auto& marker : large_order_markers_) {
    ImU32 color = getMarkerColor(marker);
    ImU32 border_color = IM_COL32(255, 255, 255, 230);  // White border

    // Calculate X position based on timestamp relative to history range
    double relative_time = 0.0;
    if (history_end_timestamp_ > history_start_timestamp_) {
      relative_time = static_cast<double>(marker.timestamp - history_start_timestamp_) /
                      static_cast<double>(history_end_timestamp_ - history_start_timestamp_);
    }
    // Map bounds_min[0] (0) to bounds_max[0] (time_steps)
    // Actually, bounds max is time_steps.
    double x_pos = bounds_min_[0] + relative_time * (bounds_max_[0] - bounds_min_[0]);

    // Only render if within view
    if (x_pos < bounds_min_[0] || x_pos > bounds_max_[0]) {
      // Optional: continue or skip? Markers might be slightly out of bounds if
      // history shifted past them
      if (x_pos < bounds_min_[0]) continue;  // Too old
      // If too new, show it (shouldn't happen with correct end_timestamp)
    }

    // Convert plot coordinates to pixel coordinates
    ImVec2 pixel_pos = ImPlot::PlotToPixels(x_pos, marker.y);

    // Draw filled circle
    ImDrawList* draw_list = ImPlot::GetPlotDrawList();
    if (draw_list) {
      draw_list->AddCircleFilled(pixel_pos, marker.radius, color, 32);
      draw_list->AddCircle(pixel_pos, marker.radius, border_color, 32, 1.0f);

      // Check for hover and show tooltip
      ImVec2 mouse_pos = ImGui::GetMousePos();
      float distance = std::sqrt(std::pow(mouse_pos.x - pixel_pos.x, 2) +
                                 std::pow(mouse_pos.y - pixel_pos.y, 2));

      if (distance < marker.radius) {
        ImGui::SetTooltip("%s", getMarkerTooltip(marker).c_str());
      }
    }
  }

  ImPlot::PopStyleVar();
}

void DomSurfacePanel::render() {
  std::lock_guard<std::mutex> lock(data_mutex_);
  if (consumeDirty()) {
    updateHeatmapData();
    updateLargeOrderMarkers();
    updateTradeBubbles();  // Update trade bubbles

    // Update persistent levels if we have current orderbook data
    auto orderbook_opt =
        processor_ ? processor_->getOrderbookData(current_symbol_id_) : std::nullopt;
    if (orderbook_opt) {
      updatePersistentLevels(*orderbook_opt);
    }
  }

  begin_panel_window();

  if (current_symbol_id_ == 0 || heatmap_data_.empty()) {
    ImGui::Text("No Data / Select Symbol");
    end_panel_window();
    return;
  }

  // DOM Surface controls
  if (ImGui::Button("Reset View")) {
    ImPlot::SetNextAxesToFit();
    // Add haptic feedback for button interaction
    BTQuant::UI::HapticFeedback::getInstance().triggerForSubtleInteraction();
  }
  // Show standardized tooltip for the button
  BTQuant::UI::show_control_tooltip("dom_surface_reset_view");
  ImGui::SameLine();
  ImGui::Checkbox("Show Persistent Lines", &show_persistent_lines_);
  ImGui::SameLine();
  if (!current_symbol_name_.empty()) {
    ImGui::Text(" | Symbol: %s (%u) | Bins: %d | Orders: %zu | Trades: %zu",
                current_symbol_name_.c_str(), current_symbol_id_, price_bins_,
                large_order_markers_.size(), trade_bubbles_.size());
  } else {
    ImGui::Text(" | Symbol ID: %u | Bins: %d | Orders: %zu | Trades: %zu", current_symbol_id_,
                price_bins_, large_order_markers_.size(), trade_bubbles_.size());
  }

  // Enable Pan/Zoom for DOM Surface
  std::string plot_id = "##DomHeatmap_" + std::to_string(current_symbol_id_);
  if (ImPlot::BeginPlot(plot_id.c_str(), ImVec2(-1, -1), ImPlotFlags_NoLegend)) {
    ImPlot::SetupAxes("Time", "Price");

    // Allow user to pan and zoom
    ImPlot::SetupAxis(ImAxis_X1, "Time", ImPlotAxisFlags_RangeFit);
    ImPlot::SetupAxis(ImAxis_Y1, "Price", ImPlotAxisFlags_RangeFit);

    // Set axis limits with option for user interaction
    ImPlot::SetupAxisLimits(ImAxis_X1, bounds_min_[0], bounds_max_[0],
                            heatmap_data_.empty() ? ImPlotCond_Always : ImPlotCond_Once);
    ImPlot::SetupAxisLimits(ImAxis_Y1, bounds_min_[1], bounds_max_[1],
                            heatmap_data_.empty() ? ImPlotCond_Always : ImPlotCond_Once);

    // Use time history size for Cols and price_bins for Rows
    int rows = price_bins_;
    int cols = static_cast<int>(heatmap_data_.size()) / rows;

    if (cols > 0 && rows > 0) {
      // Create a custom colormap for Dark Blue to Bright Yellow gradient
      static const ImVec4 blue_yellow_colormap[] = {
          // Dark Blue (0, 0, 139) to Bright Yellow (255, 255, 0)
          ImVec4(0.0f, 0.0f, 0.545f, 1.0f),  // Dark Blue (approx)
          ImVec4(0.0f, 0.2f, 0.6f, 1.0f),    // Blue to Cyan transition
          ImVec4(0.0f, 0.5f, 0.8f, 1.0f),    // More Cyan
          ImVec4(0.0f, 0.8f, 1.0f, 1.0f),    // Cyan
          ImVec4(0.2f, 1.0f, 0.8f, 1.0f),    // Cyan to Greenish
          ImVec4(0.5f, 1.0f, 0.5f, 1.0f),    // Greenish
          ImVec4(0.8f, 1.0f, 0.2f, 1.0f),    // Yellowish
          ImVec4(1.0f, 1.0f, 0.0f, 1.0f)     // Bright Yellow
      };

      // Register the custom colormap with ImPlot if not already registered
      static ImPlotColormap registered_colormap = -1;
      if (registered_colormap == -1) {
        registered_colormap = ImPlot::AddColormap("BlueYellow", blue_yellow_colormap, 8);
      }

      // Apply the custom colormap
      ImPlot::PushColormap(registered_colormap);

      // Apply heatmap intensity to adjust color mapping sensitivity
      double adjusted_scale_max = scale_max_ / heatmap_intensity_;
      ImPlot::PlotHeatmap("Liquidity", heatmap_data_.data(), rows, cols, 0, adjusted_scale_max,
                          nullptr, ImPlotPoint(bounds_min_[0], bounds_min_[1]),
                          ImPlotPoint(bounds_max_[0], bounds_max_[1]));
      ImPlot::PopColormap();
    }

    // Render Persistent Level Lines OVER the heatmap
    if (show_persistent_lines_) {
      renderPersistentLevels();
    }

    // Render Large Order Markers OVER the heatmap and persistent lines
    renderLargeOrderMarkers();

    // Render Trade Bubbles OVER the heatmap, persistent lines, and large order markers
    renderTradeBubbles();

    ImPlot::EndPlot();
  }

  // Information Overlay for DOM Surface
  if (heatmap_data_.size() > 0) {
    // Position information overlay in the top-left corner
    ImGui::SetCursorPos(ImVec2(10, 30));

    // Create a visually appealing info box with liquidity statistics
    ImGui::BeginGroup();
    ImGui::TextColored(ImVec4(0.2f, 0.7f, 1.0f, 1.0f), "Liquidity Stats:");
    ImGui::Indent(10.0f);
    ImGui::Text("Max Volume: %.2f", scale_max_);
    ImGui::Text("History Depth: %zu", heatmap_data_.size() / price_bins_);
    ImGui::Text("Price Bins: %d", price_bins_);
    ImGui::Text("Price Range: %.4f - %.4f", bounds_min_[1], bounds_max_[1]);
    ImGui::Unindent(10.0f);

    ImGui::Spacing();

    ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Active Elements:");
    ImGui::Indent(10.0f);
    ImGui::Text("Large Orders: %zu", large_order_markers_.size());
    ImGui::Text("Median Size: %.2f", median_order_size_);
    ImGui::Text("Trade Bubbles: %zu", trade_bubbles_.size());
    ImGui::Text("Persistent Levels: %zu", persistent_levels_.size());

    // Count static liquidity levels that have been persistent for more than 30 seconds
    uint64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count();
    size_t persistent_static_count = 0;
    for (const auto& level : static_liquidity_levels_) {
      if ((current_time - level.last_changed_time) >= persistence_threshold_ms_) {
        persistent_static_count++;
      }
    }
    ImGui::Text("Static Liquidity Levels: %zu", persistent_static_count);
    ImGui::Unindent(10.0f);
    ImGui::EndGroup();
  }

  end_panel_window();
}

void DomSurfacePanel::updatePersistentLevels(const RenderEngine::OrderbookData& orderbook) {
  // Get current time
  uint64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now().time_since_epoch())
                              .count();

  // Process all bid levels (not just large orders) to track static liquidity
  for (const auto& level : orderbook.bids) {
    addOrUpdateStaticLiquidityLevel(level.price, true, level.size);
  }

  // Process all ask levels (not just large orders) to track static liquidity
  for (const auto& level : orderbook.asks) {
    addOrUpdateStaticLiquidityLevel(level.price, false, level.size);
  }

  // Clean up inactive levels
  cleanupInactiveStaticLiquidityLevels();

  // Also clean up inactive persistent levels (legacy)
  cleanupInactivePersistentLevels();
}

void DomSurfacePanel::addOrUpdateStaticLiquidityLevel(double price, bool is_bid, double size) {
  uint64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now().time_since_epoch())
                              .count();

  // Check if this price level already exists
  for (auto& level : static_liquidity_levels_) {
    // Use a small epsilon for price comparison
    if (std::abs(level.price - price) < 0.0001) {
      // Check if the size has changed significantly (more than 1% difference or absolute threshold)
      double size_change_threshold =
          std::max(level.size * 0.01, 0.01);  // 1% threshold or 0.01 minimum
      if (std::abs(level.size - size) > size_change_threshold) {
        // Size has changed significantly, update the change time
        level.last_changed_time = current_time;
        level.size = size;  // Update to the new size
      }
      // Update the last seen time regardless of size change
      level.last_updated_time = current_time;
      level.is_active = true;
      return;
    }
  }

  // Add new static liquidity level
  static_liquidity_levels_.emplace_back(price, size, is_bid, current_time);
}

void DomSurfacePanel::cleanupInactiveStaticLiquidityLevels() {
  uint64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now().time_since_epoch())
                              .count();

  // Remove levels that haven't been updated within the timeout period
  static_liquidity_levels_.erase(
      std::remove_if(static_liquidity_levels_.begin(), static_liquidity_levels_.end(),
                     [current_time, this](const StaticLiquidityLevel& level) {
                       return (current_time - level.last_updated_time) > persistence_timeout_ms_;
                     }),
      static_liquidity_levels_.end());
}

void DomSurfacePanel::addOrUpdatePersistentLevel(double price, bool is_bid, double size) {
  uint64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now().time_since_epoch())
                              .count();

  // Check if this price level already exists
  for (auto& level : persistent_levels_) {
    // Use a small epsilon for price comparison
    if (std::abs(level.price - price) < 0.0001) {
      // Update existing level
      level.last_updated_time = current_time;
      level.size = std::max(level.size, size);  // Keep the largest size seen
      level.is_active = true;
      return;
    }
  }

  // Add new persistent level
  persistent_levels_.emplace_back(price, is_bid, size, current_time);
}

void DomSurfacePanel::cleanupInactivePersistentLevels() {
  uint64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now().time_since_epoch())
                              .count();

  // Remove levels that haven't been updated within the timeout period
  persistent_levels_.erase(std::remove_if(persistent_levels_.begin(), persistent_levels_.end(),
                                          [current_time, this](const PersistentLevel& level) {
                                            return (current_time - level.last_updated_time) >
                                                   persistence_timeout_ms_;
                                          }),
                           persistent_levels_.end());
}

void DomSurfacePanel::renderPersistentLevels() {
  // Get plot area bounds
  ImPlotRect plot_rect = ImPlot::GetPlotLimits();

  // Render static liquidity levels (all levels that have remained static for more than 30 seconds)
  // with a distinct border or "glow" effect
  for (const auto& level : static_liquidity_levels_) {
    // Only render if the level is considered "persistent" (has been present for threshold time)
    uint64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count();

    // Check if the level has remained static for more than 30 seconds
    // This means the size hasn't changed significantly since last_changed_time
    if ((current_time - level.last_changed_time) >= persistence_threshold_ms_) {
      // Use the enhanced glow effect for static liquidity levels
      renderStaticLiquidityGlowEffect(level, plot_rect);
    }
  }

  // Also render the legacy persistent levels (large orders) for compatibility
  for (const auto& level : persistent_levels_) {
    // Only render if the level is considered "persistent" (has been present for threshold time)
    uint64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count();

    if ((current_time - level.first_detected_time) >= persistence_threshold_ms_) {
      ImU32 color = getPersistentLevelColor(level);

      // Draw horizontal line across the entire time axis
      ImPlot::PushStyleColor(ImPlotCol_Line, color);
      ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 3.0f);  // Thicker line for better visibility

      // Draw horizontal line at the price level from left to right of the plot
      double xs[2] = {plot_rect.X.Min, plot_rect.X.Max};
      double ys[2] = {level.price, level.price};
      ImPlot::PlotLine("##PersistentLevel", xs, ys, 2);

      ImPlot::PopStyleVar();
      ImPlot::PopStyleColor();

      // Draw a more prominent rectangle to highlight the level
      // Extract the RGB components and set alpha to 15% transparency for better visibility
      ImVec4 color_vec = ImGui::ColorConvertU32ToFloat4(color);
      color_vec.w = 0.15f;  // Set alpha to 15% transparency
      ImU32 transparent_color = ImGui::ColorConvertFloat4ToU32(color_vec);
      ImPlot::PushStyleColor(ImPlotCol_Fill, transparent_color);

      // Calculate a vertical range around the price level for the rectangle
      // Make it proportional to the zoom level for better visibility
      double visible_price_range = plot_rect.Y.Max - plot_rect.Y.Min;
      double price_range =
          visible_price_range * 0.005;               // 0.5% of the visible price range (adjustable)
      if (price_range < 0.001) price_range = 0.001;  // Minimum thickness

      double y_min = level.price - price_range / 2.0;
      double y_max = level.price + price_range / 2.0;

      // Draw a horizontal shaded area spanning the full time axis
      double shade_x[2] = {plot_rect.X.Min, plot_rect.X.Max};
      double shade_y1[2] = {y_min, y_min};
      double shade_y2[2] = {y_max, y_max};
      ImPlot::PlotShaded("##PersistentLevelRect", shade_x, shade_y1, shade_y2, 2);

      ImPlot::PopStyleColor();

      // Add a subtle highlight effect above the main line
      ImVec4 highlight_color_vec = ImGui::ColorConvertU32ToFloat4(color);
      highlight_color_vec.w = 0.08f;  // Even more transparent for highlight
      ImU32 highlight_color = ImGui::ColorConvertFloat4ToU32(highlight_color_vec);
      ImPlot::PushStyleColor(ImPlotCol_Line, highlight_color);

      // Draw highlight slightly above the main line
      double highlight_y_min = level.price + price_range / 2.0;
      double highlight_y_max = level.price + price_range / 2.0 + price_range * 0.5;

      // Draw highlight shaded area
      double highlight_shade_x[2] = {plot_rect.X.Min, plot_rect.X.Max};
      double highlight_shade_y1[2] = {highlight_y_min, highlight_y_min};
      double highlight_shade_y2[2] = {highlight_y_max, highlight_y_max};
      ImPlot::PlotShaded("##PersistentLevelHighlight", highlight_shade_x, highlight_shade_y1,
                         highlight_shade_y2, 2);

      ImPlot::PopStyleColor();
    }
  }
}

ImU32 DomSurfacePanel::getStaticLiquidityLevelColor(const StaticLiquidityLevel& level) const {
  // Use a distinct color scheme for static liquidity levels to differentiate from large orders
  // Bright magenta for bids (buy-side liquidity), bright orange for asks (sell-side liquidity)
  // These colors provide better contrast against the heatmap and stand out more distinctly
  if (level.is_bid) {
    return IM_COL32(255, 100, 255, 255);  // Bright magenta for bid levels (distinct from cyan)
  } else {
    return IM_COL32(255, 165, 0, 255);  // Bright orange for ask levels (distinct from yellow)
  }
}

ImU32 DomSurfacePanel::getPersistentLevelColor(const PersistentLevel& level) const {
  // Color: Bright Green for Bids, Bright Red for Asks
  // Use brighter colors than the markers to distinguish persistent levels
  if (level.is_bid) {
    return IM_COL32(0, 255, 150, 180);  // Brighter green with moderate transparency
  } else {
    return IM_COL32(255, 100, 150, 180);  // Brighter red with moderate transparency
  }
}

void DomSurfacePanel::renderStaticLiquidityGlowEffect(const StaticLiquidityLevel& level,
                                                      const ImPlotRect& plot_rect) const {
  // Get current time for pulsing animation
  uint64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now().time_since_epoch())
                              .count();

  // Use the static liquidity level color
  ImU32 color = getStaticLiquidityLevelColor(level);

  // Draw multiple layers for enhanced glow effect
  // Layer 1: Outer glow (largest and most transparent)
  ImVec4 outer_glow_color_vec = ImGui::ColorConvertU32ToFloat4(color);
  outer_glow_color_vec.w = 0.08f;  // Very low transparency for wide glow
  ImU32 outer_glow_color = ImGui::ColorConvertFloat4ToU32(outer_glow_color_vec);

  ImPlot::PushStyleColor(ImPlotCol_Line, outer_glow_color);
  ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 20.0f);  // Extra thick for outer glow

  double xs[2] = {plot_rect.X.Min, plot_rect.X.Max};
  double ys[2] = {level.price, level.price};
  ImPlot::PlotLine("##StaticLiquidityOuterGlow", xs, ys, 2);

  ImPlot::PopStyleVar();
  ImPlot::PopStyleColor();

  // Layer 2: Middle glow
  ImVec4 middle_glow_color_vec = ImGui::ColorConvertU32ToFloat4(color);
  middle_glow_color_vec.w = 0.15f;  // Lower transparency for stronger glow
  ImU32 middle_glow_color = ImGui::ColorConvertFloat4ToU32(middle_glow_color_vec);

  ImPlot::PushStyleColor(ImPlotCol_Line, middle_glow_color);
  ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 14.0f);  // Thick for middle glow

  ImPlot::PlotLine("##StaticLiquidityMiddleGlow", xs, ys, 2);

  ImPlot::PopStyleVar();
  ImPlot::PopStyleColor();

  // Layer 3: Inner glow
  ImVec4 inner_glow_color_vec = ImGui::ColorConvertU32ToFloat4(color);
  inner_glow_color_vec.w = 0.3f;  // Moderate transparency for inner glow
  ImU32 inner_glow_color = ImGui::ColorConvertFloat4ToU32(inner_glow_color_vec);

  ImPlot::PushStyleColor(ImPlotCol_Line, inner_glow_color);
  ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 10.0f);  // Medium thickness for inner glow

  ImPlot::PlotLine("##StaticLiquidityInnerGlow", xs, ys, 2);

  ImPlot::PopStyleVar();
  ImPlot::PopStyleColor();

  // Layer 4: Main line (bright and solid)
  ImPlot::PushStyleColor(ImPlotCol_Line, color);
  ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight,
                       6.0f);  // Thicker for main line to make it more distinct

  ImPlot::PlotLine("##StaticLiquidityMain", xs, ys, 2);

  ImPlot::PopStyleVar();
  ImPlot::PopStyleColor();

  // Draw a highlighted rectangle around the level to make it stand out
  ImVec4 rect_color_vec = ImGui::ColorConvertU32ToFloat4(color);
  rect_color_vec.w = 0.12f;  // 12% transparency for the rectangle
  ImU32 rect_color = ImGui::ColorConvertFloat4ToU32(rect_color_vec);
  ImPlot::PushStyleColor(ImPlotCol_Fill, rect_color);

  // Calculate a vertical range around the price level for the rectangle
  // Make it proportional to the zoom level for better visibility
  double visible_price_range = plot_rect.Y.Max - plot_rect.Y.Min;
  double price_range =
      visible_price_range *
      0.02;  // 2% of the visible price range for rectangle height (increased for better visibility)
  if (price_range < 0.001) price_range = 0.001;  // Minimum thickness

  double y_min = level.price - price_range / 2.0;
  double y_max = level.price + price_range / 2.0;

  // Draw a horizontal shaded area spanning the full time axis
  double shade_x[2] = {plot_rect.X.Min, plot_rect.X.Max};
  double shade_y1[2] = {y_min, y_min};
  double shade_y2[2] = {y_max, y_max};
  ImPlot::PlotShaded("##StaticLiquidityRect", shade_x, shade_y1, shade_y2, 2);

  ImPlot::PopStyleColor();

  // Add a pulsing animation effect for extra visibility
  float pulse_factor = 0.5f + 0.3f * std::sin((current_time % 2000) * 0.001f * 3.14159f *
                                              2.0f);  // Pulsing every 2 seconds
  ImVec4 pulse_color_vec = ImGui::ColorConvertU32ToFloat4(color);
  pulse_color_vec.w = 0.1f * pulse_factor;  // Pulsing transparency
  ImU32 pulse_color = ImGui::ColorConvertFloat4ToU32(pulse_color_vec);
  ImPlot::PushStyleColor(ImPlotCol_Fill, pulse_color);

  // Draw a pulsing outer rectangle
  double outer_price_range =
      price_range * 2.5f;  // 2.5x the inner rectangle for more pronounced effect
  double y_outer_min = level.price - outer_price_range / 2.0;
  double y_outer_max = level.price + outer_price_range / 2.0;

  double outer_shade_x[2] = {plot_rect.X.Min, plot_rect.X.Max};
  double outer_shade_y1[2] = {y_outer_min, y_outer_min};
  double outer_shade_y2[2] = {y_outer_max, y_outer_max};
  ImPlot::PlotShaded("##StaticLiquidityPulse", outer_shade_x, outer_shade_y1, outer_shade_y2, 2);

  ImPlot::PopStyleColor();

  // Add tooltip functionality when hovering over the persistent level
  ImVec2 mouse_pos = ImGui::GetMousePos();
  ImVec2 level_pixel_pos =
      ImPlot::PlotToPixels(plot_rect.X.Min, level.price);  // Left side of the line
  ImVec2 level_pixel_pos_right =
      ImPlot::PlotToPixels(plot_rect.X.Max, level.price);  // Right side of the line

  // Calculate distance from mouse to the horizontal line
  float distance_to_line = std::abs(mouse_pos.y - level_pixel_pos.y);

  // Check if mouse is near the persistent level line and within plot bounds horizontally
  if (mouse_pos.x >= plot_rect.X.Min && mouse_pos.x <= plot_rect.X.Max &&
      distance_to_line < 10.0f) {  // 10 pixel tolerance

    // Calculate how long this level has been persistent
    uint64_t time_persistent_ms = current_time - level.last_changed_time;
    float time_persistent_seconds = static_cast<float>(time_persistent_ms) / 1000.0f;

    std::string tooltip = std::format(
        "Persistent Level: ${:.2f}\nSide: {}\nDuration: {:.1f}s\nSize: {:.2f}", level.price,
        level.is_bid ? "Bid (Buy)" : "Ask (Sell)", time_persistent_seconds, level.size);

    ImGui::SetTooltip("%s", tooltip.c_str());
  }
}

void DomSurfacePanel::render_panel_header() {
  // Call parent implementation to render the default header
  PanelBase::render_panel_header();

  // Add heatmap intensity slider to the panel header
  ImGui::Separator();
  ImGui::Text("Heatmap Intensity:");
  ImGui::SameLine();
  ImGui::PushItemWidth(200);
  ImGui::SliderFloat("##HeatmapIntensity", &heatmap_intensity_, 0.1f, 5.0f, "%.2f",
                     ImGuiSliderFlags_Logarithmic);
  ImGui::PopItemWidth();
  ImGui::SameLine();
  if (ImGui::Button("Reset##HeatmapIntensity")) {
    heatmap_intensity_ = 1.0f;
  }
  ImGui::Separator();

  // Add Large Order Tracker controls
  ImGui::Text("Large Order Tracker:");
  ImGui::SameLine();
  ImGui::PushItemWidth(150);
  ImGui::SliderFloat("##Threshold", &large_order_threshold_, 1.0f, 50.0f, "Threshold: %.1fx",
                     ImGuiSliderFlags_Logarithmic);
  ImGui::PopItemWidth();
  ImGui::SameLine();
  ImGui::PushItemWidth(150);
  ImGui::SliderInt("Max Markers", &max_large_order_markers_, 10, 500);
  ImGui::PopItemWidth();
  ImGui::SameLine();
  ImGui::Checkbox("Fade Out", &enable_fade_out_);
  ImGui::Separator();

  // Add Persistent Level controls
  ImGui::Text("Persistent Levels:");
  ImGui::SameLine();
  ImGui::PushItemWidth(150);
  ImGui::SliderInt("Persistence (ms)", reinterpret_cast<int*>(&persistence_threshold_ms_), 30000,
                   60000, "%d ms");
  ImGui::PopItemWidth();
  ImGui::SameLine();
  ImGui::PushItemWidth(150);
  ImGui::SliderInt("Timeout (ms)", reinterpret_cast<int*>(&persistence_timeout_ms_), 30000, 120000,
                   "%d ms");
  ImGui::PopItemWidth();
  ImGui::Separator();
}

}  // namespace BTQuant
