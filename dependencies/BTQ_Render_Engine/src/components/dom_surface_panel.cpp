#include "components/dom_surface_panel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>

#include <imgui.h>

namespace BTQuant {

DomSurfacePanel::DomSurfacePanel(std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(PanelConfig{.title = "DOM Surface", .type = PanelType::HEATMAP}),
      processor_(processor) {
  // Vulkan compute removed - using CPU-based heatmap rendering
}

DomSurfacePanel::~DomSurfacePanel() {
  if (subscription_id_ > 0 && processor_) {
    processor_->unsubscribe(subscription_id_);
  }
}

void DomSurfacePanel::setSymbol(uint32_t symbol_id) {
  if (current_symbol_id_ == symbol_id) return;

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

void DomSurfacePanel::onDataUpdate(uint32_t symbol_id, RenderEngine::NotificationType type) {
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
        if (bubble.timestamp == trade.timestamp && 
            std::abs(bubble.y - trade.price) < 0.0001 && 
            std::abs(bubble.volume - trade.size) < 0.0001) {
          exists = true;
          break;
        }
      }

      if (!exists) {
        TradeBubble bubble(x_pos, trade.price, trade.size, trade.price, trade.is_buy, trade.timestamp);
        bubble.radius = calculateBubbleRadius(trade.size);
        trade_bubbles_.push_back(bubble);
      }
    }
  }

  // Clean up bubbles that are outside the current view range to prevent accumulation
  cleanupOldTradeBubbles();
}

void DomSurfacePanel::cleanupOldTradeBubbles() {
  // Remove bubbles that are outside the current view range
  // This helps keep the vector size manageable
  
  trade_bubbles_.erase(
      std::remove_if(trade_bubbles_.begin(), trade_bubbles_.end(),
                     [this](const TradeBubble& bubble) {
                       // Remove if outside the current view bounds by a margin
                       double margin = (bounds_max_[0] - bounds_min_[0]) * 0.1; // 10% margin
                       return (bubble.x < bounds_min_[0] - margin || bubble.x > bounds_max_[0] + margin);
                     }),
      trade_bubbles_.end());
}

float DomSurfacePanel::calculateBubbleRadius(double volume) const {
  if (max_trade_volume_ <= 0.0) return 5.0f;  // Default radius
  
  // Calculate radius: base_radius * sqrt(volume / max_volume) to make differences more visible
  float base_radius = 8.0f;  // Base radius for smallest trades
  float calculated_radius = base_radius * std::sqrt(volume / max_trade_volume_) * 3.0f;  // Amplify effect
  
  // Clamp to reasonable range
  return std::clamp(calculated_radius, 3.0f, 20.0f);
}

ImU32 DomSurfacePanel::getBubbleColor(const TradeBubble& bubble) const {
  // Color: Green for Buys, Red for Sells
  if (bubble.is_buy) {
    return IM_COL32(0, 255, 0, 180);  // Green with transparency
  } else {
    return IM_COL32(255, 0, 0, 180);  // Red with transparency
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
        std::string tooltip = std::format("Trade: {} {:.2f} @ ${:.2f}", 
                                         bubble.is_buy ? "Buy" : "Sell", 
                                         bubble.volume, bubble.price);
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
  // Note: PopStyleVar removed - no corresponding PushStyleVar in this function
}

void DomSurfacePanel::render() {
  if (consumeDirty()) {
    updateHeatmapData();
    updateLargeOrderMarkers();
    updateTradeBubbles();  // Update trade bubbles

    // Update persistent levels if we have current orderbook data
    auto orderbook_opt = processor_ ? processor_->getOrderbookData(current_symbol_id_) : std::nullopt;
    if (orderbook_opt) {
      updatePersistentLevels(*orderbook_opt);
    }
  }

  begin_panel_window();

  if (current_symbol_id_ == 0) {
    ImGui::Text("No Data / Select Symbol");
    end_panel_window();
    return;
  }

  // DOM Surface controls
  ImGui::Checkbox("DOM Ladder", &show_dom_ladder_);
  ImGui::SameLine();
  ImGui::Checkbox("Show Heatmap BG", &show_heatmap_overlay_);
  ImGui::SameLine();
  ImGui::Checkbox("Show Persistent Lines", &show_persistent_lines_);
  if (show_dom_ladder_) {
    ImGui::SameLine();
    ImGui::PushItemWidth(120);
    ImGui::SliderInt("Rows", &ladder_visible_rows_, 5, 60);
    ImGui::PopItemWidth();
  }
  ImGui::SameLine();
  ImGui::Text(" | Symbol: %u | Orders: %zu | Trades: %zu", current_symbol_id_,
              large_order_markers_.size(), trade_bubbles_.size());

  // If ladder mode, render the 5-column DOM ladder and return
  if (show_dom_ladder_) {
    renderDOMLadder();
    end_panel_window();
    return;
  }

  if (ImGui::Button("Reset View")) {
    ImPlot::SetNextAxesToFit();
  }

  // Enable Pan/Zoom for DOM Surface (heatmap mode)
  std::string plot_id = "##DomHeatmap_" + std::to_string(current_symbol_id_);
  if (ImPlot::BeginPlot(plot_id.c_str(), ImVec2(-1, -1), ImPlotFlags_NoLegend)) {
    ImPlot::SetupAxes("Time", "Price");

    // Allow user to pan and zoom
    ImPlot::SetupAxis(ImAxis_X1, "Time", ImPlotAxisFlags_RangeFit);
    ImPlot::SetupAxis(ImAxis_Y1, "Price", ImPlotAxisFlags_RangeFit);

    // Set axis limits with option for user interaction
    ImPlot::SetupAxisLimits(ImAxis_X1, bounds_min_[0], bounds_max_[0],
                            ImPlotCond_Once);
    ImPlot::SetupAxisLimits(ImAxis_Y1, bounds_min_[1], bounds_max_[1],
                            ImPlotCond_Once);

    // Use Vulkan-accelerated heatmap texture if available
    // CPU-based heatmap rendering
    int rows = price_bins_;
    int cols = static_cast<int>(heatmap_data_.size()) / rows;

    if (cols > 0 && rows > 0) {
      ImPlot::PushColormap(ImPlotColormap_Viridis);
      // Apply heatmap intensity to adjust color mapping sensitivity
      double adjusted_scale_max = scale_max_ / heatmap_intensity_;
      ImPlot::PlotHeatmap("Liquidity", heatmap_data_.data(), rows, cols, 0, adjusted_scale_max, nullptr,
                          ImPlotPoint(bounds_min_[0], bounds_min_[1]),
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

    // Render Flush DOM Ruler OVER everything else - showing live orderbook at the right edge
    renderFlushDOMRuler();

    ImPlot::EndPlot();
  }

  // Debug Overlay for DOM troubleshooting
  if (heatmap_data_.size() > 0) {
    ImGui::SetCursorPos(ImVec2(10, 30));
    ImGui::TextColored(ImVec4(1, 1, 0, 1), "Debug: MaxVol=%.2f, Hist=%zu, Bins=%d", scale_max_,
                       heatmap_data_.size() / price_bins_, price_bins_);
    ImGui::Text("Bounds: Y=%.4f - %.4f", bounds_min_[1], bounds_max_[1]);
    ImGui::Text("Large Orders: %zu (Median: %.2f)", large_order_markers_.size(),
                median_order_size_);
    ImGui::Text("Trade Bubbles: %zu (MaxVol: %.2f)", trade_bubbles_.size(), max_trade_volume_);
    ImGui::Text("Persistent Levels: %zu", persistent_levels_.size());
  }

  end_panel_window();
}

// ============================================================
// DOM Ladder — 5-Column Price Ladder
// ============================================================
void DomSurfacePanel::renderDOMLadder() {
  if (!processor_ || current_symbol_id_ == 0) return;

  auto orderbook_opt = processor_->getOrderbookData(current_symbol_id_);
  if (!orderbook_opt || (orderbook_opt->bids.empty() && orderbook_opt->asks.empty())) {
    ImGui::Text("No orderbook data available");
    return;
  }

  const auto& book = *orderbook_opt;

  // Determine the tick size (minimum price increment) from the data
  double tick_size = 0.01;  // default fallback
  if (book.bids.size() >= 2) {
    tick_size = std::abs(book.bids[0].price - book.bids[1].price);
  } else if (book.asks.size() >= 2) {
    tick_size = std::abs(book.asks[0].price - book.asks[1].price);
  }
  if (tick_size <= 0.0) tick_size = 0.01;

  // Best bid / best ask / mid price
  double best_bid = book.bids.empty() ? 0.0 : book.bids.front().price;
  double best_ask = book.asks.empty() ? 0.0 : book.asks.front().price;
  double mid_price = (best_bid > 0.0 && best_ask > 0.0) ? (best_bid + best_ask) / 2.0
                     : best_bid > 0.0 ? best_bid
                     : best_ask;

  // Build lookup maps for size at each price
  std::map<int64_t, double> bid_map;  // tick_index -> size
  std::map<int64_t, double> ask_map;
  double max_bid_size = 0.0;
  double max_ask_size = 0.0;
  double total_bid_vol = 0.0;
  double total_ask_vol = 0.0;
  double all_sizes_sum = 0.0;
  int all_sizes_count = 0;

  for (const auto& lv : book.bids) {
    int64_t idx = static_cast<int64_t>(std::round(lv.price / tick_size));
    bid_map[idx] += lv.size;
    max_bid_size = std::max(max_bid_size, bid_map[idx]);
    total_bid_vol += lv.size;
    all_sizes_sum += lv.size;
    all_sizes_count++;
  }
  for (const auto& lv : book.asks) {
    int64_t idx = static_cast<int64_t>(std::round(lv.price / tick_size));
    ask_map[idx] += lv.size;
    max_ask_size = std::max(max_ask_size, ask_map[idx]);
    total_ask_vol += lv.size;
    all_sizes_sum += lv.size;
    all_sizes_count++;
  }

  double max_size = std::max(max_bid_size, max_ask_size);
  if (max_size <= 0.0) max_size = 1.0;

  double avg_size = (all_sizes_count > 0) ? (all_sizes_sum / all_sizes_count) : 1.0;
  double large_order_threshold = avg_size * 2.0;

  // Compute cumulative volumes for the cumulative columns
  // Asks: cumulative from top (best ask) downward (ascending price)
  // Bids: cumulative from top (best bid) downward (descending price)
  // We pre-compute for all tick indices we'll display.

  int64_t center_tick = static_cast<int64_t>(std::round(mid_price / tick_size));
  int half_rows = ladder_visible_rows_;

  // Pre-compute sorted tick indices for asks and bids
  // Asks go from center_tick+1 upward in price (ascending)
  // Bids go from center_tick downward in price (descending)

  // Cumulative ask volume: accumulate from highest displayed ask down to lowest ask
  std::map<int64_t, double> cum_ask_vol;
  double running = 0.0;
  for (int i = half_rows; i >= 0; --i) {
    int64_t tick_idx = center_tick + 1 + i;
    double sz = 0.0;
    auto it = ask_map.find(tick_idx);
    if (it != ask_map.end()) sz = it->second;
    running += sz;
    cum_ask_vol[tick_idx] = running;
  }

  // Cumulative bid volume: accumulate from lowest displayed bid up to best bid
  std::map<int64_t, double> cum_bid_vol;
  running = 0.0;
  for (int i = half_rows; i >= 0; --i) {
    int64_t tick_idx = center_tick - 1 - i;
    double sz = 0.0;
    auto it = bid_map.find(tick_idx);
    if (it != bid_map.end()) sz = it->second;
    running += sz;
    cum_bid_vol[tick_idx] = running;
  }

  // Colors
  const ImU32 col_bid_bar     = IM_COL32(0, 230, 102, 120);    // Neon Mint
  const ImU32 col_ask_bar     = IM_COL32(230, 25, 38, 120);    // Crimson
  const ImU32 col_spread_bg   = IM_COL32(60, 50, 20, 80);      // Amber tint
  const ImU32 col_large_order = IM_COL32(255, 200, 0, 40);      // Gold highlight
  const ImU32 col_price_text  = IM_COL32(200, 210, 220, 255);   // Light grey
  const ImU32 col_grid        = IM_COL32(30, 35, 40, 100);      // Grid lines
  const ImU32 col_bid_text    = IM_COL32(0, 230, 102, 255);
  const ImU32 col_ask_text    = IM_COL32(230, 25, 38, 255);
  const ImU32 col_cum_text    = IM_COL32(150, 160, 170, 200);
  const ImU32 col_spread_text = IM_COL32(255, 200, 80, 255);

  ImDrawList* dl = ImGui::GetWindowDrawList();
  ImVec2 cursor_start = ImGui::GetCursorScreenPos();
  float panel_width = ImGui::GetContentRegionAvail().x;
  float row_height = ImGui::GetTextLineHeightWithSpacing();

  // Column widths: SellVol | AskSize | Price | BidSize | BuyVol
  float col_price_w = panel_width * 0.22f;
  float col_size_w  = panel_width * 0.20f;
  float col_vol_w   = panel_width * 0.19f;

  float col_x[5];
  col_x[0] = cursor_start.x;                                    // Sell Vol (cumulative asks)
  col_x[1] = col_x[0] + col_vol_w;                              // Ask Size
  col_x[2] = col_x[1] + col_size_w;                             // Price (center)
  col_x[3] = col_x[2] + col_price_w;                            // Bid Size
  col_x[4] = col_x[3] + col_size_w;                             // Buy Vol (cumulative bids)

  // Header row
  float y = cursor_start.y;
  auto draw_centered_text = [&](float x_left, float width, const char* text, ImU32 color) {
    float tw = ImGui::CalcTextSize(text).x;
    dl->AddText(ImVec2(x_left + (width - tw) * 0.5f, y), color, text);
  };

  draw_centered_text(col_x[0], col_vol_w,   "Sell Vol",  col_ask_text);
  draw_centered_text(col_x[1], col_size_w,  "Ask",       col_ask_text);
  draw_centered_text(col_x[2], col_price_w, "Price",     col_price_text);
  draw_centered_text(col_x[3], col_size_w,  "Bid",       col_bid_text);
  draw_centered_text(col_x[4], col_vol_w,   "Buy Vol",   col_bid_text);
  y += row_height;

  // Grid line below header
  dl->AddLine(ImVec2(cursor_start.x, y), ImVec2(cursor_start.x + panel_width, y), col_grid, 1.0f);

  // Build all rows: from top (highest ask) to bottom (lowest bid)
  // Row order: asks descending from (center + half_rows) down to (center + 1),
  //            spread row at center,
  //            bids descending from center down to (center - half_rows)

  struct LadderRow {
    int64_t tick_idx;
    bool is_ask;
    bool is_spread;
    double size;
    double cum_vol;
  };

  std::vector<LadderRow> rows;
  // Asks: from highest to lowest (display top to bottom = high to low)
  for (int i = half_rows; i >= 1; --i) {
    int64_t ti = center_tick + i;
    double sz = 0.0;
    auto it = ask_map.find(ti);
    if (it != ask_map.end()) sz = it->second;
    double cv = 0.0;
    auto cit = cum_ask_vol.find(ti);
    if (cit != cum_ask_vol.end()) cv = cit->second;
    rows.push_back({ti, true, false, sz, cv});
  }

  // Spread row
  rows.push_back({center_tick, false, true, 0.0, 0.0});

  // Bids: from highest to lowest
  for (int i = 0; i < half_rows; ++i) {
    int64_t ti = center_tick - i;
    double sz = 0.0;
    auto it = bid_map.find(ti);
    if (it != bid_map.end()) sz = it->second;
    double cv = 0.0;
    auto cit = cum_bid_vol.find(ti);
    if (cit != cum_bid_vol.end()) cv = cit->second;
    rows.push_back({ti, false, false, sz, cv});
  }

  // Reserve space for the ladder
  float total_height = static_cast<float>(rows.size()) * row_height + row_height;  // +1 for header
  ImGui::Dummy(ImVec2(panel_width, total_height));

  // Draw each row
  for (const auto& row : rows) {
    double price = row.tick_idx * tick_size;
    bool is_large = row.size > large_order_threshold;

    // Background for spread row
    if (row.is_spread) {
      dl->AddRectFilled(
          ImVec2(cursor_start.x, y),
          ImVec2(cursor_start.x + panel_width, y + row_height),
          col_spread_bg);
    }
    // Background for large orders
    else if (is_large) {
      dl->AddRectFilled(
          ImVec2(cursor_start.x, y),
          ImVec2(cursor_start.x + panel_width, y + row_height),
          col_large_order);
    }

    // Grid line at bottom of row
    dl->AddLine(
        ImVec2(cursor_start.x, y + row_height),
        ImVec2(cursor_start.x + panel_width, y + row_height),
        col_grid, 0.5f);

    // Format price
    char price_buf[32];
    // Use appropriate decimal places based on tick size
    int decimals = 2;
    if (tick_size < 0.0001) decimals = 8;
    else if (tick_size < 0.001) decimals = 6;
    else if (tick_size < 0.01) decimals = 4;
    else if (tick_size < 1.0) decimals = 2;
    else decimals = 0;
    snprintf(price_buf, sizeof(price_buf), "%.*f", decimals, price);

    if (row.is_spread) {
      // Spread row
      float tw = ImGui::CalcTextSize(price_buf).x;
      dl->AddText(ImVec2(col_x[2] + (col_price_w - tw) * 0.5f, y), col_spread_text, price_buf);

      // Show spread value
      char spread_buf[64];
      snprintf(spread_buf, sizeof(spread_buf), "--- %.2f (%.4f%%) ---",
               book.spread, book.spread_percent * 100.0);
      float sw = ImGui::CalcTextSize(spread_buf).x;
      dl->AddText(ImVec2(col_x[2] + (col_price_w - sw) * 0.5f, y + row_height * 0.0f),
                  col_spread_text, spread_buf);
    } else if (row.is_ask) {
      // Ask side: columns 0 (cum), 1 (size), 2 (price)
      // Cumulative volume bar (column 0) - right-aligned from price side
      double max_cum = cum_ask_vol.empty() ? 1.0 : cum_ask_vol.rbegin()->second;
      if (max_cum <= 0.0) max_cum = 1.0;
      float bar_ratio = static_cast<float>(row.cum_vol / max_cum);
      float bar_w = bar_ratio * col_vol_w;
      if (bar_w > 1.0f) {
        dl->AddRectFilled(
            ImVec2(col_x[0] + col_vol_w - bar_w, y + 1.0f),
            ImVec2(col_x[0] + col_vol_w, y + row_height - 1.0f),
            col_ask_bar);
      }

      // Cumulative volume text
      char cv_buf[32];
      snprintf(cv_buf, sizeof(cv_buf), "%.1f", row.cum_vol);
      dl->AddText(ImVec2(col_x[0] + 4.0f, y), col_cum_text, cv_buf);

      // Ask size bar (column 1) - right-aligned from price
      float size_ratio = static_cast<float>(row.size / max_size);
      float size_bar_w = size_ratio * col_size_w;
      if (size_bar_w > 1.0f) {
        dl->AddRectFilled(
            ImVec2(col_x[1] + col_size_w - size_bar_w, y + 1.0f),
            ImVec2(col_x[1] + col_size_w, y + row_height - 1.0f),
            col_ask_bar);
      }

      // Ask size text
      if (row.size > 0.0) {
        char sz_buf[32];
        snprintf(sz_buf, sizeof(sz_buf), "%.1f", row.size);
        dl->AddText(ImVec2(col_x[1] + 4.0f, y),
                    is_large ? col_spread_text : col_ask_text, sz_buf);
      }

      // Price text (column 2)
      float tw = ImGui::CalcTextSize(price_buf).x;
      dl->AddText(ImVec2(col_x[2] + (col_price_w - tw) * 0.5f, y), col_ask_text, price_buf);

    } else {
      // Bid side: columns 2 (price), 3 (size), 4 (cum)
      // Price text
      float tw = ImGui::CalcTextSize(price_buf).x;
      dl->AddText(ImVec2(col_x[2] + (col_price_w - tw) * 0.5f, y), col_bid_text, price_buf);

      // Bid size bar (column 3) - left-aligned from price
      float size_ratio = static_cast<float>(row.size / max_size);
      float size_bar_w = size_ratio * col_size_w;
      if (size_bar_w > 1.0f) {
        dl->AddRectFilled(
            ImVec2(col_x[3], y + 1.0f),
            ImVec2(col_x[3] + size_bar_w, y + row_height - 1.0f),
            col_bid_bar);
      }

      // Bid size text
      if (row.size > 0.0) {
        char sz_buf[32];
        snprintf(sz_buf, sizeof(sz_buf), "%.1f", row.size);
        float stw = ImGui::CalcTextSize(sz_buf).x;
        dl->AddText(ImVec2(col_x[3] + col_size_w - stw - 4.0f, y),
                    is_large ? col_spread_text : col_bid_text, sz_buf);
      }

      // Cumulative volume bar (column 4) - left-aligned
      double max_cum = cum_bid_vol.empty() ? 1.0 : cum_bid_vol.rbegin()->second;
      if (max_cum <= 0.0) max_cum = 1.0;
      float bar_ratio = static_cast<float>(row.cum_vol / max_cum);
      float bar_w = bar_ratio * col_vol_w;
      if (bar_w > 1.0f) {
        dl->AddRectFilled(
            ImVec2(col_x[4], y + 1.0f),
            ImVec2(col_x[4] + bar_w, y + row_height - 1.0f),
            col_bid_bar);
      }

      // Cumulative volume text
      char cv_buf[32];
      snprintf(cv_buf, sizeof(cv_buf), "%.1f", row.cum_vol);
      float cvw = ImGui::CalcTextSize(cv_buf).x;
      dl->AddText(ImVec2(col_x[4] + col_vol_w - cvw - 4.0f, y), col_cum_text, cv_buf);
    }

    y += row_height;
  }

  // Draw vertical separator lines between columns
  for (int c = 1; c < 5; ++c) {
    dl->AddLine(
        ImVec2(col_x[c], cursor_start.y),
        ImVec2(col_x[c], y),
        col_grid, 1.0f);
  }

  // CVD (Cumulative Volume Delta) at bottom
  y += 4.0f;
  double analytics_buy_vol = 0.0;
  double analytics_sell_vol = 0.0;
  auto analytics = processor_->getSymbolAnalytics(current_symbol_id_);
  analytics_buy_vol = analytics.buy_volume;
  analytics_sell_vol = analytics.sell_volume;
  double cvd = analytics_buy_vol - analytics_sell_vol;

  char cvd_buf[128];
  snprintf(cvd_buf, sizeof(cvd_buf), "CVD: %.1f  |  Buy: %.1f  Sell: %.1f  |  Imbalance: %.2f%%",
           cvd, analytics_buy_vol, analytics_sell_vol,
           (total_bid_vol + total_ask_vol > 0.0)
               ? ((total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)) * 100.0
               : 0.0);
  ImU32 cvd_color = cvd >= 0.0 ? col_bid_text : col_ask_text;
  dl->AddText(ImVec2(cursor_start.x, y), cvd_color, cvd_buf);
}

void DomSurfacePanel::updatePersistentLevels(const RenderEngine::OrderbookData& orderbook) {
  // Get current time
  uint64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now().time_since_epoch())
                              .count();

  // Process bids
  for (const auto& level : orderbook.bids) {
    // Check if this level has a large order (using the same threshold as markers)
    double threshold = large_order_threshold_ * median_order_size_;
    if (level.size > threshold) {
      addOrUpdatePersistentLevel(level.price, true, level.size);
    }
  }

  // Process asks
  for (const auto& level : orderbook.asks) {
    // Check if this level has a large order (using the same threshold as markers)
    double threshold = large_order_threshold_ * median_order_size_;
    if (level.size > threshold) {
      addOrUpdatePersistentLevel(level.price, false, level.size);
    }
  }

  // Clean up inactive levels
  cleanupInactivePersistentLevels();
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
      level.size = std::max(level.size, size); // Keep the largest size seen
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
  persistent_levels_.erase(
      std::remove_if(persistent_levels_.begin(), persistent_levels_.end(),
                     [current_time, this](const PersistentLevel& level) {
                       return (current_time - level.last_updated_time) > persistence_timeout_ms_;
                     }),
      persistent_levels_.end());
}

void DomSurfacePanel::renderPersistentLevels() {
  if (persistent_levels_.empty()) return;

  // Get plot area bounds
  ImPlotRect plot_rect = ImPlot::GetPlotLimits();

  // Render each persistent level as a horizontal line or rectangle
  for (const auto& level : persistent_levels_) {
    // Only render if the level is considered "persistent" (has been present for threshold time)
    uint64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count();

    if ((current_time - level.first_detected_time) >= persistence_threshold_ms_) {
      ImU32 color = getPersistentLevelColor(level);

      // Draw horizontal line at the price level from left to right of the plot
      double xs[2] = {plot_rect.X.Min, plot_rect.X.Max};
      double ys[2] = {level.price, level.price};
      ImPlot::PlotLine("##PersistentLevel", xs, ys, 2);

      // Draw a more prominent rectangle to highlight the level
      // Extract the RGB components and set alpha to 15% transparency for better visibility
      ImVec4 color_vec = ImGui::ColorConvertU32ToFloat4(color);
      color_vec.w = 0.15f; // Set alpha to 15% transparency
      ImU32 transparent_color = ImGui::ColorConvertFloat4ToU32(color_vec);

      // Calculate a vertical range around the price level for the rectangle
      // Make it proportional to the zoom level for better visibility
      double visible_price_range = plot_rect.Y.Max - plot_rect.Y.Min;
      double price_range = visible_price_range * 0.005; // 0.5% of the visible price range (adjustable)
      if (price_range < 0.001) price_range = 0.001; // Minimum thickness
      
      double y_min = level.price - price_range/2.0;
      double y_max = level.price + price_range/2.0;

      // Draw a horizontal shaded area spanning the full time axis
      double shade_x[2] = {plot_rect.X.Min, plot_rect.X.Max};
      double shade_y1[2] = {y_min, y_min};
      double shade_y2[2] = {y_max, y_max};
      ImPlot::PlotShaded("##PersistentLevelRect", shade_x, shade_y1, shade_y2, 2);

      // Add a subtle highlight effect above the main line
      ImVec4 highlight_color_vec = ImGui::ColorConvertU32ToFloat4(color);
      highlight_color_vec.w = 0.08f; // Even more transparent for highlight
      ImU32 highlight_color = ImGui::ColorConvertFloat4ToU32(highlight_color_vec);

      // Draw highlight slightly above the main line
      double highlight_y_min = level.price + price_range/2.0;
      double highlight_y_max = level.price + price_range/2.0 + price_range*0.5;
      
      // Draw highlight shaded area
      double highlight_shade_x[2] = {plot_rect.X.Min, plot_rect.X.Max};
      double highlight_shade_y1[2] = {highlight_y_min, highlight_y_min};
      double highlight_shade_y2[2] = {highlight_y_max, highlight_y_max};
      ImPlot::PlotShaded("##PersistentLevelHighlight", highlight_shade_x, highlight_shade_y1, highlight_shade_y2, 2);
    }
  }
}

ImU32 DomSurfacePanel::getPersistentLevelColor(const PersistentLevel& level) const {
  // Color: Bright Green for Bids, Bright Red for Asks
  // Use brighter colors than the markers to distinguish persistent levels
  if (level.is_bid) {
    return IM_COL32(0, 255, 150, 220);  // Brighter green with higher transparency
  } else {
    return IM_COL32(255, 100, 150, 220);  // Brighter red with higher transparency
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
  ImGui::SliderFloat("##HeatmapIntensity", &heatmap_intensity_, 0.1f, 5.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
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
  ImGui::SliderFloat("##Threshold", &large_order_threshold_, 1.0f, 50.0f, "Threshold: %.1fx", ImGuiSliderFlags_Logarithmic);
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
  ImGui::SliderInt("Persistence (ms)", reinterpret_cast<int*>(&persistence_threshold_ms_), 1000, 30000, "%d ms");
  ImGui::PopItemWidth();
  ImGui::SameLine();
  ImGui::PushItemWidth(150);
  ImGui::SliderInt("Timeout (ms)", reinterpret_cast<int*>(&persistence_timeout_ms_), 10000, 120000, "%d ms");
  ImGui::PopItemWidth();
  ImGui::Separator();

  // Add Flush DOM Ruler controls
  ImGui::Text("Flush DOM Ruler:");
  ImGui::SameLine();
  ImGui::Checkbox("Show##FlushDOMRuler", &show_flush_dom_ruler_);
  ImGui::SameLine();
  ImGui::PushItemWidth(150);
  ImGui::SliderFloat("Width##FlushDOMRuler", &flush_dom_ruler_width_, 0.01f, 0.2f, "%.2f%%", ImGuiSliderFlags_Logarithmic);
  ImGui::PopItemWidth();
  ImGui::Separator();
}

void DomSurfacePanel::renderFlushDOMRuler() {
  // TODO: Implement Flush DOM Ruler rendering
  // This function should render the live orderbook at the right edge of the heatmap panel
  // For now, this is a stub implementation
}

}  // namespace BTQuant
