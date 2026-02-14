#include "components/dom_surface_panel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>

// Vulkan compute integration includes
#ifdef VK_USE_PLATFORM_WIN32_KHR
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#include <imgui.h>
#include <backends/imgui_impl_vulkan.h>

namespace BTQuant {

DomSurfacePanel::DomSurfacePanel(std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(PanelConfig{.title = "DOM Surface", .type = PanelType::HEATMAP}),
      processor_(processor) {
  // Initialize Vulkan compute resources
  initializeVulkanCompute();
}

DomSurfacePanel::~DomSurfacePanel() {
  if (subscription_id_ > 0 && processor_) {
    processor_->unsubscribe(subscription_id_);
  }
  
  // Destroy Vulkan compute resources
  destroyVulkanCompute();
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
  
  // Update Vulkan compute if initialized
  if (vulkanInitialized_) {
    needsVulkanUpdate_ = true;
    updateVulkanHeatmap();
  }
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
  if (ImGui::Button("Reset View")) {
    ImPlot::SetNextAxesToFit();
  }
  ImGui::SameLine();
  ImGui::Checkbox("Show Persistent Lines", &show_persistent_lines_);
  ImGui::SameLine();
  ImGui::Text(" | Symbols: %u | Bins: %d | Orders: %zu | Trades: %zu", current_symbol_id_, price_bins_,
              large_order_markers_.size(), trade_bubbles_.size());

  // Enable Pan/Zoom for DOM Surface
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
    if (vulkanInitialized_ && heatmapTextureId_) {
      // Render the Vulkan-generated heatmap texture
      ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(0, 0));
      
      // Calculate the UV coordinates for the entire texture
      // The texture spans the entire plot area
      ImVec2 uv0(0.0f, 0.0f);
      ImVec2 uv1(1.0f, 1.0f);
      
      // Draw the heatmap texture as a quad
      ImPlot::PlotImage("Liquidity", heatmapTextureId_,
                        ImPlotPoint(bounds_min_[0], bounds_min_[1]),
                        ImPlotPoint(bounds_max_[0], bounds_max_[1]),
                        uv0, uv1, ImVec4(1, 1, 1, 1), 0);
      
      ImPlot::PopStyleVar();
    } else {
      // Fallback to CPU-generated heatmap if Vulkan is not available
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
    if (vulkanInitialized_) {
      ImGui::Text("Vulkan: Active");
    } else {
      ImGui::Text("Vulkan: Not Initialized");
    }
  }

  end_panel_window();
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

      // Draw horizontal line across the entire time axis
      ImPlot::PushStyleColor(ImPlotCol_Line, color);
      ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 3.0f); // Thicker line for better visibility

      // Draw horizontal line at the price level from left to right of the plot
      double xs[2] = {plot_rect.X.Min, plot_rect.X.Max};
      double ys[2] = {level.price, level.price};
      ImPlot::PlotLine("##PersistentLevel", xs, ys, 2);

      ImPlot::PopStyleVar();
      ImPlot::PopStyleColor();

      // Draw a more prominent rectangle to highlight the level
      // Extract the RGB components and set alpha to 15% transparency for better visibility
      ImVec4 color_vec = ImGui::ColorConvertU32ToFloat4(color);
      color_vec.w = 0.15f; // Set alpha to 15% transparency
      ImU32 transparent_color = ImGui::ColorConvertFloat4ToU32(color_vec);
      ImPlot::PushStyleColor(ImPlotCol_Fill, transparent_color);

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

      ImPlot::PopStyleColor();

      // Add a subtle highlight effect above the main line
      ImVec4 highlight_color_vec = ImGui::ColorConvertU32ToFloat4(color);
      highlight_color_vec.w = 0.08f; // Even more transparent for highlight
      ImU32 highlight_color = ImGui::ColorConvertFloat4ToU32(highlight_color_vec);
      ImPlot::PushStyleColor(ImPlotCol_Line, highlight_color);

      // Draw highlight slightly above the main line
      double highlight_y_min = level.price + price_range/2.0;
      double highlight_y_max = level.price + price_range/2.0 + price_range*0.5;
      
      // Draw highlight shaded area
      double highlight_shade_x[2] = {plot_rect.X.Min, plot_rect.X.Max};
      double highlight_shade_y1[2] = {highlight_y_min, highlight_y_min};
      double highlight_shade_y2[2] = {highlight_y_max, highlight_y_max};
      ImPlot::PlotShaded("##PersistentLevelHighlight", highlight_shade_x, highlight_shade_y1, highlight_shade_y2, 2);

      ImPlot::PopStyleColor();
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

bool DomSurfacePanel::initializeVulkanCompute() {
  // Check if Vulkan is available through ImGui
  if (!ImGui::GetCurrentContext() || !GImGui) {
    return false;
  }
  
  ImGuiIO& io = ImGui::GetIO();
  
  // Skip initialization if not using Vulkan backend
  if (!(io.BackendFlags & ImGuiBackendFlags_RendererHasVulkan)) {
    return false;
  }
  
  // Initialize Vulkan compute resources
  try {
    createDescriptorSetLayout();
    createComputePipeline();
    createHeatmapImage();
    createHeatmapImageView();
    createSampler();
    createDescriptorPool();
    createDescriptorSet();
    createComputeCommandBuffer();
    
    vulkanInitialized_ = true;
    return true;
  } catch (...) {
    return false;
  }
}

void DomSurfacePanel::destroyVulkanCompute() {
  if (!vulkanInitialized_) return;
  
  // Wait for any pending compute operations
  if (computeFence_ != VK_NULL_HANDLE) {
    vkWaitForFences(GImGui->VulkanHandle, 1, &computeFence_, VK_TRUE, UINT64_MAX);
  }
  
  // Clean up Vulkan resources
  if (computeCommandBuffer_ != VK_NULL_HANDLE) {
    vkFreeCommandBuffers(GImGui->VulkanHandle, 
                         ImGui::GetAllocatorUserData()->CommandPool, 
                         1, &computeCommandBuffer_);
    computeCommandBuffer_ = VK_NULL_HANDLE;
  }
  
  if (computeFence_ != VK_NULL_HANDLE) {
    vkDestroyFence(GImGui->VulkanHandle, computeFence_, nullptr);
    computeFence_ = VK_NULL_HANDLE;
  }
  
  if (descriptorSetLayout_ != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(GImGui->VulkanHandle, descriptorSetLayout_, nullptr);
    descriptorSetLayout_ = VK_NULL_HANDLE;
  }
  
  if (pipelineLayout_ != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(GImGui->VulkanHandle, pipelineLayout_, nullptr);
    pipelineLayout_ = VK_NULL_HANDLE;
  }
  
  if (computePipeline_ != VK_NULL_HANDLE) {
    vkDestroyPipeline(GImGui->VulkanHandle, computePipeline_, nullptr);
    computePipeline_ = VK_NULL_HANDLE;
  }
  
  if (orderBookBuffer_ != VK_NULL_HANDLE) {
    vkDestroyBuffer(GImGui->VulkanHandle, orderBookBuffer_, nullptr);
    orderBookBuffer_ = VK_NULL_HANDLE;
  }
  
  if (orderBookBufferMemory_ != VK_NULL_HANDLE) {
    vkFreeMemory(GImGui->VulkanHandle, orderBookBufferMemory_, nullptr);
    orderBookBufferMemory_ = VK_NULL_HANDLE;
  }
  
  if (heatmapOutputBuffer_ != VK_NULL_HANDLE) {
    vkDestroyBuffer(GImGui->VulkanHandle, heatmapOutputBuffer_, nullptr);
    heatmapOutputBuffer_ = VK_NULL_HANDLE;
  }
  
  if (heatmapOutputBufferMemory_ != VK_NULL_HANDLE) {
    vkFreeMemory(GImGui->VulkanHandle, heatmapOutputBufferMemory_, nullptr);
    heatmapOutputBufferMemory_ = VK_NULL_HANDLE;
  }
  
  if (heatmapImage_ != VK_NULL_HANDLE) {
    vkDestroyImage(GImGui->VulkanHandle, heatmapImage_, nullptr);
    heatmapImage_ = VK_NULL_HANDLE;
  }
  
  if (heatmapImageMemory_ != VK_NULL_HANDLE) {
    vkFreeMemory(GImGui->VulkanHandle, heatmapImageMemory_, nullptr);
    heatmapImageMemory_ = VK_NULL_HANDLE;
  }
  
  if (heatmapImageView_ != VK_NULL_HANDLE) {
    vkDestroyImageView(GImGui->VulkanHandle, heatmapImageView_, nullptr);
    heatmapImageView_ = VK_NULL_HANDLE;
  }
  
  if (heatmapSampler_ != VK_NULL_HANDLE) {
    vkDestroySampler(GImGui->VulkanHandle, heatmapSampler_, nullptr);
    heatmapSampler_ = VK_NULL_HANDLE;
  }
  
  vulkanInitialized_ = false;
}

void DomSurfacePanel::createDescriptorSetLayout() {
  // Define bindings for the compute shader
  VkDescriptorSetLayoutBinding orderBookBinding = {};
  orderBookBinding.binding = 0;
  orderBookBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  orderBookBinding.descriptorCount = 1;
  orderBookBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  orderBookBinding.pImmutableSamplers = nullptr;
  
  VkDescriptorSetLayoutBinding heatmapOutputBinding = {};
  heatmapOutputBinding.binding = 1;
  heatmapOutputBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  heatmapOutputBinding.descriptorCount = 1;
  heatmapOutputBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  heatmapOutputBinding.pImmutableSamplers = nullptr;
  
  VkDescriptorSetLayoutBinding heatmapParamsBinding = {};
  heatmapParamsBinding.binding = 2;
  heatmapParamsBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  heatmapParamsBinding.descriptorCount = 1;
  heatmapParamsBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  heatmapParamsBinding.pImmutableSamplers = nullptr;
  
  std::array<VkDescriptorSetLayoutBinding, 3> bindings = {orderBookBinding, heatmapOutputBinding, heatmapParamsBinding};
  
  VkDescriptorSetLayoutCreateInfo layoutInfo = {};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
  layoutInfo.pBindings = bindings.data();
  
  if (vkCreateDescriptorSetLayout(GImGui->VulkanHandle, &layoutInfo, nullptr, &descriptorSetLayout_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create descriptor set layout!");
  }
}

void DomSurfacePanel::createComputePipeline() {
  // In a real implementation, you would load the compiled SPIR-V from the shaders/lob_heatmap.comp.spv file
  // For now, we'll create a placeholder implementation that assumes the shader is available
  
  // Create a simple compute shader module (placeholder - in reality you'd load from file)
  // This is a minimal SPIR-V binary for a compute shader that does nothing
  static const uint32_t dummyComputeShaderSPV[] = {
    0x07230203, 0x00010000, 0x0008000a, 0x00000014, 0x00000000, 0x00020011, 0x00000001, 0x0006000b,
    0x00000001, 0x4c534c47, 0x6474732e, 0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001,
    0x000a000f, 0x00000000, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000b, 0x0000000f, 0x00000015,
    0x00000019, 0x0000001d, 0x00050006, 0x0000000b, 0x00000000, 0x696c5f67, 0x0065746e, 0x00040006,
    0x0000000f, 0x00000000, 0x0074754f, 0x00030005, 0x00000011, 0x00786574, 0x00060005, 0x00000015,
    0x00000000, 0x63786574, 0x00657475, 0x00060005, 0x00000019, 0x00000000, 0x63786574, 0x00657475,
    0x00060005, 0x0000001d, 0x00000000, 0x63786574, 0x00657475, 0x00050048, 0x0000000b, 0x00000000,
    0x0000000b, 0x00000000, 0x00050048, 0x0000000f, 0x00000000, 0x0000000c, 0x00000000, 0x00030047,
    0x0000000f, 0x00000003, 0x00040048, 0x00000015, 0x00000000, 0x00000016, 0x00040048, 0x00000015,
    0x00000001, 0x00000016, 0x00040048, 0x00000019, 0x00000000, 0x0000001a, 0x00040048, 0x00000019,
    0x00000001, 0x0000001a, 0x00040048, 0x0000001d, 0x00000000, 0x0000001e, 0x00040048, 0x0000001d,
    0x00000001, 0x0000001e, 0x00050041, 0x00000010, 0x00000011, 0x0000000f, 0x00000010, 0x0003003e,
    0x00000011, 0x00000012, 0x000a0004, 0x475f4c47, 0x4c474f4f, 0x70635f45, 0x74735f70, 0x5f656c79,
    0x656c676e, 0x0000766e, 0x00060004, 0x475f4c47, 0x4c474f4f, 0x61625f45, 0x616d6552, 0x00000078,
    0x00050005, 0x0000000b, 0x0070766d, 0x656c706d, 0x00007465, 0x00060005, 0x0000000f, 0x00637865,
    0x646e4974, 0x78657475, 0x00000000, 0x00050005, 0x00000015, 0x00637865, 0x646e4974, 0x00007865,
    0x00050005, 0x00000019, 0x00637865, 0x646e4974, 0x00007865, 0x00050005, 0x0000001d, 0x00637865,
    0x646e4974, 0x00007865
  };
  
  VkShaderModuleCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = sizeof(dummyComputeShaderSPV);
  createInfo.pCode = dummyComputeShaderSPV;
  
  VkShaderModule shaderModule;
  if (vkCreateShaderModule(GImGui->VulkanHandle, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
    throw std::runtime_error("failed to create shader module!");
  }
  
  VkPipelineShaderStageCreateInfo shaderStageInfo = {};
  shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  shaderStageInfo.module = shaderModule;
  shaderStageInfo.pName = "main";
  
  // Create pipeline layout
  VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 1;
  pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout_;
  
  if (vkCreatePipelineLayout(GImGui->VulkanHandle, &pipelineLayoutInfo, nullptr, &pipelineLayout_) != VK_SUCCESS) {
    vkDestroyShaderModule(GImGui->VulkanHandle, shaderModule, nullptr);
    throw std::runtime_error("failed to create pipeline layout!");
  }
  
  // Create compute pipeline
  VkComputePipelineCreateInfo pipelineInfo = {};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineInfo.stage = shaderStageInfo;
  pipelineInfo.layout = pipelineLayout_;
  
  if (vkCreateComputePipelines(GImGui->VulkanHandle, nullptr, 1, &pipelineInfo, nullptr, &computePipeline_) != VK_SUCCESS) {
    vkDestroyShaderModule(GImGui->VulkanHandle, shaderModule, nullptr);
    throw std::runtime_error("failed to create compute pipeline!");
  }
  
  // Clean up shader module after pipeline creation
  vkDestroyShaderModule(GImGui->VulkanHandle, shaderModule, nullptr);
}

void DomSurfacePanel::createHeatmapImage() {
  // Get image dimensions from current heatmap requirements
  uint32_t width = static_cast<uint32_t>(history_depth_);  // Time steps
  uint32_t height = static_cast<uint32_t>(price_bins_);    // Price bins
  
  VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT; // RGBA float format for heatmap
  
  // Create image
  VkImageCreateInfo imageInfo = {};
  imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType = VK_IMAGE_TYPE_2D;
  imageInfo.extent.width = width;
  imageInfo.extent.height = height;
  imageInfo.extent.depth = 1;
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.format = format;
  imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  
  if (vkCreateImage(GImGui->VulkanHandle, &imageInfo, nullptr, &heatmapImage_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create heatmap image!");
  }
  
  // Allocate memory for image
  VkMemoryRequirements memRequirements;
  vkGetImageMemoryRequirements(GImGui->VulkanHandle, heatmapImage_, &memRequirements);
  
  VkMemoryAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = 0; // Will be determined based on requirements
  
  // Find appropriate memory type
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(GImGui->VulkanHandle, &memProperties);
  
  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((memRequirements.memoryTypeBits & (1 << i)) && 
        (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
      allocInfo.memoryTypeIndex = i;
      break;
    }
  }
  
  if (vkAllocateMemory(GImGui->VulkanHandle, &allocInfo, nullptr, &heatmapImageMemory_) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate image memory!");
  }
  
  vkBindImageMemory(GImGui->VulkanHandle, heatmapImage_, heatmapImageMemory_, 0);
}

void DomSurfacePanel::createHeatmapImageView() {
  VkImageViewCreateInfo viewInfo = {};
  viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image = heatmapImage_;
  viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT;
  viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.levelCount = 1;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount = 1;
  
  if (vkCreateImageView(GImGui->VulkanHandle, &viewInfo, nullptr, &heatmapImageView_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create texture image view!");
  }
}

void DomSurfacePanel::createSampler() {
  VkSamplerCreateInfo samplerInfo = {};
  samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerInfo.magFilter = VK_FILTER_LINEAR;
  samplerInfo.minFilter = VK_FILTER_LINEAR;
  samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  samplerInfo.anisotropyEnable = VK_FALSE;
  samplerInfo.maxAnisotropy = 1.0f;
  samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  samplerInfo.unnormalizedCoordinates = VK_FALSE;
  samplerInfo.compareEnable = VK_FALSE;
  samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  
  if (vkCreateSampler(GImGui->VulkanHandle, &samplerInfo, nullptr, &heatmapSampler_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create texture sampler!");
  }
}

void DomSurfacePanel::createDescriptorPool() {
  std::array<VkDescriptorPoolSize, 3> poolSizes = {};
  poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSizes[0].descriptorCount = 1;
  poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  poolSizes[1].descriptorCount = 1;
  poolSizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  poolSizes[2].descriptorCount = 1;
  
  VkDescriptorPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
  poolInfo.pPoolSizes = poolSizes.data();
  poolInfo.maxSets = 1;
  
  if (vkCreateDescriptorPool(GImGui->VulkanHandle, &poolInfo, nullptr, &descriptorPool_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create descriptor pool!");
  }
}

void DomSurfacePanel::createDescriptorSet() {
  // First, create the order book buffer that will be used by the compute shader
  createOrderBookBuffer();
  
  VkDescriptorSetAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = descriptorPool_;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &descriptorSetLayout_;
  
  if (vkAllocateDescriptorSets(GImGui->VulkanHandle, &allocInfo, &descriptorSet_) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate descriptor set!");
  }
  
  // Update descriptor sets
  VkDescriptorBufferInfo orderBookBufferInfo = {};
  orderBookBufferInfo.buffer = orderBookBuffer_;
  orderBookBufferInfo.offset = 0;
  orderBookBufferInfo.range = VK_WHOLE_SIZE;
  
  VkDescriptorImageInfo heatmapImageInfo = {};
  heatmapImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  heatmapImageInfo.imageView = heatmapImageView_;
  heatmapImageInfo.sampler = heatmapSampler_;
  
  std::array<VkWriteDescriptorSet, 2> descriptorWrites = {};
  
  descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  descriptorWrites[0].dstSet = descriptorSet_;
  descriptorWrites[0].dstBinding = 0;
  descriptorWrites[0].dstArrayElement = 0;
  descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  descriptorWrites[0].descriptorCount = 1;
  descriptorWrites[0].pBufferInfo = &orderBookBufferInfo;
  
  descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  descriptorWrites[1].dstSet = descriptorSet_;
  descriptorWrites[1].dstBinding = 1;
  descriptorWrites[1].dstArrayElement = 0;
  descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  descriptorWrites[1].descriptorCount = 1;
  descriptorWrites[1].pImageInfo = &heatmapImageInfo;
  
  vkUpdateDescriptorSets(GImGui->VulkanHandle, static_cast<uint32_t>(descriptorWrites.size()), 
                         descriptorWrites.data(), 0, nullptr);
  
  // Create the ImGui texture ID for the heatmap image
  if (GImGui && GImGui->BackendRendererUserData) {
    heatmapTextureId_ = ImGui_ImplVulkan_AddTexture(heatmapSampler_, heatmapImageView_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }
}

void DomSurfacePanel::createOrderBookBuffer() {
  // Calculate the size needed for the order book data
  // We'll create a buffer that can hold the current order book snapshot
  size_t bufferSize = sizeof(uint32_t) * 3 + // currentTimeIndex, priceLevelsCount, padding
                      sizeof(float) * 2 +    // basePrice, priceRange
                      100 * sizeof(OrderBookLevel); // Assuming max 100 price levels

  // Create the buffer
  VkBufferCreateInfo bufferInfo = {};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = bufferSize;
  bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateBuffer(GImGui->VulkanHandle, &bufferInfo, nullptr, &orderBookBuffer_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create order book buffer!");
  }

  // Allocate memory for the buffer
  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(GImGui->VulkanHandle, orderBookBuffer_, &memRequirements);

  VkMemoryAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex = 0; // Will be determined based on requirements

  // Find appropriate memory type
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(GImGui->VulkanHandle, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((memRequirements.memoryTypeBits & (1 << i)) && 
        (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
        (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
      allocInfo.memoryTypeIndex = i;
      break;
    }
  }

  if (vkAllocateMemory(GImGui->VulkanHandle, &allocInfo, nullptr, &orderBookBufferMemory_) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate order book buffer memory!");
  }

  vkBindBufferMemory(GImGui->VulkanHandle, orderBookBuffer_, orderBookBufferMemory_, 0);
}

void DomSurfacePanel::createComputeCommandBuffer() {
  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = ImGui::GetAllocatorUserData()->CommandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;
  
  if (vkAllocateCommandBuffers(GImGui->VulkanHandle, &allocInfo, &computeCommandBuffer_) != VK_SUCCESS) {
    throw std::runtime_error("failed to allocate compute command buffers!");
  }
  
  // Create fence for compute operations
  VkFenceCreateInfo fenceInfo = {};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = 0; // Not signaled initially
  
  if (vkCreateFence(GImGui->VulkanHandle, &fenceInfo, nullptr, &computeFence_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create compute fence!");
  }
}

void DomSurfacePanel::recordComputeCommands() {
  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  
  vkBeginCommandBuffer(computeCommandBuffer_, &beginInfo);
  
  // Bind compute pipeline
  vkCmdBindPipeline(computeCommandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline_);
  
  // Bind descriptor set
  vkCmdBindDescriptorSets(computeCommandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, 
                          pipelineLayout_, 0, 1, &descriptorSet_, 0, nullptr);
  
  // Dispatch compute shader
  // Calculate appropriate work group dimensions based on heatmap size
  uint32_t width = static_cast<uint32_t>(history_depth_);
  uint32_t height = static_cast<uint32_t>(price_bins_);
  
  // Use the local work group size defined in the shader (1, 64, 1)
  uint32_t groupX = 1;
  uint32_t groupY = (height + 63) / 64; // Round up to nearest multiple of 64
  
  vkCmdDispatch(computeCommandBuffer_, groupX, groupY, 1);
  
  vkEndCommandBuffer(computeCommandBuffer_);
}

void DomSurfacePanel::submitComputeCommands() {
  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &computeCommandBuffer_;
  
  // Submit to compute queue
  VkQueue computeQueue = ImGui::GetVulkanData()->Queue;
  vkQueueSubmit(computeQueue, 1, &submitInfo, computeFence_);
}

void DomSurfacePanel::updateVulkanHeatmap() {
  if (!vulkanInitialized_ || !needsVulkanUpdate_) return;
  
  // Update the order book buffer with current market data
  updateOrderBookBuffer();
  
  // Wait for previous compute operations to complete
  if (computeFence_ != VK_NULL_HANDLE) {
    VkResult result = vkWaitForFences(GImGui->VulkanHandle, 1, &computeFence_, VK_TRUE, 1000000000); // 1 second timeout
    if (result == VK_SUCCESS) {
      vkResetFences(GImGui->VulkanHandle, 1, &computeFence_);
    }
  }
  
  // Record and submit compute commands
  recordComputeCommands();
  submitComputeCommands();
  
  needsVulkanUpdate_ = false;
}

void DomSurfacePanel::updateOrderBookBuffer() {
  if (!vulkanInitialized_ || !processor_) return;
  
  // Get the current orderbook data
  auto orderbook_opt = processor_->getOrderbookData(current_symbol_id_);
  if (!orderbook_opt) return;
  
  const auto& orderbook = *orderbook_opt;
  
  // Prepare the data structure to match the shader expectations
  struct OrderBookSnapshot {
    uint32_t currentTimeIndex;
    uint32_t priceLevelsCount;
    float basePrice;
    float priceRange;
    OrderBookLevel levels[100]; // Fixed-size array for simplicity
  } snapshot = {};
  
  // Fill in the snapshot data
  snapshot.currentTimeIndex = static_cast<uint32_t>(heatmap_data_.size() / price_bins_); // Current time index
  snapshot.priceLevelsCount = 0;
  
  // Determine base price and price range
  double min_price = std::numeric_limits<double>::max();
  double max_price = std::numeric_limits<double>::lowest();
  
  for (const auto& level : orderbook.bids) {
    min_price = std::min(min_price, level.price);
    max_price = std::max(max_price, level.price);
  }
  for (const auto& level : orderbook.asks) {
    min_price = std::min(min_price, level.price);
    max_price = std::max(max_price, level.price);
  }
  
  if (min_price < max_price) {
    snapshot.basePrice = static_cast<float>(min_price);
    snapshot.priceRange = static_cast<float>(max_price - min_price);
  } else {
    // Fallback values
    snapshot.basePrice = 0.0f;
    snapshot.priceRange = 1.0f;
  }
  
  // Copy bid levels
  size_t level_idx = 0;
  for (const auto& level : orderbook.bids) {
    if (level_idx >= 50) break; // Limit to first 50 bids
    
    snapshot.levels[level_idx].price = static_cast<float>(level.price);
    snapshot.levels[level_idx].bidQuantity = static_cast<uint32_t>(level.size);
    snapshot.levels[level_idx].askQuantity = 0; // No ask quantity for bid levels
    snapshot.levels[level_idx].numOrders = 1; // Simplified count
    level_idx++;
  }
  
  // Copy ask levels
  for (const auto& level : orderbook.asks) {
    if (level_idx >= 100) break; // Limit to total 100 levels
    
    snapshot.levels[level_idx].price = static_cast<float>(level.price);
    snapshot.levels[level_idx].askQuantity = static_cast<uint32_t>(level.size);
    snapshot.levels[level_idx].bidQuantity = 0; // No bid quantity for ask levels
    snapshot.levels[level_idx].numOrders = 1; // Simplified count
    level_idx++;
  }
  
  snapshot.priceLevelsCount = static_cast<uint32_t>(level_idx);
  
  // Copy the data to the GPU buffer
  void* mappedData;
  vkMapMemory(GImGui->VulkanHandle, orderBookBufferMemory_, 0, sizeof(snapshot), 0, &mappedData);
  memcpy(mappedData, &snapshot, sizeof(snapshot));
  vkUnmapMemory(GImGui->VulkanHandle, orderBookBufferMemory_);
}

void DomSurfacePanel::updateFlushDOMRulerData() {
  // This method would update the data for the flush DOM ruler
  // Currently, it's handled as part of the regular orderbook updates
  // The ruler shows the current live orderbook at the right edge
}

void DomSurfacePanel::renderFlushDOMRuler() {
  if (!show_flush_dom_ruler_ || current_symbol_id_ == 0 || !processor_) return;

  // Get the current orderbook data
  auto orderbook_opt = processor_->getOrderbookData(current_symbol_id_);
  if (!orderbook_opt) return;

  const auto& orderbook = *orderbook_opt;

  // Get plot area bounds
  ImPlotRect plot_rect = ImPlot::GetPlotLimits();

  // Calculate the width of the flush DOM ruler as a percentage of the plot width
  double ruler_width = (plot_rect.X.Max - plot_rect.X.Min) * flush_dom_ruler_width_;
  
  // Calculate the X position where the ruler starts (right edge of heatmap moving inward)
  double ruler_start_x = plot_rect.X.Max - ruler_width;

  // Render bid levels (green bars extending from right edge inward)
  for (const auto& level : orderbook.bids) {
    // Calculate Y position for this price level
    double y_pos = level.price;

    // Only render if within visible price range
    if (y_pos >= plot_rect.Y.Min && y_pos <= plot_rect.Y.Max) {
      // Calculate the depth/intensity of the bar based on order size
      // Find max volume for normalization
      double max_volume = 0.0;
      for (const auto& bid_level : orderbook.bids) {
        max_volume = std::max(max_volume, bid_level.size);
      }

      // Calculate the width of the bar based on the order size (relative to max volume)
      double normalized_size = max_volume > 0 ? level.size / max_volume : 0.0;
      double bar_width = ruler_width * normalized_size;

      // Calculate the X position where the bar ends (left side of the ruler area)
      double bar_end_x = ruler_start_x + bar_width;

      // Calculate a small height for the bar based on the visible price range
      // Use a fixed small height relative to the plot height
      double bar_height = (plot_rect.Y.Max - plot_rect.Y.Min) * 0.005; // 0.5% of the plot height
      if (bar_height < 0.001) bar_height = 0.001; // Minimum height

      // Calculate the Y range for this bar
      double y_min = y_pos - bar_height / 2.0;
      double y_max = y_pos + bar_height / 2.0;

      // Create points for the rectangle
      ImPlotPoint rect_min(ruler_start_x, y_min);
      ImPlotPoint rect_max(bar_end_x, y_max);

      // Draw the bid bar as a green rectangle
      ImU32 bid_color = IM_COL32(0, 230, 118, 180); // Green with transparency
      
      ImPlot::PushStyleColor(ImPlotCol_Fill, bid_color);
      ImPlot::PlotRect("##BidRuler", rect_min.x, rect_min.y, rect_max.x, rect_max.y);
      ImPlot::PopStyleColor();
    }
  }

  // Render ask levels (red bars extending from right edge inward)
  for (const auto& level : orderbook.asks) {
    // Calculate Y position for this price level
    double y_pos = level.price;

    // Only render if within visible price range
    if (y_pos >= plot_rect.Y.Min && y_pos <= plot_rect.Y.Max) {
      // Calculate the depth/intensity of the bar based on order size
      // Find max volume for normalization
      double max_volume = 0.0;
      for (const auto& ask_level : orderbook.asks) {
        max_volume = std::max(max_volume, ask_level.size);
      }

      // Calculate the width of the bar based on the order size (relative to max volume)
      double normalized_size = max_volume > 0 ? level.size / max_volume : 0.0;
      double bar_width = ruler_width * normalized_size;

      // Calculate the X position where the bar ends (left side of the ruler area)
      double bar_end_x = ruler_start_x + bar_width;

      // Calculate a small height for the bar based on the visible price range
      // Use a fixed small height relative to the plot height
      double bar_height = (plot_rect.Y.Max - plot_rect.Y.Min) * 0.005; // 0.5% of the plot height
      if (bar_height < 0.001) bar_height = 0.001; // Minimum height

      // Calculate the Y range for this bar
      double y_min = y_pos - bar_height / 2.0;
      double y_max = y_pos + bar_height / 2.0;

      // Create points for the rectangle
      ImPlotPoint rect_min(ruler_start_x, y_min);
      ImPlotPoint rect_max(bar_end_x, y_max);

      // Draw the ask bar as a red rectangle
      ImU32 ask_color = IM_COL32(255, 59, 105, 180); // Red with transparency
      
      ImPlot::PushStyleColor(ImPlotCol_Fill, ask_color);
      ImPlot::PlotRect("##AskRuler", rect_min.x, rect_min.y, rect_max.x, rect_max.y);
      ImPlot::PopStyleColor();
    }
  }
}

}  // namespace BTQuant
