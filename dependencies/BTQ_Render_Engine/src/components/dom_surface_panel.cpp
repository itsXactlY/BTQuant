#include "components/dom_surface_panel.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <chrono>

namespace BTQuant {

DomSurfacePanel::DomSurfacePanel(
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(
          PanelConfig{.title = "DOM Surface", .type = PanelType::HEATMAP}),
      processor_(processor) {}

DomSurfacePanel::~DomSurfacePanel() {
  if (subscription_id_ > 0 && processor_) {
    processor_->unsubscribe(subscription_id_);
  }
}

void DomSurfacePanel::setSymbol(uint32_t symbol_id) {
  if (current_symbol_id_ == symbol_id)
    return;

  if (subscription_id_ > 0) {
    processor_->unsubscribe(subscription_id_);
    subscription_id_ = 0;
  }

  current_symbol_id_ = symbol_id;

  // Subscribe to ORDERBOOK updates
  if (processor_) {
    subscription_id_ = processor_->subscribe(
        symbol_id, RenderEngine::NotificationType::ORDERBOOK,
        [this](uint32_t sym, RenderEngine::NotificationType type) {
          this->onDataUpdate(sym, type);
        });
  }

  // Clear existing data to prevent mixing symbols
  heatmap_data_.clear();
  large_order_markers_.clear();
  recent_order_sizes_.clear();
  median_order_size_ = 0.0;
  markDirty();
}

void DomSurfacePanel::onDataUpdate(uint32_t symbol_id,
                                   RenderEngine::NotificationType type) {
  if (symbol_id == current_symbol_id_) {
    markDirty();
  }
}

void DomSurfacePanel::updateHeatmapData() {
  if (current_symbol_id_ == 0 || !processor_)
    return;

  // Request ALL available orderbook history (0 = no limit)
  auto history = processor_->getHistoricalOrderbooks(current_symbol_id_, 0);
  if (history.empty())
    return;

  // Determine price range based on LATEST snapshot
  const auto &latest = history.back();
  double mid_price = 0;
  if (!latest.bids.empty() && !latest.asks.empty()) {
    mid_price = (latest.bids.front().price + latest.asks.front().price) / 2.0;
  } else if (!latest.bids.empty()) {
    mid_price = latest.bids.front().price;
  } else if (!latest.asks.empty()) {
    mid_price = latest.asks.front().price;
  } else {
    return; // No price data
  }

  if (mid_price <= 0)
    return;

  double min_price = mid_price * (1.0 - price_range_);
  double max_price = mid_price * (1.0 + price_range_);

  if (max_price <= min_price)
    return;

  double price_step =
      (max_price - min_price) / static_cast<double>(price_bins_);

  // Resize data buffer: rows (price bins) * cols (time slices)
  int time_steps = static_cast<int>(history.size());
  size_t total_size =
      static_cast<size_t>(price_bins_) * static_cast<size_t>(time_steps);

  if (heatmap_data_.size() != total_size) {
    heatmap_data_.assign(total_size, 0.0);
  } else {
    std::fill(heatmap_data_.begin(), heatmap_data_.end(), 0.0);
  }

  // Populate data
  // Map: X-axis = Time (index), Y-axis = Price (bin)
  // ImPlot PlotHeatmap default: data[row * cols + col]
  // where row is Y-axis (price) and col is X-axis (time)

  double max_vol = 0;

  for (int t = 0; t < time_steps; ++t) {
    const auto &book = history[t];

    // Process Bids
    for (const auto &level : book.bids) {
      if (level.price >= min_price && level.price < max_price) {
        int bin = static_cast<int>((level.price - min_price) / price_step);
        if (bin >= 0 && bin < price_bins_) {
          heatmap_data_[bin * time_steps + t] += level.size;
          max_vol = std::max(max_vol, heatmap_data_[bin * time_steps + t]);
        }
      }
    }

    // Process Asks
    for (const auto &level : book.asks) {
      if (level.price >= min_price && level.price < max_price) {
        int bin = static_cast<int>((level.price - min_price) / price_step);
        if (bin >= 0 && bin < price_bins_) {
          heatmap_data_[bin * time_steps + t] += level.size;
          max_vol = std::max(max_vol, heatmap_data_[bin * time_steps + t]);
        }
      }
    }
  }

  // Update bounds for plotting
  bounds_min_[0] = 0;
  bounds_min_[1] = min_price;
  bounds_max_[0] = static_cast<double>(time_steps);
  bounds_max_[1] = max_price;

  scale_max_ = max_vol > 0 ? max_vol : 1.0;
}

void DomSurfacePanel::updateLargeOrderMarkers() {
  if (current_symbol_id_ == 0 || !processor_)
    return;

  // Get latest orderbook for large order detection
  auto orderbook_opt = processor_->getOrderbookData(current_symbol_id_);
  if (!orderbook_opt)
    return;

  const auto &orderbook = *orderbook_opt;

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
    median_order_size_ = 1.0; // Default fallback
    return;
  }

  // Sort and find median
  std::vector<double> sorted_sizes(recent_order_sizes_.begin(), 
                                  recent_order_sizes_.end());
  std::sort(sorted_sizes.begin(), sorted_sizes.end());

  size_t n = sorted_sizes.size();
  if (n % 2 == 0) {
    median_order_size_ = (sorted_sizes[n/2 - 1] + sorted_sizes[n/2]) / 2.0;
  } else {
    median_order_size_ = sorted_sizes[n/2];
  }
}

void DomSurfacePanel::detectLargeOrders(const RenderEngine::OrderbookData& orderbook) {
  // Collect all order sizes for median calculation
  for (const auto &level : orderbook.bids) {
    recent_order_sizes_.push_back(level.size);
  }
  for (const auto &level : orderbook.asks) {
    recent_order_sizes_.push_back(level.size);
  }

  // Keep only the last MEDIAN_WINDOW_SIZE sizes
  while (recent_order_sizes_.size() > MEDIAN_WINDOW_SIZE) {
    recent_order_sizes_.pop_front();
  }

  // Detect large orders (threshold: >10x median)
  double threshold = large_order_threshold_ * median_order_size_;
  uint64_t current_time = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();

  // Check bids
  for (const auto &level : orderbook.bids) {
    if (level.size > threshold) {
      // Check if we already have a marker at this price level
      bool exists = false;
      for (const auto &marker : large_order_markers_) {
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
  for (const auto &level : orderbook.asks) {
    if (level.size > threshold) {
      // Check if we already have a marker at this price level
      bool exists = false;
      for (const auto &marker : large_order_markers_) {
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
      std::chrono::steady_clock::now().time_since_epoch()).count();

  // Remove markers older than FADE_OUT_DURATION_US
  large_order_markers_.erase(
      std::remove_if(large_order_markers_.begin(), large_order_markers_.end(),
          [current_time](const LargeOrderMarker& marker) {
            return (current_time - marker.timestamp) > FADE_OUT_DURATION_US;
          }),
      large_order_markers_.end());
}

float DomSurfacePanel::calculateMarkerRadius(double order_size) const {
  if (median_order_size_ <= 0.0)
    return BASE_RADIUS;

  // Calculate radius: base_radius * sqrt(order_size / median_size)
  float radius = BASE_RADIUS * std::sqrt(order_size / median_order_size_);

  // Clamp to min/max range
  return std::clamp(radius, MIN_RADIUS, MAX_RADIUS);
}

ImU32 DomSurfacePanel::getMarkerColor(const LargeOrderMarker& marker) const {
  // Calculate alpha based on fade-out (if enabled)
  float alpha = 0.7f; // Default alpha
  if (enable_fade_out_) {
    uint64_t current_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    uint64_t age = current_time - marker.timestamp;
    float fade_ratio = 1.0f - static_cast<float>(age) / static_cast<float>(FADE_OUT_DURATION_US);
    alpha = std::clamp(fade_ratio * 0.8f, 0.3f, 0.8f);
  }

  // Color: Green for Bids, Red for Asks
  if (marker.is_bid) {
    return IM_COL32(0, 255, 0, static_cast<int>(alpha * 255)); // Green
  } else {
    return IM_COL32(255, 0, 0, static_cast<int>(alpha * 255)); // Red
  }
}

std::string DomSurfacePanel::getMarkerTooltip(const LargeOrderMarker& marker) const {
  std::string side = marker.is_bid ? "Bid" : "Ask";
  return std::format("Whale {}: {:.2f} @ ${:.2f}", side, marker.size, marker.price);
}

void DomSurfacePanel::renderLargeOrderMarkers() {
  if (large_order_markers_.empty())
    return;

  ImPlot::PushStyleVar(ImPlotStyleVar_MarkerSize, 1.0f);

  // Get plot area for manual circle rendering
  ImPlotRect plot_rect = ImPlot::GetPlotLimits();

  // Render each large order marker as a circle
  for (const auto &marker : large_order_markers_) {
    ImU32 color = getMarkerColor(marker);
    ImU32 border_color = IM_COL32(255, 255, 255, 230); // White border

    // Convert plot coordinates to pixel coordinates
    ImVec2 pixel_pos = ImPlot::PlotToPixels(marker.x, marker.y);

    // Draw filled circle
    ImDrawList* draw_list = ImPlot::GetPlotDrawList();
    if (draw_list) {
      draw_list->AddCircleFilled(pixel_pos, marker.radius, color, 32);
      draw_list->AddCircle(pixel_pos, marker.radius, border_color, 32, 1.0f);

      // Check for hover and show tooltip
      ImVec2 mouse_pos = ImGui::GetMousePos();
      float distance = std::sqrt(
          std::pow(mouse_pos.x - pixel_pos.x, 2) + 
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
  }

  begin_panel_window();

  if (current_symbol_id_ == 0 || heatmap_data_.empty()) {
    ImGui::Text("No Data / Select Symbol");
    end_panel_window();
    return;
  }

  // Enable Pan/Zoom for DOM Surface
  std::string plot_id = "##DomHeatmap_" + std::to_string(current_symbol_id_);
  if (ImPlot::BeginPlot(plot_id.c_str(), ImVec2(-1, -1),
                        ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText)) {
    ImPlot::SetupAxes("Time Step", "Price");
    ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoTickLabels);

    // Always fit axes to data bounds (fills plot area)
    ImPlot::SetupAxisLimits(ImAxis_X1, bounds_min_[0], bounds_max_[0],
                            ImPlotCond_Always);
    ImPlot::SetupAxisLimits(ImAxis_Y1, bounds_min_[1], bounds_max_[1],
                            ImPlotCond_Always);

    // Use time history size for Cols and price_bins for Rows
    int rows = price_bins_;
    int cols = static_cast<int>(heatmap_data_.size()) / rows;

    if (cols > 0 && rows > 0) {
      ImPlot::PushColormap(ImPlotColormap_Viridis);
      ImPlot::PlotHeatmap("Liquidity", heatmap_data_.data(), rows, cols, 0,
                          scale_max_, nullptr,
                          ImPlotPoint(bounds_min_[0], bounds_min_[1]),
                          ImPlotPoint(bounds_max_[0], bounds_max_[1]));
      ImPlot::PopColormap();
    }

    // Render Large Order Markers OVER the heatmap
    renderLargeOrderMarkers();

    ImPlot::EndPlot();
  }

  // Debug Overlay for DOM troubleshooting
  if (heatmap_data_.size() > 0) {
    ImGui::SetCursorPos(ImVec2(10, 30));
    ImGui::TextColored(ImVec4(1, 1, 0, 1),
                       "Debug: MaxVol=%.2f, Hist=%zu, Bins=%d", scale_max_,
                       heatmap_data_.size() / price_bins_, price_bins_);
    ImGui::Text("Bounds: Y=%.4f - %.4f", bounds_min_[1], bounds_max_[1]);
    ImGui::Text("Large Orders: %zu (Median: %.2f)", large_order_markers_.size(), median_order_size_);
  }

  end_panel_window();
}

} // namespace BTQuant
