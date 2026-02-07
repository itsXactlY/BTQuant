#include "components/domsurfacepanel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>

#include "vulkan_base_types.hpp"
#include "backends/imgui_impl_vulkan.h"
#include "data/TradeData.h"

namespace BTQuant {
namespace RenderEngine {

DomSurfacePanel::DomSurfacePanel(std::shared_ptr<MarketDataProcessor> processor)
    : PanelBase(PanelConfig{.title = "DOM Surface", .type = PanelType::HEATMAP}),
      processor_(processor) {}

DomSurfacePanel::~DomSurfacePanel() {
  cleanupVulkanResources();

  if (subscription_id_ > 0 && processor_) {
    processor_->unsubscribe(subscription_id_);
  }
  
  // Clear trade bubbles to ensure proper cleanup
  trade_bubbles_.clear();
}

void DomSurfacePanel::setSymbol(uint32_t symbol_id) {
  // Get the symbol name from the processor or symbol registry if available
  std::string symbol_name;
  if (processor_) {
    symbol_name = processor_->getSymbolName(symbol_id);
  }
  
  setSymbol(symbol_id, symbol_name);
}

void DomSurfacePanel::setSymbol(uint32_t symbol_id, const std::string& symbol_name) {
  // Store the symbol name for potential use in UI elements
  current_symbol_name_ = symbol_name;
  
  // Call the original setSymbol method with the ID to handle subscriptions and data clearing
  if (current_symbol_id_ != symbol_id) {
    if (subscription_id_ > 0) {
      processor_->unsubscribe(subscription_id_);
      subscription_id_ = 0;
    }

    current_symbol_id_ = symbol_id;

    // Subscribe to both ORDERBOOK and TRADE updates
    if (processor_) {
      subscription_id_ =
          processor_->subscribe(symbol_id, NotificationType::ORDERBOOK,
                                [this](uint32_t sym, NotificationType type) {
                                  this->onDataUpdate(sym, type);
                                });

      // Also subscribe to trade updates for trade bubbles
      processor_->subscribe(symbol_id, NotificationType::TRADE,
                            [this](uint32_t sym, NotificationType type) {
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
}

void DomSurfacePanel::onDataUpdate(uint32_t symbol_id, NotificationType type) {
  if (symbol_id == current_symbol_id_) {
    if (type == NotificationType::TRADE) {
      // For trade updates, we'll update trade bubbles specifically
      updateTradeBubbles();
    }
    markDirty();
  }
}

void DomSurfacePanel::updateHeatmapData() {
  if (current_symbol_id_ == 0 || !processor_) return;

  // Request ALL available orderbook history (0 = no limit)
  auto history = processor_->getHistoricalOrderbooks(current_symbol_id_, 0);
  if (history.empty()) return;

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
  if (!history.empty()) {
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
  
  // Update Vulkan texture with new heatmap data
  updateVulkanTexture();
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

void DomSurfacePanel::detectLargeOrders(const OrderbookData& orderbook) {
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
    double x_pos = bounds_min_[0] + relative_time * (bounds_max_[0] - bounds_min_[0]);

    // Only render if within view
    if (x_pos < bounds_min_[0] || x_pos > bounds_max_[0]) {
      if (x_pos < bounds_min_[0]) continue;  // Too old
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

void DomSurfacePanel::renderLiquidityBars() {
  if (current_symbol_id_ == 0 || !processor_) return;

  // Get latest orderbook for liquidity bars
  auto orderbook_opt = processor_->getOrderbookData(current_symbol_id_);
  if (!orderbook_opt) return;

  const auto& orderbook = *orderbook_opt;

  // Calculate max volume across all levels for normalization
  double max_total_volume = 0.0;
  for (const auto& level : orderbook.bids) {
    max_total_volume = std::max(max_total_volume, level.size);
  }
  for (const auto& level : orderbook.asks) {
    max_total_volume = std::max(max_total_volume, level.size);
  }

  if (max_total_volume <= 0) return;

  ImDrawList* draw_list = ImPlot::GetPlotDrawList();
  if (!draw_list) return;

  // Get plot limits to determine where to draw the bars
  ImPlotRect plot_rect = ImPlot::GetPlotLimits();
  
  // Calculate bar width based on plot dimensions (10% of plot width as max)
  float max_bar_width = ImPlot::GetPlotSize().x * 0.1f;
  if (max_bar_width < 5.0f) max_bar_width = 5.0f;

  // Calculate the pixel height that corresponds to a small price range
  float bar_height_px = 3.0f; // Fixed height in pixels for each bar
  
  // Calculate price range per pixel to convert bar height
  double price_per_px = (plot_rect.Y.Max - plot_rect.Y.Min) / ImPlot::GetPlotSize().y;

  // Render bid liquidity bars (green) on the right side
  for (const auto& level : orderbook.bids) {
    // Calculate bar width based on volume
    float volume_ratio = static_cast<float>(level.size / max_total_volume);
    float bar_width = volume_ratio * max_bar_width;
    
    // Calculate the price range that corresponds to the bar height in pixels
    double price_range = price_per_px * bar_height_px;
    
    // Calculate the price range for the bar (centered at the price level)
    double top_price = level.price + price_range / 2.0;
    double bottom_price = level.price - price_range / 2.0;
    
    // Convert to pixel coordinates - note: ImPlot Y-axis is inverted (higher values are lower on screen)
    ImVec2 top_right = ImPlot::PlotToPixels(plot_rect.X.Max, top_price);
    ImVec2 bottom_right = ImPlot::PlotToPixels(plot_rect.X.Max, bottom_price);
    
    // Calculate left edge of the bar (extending left from the right edge)
    ImVec2 top_left = ImVec2(top_right.x - bar_width, top_right.y);
    ImVec2 bottom_left = ImVec2(bottom_right.x - bar_width, bottom_right.y);
    
    // Draw the bar - need to ensure correct rectangle orientation
    // In ImDrawList, the rectangle is drawn from top-left to bottom-right
    ImVec2 rect_min = ImVec2(top_left.x, std::min(top_right.y, bottom_right.y));
    ImVec2 rect_max = ImVec2(top_right.x, std::max(top_right.y, bottom_right.y));
    
    // Draw the bar
    ImU32 bid_color = IM_COL32(0, 255, 0, 180); // Green with transparency
    draw_list->AddRectFilled(rect_min, rect_max, bid_color);
  }

  // Render ask liquidity bars (red) on the right side
  for (const auto& level : orderbook.asks) {
    // Calculate bar width based on volume
    float volume_ratio = static_cast<float>(level.size / max_total_volume);
    float bar_width = volume_ratio * max_bar_width;
    
    // Calculate the price range that corresponds to the bar height in pixels
    double price_range = price_per_px * bar_height_px;
    
    // Calculate the price range for the bar (centered at the price level)
    double top_price = level.price + price_range / 2.0;
    double bottom_price = level.price - price_range / 2.0;
    
    // Convert to pixel coordinates - note: ImPlot Y-axis is inverted (higher values are lower on screen)
    ImVec2 top_right = ImPlot::PlotToPixels(plot_rect.X.Max, top_price);
    ImVec2 bottom_right = ImPlot::PlotToPixels(plot_rect.X.Max, bottom_price);
    
    // Calculate left edge of the bar (extending left from the right edge)
    ImVec2 top_left = ImVec2(top_right.x - bar_width, top_right.y);
    ImVec2 bottom_left = ImVec2(bottom_right.x - bar_width, bottom_right.y);
    
    // Draw the bar - need to ensure correct rectangle orientation
    // In ImDrawList, the rectangle is drawn from top-left to bottom-right
    ImVec2 rect_min = ImVec2(top_left.x, std::min(top_right.y, bottom_right.y));
    ImVec2 rect_max = ImVec2(top_right.x, std::max(top_right.y, bottom_right.y));
    
    // Draw the bar
    ImU32 ask_color = IM_COL32(255, 0, 0, 180); // Red with transparency
    draw_list->AddRectFilled(rect_min, rect_max, ask_color);
  }
}

double DomSurfacePanel::getMaxVolumeAtPrice(const OrderbookData& orderbook, double price) const {
  // Find the volume at the specified price level
  for (const auto& level : orderbook.bids) {
    if (std::abs(level.price - price) < 0.0001) { // Using small epsilon for floating point comparison
      return level.size;
    }
  }
  
  for (const auto& level : orderbook.asks) {
    if (std::abs(level.price - price) < 0.0001) { // Using small epsilon for floating point comparison
      return level.size;
    }
  }
  
  return 0.0; // Return 0 if price level not found
}

void DomSurfacePanel::render() {
  if (consumeDirty()) {
    updateHeatmapData();
    updateLargeOrderMarkers();
    updateTradeBubbles();
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
  }
  ImGui::SameLine();
  if (!current_symbol_name_.empty()) {
    ImGui::Text(" | Symbol: %s (%u) | Bins: %d | Orders: %zu", current_symbol_name_.c_str(), 
                current_symbol_id_, price_bins_, large_order_markers_.size());
  } else {
    ImGui::Text(" | Symbol ID: %u | Bins: %d | Orders: %zu", current_symbol_id_, price_bins_,
                large_order_markers_.size());
  }

  // Enable Pan/Zoom for DOM Surface
  std::string plot_id = "##DomHeatmap_" + std::to_string(current_symbol_id_);
  if (ImPlot::BeginPlot(plot_id.c_str(), ImVec2(-1, -1), ImPlotFlags_NoLegend)) {
    ImPlot::SetupAxes("Time", "Price");

    // Allow user to pan and zoom
    ImPlot::SetupAxis(ImAxis_X1, "Time", ImPlotAxisFlags_RangeFit);
    ImPlot::SetupAxis(ImAxis_Y1, "Price", ImPlotAxisFlags_RangeFit);

    // Add right-side Y-axis for liquidity bars
    ImPlot::SetupAxis(ImAxis_Y2, "Liquidity", ImPlotAxisFlags_AuxDefault | ImPlotAxisFlags_Opposite);

    // Set axis limits with option for user interaction
    ImPlot::SetupAxisLimits(ImAxis_X1, bounds_min_[0], bounds_max_[0],
                            heatmap_data_.empty() ? ImPlotCond_Always : ImPlotCond_Once);
    ImPlot::SetupAxisLimits(ImAxis_Y1, bounds_min_[1], bounds_max_[1],
                            heatmap_data_.empty() ? ImPlotCond_Always : ImPlotCond_Once);
    // Y2 axis should match Y1 limits
    ImPlot::SetupAxisLimits(ImAxis_Y2, bounds_min_[1], bounds_max_[1],
                            heatmap_data_.empty() ? ImPlotCond_Always : ImPlotCond_Once);

    // Use time history size for Cols and price_bins for Rows
    int rows = price_bins_;
    int cols = static_cast<int>(heatmap_data_.size()) / rows;

    if (cols > 0 && rows > 0) {
      ImPlot::PushColormap(ImPlotColormap_Viridis);

      // Use the Vulkan-accelerated texture if available
      if (vulkan_texture_id_ != nullptr) {
        // Render using the Vulkan texture
        // The texture represents the liquidity heatmap where X=Time, Y=Price
        ImPlot::PlotImage("Liquidity", vulkan_texture_id_,
                         ImPlotPoint(bounds_min_[0], bounds_min_[1]),
                         ImPlotPoint(bounds_max_[0], bounds_max_[1]));
      } else {
        // Fallback to CPU rendering
        ImPlot::PlotHeatmap("Liquidity", heatmap_data_.data(), rows, cols, 0, scale_max_, nullptr,
                            ImPlotPoint(bounds_min_[0], bounds_min_[1]),
                            ImPlotPoint(bounds_max_[0], bounds_max_[1]));
      }
      ImPlot::PopColormap();
    }

    // Render Large Order Markers OVER the heatmap
    renderLargeOrderMarkers();

    // Render Trade Bubbles OVER the heatmap and large order markers
    renderTradeBubbles();

    // Render Liquidity Bars on the right-hand price axis
    renderLiquidityBars();

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
    ImGui::Unindent(10.0f);
    ImGui::EndGroup();
  }

  end_panel_window();
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
}

void DomSurfacePanel::initializeVulkanResources(VulkanCore* core) {
  if (!core) return;
  
  vulkan_core_ = core;
  
  // Create Vulkan texture for heatmap
  createVulkanTexture();
}

void DomSurfacePanel::createVulkanTexture() {
  if (!vulkan_core_) return;

  // Initialize texture dimensions
  int width = price_bins_;
  int height = static_cast<int>(heatmap_data_.size() / price_bins_);
  if (height <= 0) height = 1; // Default to 1 if no data yet

  auto device = vulkan_core_->get_device();
  auto physicalDevice = vulkan_core_->get_physical_device();

  // Create the heatmap texture with appropriate dimensions
  VkImageCreateInfo imageInfo{.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                              .imageType = VK_IMAGE_TYPE_2D,
                              .format = VK_FORMAT_R8G8B8A8_UNORM,
                              .extent = {.width = static_cast<uint32_t>(width),
                                         .height = static_cast<uint32_t>(height),
                                         .depth = 1},
                              .mipLevels = 1,
                              .arrayLayers = 1,
                              .samples = VK_SAMPLE_COUNT_1_BIT,
                              .tiling = VK_IMAGE_TILING_OPTIMAL,
                              .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                       VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                              .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                              .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};

  if (vkCreateImage(device, &imageInfo, nullptr, &heatmap_image_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create heatmap image");
  }

  VkMemoryRequirements memRequirements;
  vkGetImageMemoryRequirements(device, heatmap_image_, &memRequirements);

  VkMemoryAllocateInfo allocInfo{
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = memRequirements.size,
      .memoryTypeIndex = vulkan_core_->find_memory_type(memRequirements.memoryTypeBits,
                                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};

  if (vkAllocateMemory(device, &allocInfo, nullptr, &heatmap_image_memory_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate heatmap image memory");
  }

  vkBindImageMemory(device, heatmap_image_, heatmap_image_memory_, 0);

  // Create Image View
  VkImageViewCreateInfo viewInfo{.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                 .image = heatmap_image_,
                                 .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                 .format = VK_FORMAT_R8G8B8A8_UNORM,
                                 .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                      .baseMipLevel = 0,
                                                      .levelCount = 1,
                                                      .baseArrayLayer = 0,
                                                      .layerCount = 1}};

  if (vkCreateImageView(device, &viewInfo, nullptr, &heatmap_image_view_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create heatmap image view");
  }

  // Create Sampler
  VkSamplerCreateInfo samplerInfo{.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                  .magFilter = VK_FILTER_LINEAR,
                                  .minFilter = VK_FILTER_LINEAR,
                                  .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                  .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                  .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                  .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                  .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
                                  .unnormalizedCoordinates = VK_FALSE};

  if (vkCreateSampler(device, &samplerInfo, nullptr, &heatmap_sampler_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create heatmap sampler");
  }

  // Register the texture with ImGui
  VkDescriptorSet descriptor_set = ImGui_ImplVulkan_AddTexture(heatmap_sampler_, heatmap_image_view_,
                                                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  vulkan_texture_id_ = (void*)descriptor_set;
  
  // Store current dimensions
  current_texture_width_ = width;
  current_texture_height_ = height;
}

void DomSurfacePanel::recreateVulkanTexture(int new_width, int new_height) {
  if (!vulkan_core_) return;

  // Clean up existing resources
  auto device = vulkan_core_->get_device();

  if (heatmap_sampler_ != VK_NULL_HANDLE) {
    vkDestroySampler(device, heatmap_sampler_, nullptr);
    heatmap_sampler_ = VK_NULL_HANDLE;
  }

  if (heatmap_image_view_ != VK_NULL_HANDLE) {
    vkDestroyImageView(device, heatmap_image_view_, nullptr);
    heatmap_image_view_ = VK_NULL_HANDLE;
  }

  if (heatmap_image_ != VK_NULL_HANDLE) {
    vkDestroyImage(device, heatmap_image_, nullptr);
    heatmap_image_ = VK_NULL_HANDLE;
  }

  if (heatmap_image_memory_ != VK_NULL_HANDLE) {
    vkFreeMemory(device, heatmap_image_memory_, nullptr);
    heatmap_image_memory_ = VK_NULL_HANDLE;
  }

  // Create new texture with updated dimensions
  VkImageCreateInfo imageInfo{.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                              .imageType = VK_IMAGE_TYPE_2D,
                              .format = VK_FORMAT_R8G8B8A8_UNORM,
                              .extent = {.width = static_cast<uint32_t>(new_width),
                                         .height = static_cast<uint32_t>(new_height),
                                         .depth = 1},
                              .mipLevels = 1,
                              .arrayLayers = 1,
                              .samples = VK_SAMPLE_COUNT_1_BIT,
                              .tiling = VK_IMAGE_TILING_OPTIMAL,
                              .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                       VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                              .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                              .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};

  if (vkCreateImage(device, &imageInfo, nullptr, &heatmap_image_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to recreate heatmap image");
  }

  VkMemoryRequirements memRequirements;
  vkGetImageMemoryRequirements(device, heatmap_image_, &memRequirements);

  VkMemoryAllocateInfo allocInfo{
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = memRequirements.size,
      .memoryTypeIndex = vulkan_core_->find_memory_type(memRequirements.memoryTypeBits,
                                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};

  if (vkAllocateMemory(device, &allocInfo, nullptr, &heatmap_image_memory_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to allocate heatmap image memory");
  }

  vkBindImageMemory(device, heatmap_image_, heatmap_image_memory_, 0);

  // Create Image View
  VkImageViewCreateInfo viewInfo{.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                 .image = heatmap_image_,
                                 .viewType = VK_IMAGE_VIEW_TYPE_2D,
                                 .format = VK_FORMAT_R8G8B8A8_UNORM,
                                 .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                                      .baseMipLevel = 0,
                                                      .levelCount = 1,
                                                      .baseArrayLayer = 0,
                                                      .layerCount = 1}};

  if (vkCreateImageView(device, &viewInfo, nullptr, &heatmap_image_view_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create heatmap image view");
  }

  // Create Sampler
  VkSamplerCreateInfo samplerInfo{.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                  .magFilter = VK_FILTER_LINEAR,
                                  .minFilter = VK_FILTER_LINEAR,
                                  .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                  .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                  .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                  .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                  .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
                                  .unnormalizedCoordinates = VK_FALSE};

  if (vkCreateSampler(device, &samplerInfo, nullptr, &heatmap_sampler_) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create heatmap sampler");
  }

  // Register the texture with ImGui
  VkDescriptorSet descriptor_set = ImGui_ImplVulkan_AddTexture(heatmap_sampler_, heatmap_image_view_,
                                                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  vulkan_texture_id_ = (void*)descriptor_set;
  
  // Update stored dimensions
  current_texture_width_ = new_width;
  current_texture_height_ = new_height;
}

void DomSurfacePanel::updateVulkanTexture() {
  if (!vulkan_core_ || heatmap_data_.empty()) return;

  // Convert heatmap data to RGBA format for the texture
  int time_steps = static_cast<int>(heatmap_data_.size()) / price_bins_;
  int height = price_bins_;

  if (time_steps <= 0 || height <= 0) return;

  // Check if texture needs to be recreated due to size change
  if (time_steps != current_texture_width_ || height != current_texture_height_) {
    recreateVulkanTexture(time_steps, height);
  }

  // Create temporary RGBA data
  std::vector<uint32_t> rgba_data(time_steps * height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < time_steps; ++x) {
      // heatmap_data_ is organized as [price_bin * time_steps + time_step]
      // So for position (x=time, y=price), we access [y * time_steps + x]
      double value = 0.0;
      if (y * time_steps + x < heatmap_data_.size()) {
        value = heatmap_data_[y * time_steps + x];
      }
      float normalized = static_cast<float>(value / scale_max_);

      // Apply heatmap intensity adjustment to sensitivity
      normalized = std::pow(normalized, 1.0f / heatmap_intensity_);

      // Clamp normalized value to [0, 1] range
      normalized = std::clamp(normalized, 0.0f, 1.0f);

      // Apply colormap (Dark Blue to Bright Yellow gradient)
      uint8_t r, g, b, a = 255;

      // Dark Blue (0, 0, 139) to Bright Yellow (255, 255, 0) gradient
      if (normalized <= 0.0f) {
        r = 0; g = 0; b = 139; // Dark Blue
      } else if (normalized < 0.5f) {
        // Transition from Dark Blue to Cyan
        float t = normalized * 2.0f; // Scale to 0-1 range for this segment
        r = static_cast<uint8_t>(0 + (t * (0 - 0)));   // Stay at 0
        g = static_cast<uint8_t>(0 + (t * (255 - 0))); // Go from 0 to 255
        b = static_cast<uint8_t>(139 + (t * (255 - 139))); // Go from 139 to 255
      } else if (normalized < 1.0f) {
        // Transition from Cyan to Bright Yellow
        float t = (normalized - 0.5f) * 2.0f; // Scale to 0-1 range for this segment
        r = static_cast<uint8_t>(0 + (t * (255 - 0))); // Go from 0 to 255
        g = static_cast<uint8_t>(255 + (t * (255 - 255))); // Stay at 255
        b = static_cast<uint8_t>(255 + (t * (0 - 255))); // Go from 255 to 0
      } else {
        // At maximum intensity - Bright Yellow
        r = 255; g = 255; b = 0; // Bright Yellow
      }

      // Store in row-major order for texture (x = column, y = row)
      // Flip vertically to match OpenGL/Vulkan coordinate system
      rgba_data[(height - 1 - y) * time_steps + x] = (a << 24) | (b << 16) | (g << 8) | r;
    }
  }

  // Upload the texture data to the GPU using staging buffer
  auto device = vulkan_core_->get_device();

  // Allocate staging buffer
  VkDeviceSize imageSize = static_cast<VkDeviceSize>(time_steps * height * sizeof(uint32_t));
  auto staging_buffer = vulkan_core_->get_memory_manager().allocate_staging_buffer(imageSize);

  // Copy image data to staging buffer
  memcpy(staging_buffer.mapped_ptr, rgba_data.data(), static_cast<size_t>(imageSize));

  // Create command buffer for transfer
  VkCommandBuffer commandBuffer = vulkan_core_->begin_single_time_commands();

  // Transition image layout to TRANSFER_DST_OPTIMAL
  VkImageMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; // Start with undefined layout
  barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = heatmap_image_;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;
  barrier.srcAccessMask = 0;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

  vkCmdPipelineBarrier(
      commandBuffer,
      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
      0,
      0, nullptr,
      0, nullptr,
      1, &barrier
  );

  // Copy buffer to image
  VkBufferImageCopy region = {};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;
  region.imageOffset = {0, 0, 0};
  region.imageExtent = {
      static_cast<uint32_t>(time_steps),
      static_cast<uint32_t>(height),
      1
  };

  vkCmdCopyBufferToImage(
      commandBuffer,
      staging_buffer.buffer,
      heatmap_image_,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      1, &region
  );

  // Transition image layout to SHADER_READ_ONLY_OPTIMAL
  VkImageMemoryBarrier shader_barrier = {};
  shader_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  shader_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  shader_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  shader_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  shader_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  shader_barrier.image = heatmap_image_;
  shader_barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  shader_barrier.subresourceRange.baseMipLevel = 0;
  shader_barrier.subresourceRange.levelCount = 1;
  shader_barrier.subresourceRange.baseArrayLayer = 0;
  shader_barrier.subresourceRange.layerCount = 1;
  shader_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  shader_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

  vkCmdPipelineBarrier(
      commandBuffer,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
      0,
      0, nullptr,
      0, nullptr,
      1, &shader_barrier
  );

  // Submit command buffer
  vulkan_core_->end_single_time_commands(commandBuffer);

  // Clean up staging buffer
  vulkan_core_->get_memory_manager().deallocate_buffer(staging_buffer);
}

void DomSurfacePanel::cleanupVulkanResources() {
  if (!vulkan_core_) return;
  
  auto device = vulkan_core_->get_device();
  
  if (heatmap_sampler_ != VK_NULL_HANDLE) {
    vkDestroySampler(device, heatmap_sampler_, nullptr);
    heatmap_sampler_ = VK_NULL_HANDLE;
  }
  
  if (heatmap_image_view_ != VK_NULL_HANDLE) {
    vkDestroyImageView(device, heatmap_image_view_, nullptr);
    heatmap_image_view_ = VK_NULL_HANDLE;
  }
  
  if (heatmap_image_ != VK_NULL_HANDLE) {
    vkDestroyImage(device, heatmap_image_, nullptr);
    heatmap_image_ = VK_NULL_HANDLE;
  }
  
  if (heatmap_image_memory_ != VK_NULL_HANDLE) {
    vkFreeMemory(device, heatmap_image_memory_, nullptr);
    heatmap_image_memory_ = VK_NULL_HANDLE;
  }

  vulkan_texture_id_ = nullptr;
}

void DomSurfacePanel::updateTradeBubbles() {
  if (current_symbol_id_ == 0 || !processor_) return;

  // Get recent trades for the current symbol from the analytics
  auto symbol_analytics = processor_->getSymbolAnalytics(current_symbol_id_);

  // Process recent trades from the analytics
  for (const auto& trade : symbol_analytics.recent_trades) {
    // Only add trades that are newer than our last processed timestamp
    if (trade.timestamp > last_trade_timestamp_) {
      // Convert from the internal TradeData to the external TradeData format
      BTQuant::Data::TradeData external_trade;
      external_trade.timestamp = trade.timestamp;
      external_trade.price = trade.price;
      external_trade.volume = static_cast<float>(trade.size);
      external_trade.side = trade.is_buy ? BTQuant::Data::TradeSide::BUY : BTQuant::Data::TradeSide::SELL;
      external_trade.exchange_id = 0; // Default exchange ID
      external_trade.flags = 0; // Default flags

      trade_bubbles_.push_back(external_trade);

      // Keep only recent trades to prevent unlimited growth
      if (trade_bubbles_.size() > TRADE_HISTORY_SIZE) {
        trade_bubbles_.erase(trade_bubbles_.begin());
      }

      last_trade_timestamp_ = trade.timestamp;
    }
  }
  
  // Clean up fully faded out trade bubbles
  uint64_t current_time = std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now().time_since_epoch())
                              .count();
  
  trade_bubbles_.erase(
      std::remove_if(trade_bubbles_.begin(), trade_bubbles_.end(),
                     [current_time](const BTQuant::Data::TradeData& trade) {
                       uint64_t age_us = current_time - trade.timestamp;
                       return age_us >= TRADE_BUBBLE_FADE_DURATION_US;
                     }),
      trade_bubbles_.end());
}

void DomSurfacePanel::renderTradeBubbles() {
  if (trade_bubbles_.empty()) return;

  ImPlot::PushStyleVar(ImPlotStyleVar_MarkerSize, 1.0f);

  // Get plot area for manual circle rendering
  ImPlotRect plot_rect = ImPlot::GetPlotLimits();

  // Calculate time range for mapping timestamps to X coordinates
  double time_range = bounds_max_[0] - bounds_min_[0];
  uint64_t min_timestamp = history_start_timestamp_;
  uint64_t max_timestamp = history_end_timestamp_;
  double timestamp_range = static_cast<double>(max_timestamp - min_timestamp);

  // Render each trade bubble as a circle
  for (const auto& trade : trade_bubbles_) {
    ImU32 color = getTradeBubbleColor(trade);
    ImU32 border_color = IM_COL32(255, 255, 255, 200);  // White semi-transparent border

    // Calculate X position based on timestamp relative to history range
    double relative_time = 0.0;
    if (timestamp_range > 0) {
      relative_time = static_cast<double>(trade.timestamp - min_timestamp) / timestamp_range;
    }
    
    // Map to plot coordinates: X = time, Y = price
    double x_pos = bounds_min_[0] + relative_time * time_range;
    double y_pos = trade.price;

    // Only render if within view bounds
    if (x_pos < bounds_min_[0] || x_pos > bounds_max_[0] || 
        y_pos < bounds_min_[1] || y_pos > bounds_max_[1]) {
      continue;
    }

    // Convert plot coordinates to pixel coordinates
    ImVec2 pixel_pos = ImPlot::PlotToPixels(x_pos, y_pos);

    // Calculate bubble radius based on volume
    float radius = calculateTradeBubbleRadius(trade.volume);

    // Draw filled circle
    ImDrawList* draw_list = ImPlot::GetPlotDrawList();
    if (draw_list) {
      draw_list->AddCircleFilled(pixel_pos, radius, color, 32);
      draw_list->AddCircle(pixel_pos, radius, border_color, 32, 1.5f);

      // Check for hover and show tooltip
      ImVec2 mouse_pos = ImGui::GetMousePos();
      float distance = std::sqrt(std::pow(mouse_pos.x - pixel_pos.x, 2) +
                                 std::pow(mouse_pos.y - pixel_pos.y, 2));

      if (distance < radius) {
        ImGui::SetTooltip("%s", getTradeBubbleTooltip(trade).c_str());
      }
    }
  }

  ImPlot::PopStyleVar();
}

float DomSurfacePanel::calculateTradeBubbleRadius(float volume) const {
  if (volume <= 0.0f) return TRADE_BUBBLE_BASE_RADIUS;  // Base radius for invalid volumes

  // Use logarithmic scaling: log(volume + 1) to handle volume = 0 gracefully
  // This prevents massive trades from covering the entire price axis
  float log_volume = std::log(volume + 1.0f);

  // Use a more adaptive normalization that scales with actual market conditions
  // Rather than a fixed maximum, we'll use a dynamic reference that adapts to recent trade volumes
  float max_log_volume = std::log(TRADE_BUBBLE_MAX_VOLUME + 1.0f);

  // Handle edge case where max_log_volume might be 0
  if (max_log_volume <= 0.0f) {
    max_log_volume = std::log(1000.0f + 1.0f); // Default to 1000 as reference max volume
  }

  float normalized_log_volume = log_volume / max_log_volume;

  // Clamp the normalized value to prevent exceeding intended radius range
  normalized_log_volume = std::clamp(normalized_log_volume, 0.0f, 1.0f);

  // Apply additional curve to make the scaling more gradual at low volumes
  // and steeper at high volumes for better visual distinction
  // Using a power function to adjust the distribution of bubble sizes
  // Reduced the exponent to make the scaling even more conservative for large trades
  normalized_log_volume = std::pow(normalized_log_volume, 0.5f);

  // Scale radius from base to max based on normalized log volume
  // Use calibrated scaling for better visual representation of trade volumes
  return TRADE_BUBBLE_BASE_RADIUS + (TRADE_BUBBLE_MAX_RADIUS - TRADE_BUBBLE_BASE_RADIUS) * normalized_log_volume;
}

ImU32 DomSurfacePanel::getTradeBubbleColor(const BTQuant::Data::TradeData& trade) const {
  // Get current time to calculate age of the trade
  uint64_t current_time = std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::steady_clock::now().time_since_epoch())
                              .count();

  // Calculate age of the trade in microseconds
  uint64_t age_us = current_time - trade.timestamp;

  // Calculate fade ratio (0.0 = fully faded, 1.0 = fully opaque)
  float fade_ratio = 1.0f - static_cast<float>(age_us) / static_cast<float>(TRADE_BUBBLE_FADE_DURATION_US);
  fade_ratio = std::clamp(fade_ratio, 0.0f, 1.0f);

  // Apply smooth easing function for more natural fade-out
  // Using exponential decay for smoother initial fade with more gradual tail-off
  // This creates a more natural fade that starts slowly and accelerates toward the end
  float eased_fade_ratio = std::exp(-2.0f * (1.0f - fade_ratio)) * fade_ratio;

  // Alternative: Enhanced quintic easing for even smoother transitions
  // Using smoother quintic easing function: 6t⁵ - 15t⁴ + 10t³ for even smoother transitions
  float quintic_ease = fade_ratio * fade_ratio * fade_ratio * (fade_ratio * (fade_ratio * 6.0f - 15.0f) + 10.0f);
  
  // Blend both easing functions for optimal smoothness
  float final_fade_ratio = 0.7f * eased_fade_ratio + 0.3f * quintic_ease;

  // Calculate alpha based on fade ratio
  uint8_t base_alpha = 200;  // Increased base alpha value for better visibility
  uint8_t alpha = static_cast<uint8_t>(base_alpha * final_fade_ratio);

  // Color based on trade side: Green for BUY, Red for SELL with fade-out effect
  if (trade.side == BTQuant::Data::TradeSide::BUY) {
    return IM_COL32(0, 255, 0, alpha);  // Green with fade-out transparency
  } else {
    return IM_COL32(255, 0, 0, alpha);  // Red with fade-out transparency
  }
}

std::string DomSurfacePanel::getTradeBubbleTooltip(const BTQuant::Data::TradeData& trade) const {
  std::string side = (trade.side == BTQuant::Data::TradeSide::BUY) ? "Buy" : "Sell";
  return std::format("Trade {}: {} @ ${:.2f}", side, trade.volume, trade.price);
}

}  // namespace RenderEngine
}  // namespace BTQuant