#include "components/dom_surface_panel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>

#include <imgui.h>
#include "backends/imgui_impl_vulkan.h"
#include "symbol_registry.hpp"
#include "../../include/analytics/cluster_engine.hpp"

namespace BTQuant {

DomSurfacePanel::DomSurfacePanel(std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(PanelConfig{.title = "DOM Surface", .type = PanelType::HEATMAP}),
      processor_(processor),
      heatmap_texture_{} {
  // Initialize Vulkan texture if Vulkan core is available
  // Vulkan compute removed - using CPU-based heatmap rendering initially
  // But we'll prepare for Vulkan-accelerated texture rendering
}

DomSurfacePanel::~DomSurfacePanel() {
  if (subscription_id_ > 0 && processor_) {
    processor_->unsubscribe(subscription_id_);
  }

  // Clean up Vulkan texture if initialized
  destroyVulkanTexture();
  
  // Clean up cluster engine
  cluster_engine_.reset();
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

  // Initialize cluster engine with appropriate tick size for the symbol
  if (processor_) {
    auto symbol_info_opt = SymbolRegistry::instance().get_symbol_info(current_symbol_id_);
    if (symbol_info_opt) {
      double tick_size = symbol_info_opt->tick_size > 0.0 ? symbol_info_opt->tick_size : 0.25; // Default tick size
      cluster_engine_ = std::make_unique<Analytics::ClusterEngine>(tick_size);
    } else {
      cluster_engine_ = std::make_unique<Analytics::ClusterEngine>(0.25); // Default tick size
    }
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
      
      // Process trade through cluster engine for cumulative volume data
      if (cluster_engine_ && processor_) {
        auto analytics = processor_->getSymbolAnalytics(current_symbol_id_);
        // Process the most recent trade through the cluster engine
        if (!analytics.recent_trades.empty()) {
          const auto& latest_trade = analytics.recent_trades.back();
          
          // Convert RenderEngine::TradeData to MarketData::Trade for cluster engine
          MarketData::Trade converted_trade;
          converted_trade.price = latest_trade.price;
          converted_trade.quantity = latest_trade.size;  // Use 'size' instead of 'quantity'
          converted_trade.timestamp_us = latest_trade.timestamp;
          converted_trade.is_buyer_maker = latest_trade.is_buy;  // Use 'is_buy' instead of 'is_buyer_maker'
          
          // Process trade with default time bucket (0 for now)
          cluster_engine_->processTrade(converted_trade, 0);
        }
      }
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

  // If multi-exchange aggregation is enabled, aggregate data from multiple exchanges
  if (multi_exchange_aggregation_enabled_ && !selected_exchanges_.empty()) {
    // Aggregate data from multiple exchanges
    aggregateMultiExchangeData();
  } else {
    // Use single exchange data (original behavior)
    aggregateSingleExchangeData();
  }
}

void DomSurfacePanel::aggregateSingleExchangeData() {
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

void DomSurfacePanel::aggregateMultiExchangeData() {
  // For multi-exchange aggregation, we need to get data from multiple exchanges
  // This is a simplified implementation - in a real system, we'd need to:
  // 1. Get historical data for the same symbol from multiple exchanges
  // 2. Align timestamps across exchanges
  // 3. Aggregate volumes appropriately
  
  std::vector<RenderEngine::OrderbookData> combined_history;
  
  // Get the symbol name for the current symbol ID to find equivalent symbols on other exchanges
  std::string base_symbol_name = "UNKNOWN";
  auto symbol_info_opt = SymbolRegistry::instance().get_symbol_info(current_symbol_id_);
  if (symbol_info_opt) {
    base_symbol_name = symbol_info_opt->symbol;
  }
  
  // If we have selected exchanges, try to get data from them
  if (!selected_exchanges_.empty()) {
    // For each selected exchange, get the corresponding symbol data
    for (const auto& exchange : selected_exchanges_) {
      // Find the symbol ID for the same symbol on this exchange
      auto exchange_symbol_id_opt = SymbolRegistry::instance().get_symbol_id(exchange, base_symbol_name);
      
      if (exchange_symbol_id_opt.has_value()) {
        auto exchange_history = processor_->getHistoricalOrderbooks(exchange_symbol_id_opt.value(), 0);
        
        // For now, we'll just append the data from each exchange
        // In a real implementation, we would need to align timestamps and merge the data properly
        combined_history.insert(combined_history.end(), exchange_history.begin(), exchange_history.end());
      }
    }
  } else {
    // If no specific exchanges are selected, use the current symbol's data as a fallback
    auto base_history = processor_->getHistoricalOrderbooks(current_symbol_id_, 0);
    combined_history = base_history;
  }
  
  // If no data was found, return early
  if (combined_history.empty()) {
    auto base_history = processor_->getHistoricalOrderbooks(current_symbol_id_, 0);
    if (base_history.empty()) return;
    combined_history = base_history;
  }
  
  // Sort combined history by timestamp to ensure proper chronological order
  std::sort(combined_history.begin(), combined_history.end(), 
            [](const auto& a, const auto& b) {
              return a.timestamp < b.timestamp;
            });
  
  // Determine price range across all exchanges
  double min_price = std::numeric_limits<double>::max();
  double max_price = std::numeric_limits<double>::lowest();

  if (auto_scale_price_) {
    for (const auto& book : combined_history) {
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
      auto latest = combined_history.back();
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
    const auto& latest = combined_history.back();
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
  if (!combined_history.empty()) {
    history_start_timestamp_ = combined_history.front().timestamp;
    history_end_timestamp_ = combined_history.back().timestamp;
  }

  // Ensure valid time range
  if (history_end_timestamp_ <= history_start_timestamp_) {
    history_end_timestamp_ = history_start_timestamp_ + 1;
  }

  double price_step = (max_price - min_price) / static_cast<double>(price_bins_);
  int time_steps = static_cast<int>(combined_history.size());
  size_t total_size = static_cast<size_t>(price_bins_) * static_cast<size_t>(time_steps);

  if (heatmap_data_.size() != total_size) {
    heatmap_data_.assign(total_size, 0.0);
  } else {
    std::fill(heatmap_data_.begin(), heatmap_data_.end(), 0.0);
  }

  double max_vol = 0;

  // Aggregate data from all exchanges at each time step
  for (int t = 0; t < time_steps; ++t) {
    const auto& book = combined_history[t];

    // Bids
    for (const auto& level : book.bids) {
      if (level.price >= min_price && level.price < max_price) {
        int bin = static_cast<int>((level.price - min_price) / price_step);
        if (bin >= 0 && bin < price_bins_) {
          // In multi-exchange mode, we aggregate volumes from all exchanges
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
          // In multi-exchange mode, we aggregate volumes from all exchanges
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

    // Initialize Vulkan texture if not already done and if we have the Vulkan core
    if (!texture_initialized_ && vulkan_core_) {
      initializeVulkanTexture();
    }

    // Update Vulkan texture if available
    if (texture_initialized_) {
      updateVulkanTexture();
    }
    
    // Update MMT layout data if enabled
    if (show_mmt_layout_) {
      updateMMTLayoutData();
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
  ImGui::Checkbox("Show MMT 5-Column Layout", &show_mmt_layout_);
  ImGui::SameLine();
  ImGui::Text(" | Symbols: %u | Bins: %d | Orders: %zu | Trades: %zu", current_symbol_id_, price_bins_,
              large_order_markers_.size(), trade_bubbles_.size());

  // If MMT layout is enabled, render it instead of the heatmap
  if (show_mmt_layout_) {
    renderMMTLayout();
  } else {
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
      // Apply center mode if enabled
      if (mmt_center_mode_) {
        // Calculate center price
        double center_price = 0.0;
        auto orderbook_opt = processor_ ? processor_->getOrderbookData(current_symbol_id_) : std::nullopt;
        if (orderbook_opt && !orderbook_opt->bids.empty() && !orderbook_opt->asks.empty()) {
          center_price = (orderbook_opt->bids.front().price + orderbook_opt->asks.front().price) / 2.0;
        } else if (orderbook_opt && !orderbook_opt->bids.empty()) {
          center_price = orderbook_opt->bids.front().price;
        } else if (orderbook_opt && !orderbook_opt->asks.empty()) {
          center_price = orderbook_opt->asks.front().price;
        } else {
          // Fallback if no orderbook data available
          center_price = (bounds_min_[1] + bounds_max_[1]) / 2.0;
        }

        // Calculate range based on center price and mmt_center_range_
        double range = center_price * mmt_center_range_;
        ImPlot::SetupAxisLimits(ImAxis_Y1, center_price - range, center_price + range,
                                ImPlotCond_Always); // Use Always to enforce center mode
      } else {
        ImPlot::SetupAxisLimits(ImAxis_Y1, bounds_min_[1], bounds_max_[1],
                                ImPlotCond_Once);
      }

      // Use Vulkan-accelerated heatmap texture if available
      if (texture_initialized_ && heatmap_texture_id_) {
        // Render the heatmap using the Vulkan texture
        // First, ensure the texture is updated with current data
        updateVulkanTexture();

        // Render the texture as an image overlay on the plot
        // We'll use ImPlot::PlotImage to draw the texture
        ImPlot::PlotImage("Liquidity",
                          heatmap_texture_id_,
                          ImPlotPoint(bounds_min_[0], bounds_min_[1]),
                          ImPlotPoint(bounds_max_[0], bounds_max_[1]));
      } else {
        // Fallback to CPU-based heatmap rendering if Vulkan texture is not available
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
  }

  // Debug Overlay for DOM troubleshooting
  if (heatmap_data_.size() > 0 && !show_mmt_layout_) {
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

  // Create a dummy invisible button to capture right-clicks for the context menu
  // This ensures the context menu appears when right-clicking anywhere in the header area
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));  // Transparent button
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.3f, 0.2f));  // Slightly highlighted on hover
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.4f, 0.4f, 0.4f, 0.3f));   // More highlighted when active
  
  // Invisible button that spans the entire header area to capture right-clicks
  if (ImGui::InvisibleButton("##DOMHeaderArea", ImVec2(ImGui::GetContentRegionAvail().x, 25.0f))) {
    // Left click action - could be used for other interactions if needed
  }
  
  // Add context menu for the DOM header area
  if (ImGui::BeginPopupContextItem("##DOMHeaderArea")) {
    if (ImGui::MenuItem("Aggregated Heatmap", nullptr, &multi_exchange_aggregation_enabled_)) {
      // Toggle multi-exchange aggregation
    }

    // Add exchange selection submenu if multi-exchange aggregation is enabled
    if (multi_exchange_aggregation_enabled_) {
      if (ImGui::BeginMenu("Select Exchanges")) {
        // Get available exchanges from the processor or symbol registry
        std::vector<std::string> available_exchanges;

        // Try to get exchanges from the processor if available
        if (processor_) {
          // Attempt to get exchanges from the processor's symbol registry
          // This assumes the processor has access to a symbol registry
          auto& registry = SymbolRegistry::instance();
          available_exchanges = registry.get_exchanges();

          // If no exchanges were found, use a default list
          if (available_exchanges.empty()) {
            available_exchanges = {
              "Binance", "Coinbase", "Kraken", "Bybit", "OKX", "Bitfinex", "Huobi"
            };
          }
        } else {
          // Use default exchanges if processor is not available
          available_exchanges = {
            "Binance", "Coinbase", "Kraken", "Bybit", "OKX", "Bitfinex", "Huobi"
          };
        }

        for (auto& exchange : available_exchanges) {
          bool is_selected = std::find(selected_exchanges_.begin(), selected_exchanges_.end(), exchange) != selected_exchanges_.end();
          if (ImGui::MenuItem(exchange.c_str(), nullptr, &is_selected)) {
            if (is_selected) {
              // Add exchange to selection if not already present
              if (std::find(selected_exchanges_.begin(), selected_exchanges_.end(), exchange) == selected_exchanges_.end()) {
                selected_exchanges_.push_back(exchange);
              }
            } else {
              // Remove exchange from selection
              selected_exchanges_.erase(
                std::remove(selected_exchanges_.begin(), selected_exchanges_.end(), exchange),
                selected_exchanges_.end()
              );
            }
          }
        }
        ImGui::EndMenu();
      }
    }
    
    // Add MMT layout options to the context menu
    if (ImGui::BeginMenu("MMT Layout")) {
      ImGui::MenuItem("Enable 5-Column Layout", nullptr, &show_mmt_layout_);
      ImGui::MenuItem("Center Mode", nullptr, &mmt_center_mode_);
      if (ImGui::BeginMenu("Display Levels")) {
        if (ImGui::MenuItem("5 Levels", nullptr, mmt_display_levels_ == 5)) mmt_display_levels_ = 5;
        if (ImGui::MenuItem("10 Levels", nullptr, mmt_display_levels_ == 10)) mmt_display_levels_ = 10;
        if (ImGui::MenuItem("20 Levels", nullptr, mmt_display_levels_ == 20)) mmt_display_levels_ = 20;
        if (ImGui::MenuItem("30 Levels", nullptr, mmt_display_levels_ == 30)) mmt_display_levels_ = 30;
        if (ImGui::MenuItem("50 Levels", nullptr, mmt_display_levels_ == 50)) mmt_display_levels_ = 50;
        ImGui::EndMenu();
      }
      ImGui::EndMenu();
    }

    ImGui::EndPopup();
  }

  ImGui::PopStyleColor(3); // Restore button colors

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
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(5, 0));
  ImGui::PushItemWidth(150);
  ImGui::SliderFloat("##Threshold", &large_order_threshold_, 1.0f, 50.0f, "Threshold: %.1fx", ImGuiSliderFlags_Logarithmic);
  ImGui::PopItemWidth();
  ImGui::SameLine();
  ImGui::PushItemWidth(150);
  ImGui::SliderInt("Max Markers", &max_large_order_markers_, 10, 500);
  ImGui::PopItemWidth();
  ImGui::SameLine();
  ImGui::Checkbox("Fade Out", &enable_fade_out_);
  ImGui::PopStyleVar();
  ImGui::Separator();

  // Add Multi-Exchange Aggregation controls
  if (multi_exchange_aggregation_enabled_) {
    ImGui::Text("Multi-Exchange Aggregation:");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "ACTIVE"); // Green indicator
    ImGui::SameLine();
    if (ImGui::SmallButton("Configure##Exchanges")) {
      // This would open a modal dialog or expand controls, but for now we'll just show the context menu
      ImGui::OpenPopup("##DOMHeaderArea");
    }
    ImGui::Separator();
  }

  // Add Persistent Level controls
  ImGui::Text("Persistent Levels:");
  ImGui::SameLine();
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(5, 0));
  ImGui::PushItemWidth(150);
  ImGui::SliderInt("Persistence (ms)", reinterpret_cast<int*>(&persistence_threshold_ms_), 1000, 30000, "%d ms");
  ImGui::PopItemWidth();
  ImGui::SameLine();
  ImGui::PushItemWidth(150);
  ImGui::SliderInt("Timeout (ms)", reinterpret_cast<int*>(&persistence_timeout_ms_), 10000, 120000, "%d ms");
  ImGui::PopItemWidth();
  ImGui::PopStyleVar();
  ImGui::Separator();

  // Add Flush DOM Ruler controls
  ImGui::Text("Flush DOM Ruler:");
  ImGui::SameLine();
  ImGui::Checkbox("Show##FlushDOMRuler", &show_flush_dom_ruler_);
  ImGui::SameLine();
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(5, 0));
  ImGui::PushItemWidth(150);
  ImGui::SliderFloat("Width##FlushDOMRuler", &flush_dom_ruler_width_, 0.01f, 0.2f, "%.2f%%", ImGuiSliderFlags_Logarithmic);
  ImGui::PopItemWidth();
  ImGui::PopStyleVar();
  ImGui::Separator();

  // Add MMT Layout controls
  ImGui::Text("MMT Layout:");
  ImGui::SameLine();
  ImGui::Checkbox("Show##MMTLayout", &show_mmt_layout_);
  ImGui::SameLine();
  if (show_mmt_layout_) {
    ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "ACTIVE"); // Cyan indicator when active
  }
  ImGui::Separator();
}

void DomSurfacePanel::renderFlushDOMRuler() {
  // TODO: Implement Flush DOM Ruler rendering
  // This function should render the live orderbook at the right edge of the heatmap panel
  // For now, this is a stub implementation
}

void DomSurfacePanel::initializeVulkanTexture() {
  // This method would be called when we have access to the VulkanCore
  // For now, we'll implement it assuming we have access to vulkan_core_
  if (!vulkan_core_ || texture_initialized_) {
    return;
  }

  try {
    // Get reference to GPUMemoryManager
    GPUMemoryManager& memory_manager = vulkan_core_->get_memory_manager();

    // Allocate a texture for the heatmap (initial size, will be resized as needed)
    uint32_t width = 1024;  // Default width
    uint32_t height = 1024; // Default height

    heatmap_texture_ = memory_manager.allocate_image(
        width, height,
        VK_FORMAT_R32G32B32A32_SFLOAT,  // Format for heatmap data
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    // Create image view for the heatmap texture
    VkImageViewCreateInfo view_info = {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = heatmap_texture_.image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;

    VkResult result = vkCreateImageView(vulkan_core_->get_device(), &view_info, nullptr, &heatmap_image_view_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image view for heatmap texture");
    }

    // Create sampler for the heatmap texture
    VkSamplerCreateInfo sampler_info = {};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.anisotropyEnable = VK_FALSE;
    sampler_info.maxAnisotropy = 1.0f;
    sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    sampler_info.unnormalizedCoordinates = VK_FALSE;
    sampler_info.compareEnable = VK_FALSE;
    sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    result = vkCreateSampler(vulkan_core_->get_device(), &sampler_info, nullptr, &heatmap_sampler_);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to create sampler for heatmap texture");
    }

    // Register the texture with ImGui using ImGui_ImplVulkan_AddTexture
    // This creates an ImTextureID that can be used with ImGui::Image
    //heatmap_texture_id_ = ImGui_ImplVulkan_AddTexture(
    //    heatmap_sampler_,
    //    heatmap_image_view_,
    //    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    //);

    texture_initialized_ = true;
    std::cout << "[DomSurfacePanel] Vulkan texture initialized successfully" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "[DomSurfacePanel] Failed to initialize Vulkan texture: " << e.what() << std::endl;
    texture_initialized_ = false;
  }
}

void DomSurfacePanel::updateVulkanTexture() {
  if (!texture_initialized_) {
    return;
  }

  // This method updates the texture with new heatmap data
  // Implementation involves copying heatmap_data_ to the GPU texture
  
  if (heatmap_data_.empty()) {
    return; // Nothing to update
  }

  // Calculate dimensions based on heatmap data
  int rows = price_bins_;
  int cols = static_cast<int>(heatmap_data_.size()) / rows;
  
  if (rows <= 0 || cols <= 0) {
    return; // Invalid dimensions
  }

  try {
    // Get Vulkan device and memory manager
    VkDevice device = vulkan_core_->get_device();
    GPUMemoryManager& memory_manager = vulkan_core_->get_memory_manager();
    
    // Prepare heatmap data for GPU upload
    // Convert double values to RGBA float format for the texture
    std::vector<float> texture_data(rows * cols * 4, 0.0f); // 4 channels (RGBA)
    
    // Map heatmap values to color based on intensity and colormap
    double max_val = scale_max_ / heatmap_intensity_;
    if (max_val <= 0.0) max_val = 1.0; // Prevent division by zero
    
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        size_t idx = i * cols + j;
        double val = (idx < heatmap_data_.size()) ? heatmap_data_[idx] : 0.0;
        
        // Normalize value to [0, 1]
        float norm_val = static_cast<float>(std::min(val / max_val, 1.0));
        
        // Map to Viridis-like color (simplified)
        // This is a simplified approximation of the Viridis colormap
        float r = std::min(1.0f, 0.8f * norm_val);
        float g = std::min(1.0f, 0.9f * norm_val * norm_val);
        float b = std::min(1.0f, norm_val * norm_val * norm_val);
        float a = norm_val; // Alpha based on intensity
        
        // Set RGBA values
        size_t tex_idx = (i * cols + j) * 4;
        texture_data[tex_idx + 0] = r; // R
        texture_data[tex_idx + 1] = g; // G
        texture_data[tex_idx + 2] = b; // B
        texture_data[tex_idx + 3] = a; // A
      }
    }
    
    // Upload data to the GPU texture
    // This would typically involve:
    // 1. Creating a staging buffer
    // 2. Copying data to the staging buffer
    // 3. Submitting a command buffer to copy from staging to the texture
    // 4. Properly transitioning image layouts
    
    // For now, we'll just log that the update should happen
    std::cout << "[DomSurfacePanel] Prepared " << rows << "x" << cols 
              << " texture data for GPU upload" << std::endl;
              
  } catch (const std::exception& e) {
    std::cerr << "[DomSurfacePanel] Failed to update Vulkan texture: " << e.what() << std::endl;
  }
}

void DomSurfacePanel::destroyVulkanTexture() {
  if (texture_initialized_ && vulkan_core_) {
    try {
      // Remove the texture from ImGui's texture registry if needed
      // Note: ImGui_ImplVulkan_RemoveTexture is available but typically not needed
      // as the descriptor sets are managed by the pool
      
      // Destroy sampler
      if (heatmap_sampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(vulkan_core_->get_device(), heatmap_sampler_, nullptr);
        heatmap_sampler_ = VK_NULL_HANDLE;
      }
      
      // Destroy image view
      if (heatmap_image_view_ != VK_NULL_HANDLE) {
        vkDestroyImageView(vulkan_core_->get_device(), heatmap_image_view_, nullptr);
        heatmap_image_view_ = VK_NULL_HANDLE;
      }
      
      // Deallocate the image
      GPUMemoryManager& memory_manager = vulkan_core_->get_memory_manager();
      memory_manager.deallocate_image(heatmap_texture_);
      
      texture_initialized_ = false;
      std::cout << "[DomSurfacePanel] Vulkan texture destroyed successfully" << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "[DomSurfacePanel] Failed to destroy Vulkan texture: " << e.what() << std::endl;
    }
  }
}

void DomSurfacePanel::updateMMTLayoutData() {
  // Update data for the 5-column MMT layout
  // This method prepares the data needed for the MMT-style table view
  if (current_symbol_id_ == 0 || !processor_) return;

  // Get the latest orderbook data for the current symbol
  auto orderbook_opt = processor_->getOrderbookData(current_symbol_id_);
  if (!orderbook_opt) return;

  const auto& orderbook = *orderbook_opt;
  
  // The data is already available in the orderbook, so we just need to prepare for rendering
  // The MMT layout will render the orderbook data in 5 columns: [Buys | Asks | Price | Bids | Sells]
}

void DomSurfacePanel::renderHorizontalVolumeBars(ImDrawList* draw_list, ImVec2 pos, float width, float height, 
                                                 double buy_volume, double sell_volume, 
                                                 double max_possible_volume) {
  // Calculate normalized bar widths based on volumes
  float normalized_buy_width = 0.0f;
  float normalized_sell_width = 0.0f;
  
  if (max_possible_volume > 0.0) {
    normalized_buy_width = static_cast<float>(buy_volume / max_possible_volume) * width;
    normalized_sell_width = static_cast<float>(sell_volume / max_possible_volume) * width;
  }
  
  // Draw buy volume bar (green, extending right from the left side)
  ImVec2 buy_bar_start = ImVec2(pos.x, pos.y);
  ImVec2 buy_bar_end = ImVec2(pos.x + normalized_buy_width, pos.y + height);
  if (normalized_buy_width > 0) {
    draw_list->AddRectFilled(buy_bar_start, buy_bar_end, IM_COL32(0, 255, 0, 100)); // Green with transparency
  }
  
  // Draw sell volume bar (red, extending left from the right side)
  ImVec2 sell_bar_start = ImVec2(pos.x + width - normalized_sell_width, pos.y);
  ImVec2 sell_bar_end = ImVec2(pos.x + width, pos.y + height);
  if (normalized_sell_width > 0) {
    draw_list->AddRectFilled(sell_bar_start, sell_bar_end, IM_COL32(255, 0, 0, 100)); // Red with transparency
  }
}

void DomSurfacePanel::renderMMTLayout() {
  // Render the 5-column MMT layout as a table
  if (current_symbol_id_ == 0 || !processor_) return;

  // Get the latest orderbook data for the current symbol
  auto orderbook_opt = processor_->getOrderbookData(current_symbol_id_);
  if (!orderbook_opt) {
    ImGui::Text("No orderbook data available");
    return;
  }

  const auto& orderbook = *orderbook_opt;

  // Calculate how many levels to display
  int display_levels = std::min(mmt_display_levels_, static_cast<int>(std::max(orderbook.bids.size(), orderbook.asks.size()))); // Use a reasonable default

  // Create the 5-column table
  if (ImGui::BeginTable("MMTLayoutTable", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_SizingStretchSame)) {
    ImGui::TableSetupColumn("Buys", ImGuiTableColumnFlags_WidthStretch, 0.2f);
    ImGui::TableSetupColumn("Asks", ImGuiTableColumnFlags_WidthStretch, 0.2f);
    ImGui::TableSetupColumn("Price", ImGuiTableColumnFlags_WidthStretch, 0.2f);
    ImGui::TableSetupColumn("Bids", ImGuiTableColumnFlags_WidthStretch, 0.2f);
    ImGui::TableSetupColumn("Sells", ImGuiTableColumnFlags_WidthStretch, 0.2f);

    ImGui::TableHeadersRow();

    // Calculate midpoint price for reference
    double mid_price = 0.0;
    if (!orderbook.bids.empty() && !orderbook.asks.empty()) {
      mid_price = (orderbook.bids.front().price + orderbook.asks.front().price) / 2.0;
    } else if (!orderbook.bids.empty()) {
      mid_price = orderbook.bids.front().price;
    } else if (!orderbook.asks.empty()) {
      mid_price = orderbook.asks.front().price;
    }

    // Find maximum volume for normalization across all displayed levels
    double max_volume = 0.0;
    for (int i = 0; i < display_levels; ++i) {
      if (i < orderbook.bids.size()) {
        max_volume = std::max(max_volume, orderbook.bids[i].size);
      }
      if (i < orderbook.asks.size()) {
        max_volume = std::max(max_volume, orderbook.asks[i].size);
      }
    }

    // Render the orderbook levels in the 5-column format
    for (int i = 0; i < display_levels; ++i) {
      ImGui::TableNextRow();

      // Column 1: Buys (aggregated buy volume from recent trades)
      ImGui::TableSetColumnIndex(0);
      if (i < orderbook.bids.size()) {
        // Calculate buy pressure based on bid size and recent trades
        double buy_pressure = orderbook.bids[i].size; // Placeholder for actual buy pressure calculation
        
        // Get cumulative buy volume from ClusterEngine if available
        double cumulative_buy_volume = 0.0;
        if (cluster_engine_) {
          int64_t tick_index = static_cast<int64_t>(std::round(orderbook.bids[i].price / cluster_engine_->get_tick_size()));
          int64_t relative_index = tick_index - cluster_engine_->get_min_tick_index();
          
          if (relative_index >= 0 && static_cast<size_t>(relative_index) < cluster_engine_->getClusterCanvas().size()) {
            for (const auto& time_bucket : cluster_engine_->getClusterCanvas()[relative_index]) {
              cumulative_buy_volume += time_bucket.getBuyVolume();
            }
          }
        }
        
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 255, 0, 255));
        ImGui::Text("%.4f", cumulative_buy_volume > 0 ? cumulative_buy_volume : buy_pressure);
        ImGui::PopStyleColor();

        // Render cumulative volume bar extending right for buys
        ImVec2 pos = ImGui::GetCursorScreenPos();
        float bar_height = ImGui::GetTextLineHeight() * 0.8f;
        float max_bar_width = 100.0f; // Maximum width for the bar

        // Calculate normalized volume for the bar width based on cumulative volume if available
        double volume_for_bar = cumulative_buy_volume > 0 ? cumulative_buy_volume : buy_pressure;
        if (max_volume > 0) {
            float bar_width = static_cast<float>((volume_for_bar / max_volume) * max_bar_width);

            // Draw the cumulative volume bar
            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            ImVec2 bar_start = ImVec2(pos.x, pos.y + (ImGui::GetTextLineHeight() - bar_height) / 2);
            ImVec2 bar_end = ImVec2(bar_start.x + bar_width, bar_start.y + bar_height);
            draw_list->AddRectFilled(bar_start, bar_end, IM_COL32(0, 255, 0, 100)); // Green with transparency
        }
      } else {
        ImGui::Text("--");
      }

      // Column 2: Asks (from orderbook asks)
      ImGui::TableSetColumnIndex(1);
      if (i < orderbook.asks.size()) {
        // Show ask volume in red
        double ask_volume = orderbook.asks[i].size;
        
        // Get cumulative sell volume from ClusterEngine if available
        double cumulative_sell_volume = 0.0;
        if (cluster_engine_) {
          int64_t tick_index = static_cast<int64_t>(std::round(orderbook.asks[i].price / cluster_engine_->get_tick_size()));
          int64_t relative_index = tick_index - cluster_engine_->get_min_tick_index();
          
          if (relative_index >= 0 && static_cast<size_t>(relative_index) < cluster_engine_->getClusterCanvas().size()) {
            for (const auto& time_bucket : cluster_engine_->getClusterCanvas()[relative_index]) {
              cumulative_sell_volume += time_bucket.getSellVolume();
            }
          }
        }
        
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 100, 100, 255));
        ImGui::Text("%.4f", cumulative_sell_volume > 0 ? cumulative_sell_volume : ask_volume);
        ImGui::PopStyleColor();

        // Render cumulative volume bar extending left for sells
        ImVec2 pos = ImGui::GetCursorScreenPos();
        float bar_height = ImGui::GetTextLineHeight() * 0.8f;
        float max_bar_width = 100.0f; // Maximum width for the bar

        // Calculate normalized volume for the bar width based on cumulative volume if available
        double volume_for_bar = cumulative_sell_volume > 0 ? cumulative_sell_volume : ask_volume;
        if (max_volume > 0) {
            float bar_width = static_cast<float>((volume_for_bar / max_volume) * max_bar_width);

            // Draw the cumulative volume bar extending to the left
            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            ImVec2 bar_start = ImVec2(pos.x - bar_width, pos.y + (ImGui::GetTextLineHeight() - bar_height) / 2);
            ImVec2 bar_end = ImVec2(pos.x, bar_start.y + bar_height);
            draw_list->AddRectFilled(bar_start, bar_end, IM_COL32(255, 0, 0, 100)); // Red with transparency
        }
      } else {
        ImGui::Text("--");
      }

      // Column 3: Price (center column - actual price level)
      ImGui::TableSetColumnIndex(2);
      // Show the price in the middle - this represents the actual price level
      if (i < orderbook.bids.size() && i < orderbook.asks.size()) {
        // Average of bid and ask at this level
        double avg_price = (orderbook.bids[i].price + orderbook.asks[i].price) / 2.0;
        // Highlight if center mode is active and this is near the center
        if (mmt_center_mode_) {
          double center_price = (orderbook.bids.front().price + orderbook.asks.front().price) / 2.0;
          double range = center_price * mmt_center_range_;
          if (avg_price >= (center_price - range) && avg_price <= (center_price + range)) {
            ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 255, 0, 255)); // Yellow for center
          }
        }
        ImGui::Text("%.4f", avg_price);
        if (mmt_center_mode_) {
          ImGui::PopStyleColor(); // Pop the yellow color if we pushed it
        }
      } else if (i < orderbook.bids.size()) {
        // Only bid exists at this level
        ImGui::Text("%.4f", orderbook.bids[i].price);
      } else if (i < orderbook.asks.size()) {
        // Only ask exists at this level
        ImGui::Text("%.4f", orderbook.asks[i].price);
      } else {
        ImGui::Text("--");
      }

      // Column 4: Bids (from orderbook bids)
      ImGui::TableSetColumnIndex(3);
      if (i < orderbook.bids.size()) {
        // Show bid volume in green
        double bid_volume = orderbook.bids[i].size;
        
        // Get cumulative bid volume from ClusterEngine if available
        double cumulative_bid_volume = 0.0;
        if (cluster_engine_) {
          int64_t tick_index = static_cast<int64_t>(std::round(orderbook.bids[i].price / cluster_engine_->get_tick_size()));
          int64_t relative_index = tick_index - cluster_engine_->get_min_tick_index();
          
          if (relative_index >= 0 && static_cast<size_t>(relative_index) < cluster_engine_->getClusterCanvas().size()) {
            for (const auto& time_bucket : cluster_engine_->getClusterCanvas()[relative_index]) {
              cumulative_bid_volume += time_bucket.getBuyVolume();
            }
          }
        }
        
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(0, 255, 0, 255));
        ImGui::Text("%.4f", cumulative_bid_volume > 0 ? cumulative_bid_volume : bid_volume);
        ImGui::PopStyleColor();

        // Render cumulative volume bar extending right for bids
        ImVec2 pos = ImGui::GetCursorScreenPos();
        float bar_height = ImGui::GetTextLineHeight() * 0.8f;
        float max_bar_width = 100.0f; // Maximum width for the bar

        // Calculate normalized volume for the bar width based on cumulative volume if available
        double volume_for_bar = cumulative_bid_volume > 0 ? cumulative_bid_volume : bid_volume;
        if (max_volume > 0) {
            float bar_width = static_cast<float>((volume_for_bar / max_volume) * max_bar_width);

            // Draw the cumulative volume bar
            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            ImVec2 bar_start = ImVec2(pos.x, pos.y + (ImGui::GetTextLineHeight() - bar_height) / 2);
            ImVec2 bar_end = ImVec2(bar_start.x + bar_width, bar_start.y + bar_height);
            draw_list->AddRectFilled(bar_start, bar_end, IM_COL32(0, 255, 0, 100)); // Green with transparency
        }
      } else {
        ImGui::Text("--");
      }

      // Column 5: Sells (aggregated sell volume from recent trades)
      ImGui::TableSetColumnIndex(4);
      if (i < orderbook.asks.size()) {
        // Calculate sell pressure based on ask size and recent trades
        double sell_pressure = orderbook.asks[i].size; // Placeholder for actual sell pressure calculation
        
        // Get cumulative sell volume from ClusterEngine if available
        double cumulative_sell_volume_col5 = 0.0;
        if (cluster_engine_) {
          int64_t tick_index = static_cast<int64_t>(std::round(orderbook.asks[i].price / cluster_engine_->get_tick_size()));
          int64_t relative_index = tick_index - cluster_engine_->get_min_tick_index();
          
          if (relative_index >= 0 && static_cast<size_t>(relative_index) < cluster_engine_->getClusterCanvas().size()) {
            for (const auto& time_bucket : cluster_engine_->getClusterCanvas()[relative_index]) {
              cumulative_sell_volume_col5 += time_bucket.getSellVolume();
            }
          }
        }
        
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 100, 100, 255));
        ImGui::Text("%.4f", cumulative_sell_volume_col5 > 0 ? cumulative_sell_volume_col5 : sell_pressure);
        ImGui::PopStyleColor();

        // Render cumulative volume bar extending left for sells
        ImVec2 pos = ImGui::GetCursorScreenPos();
        float bar_height = ImGui::GetTextLineHeight() * 0.8f;
        float max_bar_width = 100.0f; // Maximum width for the bar

        // Calculate normalized volume for the bar width based on cumulative volume if available
        double volume_for_bar = cumulative_sell_volume_col5 > 0 ? cumulative_sell_volume_col5 : sell_pressure;
        if (max_volume > 0) {
            float bar_width = static_cast<float>((volume_for_bar / max_volume) * max_bar_width);

            // Draw the cumulative volume bar extending to the left
            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            ImVec2 bar_start = ImVec2(pos.x - bar_width, pos.y + (ImGui::GetTextLineHeight() - bar_height) / 2);
            ImVec2 bar_end = ImVec2(pos.x, bar_start.y + bar_height);
            draw_list->AddRectFilled(bar_start, bar_end, IM_COL32(255, 0, 0, 100)); // Red with transparency
        }
      } else {
        ImGui::Text("--");
      }
    }

    ImGui::EndTable();
  }

  // Add controls for the MMT layout
  ImGui::Separator();
  ImGui::Text("MMT Layout Controls:");
  ImGui::SameLine();
  ImGui::PushItemWidth(100);
  ImGui::SliderInt("##Levels", &mmt_display_levels_, 5, 50, "Levels: %d");
  ImGui::PopItemWidth();
  ImGui::SameLine();
  ImGui::Checkbox("Center##MMTCenter", &mmt_center_mode_);
  ImGui::SameLine();
  if (ImGui::Button("Refresh")) {
    markDirty();
  }

  // Add center mode range control if center mode is enabled
  if (mmt_center_mode_) {
    ImGui::Separator();
    ImGui::Text("Center Mode Range:");
    ImGui::SameLine();
    ImGui::PushItemWidth(150);
    ImGui::SliderFloat("##CenterRange", reinterpret_cast<float*>(&mmt_center_range_), 0.001f, 0.1f, "%.3f", ImGuiSliderFlags_Logarithmic);
    ImGui::PopItemWidth();
    ImGui::SameLine();
    ImGui::Text("(%.2f%%)", mmt_center_range_ * 100);
  }

  // Display cumulative volume information from ClusterEngine if available
  if (cluster_engine_) {
    ImGui::Separator();
    ImGui::Text("Cumulative Volume Data:");

    // Show a simple representation of cumulative volume data
    const auto& cluster_canvas = cluster_engine_->getClusterCanvas();
    if (!cluster_canvas.empty()) {
      // Show some summary statistics
      double total_buy_volume = 0.0;
      double total_sell_volume = 0.0;

      for (const auto& price_level : cluster_canvas) {
        for (const auto& time_bucket : price_level) {
          // Access the data without mutex since we're just reading using atomic operations
          total_buy_volume += time_bucket.getBuyVolume();
          total_sell_volume += time_bucket.getSellVolume();
        }
      }

      ImGui::Text("Total Buy Volume: %.2f", total_buy_volume);
      ImGui::Text("Total Sell Volume: %.2f", total_sell_volume);
      ImGui::Text("Net Delta: %.2f", total_buy_volume - total_sell_volume);
    }

    // Enhanced cumulative volume columns rendering
    ImGui::Separator();
    ImGui::Text("Cumulative Volume Columns:");

    // Render cumulative volume bars for each price level
    if (cluster_engine_) {
      // Calculate the current viewport to determine which price levels to display
      // Get the current orderbook to determine the price range
      auto orderbook_opt = processor_->getOrderbookData(current_symbol_id_);
      if (orderbook_opt) {
        const auto& orderbook = *orderbook_opt;

        // Determine the price range to display
        double min_price = std::numeric_limits<double>::max();
        double max_price = std::numeric_limits<double>::lowest();

        for (const auto& bid : orderbook.bids) {
          min_price = std::min(min_price, bid.price);
          max_price = std::max(max_price, bid.price);
        }
        for (const auto& ask : orderbook.asks) {
          min_price = std::min(min_price, ask.price);
          max_price = std::max(max_price, ask.price);
        }

        // Add some padding to the range
        double price_range = max_price - min_price;
        if (price_range > 0) {
          min_price -= price_range * 0.1;
          max_price += price_range * 0.1;
        } else {
          // Fallback if no price range
          if (!orderbook.bids.empty()) {
            min_price = orderbook.bids.front().price * 0.99;
            max_price = orderbook.bids.front().price * 1.01;
          } else if (!orderbook.asks.empty()) {
            min_price = orderbook.asks.front().price * 0.99;
            max_price = orderbook.asks.front().price * 1.01;
          }
        }

        // Calculate the conversion factor from price to index
        int64_t min_tick_index = static_cast<int64_t>(std::round(min_price / cluster_engine_->get_tick_size()));
        int64_t max_tick_index = static_cast<int64_t>(std::round(max_price / cluster_engine_->get_tick_size()));

        // Calculate cumulative volumes for the displayed price levels
        std::vector<std::pair<double, std::pair<double, double>>> cumulative_data; // {price, {cumulative_buy, cumulative_sell}}

        // Iterate through the price levels in the cluster canvas that correspond to our display range
        for (int64_t tick_idx = min_tick_index; tick_idx <= max_tick_index; ++tick_idx) {
          int64_t relative_idx = tick_idx - cluster_engine_->get_min_tick_index();

          if (relative_idx >= 0 && static_cast<size_t>(relative_idx) < cluster_engine_->getClusterCanvas().size()) {
            // Calculate cumulative volumes for this price level across all time buckets
            double level_buy_volume = 0.0;
            double level_sell_volume = 0.0;

            for (const auto& time_bucket : cluster_engine_->getClusterCanvas()[relative_idx]) {
              level_buy_volume += time_bucket.getBuyVolume();
              level_sell_volume += time_bucket.getSellVolume();
            }

            double price = tick_idx * cluster_engine_->get_tick_size();
            cumulative_data.push_back({price, {level_buy_volume, level_sell_volume}});
          }
        }

        // Render the cumulative volume bars
        if (!cumulative_data.empty()) {
          // Find the maximum cumulative volume for normalization
          double max_volume = 0.0;
          for (const auto& data : cumulative_data) {
            max_volume = std::max(max_volume, std::max(data.second.first, data.second.second));
          }

          if (max_volume > 0.0) {
            // Create a child window to contain the cumulative volume visualization
            ImGui::BeginChild("CumulativeVolumeVisualization", ImVec2(0, 200), true);

            // Draw the cumulative volume bars
            for (const auto& data : cumulative_data) {
              double price = data.first;
              double buy_vol = data.second.first;
              double sell_vol = data.second.second;

              // Calculate bar widths based on volumes
              float bar_width = 200.0f; // Maximum width for both bars

              // Draw the price label
              ImGui::Text("%.2f", price);
              ImGui::SameLine();

              // Draw buy and sell volume bars using the new helper function
              ImVec2 pos = ImGui::GetCursorScreenPos();
              ImDrawList* draw_list = ImGui::GetWindowDrawList();

              // Call the new function to render horizontal bars
              renderHorizontalVolumeBars(draw_list,
                                        ImVec2(pos.x, pos.y + (ImGui::GetTextLineHeight() - ImGui::GetTextLineHeight() * 0.6f) / 2),
                                        bar_width,
                                        ImGui::GetTextLineHeight() * 0.6f,
                                        buy_vol,
                                        sell_vol,
                                        max_volume);

              // Add some spacing
              ImGui::Dummy(ImVec2(0, ImGui::GetTextLineHeight() * 0.2f));
            }

            ImGui::EndChild();
          }
        }
      }
    }
  }
}

}  // namespace BTQuant
