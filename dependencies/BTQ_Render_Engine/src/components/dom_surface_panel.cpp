#include "components/dom_surface_panel.hpp"
#include "../../include/components/MarketMicrostructureRenderer.h"
#include "../../include/symbol_registry.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace BTQuant {

DomSurfacePanel::DomSurfacePanel(
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
    RenderEngine::MarketMicrostructureRenderer *renderer)
    : PanelBase(
          PanelConfig{.title = "LOB Heatmap", .type = PanelType::HEATMAP}),
      renderer_(renderer) {}

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

  // Determine price range based on professional tick-window (Quantower style)
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

  // Use fixed tick resolution and window (e.g., ±200 ticks)
  constexpr double tickSize = 0.5;
  constexpr double tickWindow = 200.0;

  double min_price = mid_price - (tickWindow * tickSize);
  double max_price = mid_price + (tickWindow * tickSize);

  // Increase bin resolution for "smooth" look
  price_bins_ = 256;

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

void DomSurfacePanel::render() {
  // ONLY run CPU logic if Vulkan renderer is NOT available
  if (!renderer_ && consumeDirty()) {
    updateHeatmapData();
  }

  begin_panel_window();

  if (current_symbol_id_ == 0 || (heatmap_data_.empty() && !renderer_)) {
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

    if (renderer_) {
      auto [baseP, rangeP] = renderer_->getLOBPriceBounds();
      bounds_min_[1] = baseP;
      bounds_max_[1] = baseP + rangeP;
      bounds_min_[0] = 0;
      bounds_max_[0] = 600; // time steps
    }

    // Always fit axes to data bounds (fills the plot area)
    ImPlot::SetupAxisLimits(ImAxis_X1, bounds_min_[0], bounds_max_[0],
                            ImPlotCond_Always);
    ImPlot::SetupAxisLimits(ImAxis_Y1, (double)bounds_min_[1],
                            (double)bounds_max_[1], ImPlotCond_Always);

    // Use time history size for Cols and price_bins for Rows
    int rows = price_bins_;
    int cols = static_cast<int>(heatmap_data_.size()) / rows;

    // Use high-performance Vulkan Heatmap if available (Quantower style)
    if (renderer_) {
      void *texID = renderer_->getHeatmapTextureID();
      if (texID) {
        ImPlot::PlotImage("Heatmap", texID,
                          ImPlotPoint(bounds_min_[0], bounds_min_[1]),
                          ImPlotPoint(bounds_max_[0], bounds_max_[1]));
      }
    } else {
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
    }

    ImPlot::EndPlot();
  }

  // Debug Overlay for DOM troubleshooting
  if (heatmap_data_.size() > 0) {
    ImGui::SetCursorPos(ImVec2(10, 30));
    ImGui::TextColored(ImVec4(1, 1, 0, 1),
                       "Debug: MaxVol=%.2f, Hist=%zu, Bins=%d", scale_max_,
                       heatmap_data_.size() / price_bins_, price_bins_);
    ImGui::Text("Bounds: Y=%.4f - %.4f", bounds_min_[1], bounds_max_[1]);
  }

  end_panel_window();
}

} // namespace BTQuant
