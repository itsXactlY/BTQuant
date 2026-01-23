#include "components/dom_surface_panel.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

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

  auto history =
      processor_->getHistoricalOrderbooks(current_symbol_id_, history_depth_);
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

void DomSurfacePanel::render() {
  if (consumeDirty()) {
    updateHeatmapData();
  }

  begin_panel_window();

  if (current_symbol_id_ == 0 || heatmap_data_.empty()) {
    ImGui::Text("No Data / Select Symbol");
    end_panel_window();
    return;
  }

  if (ImPlot::BeginPlot("##DomHeatmap", ImVec2(-1, -1),
                        ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText)) {
    ImPlot::SetupAxes("Time Step", "Price");
    ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoTickLabels);

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

    ImPlot::EndPlot();
  }

  end_panel_window();
}

} // namespace BTQuant
