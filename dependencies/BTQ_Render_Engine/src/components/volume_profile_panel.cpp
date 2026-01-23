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

  // Find price range
  double min_price = std::numeric_limits<double>::max();
  double max_price = std::numeric_limits<double>::lowest();

  for (const auto &trade : trades) {
    min_price = std::min(min_price, trade.price);
    max_price = std::max(max_price, trade.price);
  }

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

void VolumeProfilePanel::render_controls() {
  ImGui::Text("Symbol: %s", symbol_name_.c_str());
  ImGui::SameLine();
  ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "| POC: %.4f", poc_price_);
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
                        ImPlotFlags_NoTitle | ImPlotFlags_NoLegend |
                            ImPlotFlags_NoMouseText)) {

    ImPlot::SetupAxes("Volume", "Price",
                      ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_Invert,
                      ImPlotAxisFlags_AutoFit);

    // Bar height based on price bucket size
    double bar_height = price_bucket_size_ * 0.8;

    // Buy bars (green, positive X) - using PlotBars with horizontal flag
    ImPlot::SetNextFillStyle(ImVec4(0.1f, 0.8f, 0.1f, 0.7f));
    ImPlot::PlotBars("Buy", buy_volumes.data(), prices.data(),
                     static_cast<int>(prices.size()), bar_height,
                     ImPlotBarsFlags_Horizontal);

    // Sell bars (red, negative X)
    ImPlot::SetNextFillStyle(ImVec4(0.8f, 0.1f, 0.1f, 0.7f));
    ImPlot::PlotBars("Sell", sell_volumes.data(), prices.data(),
                     static_cast<int>(prices.size()), bar_height,
                     ImPlotBarsFlags_Horizontal);

    // POC line
    if (poc_price_ > 0) {
      double poc_line_x[2] = {-max_volume_, max_volume_};
      double poc_line_y[2] = {poc_price_, poc_price_};
      ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(1.0f, 0.8f, 0.0f, 1.0f));
      ImPlot::PlotLine("POC", poc_line_x, poc_line_y, 2);
      ImPlot::PopStyleColor();
    }

    ImPlot::EndPlot();
  }
}

} // namespace BTQuant
