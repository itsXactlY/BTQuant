#include "../../include/components/volume_profile_panel.hpp"
#include "imgui.h"
#include <algorithm>
#include <cmath>

namespace BTQuant {

VolumeProfilePanel::VolumeProfilePanel(
    const PanelConfig &config, std::shared_ptr<HotSpineDataBridge> bridge,
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(config), bridge_(bridge), processor_(processor) {}

void VolumeProfilePanel::update(float dt) {
  update_timer_ += dt;
  if (update_timer_ >= UPDATE_INTERVAL) {
    build_volume_profile();
    update_timer_ = 0.0f;
  }
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
  render_volume_bars();

  end_panel_window();
}

void VolumeProfilePanel::set_symbol(uint32_t symbol_id,
                                    const std::string &symbol_name) {
  symbol_id_ = symbol_id;
  symbol_name_ = symbol_name;
  volume_profile_.clear();
}

void VolumeProfilePanel::build_volume_profile() {
  if (!processor_ || symbol_id_ == 0)
    return;

  auto analytics = processor_->getSymbolAnalytics(symbol_id_);
  const auto &trades = analytics.recent_trades;

  if (trades.empty()) {
    volume_profile_.clear();
    return;
  }

  // Find price range from recent trades
  double min_price = trades.front().price;
  double max_price = trades.front().price;
  for (const auto &trade : trades) {
    min_price = std::min(min_price, trade.price);
    max_price = std::max(max_price, trade.price);
  }

  // Calculate bucket size
  double range = max_price - min_price;
  if (range <= 0)
    range = 1.0;
  price_bucket_size_ = range / NUM_PRICE_LEVELS;
  if (price_bucket_size_ <= 0)
    price_bucket_size_ = 1.0;

  // Initialize volume levels
  volume_profile_.clear();
  volume_profile_.resize(NUM_PRICE_LEVELS);
  for (size_t i = 0; i < NUM_PRICE_LEVELS; ++i) {
    volume_profile_[i].price =
        min_price + (i + 0.5) * price_bucket_size_; // Midpoint
    volume_profile_[i].buy_volume = 0.0;
    volume_profile_[i].sell_volume = 0.0;
    volume_profile_[i].total_volume = 0.0;
  }

  // Aggregate volume into buckets
  for (const auto &trade : trades) {
    size_t bucket_idx =
        static_cast<size_t>((trade.price - min_price) / price_bucket_size_);
    bucket_idx = std::min(bucket_idx, NUM_PRICE_LEVELS - 1);

    if (trade.is_buy) {
      volume_profile_[bucket_idx].buy_volume += trade.size;
    } else {
      volume_profile_[bucket_idx].sell_volume += trade.size;
    }
    volume_profile_[bucket_idx].total_volume += trade.size;
  }

  // Find POC and max volume
  max_volume_ = 0.0;
  poc_price_ = 0.0;
  double poc_volume = 0.0;
  for (const auto &level : volume_profile_) {
    max_volume_ = std::max(max_volume_, level.total_volume);
    if (level.total_volume > poc_volume) {
      poc_volume = level.total_volume;
      poc_price_ = level.price;
    }
  }
}

void VolumeProfilePanel::render_controls() {
  ImGui::Text("Symbol: %s", symbol_name_.c_str());
  ImGui::SameLine();
  ImGui::Text("| POC: %.4f", poc_price_);
}

void VolumeProfilePanel::render_volume_bars() {
  if (volume_profile_.empty() || max_volume_ <= 0) {
    ImGui::Text("No volume data available");
    return;
  }

  ImVec2 region = ImGui::GetContentRegionAvail();
  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  ImVec2 cursor = ImGui::GetCursorScreenPos();

  float row_height = region.y / static_cast<float>(volume_profile_.size());
  float bar_max_width = region.x / 2.0f - 10.0f; // Half width for each side

  // Render from top (highest price) to bottom (lowest price)
  for (size_t i = 0; i < volume_profile_.size(); ++i) {
    size_t idx = volume_profile_.size() - 1 - i; // Reverse order
    const auto &level = volume_profile_[idx];

    float y = cursor.y + i * row_height;
    float center_x = cursor.x + region.x / 2.0f;

    // Buy volume bar (left side, green)
    float buy_width = (level.buy_volume / max_volume_) * bar_max_width;
    ImVec4 buy_color = (level.price == poc_price_)
                           ? ImVec4(0.3f, 1.0f, 0.3f, 1.0f)
                           : ImVec4(0.2f, 0.6f, 0.2f, 0.8f);
    draw_list->AddRectFilled(ImVec2(center_x - buy_width, y),
                             ImVec2(center_x - 2, y + row_height - 2),
                             ImGui::ColorConvertFloat4ToU32(buy_color));

    // Sell volume bar (right side, red)
    float sell_width = (level.sell_volume / max_volume_) * bar_max_width;
    ImVec4 sell_color = (level.price == poc_price_)
                            ? ImVec4(1.0f, 0.3f, 0.3f, 1.0f)
                            : ImVec4(0.6f, 0.2f, 0.2f, 0.8f);
    draw_list->AddRectFilled(
        ImVec2(center_x + 2, y),
        ImVec2(center_x + 2 + sell_width, y + row_height - 2),
        ImGui::ColorConvertFloat4ToU32(sell_color));

    // Price label (center)
    char price_str[32];
    snprintf(price_str, sizeof(price_str), "%.2f", level.price);
    ImVec2 text_size = ImGui::CalcTextSize(price_str);
    draw_list->AddText(ImVec2(center_x - text_size.x / 2, y + 2),
                       IM_COL32(200, 200, 200, 255), price_str);
  }

  // Reserve space for the rendered content
  ImGui::Dummy(region);
}

} // namespace BTQuant
