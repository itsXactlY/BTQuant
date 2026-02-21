#include "components/volume_profile_panel.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace BTQuant {

// ============================================================================
// MMT Color Constants
// ============================================================================
static constexpr ImU32 VP_COLOR_BUY = IM_COL32(0x00, 0xE5, 0x66, 0xB0);
static constexpr ImU32 VP_COLOR_SELL = IM_COL32(0xE6, 0x19, 0x26, 0xB0);
static constexpr ImU32 VP_COLOR_POC = IM_COL32(0xFF, 0xD7, 0x00, 0xFF);
static constexpr ImU32 VP_COLOR_VA = IM_COL32(0x30, 0x60, 0xA0, 0x30);
static constexpr ImU32 VP_COLOR_TEXT = IM_COL32(0xE0, 0xE0, 0xE0, 0xFF);
static constexpr ImU32 VP_COLOR_TEXT_DIM = IM_COL32(0x70, 0x75, 0x80, 0xFF);

// ============================================================================
// Construction
// ============================================================================

VolumeProfilePanel::VolumeProfilePanel(const PanelConfig& config,
                                       std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(config), processor_(processor) {}

VolumeProfilePanel::~VolumeProfilePanel() {}

void VolumeProfilePanel::set_symbol(uint32_t symbol_id, const std::string& symbol_name) {
  symbol_id_ = symbol_id;
  symbol_name_ = symbol_name;
}

// ============================================================================
// Main Render — live volume profile from ClusterEngine
// ============================================================================

void VolumeProfilePanel::render_content() {
  if (!cluster_engine_) {
    ImGui::TextDisabled("No cluster engine");
    return;
  }

  auto profile = cluster_engine_->get_volume_profile(symbol_id_);
  if (profile.levels.empty()) {
    ImGui::TextDisabled("Waiting for trade data (symbol %u)...", symbol_id_);
    return;
  }

  // Header
  ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "Volume Profile");
  ImGui::SameLine();
  ImGui::TextDisabled("| POC: %.2f | VA: %.2f - %.2f", profile.poc_price, profile.val, profile.vah);
  ImGui::Separator();

  ImVec2 avail = ImGui::GetContentRegionAvail();
  if (avail.x < 100.0f || avail.y < 80.0f) return;

  constexpr float PRICE_COL_W = 70.0f;  // Width for price labels
  float bar_area_w = avail.x - PRICE_COL_W;
  float half_bar_w = bar_area_w / 2.0f;

  // Compute row height from available levels
  size_t visible_levels = profile.levels.size();
  float row_h = std::max(2.0f, avail.y / static_cast<float>(visible_levels));
  row_h = std::min(row_h, 18.0f);

  float content_h = visible_levels * row_h;

  ImGui::BeginChild("VPScroll", avail, ImGuiChildFlags_None, ImGuiWindowFlags_None);

  ImVec2 origin = ImGui::GetCursorScreenPos();
  ImDrawList* dl = ImGui::GetWindowDrawList();
  ImGui::Dummy({avail.x, content_h});

  float max_vol = static_cast<float>(profile.max_volume);
  if (max_vol <= 0.0f) max_vol = 1.0f;

  for (size_t i = 0; i < visible_levels; ++i) {
    const auto& lvl = profile.levels[visible_levels - 1 - i];  // Top = highest price
    float y = origin.y + i * row_h;

    float buy_ratio = static_cast<float>(lvl.buy_vol) / max_vol;
    float sell_ratio = static_cast<float>(lvl.sell_vol) / max_vol;

    // Value Area shading
    if (lvl.price >= profile.val && lvl.price <= profile.vah) {
      dl->AddRectFilled({origin.x, y}, {origin.x + avail.x, y + row_h}, VP_COLOR_VA);
    }

    // Buy volume bar (extending right from center)
    float center_x = origin.x + PRICE_COL_W + half_bar_w;
    float buy_w = half_bar_w * buy_ratio;
    dl->AddRectFilled({center_x, y + 1}, {center_x + buy_w, y + row_h - 1}, VP_COLOR_BUY);

    // Sell volume bar (extending left from center)
    float sell_w = half_bar_w * sell_ratio;
    dl->AddRectFilled({center_x - sell_w, y + 1}, {center_x, y + row_h - 1}, VP_COLOR_SELL);

    // POC line
    if (std::abs(lvl.price - profile.poc_price) < 1e-12) {
      dl->AddLine({origin.x, y + row_h / 2}, {origin.x + avail.x, y + row_h / 2}, VP_COLOR_POC,
                  2.0f);
    }

    // Price label (only if row is tall enough)
    if (row_h >= 12.0f) {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%.2f", lvl.price);
      ImU32 color =
          (std::abs(lvl.price - profile.poc_price) < 1e-12) ? VP_COLOR_POC : VP_COLOR_TEXT_DIM;
      dl->AddText({origin.x + 2, y + 1}, color, buf);
    }
  }

  // Center line
  float center_x = origin.x + PRICE_COL_W + half_bar_w;
  dl->AddLine({center_x, origin.y}, {center_x, origin.y + content_h},
              IM_COL32(0x50, 0x55, 0x60, 0xFF), 1.0f);

  ImGui::EndChild();
}

// ============================================================================
// Skeleton stubs — kept for link compatibility, will be implemented when needed
// ============================================================================
void VolumeProfilePanel::render_mini_histograms_on_candles(
    ImDrawList*, const std::vector<RenderEngine::OHLCVCandle>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&) {}
void VolumeProfilePanel::render_step_profile_histograms(
    ImDrawList*, const std::vector<RenderEngine::OHLCVCandle>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&, bool, int) {}
void VolumeProfilePanel::render_candle_volume_distribution(
    ImDrawList*, const std::vector<RenderEngine::OHLCVCandle>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&, bool, int) {}
void VolumeProfilePanel::render_step_profile_on_candles(
    ImDrawList*, const std::vector<RenderEngine::OHLCVCandle>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&, bool, int) {}
void VolumeProfilePanel::render_mini_histograms_direct(
    ImDrawList*, const std::vector<RenderEngine::OHLCVCandle>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&, const std::vector<TradeData>&, bool,
    int) {}
void VolumeProfilePanel::render_step_profile_on_candles_static(
    ImDrawList*, const std::vector<RenderEngine::OHLCVCandle>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&, const std::vector<TradeData>&, bool,
    int) {}
void VolumeProfilePanel::render_enhanced_step_profile_on_candles(
    ImDrawList*, const std::vector<RenderEngine::OHLCVCandle>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&, const std::vector<TradeData>&, bool,
    int, float) {}
void VolumeProfilePanel::render_step_profile_on_candles_with_volume_distribution(
    ImDrawList*, const std::vector<RenderEngine::OHLCVCandle>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&, bool, int) {}
void VolumeProfilePanel::render_enhanced_step_profile_with_volume_distribution(
    ImDrawList*, const std::vector<RenderEngine::OHLCVCandle>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&, bool, int, float, bool) {}
void VolumeProfilePanel::render_step_profile_histograms_on_candle_bars(
    ImDrawList*, const std::vector<RenderEngine::OHLCVCandle>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&, bool, int) {}
void VolumeProfilePanel::render_enhanced_step_profile_histograms_on_candle_bars(
    ImDrawList*, const std::vector<RenderEngine::OHLCVCandle>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&, bool, int, float, bool) {}
void VolumeProfilePanel::drawMiniHistogramOverlay(ImDrawList*,
                                                  const std::vector<RenderEngine::OHLCVCandle>&,
                                                  const std::vector<double>&,
                                                  const std::vector<double>&,
                                                  const std::vector<double>&, bool, int) {}
void VolumeProfilePanel::handleMouseDragInteraction() {}
void VolumeProfilePanel::renderCustomProfileOverlay(ImDrawList*) {}
void VolumeProfilePanel::calculateProfileForTimeRange(double, double) {}
double VolumeProfilePanel::getMinTimeAvailable() { return 0.0; }
double VolumeProfilePanel::getMaxTimeAvailable() { return 0.0; }
double VolumeProfilePanel::getTimeRangeAvailable() { return 0.0; }
void VolumeProfilePanel::build_volume_profile() {}
void VolumeProfilePanel::render_volume_bars() {}
void VolumeProfilePanel::render_controls() {}
void VolumeProfilePanel::render_step_profile(const double*, const double*, const double*, int,
                                             double) {}
void VolumeProfilePanel::render_split_profile(const double*, const double*, const double*, int,
                                              double) {}
void VolumeProfilePanel::render_yesterday_step_profile(const double*, const double*, const double*,
                                                       int, double, float) {}
void VolumeProfilePanel::render_yesterday_split_profile(const double*, const double*, const double*,
                                                        int, double, float) {}
void VolumeProfilePanel::highlight_profile_divergence(ImDrawList*) {}
void VolumeProfilePanel::calculate_value_area() {}
void VolumeProfilePanel::calculate_value_area_for_session(SessionProfile&) {}
void VolumeProfilePanel::store_current_as_yesterday_profile() {}
void VolumeProfilePanel::calculate_yesterday_value_area() {}
void VolumeProfilePanel::build_composite_profile() {}
void VolumeProfilePanel::add_daily_profile(const std::vector<VolumeLevel>&, time_t) {}
void VolumeProfilePanel::calculate_composite_value_area() {}
time_t VolumeProfilePanel::get_date_from_timestamp(double timestamp) {
  return static_cast<time_t>(timestamp);
}
void VolumeProfilePanel::clear_daily_profiles() {}
void VolumeProfilePanel::update_composite_profile() {}
void VolumeProfilePanel::initialize_predefined_sessions() {}
int VolumeProfilePanel::get_session_index_for_timestamp(double) { return -1; }
bool VolumeProfilePanel::is_new_session_boundary(double, double) { return false; }
void VolumeProfilePanel::create_new_session_profile(double, double, const std::string&) {}
void VolumeProfilePanel::switch_to_session(int) {}
void VolumeProfilePanel::reset_session_profiles() {}
void VolumeProfilePanel::detect_and_handle_session_boundaries() {}
void VolumeProfilePanel::calculate_virgin_poc() {}
void VolumeProfilePanel::subscribe_to_updates() {}

}  // namespace BTQuant
