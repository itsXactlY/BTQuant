#include "components/footprint_panel.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace BTQuant {

// ============================================================================
// MMT Color Constants
// ============================================================================
static constexpr ImU32 FP_COLOR_BUY = IM_COL32(0x00, 0xE5, 0x66, 0xFF);
static constexpr ImU32 FP_COLOR_SELL = IM_COL32(0xE6, 0x19, 0x26, 0xFF);
static constexpr ImU32 FP_COLOR_DELTA_POS = IM_COL32(0x00, 0xE5, 0x66, 0x80);
static constexpr ImU32 FP_COLOR_DELTA_NEG = IM_COL32(0xE6, 0x19, 0x26, 0x80);
static constexpr ImU32 FP_COLOR_GRID = IM_COL32(0x30, 0x35, 0x40, 0xFF);
static constexpr ImU32 FP_COLOR_POC = IM_COL32(0xFF, 0xD7, 0x00, 0xA0);
static constexpr ImU32 FP_COLOR_TEXT_DIM = IM_COL32(0x80, 0x85, 0x90, 0xFF);

// ============================================================================
// Construction
// ============================================================================

FootprintPanel::FootprintPanel(const PanelConfig& config, ClusterEngine* engine)
    : PanelBase(config), engine_(engine) {}

void FootprintPanel::update(float /*dt*/) {}

// ============================================================================
// Rendering
// ============================================================================

void FootprintPanel::render_content() {
  if (!engine_) {
    ImGui::TextDisabled("No cluster engine");
    return;
  }

  auto columns = engine_->get_footprint_columns(symbol_id_, 30);
  if (columns.empty()) {
    ImGui::TextDisabled("Waiting for trade data (symbol %u)...", symbol_id_);
    return;
  }

  // Header info
  double cvd = engine_->get_cvd(symbol_id_);
  ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "Footprint | CVD: ");
  ImGui::SameLine();
  ImGui::TextColored(cvd >= 0 ? ImVec4(0.0f, 0.9f, 0.4f, 1.0f) : ImVec4(0.9f, 0.1f, 0.15f, 1.0f),
                     "%.2f", cvd);
  ImGui::Separator();

  ImVec2 avail = ImGui::GetContentRegionAvail();
  if (avail.x < 100.0f || avail.y < 100.0f) return;

  // Define cell dimensions
  constexpr float COL_WIDTH = 120.0f;   // Width per candle column
  constexpr float ROW_HEIGHT = 16.0f;   // Height per price row
  constexpr float DELTA_BAR_H = 20.0f;  // Height of delta bar at bottom

  // Find global price range across visible columns
  double global_high = -1e18, global_low = 1e18;
  double max_vol = 0.0;
  for (const auto& col : columns) {
    for (const auto& t : col.ticks) {
      global_high = std::max(global_high, t.price);
      global_low = std::min(global_low, t.price);
      max_vol = std::max(max_vol, t.total_vol());
    }
  }
  if (global_high <= global_low || max_vol <= 0.0) return;

  // Determine tick size from first column with data
  double tick_size = 1.0;
  for (const auto& col : columns) {
    if (col.ticks.size() >= 2) {
      tick_size = std::abs(col.ticks[1].price - col.ticks[0].price);
      if (tick_size > 0) break;
    }
  }
  if (tick_size <= 0) tick_size = 1.0;

  int total_rows = static_cast<int>((global_high - global_low) / tick_size) + 1;
  float content_height = total_rows * ROW_HEIGHT + DELTA_BAR_H;
  float content_width = columns.size() * COL_WIDTH;

  // Scrollable child region
  ImGui::BeginChild("FootprintScroll", avail, ImGuiChildFlags_None,
                    ImGuiWindowFlags_HorizontalScrollbar);

  // Auto-scroll to right (newest data)
  if (ImGui::GetScrollX() < ImGui::GetScrollMaxX() - COL_WIDTH) {
    ImGui::SetScrollX(ImGui::GetScrollMaxX());
  }

  ImVec2 origin = ImGui::GetCursorScreenPos();
  ImDrawList* dl = ImGui::GetWindowDrawList();

  // Reserve space
  ImGui::Dummy({content_width, content_height});

  // Render each column
  for (size_t ci = 0; ci < columns.size(); ++ci) {
    const auto& col = columns[ci];
    float cx = origin.x + ci * COL_WIDTH;

    // Column separator
    dl->AddLine({cx, origin.y}, {cx, origin.y + content_height}, FP_COLOR_GRID, 1.0f);

    // Find max volume in this column for color intensity
    double col_max_vol = 0.0;
    for (const auto& t : col.ticks) col_max_vol = std::max(col_max_vol, t.total_vol());
    if (col_max_vol <= 0.0) continue;

    // Render each tick row
    for (const auto& tick : col.ticks) {
      int row = static_cast<int>((global_high - tick.price) / tick_size);
      if (row < 0 || row >= total_rows) continue;

      float ry = origin.y + row * ROW_HEIGHT;
      float intensity = static_cast<float>(tick.total_vol() / col_max_vol);

      // Background intensity
      uint8_t bg_alpha = static_cast<uint8_t>(intensity * 40);
      ImU32 bg_color = (tick.delta() >= 0) ? IM_COL32(0x00, 0xE5, 0x66, bg_alpha)
                                           : IM_COL32(0xE6, 0x19, 0x26, bg_alpha);
      dl->AddRectFilled({cx + 1, ry}, {cx + COL_WIDTH - 1, ry + ROW_HEIGHT - 1}, bg_color);

      // POC highlight (highest volume tick in column)
      if (std::abs(tick.total_vol() - col_max_vol) < 1e-12) {
        dl->AddRect({cx + 1, ry}, {cx + COL_WIDTH - 1, ry + ROW_HEIGHT - 1}, FP_COLOR_POC, 0.0f, 0,
                    1.0f);
      }

      // Text: buy_vol | price | sell_vol
      float half_w = (COL_WIDTH - 2) / 2.0f;
      char buf[32];

      // Buy volume (left side, green)
      if (tick.buy_vol > 0.001) {
        std::snprintf(buf, sizeof(buf), "%.1f", tick.buy_vol);
        dl->AddText({cx + 3, ry + 1}, FP_COLOR_BUY, buf);
      }

      // Sell volume (right side, red)
      if (tick.sell_vol > 0.001) {
        std::snprintf(buf, sizeof(buf), "%.1f", tick.sell_vol);
        ImVec2 text_size = ImGui::CalcTextSize(buf);
        dl->AddText({cx + COL_WIDTH - text_size.x - 3, ry + 1}, FP_COLOR_SELL, buf);
      }

      // Price in center (dimmed)
      std::snprintf(buf, sizeof(buf), "%.2f", tick.price);
      ImVec2 price_size = ImGui::CalcTextSize(buf);
      dl->AddText({cx + half_w - price_size.x / 2, ry + 1}, FP_COLOR_TEXT_DIM, buf);
    }

    // Delta bar at bottom
    float delta_y = origin.y + total_rows * ROW_HEIGHT;
    float delta_ratio = static_cast<float>(std::abs(col.delta) / col_max_vol);
    delta_ratio = std::min(delta_ratio, 1.0f);
    float bar_w = (COL_WIDTH - 4) * delta_ratio;
    ImU32 delta_color = (col.delta >= 0) ? FP_COLOR_DELTA_POS : FP_COLOR_DELTA_NEG;
    dl->AddRectFilled({cx + 2, delta_y + 2}, {cx + 2 + bar_w, delta_y + DELTA_BAR_H - 2},
                      delta_color);

    // Delta text
    char delta_buf[32];
    std::snprintf(delta_buf, sizeof(delta_buf), "Δ%.1f", col.delta);
    dl->AddText({cx + 4, delta_y + 3}, col.delta >= 0 ? FP_COLOR_BUY : FP_COLOR_SELL, delta_buf);
  }

  ImGui::EndChild();
}

}  // namespace BTQuant
