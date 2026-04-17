#include "../../include/components/tpo_panel.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <format>
#include <numeric>
#include <vector>

#include "components/theme_manager.hpp"
#include "imgui.h"
#include "implot.h"

namespace BTQuant {

// ==========================================================================
// Constants
// ==========================================================================

// TPO bracket letters (A through P = 16 brackets max)
static constexpr char TPO_LETTERS[] = "ABCDEFGHIJKLMNOP";

// Colors
static constexpr ImU32 COLOR_TPO_FIRST_HALF  = IM_COL32(180, 200, 255, 255);  // Light blue  A-H
static constexpr ImU32 COLOR_TPO_SECOND_HALF = IM_COL32(120, 160, 240, 255);  // Deeper blue I-P
static constexpr ImU32 COLOR_VALUE_AREA      = IM_COL32(30, 40, 60, 80);      // Dark blue tint
static constexpr ImU32 COLOR_POC             = IM_COL32(255, 200, 0, 220);     // Gold
static constexpr ImU32 COLOR_SINGLE_PRINT   = IM_COL32(74, 144, 255, 60);     // Blue highlight
static constexpr ImU32 COLOR_SESSION_EXTREME= IM_COL32(200, 200, 200, 160);   // Light gray
static constexpr ImU32 COLOR_IB             = IM_COL32(255, 140, 0, 140);     // Orange
static constexpr ImU32 COLOR_VAH            = IM_COL32(0, 200, 100, 200);     // Green
static constexpr ImU32 COLOR_VAL            = IM_COL32(200, 50, 50, 200);     // Red

// Value area percentage (68%)
static constexpr double VA_PERCENT = 0.68;

// ==========================================================================
// Helper: popcount for uint16_t
// ==========================================================================
static inline int popcount16(uint16_t v) {
  return std::popcount(v);
}

// ==========================================================================
// Constructor / update
// ==========================================================================

TpoPanel::TpoPanel(const PanelConfig& config) : PanelBase(config) {}

void TpoPanel::update(float /*dt*/) {
  // No periodic update needed; data read from ClusterEngine on render
}

// ==========================================================================
// Compute TPO levels and value area from ClusterEngine canvas
// ==========================================================================

static bool computeTpoProfile(
    const Analytics::ClusterEngine& engine,
    std::vector<TpoPanel::TpoLevel>& levels,
    TpoPanel::ValueArea& va)
{
  const auto& canvas = engine.getCanvas();
  if (canvas.empty()) return false;

  // 1. Collect all price levels with non-zero tpo_bits
  levels.clear();
  levels.reserve(canvas.size());

  int total_tpo = 0;
  int max_tpo = 0;
  size_t poc_index = 0;

  double session_high = -1e30;
  double session_low = 1e30;

  for (size_t i = 0; i < canvas.size(); ++i) {
    uint16_t bits = canvas[i].tpo_bits;
    if (bits == 0) continue;

    int pc = popcount16(bits);
    double price = engine.priceAtIndex(i);

    levels.push_back({price, bits, pc});
    total_tpo += pc;

    if (pc > max_tpo) {
      max_tpo = pc;
      poc_index = levels.size() - 1;
    }

    if (price > session_high) session_high = price;
    if (price < session_low) session_low = price;
  }

  if (levels.empty()) return false;

  va.session_high = session_high;
  va.session_low = session_low;

  // 2. POC price
  va.poc_price = levels[poc_index].price;

  // 3. Initial Balance (first hour = first two 30-min brackets, bits 0 and 1)
  double ib_high = -1e30;
  double ib_low = 1e30;
  bool ib_found = false;
  for (const auto& lvl : levels) {
    if (lvl.tpo_bits & 0x0003) {  // bit 0 or bit 1 set
      ib_found = true;
      if (lvl.price > ib_high) ib_high = lvl.price;
      if (lvl.price < ib_low) ib_low = lvl.price;
    }
  }
  va.ib_high = ib_found ? ib_high : 0.0;
  va.ib_low  = ib_found ? ib_low  : 0.0;

  // 4. Value Area: start at POC, expand outward until 68% of TPO enclosed.
  //    At each step, pick the direction (up or down) that adds more TPO count.
  int target_tpo = static_cast<int>(total_tpo * VA_PERCENT);
  int accumulated = levels[poc_index].popcount;

  int lo = static_cast<int>(poc_index);
  int hi = static_cast<int>(poc_index);

  while (accumulated < target_tpo) {
    int up_count = 0;
    int dn_count = 0;

    if (hi + 1 < static_cast<int>(levels.size()))
      up_count = levels[hi + 1].popcount;
    if (lo - 1 >= 0)
      dn_count = levels[lo - 1].popcount;

    bool can_up = (hi + 1 < static_cast<int>(levels.size()));
    bool can_dn = (lo - 1 >= 0);

    if (!can_up && !can_dn) break;

    if (can_up && (!can_dn || up_count >= dn_count)) {
      hi++;
      accumulated += levels[hi].popcount;
    } else {
      lo--;
      accumulated += levels[lo].popcount;
    }
  }

  va.vah = levels[hi].price;
  va.val = levels[lo].price;

  return true;
}

// ==========================================================================
// Detect single prints: price levels where popcount == 1 AND bracketed
// by levels with popcount > 1 above AND below.
// ==========================================================================

static std::vector<double> detectSinglePrints(
    const std::vector<TpoPanel::TpoLevel>& levels)
{
  std::vector<double> singles;
  if (levels.size() < 3) return singles;

  for (size_t i = 1; i + 1 < levels.size(); ++i) {
    if (levels[i].popcount == 1 &&
        levels[i - 1].popcount > 1 &&
        levels[i + 1].popcount > 1) {
      singles.push_back(levels[i].price);
    }
  }
  return singles;
}

// ==========================================================================
// render()
// ==========================================================================

void TpoPanel::render() {
  begin_panel_window();

  // ---- Toolbar ----
  if (ImGui::Button("Reset View")) {
    ImPlot::SetNextAxesToFit();
  }
  ImGui::SameLine();
  static bool show_letters = true;
  ImGui::Checkbox("Letters", &show_letters);
  ImGui::SameLine();
  static bool show_va = true;
  ImGui::Checkbox("Value Area", &show_va);
  ImGui::SameLine();
  static bool show_single_prints = true;
  ImGui::Checkbox("Single Prints", &show_single_prints);
  ImGui::SameLine();
  static bool show_ib = true;
  ImGui::Checkbox("IB", &show_ib);

  // ---- If no engine, show placeholder and exit early ----
  if (!cluster_engine_ || cluster_engine_->getCanvas().empty()) {
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                       "No TPO data available. Set ClusterEngine on this panel.");
    end_panel_window();
    return;
  }

  // ---- Compute TPO profile ----
  std::vector<TpoLevel> levels;
  ValueArea va;
  if (!computeTpoProfile(*cluster_engine_, levels, va)) {
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No TPO data in engine canvas.");
    end_panel_window();
    return;
  }

  std::vector<double> single_prints = detectSinglePrints(levels);

  // ---- Determine axis range ----
  double price_min = va.session_low;
  double price_max = va.session_high;
  double price_pad = (price_max - price_min) * 0.05;
  if (price_pad < cluster_engine_->getTickSize()) price_pad = cluster_engine_->getTickSize();
  price_min -= price_pad;
  price_max += price_pad;

  // X axis: we use a nominal range [0, num_brackets] for letter placement
  int max_bracket = 0;
  for (const auto& lvl : levels) {
    for (int b = 0; b < 16; ++b) {
      if (lvl.tpo_bits & (1 << b) && b + 1 > max_bracket) {
        max_bracket = b + 1;
      }
    }
  }
  if (max_bracket < 1) max_bracket = 16;
  double x_max = static_cast<double>(max_bracket) + 1.0;

  // ---- ImPlot ----
  if (ImPlot::BeginPlot("##TPOProfile", ImVec2(-1, -1),
                        ImPlotFlags_NoLegend | ImPlotFlags_Crosshairs)) {
    ImPlot::SetupAxes("Bracket", "Price", ImPlotAxisFlags_NoGridLines,
                      ImPlotAxisFlags_None);
    ImPlot::SetupAxisLimits(ImAxis_X1, 0, x_max, ImPlotCond_Always);
    ImPlot::SetupAxisLimits(ImAxis_Y1, price_min, price_max, ImPlotCond_Once);

    auto* draw_list = ImPlot::GetPlotDrawList();

    // ---- Value Area shading ----
    if (show_va && va.vah > va.val) {
      // Shade from x=0..x_max between VAL and VAH
      ImVec2 p_lo_left  = ImPlot::PlotToPixels(0.0, va.val);
      ImVec2 p_hi_right = ImPlot::PlotToPixels(x_max, va.vah);
      draw_list->AddRectFilled(p_lo_left, p_hi_right, COLOR_VALUE_AREA);
    }

    // ---- Session High / Low lines ----
    {
      ImVec2 p1 = ImPlot::PlotToPixels(0.0, va.session_high);
      ImVec2 p2 = ImPlot::PlotToPixels(x_max, va.session_high);
      draw_list->AddLine(p1, p2, COLOR_SESSION_EXTREME, 1.0f);

      p1 = ImPlot::PlotToPixels(0.0, va.session_low);
      p2 = ImPlot::PlotToPixels(x_max, va.session_low);
      draw_list->AddLine(p1, p2, COLOR_SESSION_EXTREME, 1.0f);
    }

    // ---- Value Area lines (VAH / VAL) ----
    if (show_va && va.vah > va.val) {
      ImVec2 p1 = ImPlot::PlotToPixels(0.0, va.vah);
      ImVec2 p2 = ImPlot::PlotToPixels(x_max, va.vah);
      draw_list->AddLine(p1, p2, COLOR_VAH, 2.0f);

      p1 = ImPlot::PlotToPixels(0.0, va.val);
      p2 = ImPlot::PlotToPixels(x_max, va.val);
      draw_list->AddLine(p1, p2, COLOR_VAL, 2.0f);
    }

    // ---- POC line (thick gold) ----
    if (va.poc_price > 0.0) {
      ImVec2 p1 = ImPlot::PlotToPixels(0.0, va.poc_price);
      ImVec2 p2 = ImPlot::PlotToPixels(x_max, va.poc_price);
      draw_list->AddLine(p1, p2, COLOR_POC, 3.0f);
    }

    // ---- Initial Balance lines ----
    if (show_ib && va.ib_high > va.ib_low) {
      ImVec2 p1 = ImPlot::PlotToPixels(0.0, va.ib_high);
      ImVec2 p2 = ImPlot::PlotToPixels(x_max, va.ib_high);
      draw_list->AddLine(p1, p2, COLOR_IB, 1.5f);

      p1 = ImPlot::PlotToPixels(0.0, va.ib_low);
      p2 = ImPlot::PlotToPixels(x_max, va.ib_low);
      draw_list->AddLine(p1, p2, COLOR_IB, 1.5f);
    }

    // ---- Single print highlights ----
    if (show_single_prints) {
      double half_tick = cluster_engine_->getTickSize() * 0.5;
      for (double sp_price : single_prints) {
        ImVec2 p1 = ImPlot::PlotToPixels(0.0, sp_price - half_tick);
        ImVec2 p2 = ImPlot::PlotToPixels(x_max, sp_price + half_tick);
        draw_list->AddRectFilled(p1, p2, COLOR_SINGLE_PRINT);
      }
    }

    // ---- TPO Letter Grid ----
    if (show_letters) {
      // Estimate pixel height per price level to decide if we can render text
      ImVec2 p_test_top = ImPlot::PlotToPixels(0.0, price_max);
      ImVec2 p_test_bot = ImPlot::PlotToPixels(0.0, price_min);
      double plot_height_px = std::abs(p_test_bot.y - p_test_top.y);
      double price_range = price_max - price_min;
      double px_per_price = (price_range > 0) ? plot_height_px / price_range : 1.0;

      // Each letter occupies about 14px wide. Check if we have enough vertical space.
      bool can_draw_text = (px_per_price >= 8.0);

      // We need to get the plot draw position to compute letter size
      ImPlotRect plot_limits = ImPlot::GetPlotLimits();

      for (const auto& lvl : levels) {
        for (int b = 0; b < 16; ++b) {
          if (!(lvl.tpo_bits & (1 << b))) continue;

          double cx = static_cast<double>(b) + 0.5;
          double cy = lvl.price;

          ImVec2 center_px = ImPlot::PlotToPixels(cx, cy);

          ImU32 color = (b < 8) ? COLOR_TPO_FIRST_HALF : COLOR_TPO_SECOND_HALF;

          if (can_draw_text) {
            // Draw the letter
            char letter_str[2] = { TPO_LETTERS[b], '\0' };
            ImVec2 text_size = ImGui::CalcTextSize(letter_str);
            draw_list->AddText(
                ImVec2(center_px.x - text_size.x * 0.5f,
                       center_px.y - text_size.y * 0.5f),
                color, letter_str);
          } else {
            // Zoomed out: draw a small filled rectangle instead of text
            float half_h = static_cast<float>(px_per_price * 0.45);
            float half_w = 5.0f;
            ImVec2 p1(center_px.x - half_w, center_px.y - half_h);
            ImVec2 p2(center_px.x + half_w, center_px.y + half_h);
            draw_list->AddRectFilled(p1, p2, color);
          }
        }
      }
    }

    ImPlot::EndPlot();
  }

  // ---- Overlay info ----
  ImGui::SetCursorPos(ImVec2(10, 45));
  ImGui::TextColored(ImVec4(1, 1, 0, 0.6f),
                     "TPO | POC: %.2f | VAH: %.2f | VAL: %.2f | Hi: %.2f | Lo: %.2f",
                     va.poc_price, va.vah, va.val, va.session_high, va.session_low);

  end_panel_window();
}

}  // namespace BTQuant
