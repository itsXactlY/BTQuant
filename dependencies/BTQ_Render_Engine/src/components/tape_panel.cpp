#include "components/tape_panel.hpp"

#include <imgui.h>

#include <algorithm>
#include <cstdio>

namespace BTQuant {

// ============================================================================
// MMT Color Constants
// ============================================================================
static constexpr ImU32 COLOR_BUY = IM_COL32(0x00, 0xE5, 0x66, 0xFF);   // Neon Mint
static constexpr ImU32 COLOR_SELL = IM_COL32(0xE6, 0x19, 0x26, 0xFF);  // Crimson
static constexpr ImU32 COLOR_SWEEP_BRACKET = IM_COL32(0xFF, 0xFF, 0xFF, 0xC8);

// ============================================================================
// Construction
// ============================================================================

TapePanel::TapePanel(const PanelConfig& config,
                     std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(config), processor_(std::move(processor)) {}

TapePanel::~TapePanel() = default;

// ============================================================================
// Data Update (called once per frame, before render)
// ============================================================================

void TapePanel::update(float /*dt*/) {
  if (!processor_) return;
  trade_count_ = processor_->peek_trades(symbol_id_, MAX_VISIBLE_TRADES, recent_trades_.data());
}

void TapePanel::set_symbol(uint32_t symbol_id, const std::string& symbol_name) {
  symbol_id_ = symbol_id;
  symbol_name_ = symbol_name;
}

// ============================================================================
// Alpha Mapping — trade size → row opacity [0.05, 0.50]
// ============================================================================

float TapePanel::compute_size_alpha(const TradeData& trade) const {
  if (trade_count_ == 0) return 0.06f;
  size_t count = 0;
  const size_t n = std::min(trade_count_, MAX_VISIBLE_TRADES);
  for (size_t i = 0; i < n; ++i) {
    if (recent_trades_[i].volume <= trade.volume) ++count;
  }
  float percentile = static_cast<float>(count) / static_cast<float>(n);
  return 0.03f + 0.15f * percentile;  // [0.03, 0.18] — subtle tinted rows
}

// ============================================================================
// Single Tape Row
// ============================================================================

void TapePanel::render_tape_row(const TradeData& trade, int row_index,
                                const TradeData* prev_trade) const {
  ImGui::TableNextRow();

  const bool is_buy = trade.is_buy();
  const ImU32 side_color = is_buy ? COLOR_BUY : COLOR_SELL;
  const ImU32 text_color = IM_COL32(0xE0, 0xE0, 0xE0, 0xFF);  // Light gray for readability

  // Subtle alpha-mapped row background
  float alpha = compute_size_alpha(trade);
  ImU32 row_bg = is_buy ? IM_COL32(0x00, 0xE5, 0x66, static_cast<uint8_t>(alpha * 255))
                        : IM_COL32(0xE6, 0x19, 0x26, static_cast<uint8_t>(alpha * 255));
  ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, row_bg);

  // Sweep detection: Δt < 50ms and price changed
  bool is_sweep = false;
  if (prev_trade) {
    int64_t delta_us =
        static_cast<int64_t>(trade.timestamp_us) - static_cast<int64_t>(prev_trade->timestamp_us);
    is_sweep = (std::abs(delta_us) < 50000) && (trade.price != prev_trade->price);
  }

  // Column 0: Time (HH:MM:SS.mmm)
  ImGui::TableSetColumnIndex(0);
  {
    uint64_t total_ms = trade.timestamp_us / 1000;
    uint32_t ms = total_ms % 1000;
    uint32_t sec = (total_ms / 1000) % 60;
    uint32_t min = (total_ms / 60000) % 60;
    uint32_t hr = (total_ms / 3600000) % 24;
    ImGui::TextColored(ImColor(text_color), "%02u:%02u:%02u.%03u", hr, min, sec, ms);
  }

  // Column 1: Price
  ImGui::TableSetColumnIndex(1);
  ImGui::TextColored(ImColor(text_color), "%.2f", trade.price);

  // Column 2: Size
  ImGui::TableSetColumnIndex(2);
  ImGui::TextColored(ImColor(text_color), "%.4f", static_cast<double>(trade.volume));

  // Column 3: Side (colored)
  ImGui::TableSetColumnIndex(3);
  ImGui::TextColored(ImColor(side_color), is_buy ? "BUY" : "SELL");

  // Sweep bracket: 1px white line on left edge
  if (is_sweep) {
    ImVec2 p_min = ImGui::GetItemRectMin();
    ImVec2 p_max = ImGui::GetItemRectMax();
    // Extend to row height
    p_min.x -= ImGui::GetStyle().CellPadding.x + ImGui::GetColumnWidth(0) +
               ImGui::GetColumnWidth(1) + ImGui::GetColumnWidth(2) + ImGui::GetColumnWidth(3);
    ImGui::GetWindowDrawList()->AddLine({p_min.x, p_min.y}, {p_min.x, p_max.y}, COLOR_SWEEP_BRACKET,
                                        1.0f);
  }

  (void)row_index;
}

// ============================================================================
// Main Render
// ============================================================================

void TapePanel::render_content() {
  // Size filter slider
  ImGui::SliderFloat("Min Size", &size_filter_, 0.0f, 100.0f, "%.1f");
  ImGui::Separator();

  // Trade count info
  ImGui::TextDisabled("Trades: %zu | Symbol: %s", trade_count_, symbol_name_.c_str());

  const ImGuiTableFlags table_flags =
      ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH |
      ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_NoSavedSettings;

  ImVec2 avail = ImGui::GetContentRegionAvail();
  if (ImGui::BeginTable("##TapeTable", 4, table_flags, avail)) {
    ImGui::TableSetupColumn("Time", ImGuiTableColumnFlags_WidthFixed, 100.0f);
    ImGui::TableSetupColumn("Price", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, 80.0f);
    ImGui::TableSetupColumn("Side", ImGuiTableColumnFlags_WidthFixed, 45.0f);
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableHeadersRow();

    // Count visible rows (after size filter) for clipper
    // Build filtered index list
    // For performance, we iterate with clipper over all trades and skip filtered ones inline
    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(trade_count_));
    while (clipper.Step()) {
      for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
        // Newest first: reverse order
        const int idx = static_cast<int>(trade_count_) - 1 - i;
        if (idx < 0) break;
        const TradeData& trade = recent_trades_[idx];

        // Size filter
        if (trade.volume < size_filter_) {
          // Skip but still need table row for clipper consistency
          ImGui::TableNextRow();
          ImGui::TableSetColumnIndex(0);
          // Empty row (clipper expects fixed row count)
          continue;
        }

        // Previous trade for sweep detection
        const TradeData* prev = (idx > 0) ? &recent_trades_[idx - 1] : nullptr;
        render_tape_row(trade, i, prev);
      }
    }

    ImGui::EndTable();
  }
}

}  // namespace BTQuant
