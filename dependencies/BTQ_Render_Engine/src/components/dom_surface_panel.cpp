#include "components/dom_surface_panel.hpp"

#include <imgui.h>

#include <algorithm>
#include <cstdint>

namespace BTQuant {

// ============================================================================
// MMT Color Constants
// ============================================================================
static constexpr ImU32 DOM_COLOR_BID = IM_COL32(0x00, 0xE5, 0x66, 0xFF);      // Neon Mint
static constexpr ImU32 DOM_COLOR_ASK = IM_COL32(0xE6, 0x19, 0x26, 0xFF);      // Crimson
static constexpr ImU32 DOM_COLOR_BID_BAR = IM_COL32(0x00, 0xE5, 0x66, 0x40);  // Mint translucent
static constexpr ImU32 DOM_COLOR_ASK_BAR = IM_COL32(0xE6, 0x19, 0x26, 0x40);  // Crimson translucent
static constexpr ImU32 DOM_COLOR_SPREAD = IM_COL32(0xFF, 0xD7, 0x00, 0xFF);   // Gold

// ============================================================================
// Construction
// ============================================================================

DomSurfacePanel::DomSurfacePanel(const PanelConfig& config,
                                 std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(config), processor_(std::move(processor)) {}

DomSurfacePanel::~DomSurfacePanel() = default;

void DomSurfacePanel::setSymbol(uint32_t symbol_id) { current_symbol_id_ = symbol_id; }

void DomSurfacePanel::render_panel_header() { PanelBase::render_panel_header(); }

void DomSurfacePanel::updateLivePrice(double price) {
  live_price_.store(price, std::memory_order_release);
}

// ============================================================================
// Main Render
// ============================================================================

void DomSurfacePanel::render_content() {
  if (!processor_) {
    ImGui::Text("No processor");
    return;
  }

  // Read the atomic orderbook snapshot (lock-free pointer load)
  const auto* snapshot = processor_->get_active_orderbook(current_symbol_id_);
  if (!snapshot) {
    ImGui::TextDisabled("No orderbook data for symbol %u", current_symbol_id_);
    return;
  }

  // Read book metrics
  double spread = snapshot->spread;
  double best_bid = snapshot->best_bid;

  uint32_t ask_count =
      static_cast<uint32_t>(std::min(snapshot->asks_size(), static_cast<size_t>(max_levels_)));
  uint32_t bid_count =
      static_cast<uint32_t>(std::min(snapshot->bids_size(), static_cast<size_t>(max_levels_)));

  // Header info
  ImGui::TextColored(ImColor(DOM_COLOR_SPREAD), "Spread: %.2f (%.3f%%)", spread,
                     best_bid > 0 ? (spread / best_bid * 100.0) : 0.0);
  ImGui::Separator();

  // Determine max size for histogram scaling
  double max_sz = 0.0;
  for (uint32_t i = 0; i < ask_count; ++i) max_sz = std::max(max_sz, snapshot->asks[i].size);
  for (uint32_t i = 0; i < bid_count; ++i) max_sz = std::max(max_sz, snapshot->bids[i].size);
  if (max_sz == 0.0) max_sz = 1.0;

  const ImGuiTableFlags table_flags =
      ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH |
      ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_NoSavedSettings;

  ImVec2 avail = ImGui::GetContentRegionAvail();
  if (ImGui::BeginTable("##DOMTable", 3, table_flags, avail)) {
    ImGui::TableSetupColumn("Price", ImGuiTableColumnFlags_WidthFixed, 90.0f);
    ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, 80.0f);
    ImGui::TableSetupColumn("Depth", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableHeadersRow();

    // ---- ASKS (reversed: worst ask at top, best ask at bottom) ----
    for (int i = static_cast<int>(ask_count) - 1; i >= 0; --i) {
      const auto& lvl = snapshot->asks[i];
      float ratio = static_cast<float>(lvl.size / max_sz);

      ImGui::TableNextRow();
      ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0,
                             IM_COL32(0xE6, 0x19, 0x26, static_cast<uint8_t>(ratio * 80)));

      ImGui::TableSetColumnIndex(0);
      ImGui::TextColored(ImColor(DOM_COLOR_ASK), "%.2f", lvl.price);

      ImGui::TableSetColumnIndex(1);
      ImGui::TextColored(ImColor(DOM_COLOR_ASK), "%.4f", lvl.size);

      // Histogram bar
      ImGui::TableSetColumnIndex(2);
      ImVec2 cursor = ImGui::GetCursorScreenPos();
      float bar_w = ImGui::GetContentRegionAvail().x * ratio;
      float row_h = ImGui::GetTextLineHeightWithSpacing();
      ImGui::GetWindowDrawList()->AddRectFilled(cursor, {cursor.x + bar_w, cursor.y + row_h},
                                                DOM_COLOR_ASK_BAR);
      ImGui::Dummy({0, row_h});
    }

    // ---- SPREAD ROW ----
    ImGui::TableNextRow();
    ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0, IM_COL32(0xFF, 0xD7, 0x00, 0x18));
    ImGui::TableSetColumnIndex(0);
    ImGui::TextColored(ImColor(DOM_COLOR_SPREAD), "── SPREAD ──");
    ImGui::TableSetColumnIndex(1);
    ImGui::TextColored(ImColor(DOM_COLOR_SPREAD), "%.2f", spread);

    // ---- BIDS (best bid at top, worst at bottom) ----
    for (uint32_t i = 0; i < bid_count; ++i) {
      const auto& lvl = snapshot->bids[i];
      float ratio = static_cast<float>(lvl.size / max_sz);

      ImGui::TableNextRow();
      ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0,
                             IM_COL32(0x00, 0xE5, 0x66, static_cast<uint8_t>(ratio * 80)));

      ImGui::TableSetColumnIndex(0);
      ImGui::TextColored(ImColor(DOM_COLOR_BID), "%.2f", lvl.price);

      ImGui::TableSetColumnIndex(1);
      ImGui::TextColored(ImColor(DOM_COLOR_BID), "%.4f", lvl.size);

      // Histogram bar
      ImGui::TableSetColumnIndex(2);
      ImVec2 cursor = ImGui::GetCursorScreenPos();
      float bar_w = ImGui::GetContentRegionAvail().x * ratio;
      float row_h = ImGui::GetTextLineHeightWithSpacing();
      ImGui::GetWindowDrawList()->AddRectFilled(cursor, {cursor.x + bar_w, cursor.y + row_h},
                                                DOM_COLOR_BID_BAR);
      ImGui::Dummy({0, row_h});
    }

    ImGui::EndTable();
  }
}

}  // namespace BTQuant
