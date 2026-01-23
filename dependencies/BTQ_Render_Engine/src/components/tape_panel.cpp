#include "../../include/components/tape_panel.hpp"
#include "imgui.h"
#include <algorithm>
#include <ctime>

namespace BTQuant {

TapePanel::TapePanel(
    const PanelConfig &config, std::shared_ptr<HotSpineDataBridge> bridge,
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(config), bridge_(bridge), processor_(processor) {}

void TapePanel::update(float dt) {
  update_timer_ += dt;
  if (update_timer_ >= UPDATE_INTERVAL) {
    // Refresh trade data from processor
    if (processor_ && symbol_id_ != 0) {
      auto analytics = processor_->getSymbolAnalytics(symbol_id_);
      cached_trades_ = analytics.recent_trades;

      // Keep only most recent trades for display
      if (cached_trades_.size() > MAX_VISIBLE_TRADES) {
        cached_trades_.erase(cached_trades_.begin(),
                             cached_trades_.begin() +
                                 (cached_trades_.size() - MAX_VISIBLE_TRADES));
      }
    }
    update_timer_ = 0.0f;
  }
}

void TapePanel::render() {
  begin_panel_window();

  if (!is_visible()) {
    end_panel_window();
    return;
  }

  render_panel_header();
  render_controls();
  ImGui::Separator();
  render_trade_table();

  end_panel_window();
}

void TapePanel::set_symbol(uint32_t symbol_id, const std::string &symbol_name) {
  symbol_id_ = symbol_id;
  symbol_name_ = symbol_name;
  cached_trades_.clear();
}

void TapePanel::render_controls() {
  ImGui::Text("Symbol: %s", symbol_name_.c_str());
  ImGui::SameLine();
  ImGui::Checkbox("Auto-scroll", &auto_scroll_);
  ImGui::SameLine();
  ImGui::Text("| Trades: %zu", cached_trades_.size());
}

void TapePanel::render_trade_table() {
  if (ImGui::BeginTable("TapeTable", 4,
                        ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_BordersInnerV |
                            ImGuiTableFlags_Resizable)) {

    ImGui::TableSetupColumn("Time", ImGuiTableColumnFlags_WidthFixed, 80.0f);
    ImGui::TableSetupColumn("Price", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Side", ImGuiTableColumnFlags_WidthFixed, 40.0f);
    ImGui::TableHeadersRow();

    // Render trades in reverse order (newest first)
    for (auto it = cached_trades_.rbegin(); it != cached_trades_.rend(); ++it) {
      const auto &trade = *it;

      ImGui::TableNextRow();

      // Time column (HH:MM:SS.mmm)
      ImGui::TableSetColumnIndex(0);
      if (trade.timestamp > 0) {
        time_t time_sec = trade.timestamp / 1000000; // micros to seconds
        uint64_t millis = (trade.timestamp / 1000) % 1000;
        char time_str[16];
        strftime(time_str, sizeof(time_str), "%H:%M:%S", localtime(&time_sec));
        ImGui::Text("%s.%03lu", time_str, static_cast<unsigned long>(millis));
      } else {
        ImGui::Text("-");
      }

      // Price column
      ImGui::TableSetColumnIndex(1);
      ImVec4 price_color = trade.is_buy ? ImVec4(0.2f, 0.8f, 0.2f, 1.0f)
                                        : ImVec4(0.8f, 0.2f, 0.2f, 1.0f);
      ImGui::TextColored(price_color, "%.4f", trade.price);

      // Size column
      ImGui::TableSetColumnIndex(2);
      ImGui::Text("%.4f", trade.size);

      // Side column
      ImGui::TableSetColumnIndex(3);
      if (trade.is_buy) {
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "BUY");
      } else {
        ImGui::TextColored(ImVec4(0.8f, 0.2f, 0.2f, 1.0f), "SELL");
      }
    }

    // Auto-scroll to bottom (newest trades)
    if (auto_scroll_ && !cached_trades_.empty()) {
      ImGui::SetScrollHereY(0.0f);
    }

    ImGui::EndTable();
  }
}

} // namespace BTQuant
