#include "../../include/components/tape_panel.hpp"

#include <algorithm>
#include <ctime>

#include "imgui.h"

namespace BTQuant {

TapePanel::TapePanel(const PanelConfig& config, std::shared_ptr<HotSpineDataBridge> bridge,
                     std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : PanelBase(config), bridge_(bridge), processor_(processor) {
  cached_trades_.reserve(MAX_VISIBLE_TRADES);

  // C++26: Subscribe to push notifications instead of polling
  subscribe_to_updates();
}

TapePanel::~TapePanel() {
  // C++26: Clean unsubscription on destruction
  if (processor_ && subscription_id_ != 0) {
    processor_->unsubscribe(subscription_id_);
  }
}

void TapePanel::subscribe_to_updates() {
  if (!processor_ || symbol_id_ == 0) return;

  // Unsubscribe from previous symbol if any
  if (subscription_id_ != 0) {
    processor_->unsubscribe(subscription_id_);
  }

  // Subscribe to TRADE notifications for this symbol
  subscription_id_ = processor_->subscribe(
      symbol_id_, RenderEngine::NotificationType::TRADE,
      [this](uint32_t /*symbol_id*/, RenderEngine::NotificationType /*type*/) {
        // Thread-safe: atomic flag set from worker thread
        this->markDirty();
      });
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

  // C++26 Reactive: Refresh data when new trades arrive or first load
  if (processor_ && symbol_id_ != 0) {
    if (consumeDirty() || cached_trades_.empty()) {
      auto analytics = processor_->getSymbolAnalytics(symbol_id_);
      cached_trades_ = analytics.recent_trades;

      // Keep only most recent trades for display
      if (cached_trades_.size() > MAX_VISIBLE_TRADES) {
        cached_trades_.erase(cached_trades_.begin(),
                             cached_trades_.begin() + (cached_trades_.size() - MAX_VISIBLE_TRADES));
      }
    }
  }

  render_trade_table();

  end_panel_window();
}

void TapePanel::set_symbol(uint32_t symbol_id, const std::string& symbol_name) {
  symbol_id_ = symbol_id;
  symbol_name_ = symbol_name;
  cached_trades_.clear();

  // Re-subscribe to new symbol
  subscribe_to_updates();
  markDirty();  // Force immediate refresh
}

void TapePanel::render_controls() {
  ImGui::Text("Symbol: %s", symbol_name_.c_str());
  ImGui::SameLine();
  ImGui::Checkbox("Auto-scroll", &auto_scroll_);
  ImGui::SameLine();
  ImGui::Text("| Trades: %zu", cached_trades_.size());
}

void TapePanel::render_trade_table() {
  // Unique table ID per panel instance to avoid ID conflicts
  char table_id[64];
  snprintf(table_id, sizeof(table_id), "TapeTable##%s", config_.title.c_str());

  if (ImGui::BeginTable(table_id, 4,
                        ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_Resizable)) {
    ImGui::TableSetupColumn("Time", ImGuiTableColumnFlags_WidthFixed, 80.0f);
    ImGui::TableSetupColumn("Price", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Side", ImGuiTableColumnFlags_WidthFixed, 40.0f);
    ImGui::TableHeadersRow();

    // Render trades in reverse order (newest first)
    int row_index = 0;
    for (auto it = cached_trades_.rbegin(); it != cached_trades_.rend(); ++it, ++row_index) {
      const auto& trade = *it;

      // Push unique ID for this row to avoid conflicts
      ImGui::PushID(row_index);
      ImGui::TableNextRow();

      // Time column (HH:MM:SS.mmm)
      ImGui::TableSetColumnIndex(0);
      if (trade.timestamp > 0) {
        time_t time_sec = trade.timestamp / 1000000;  // micros to seconds
        uint64_t millis = (trade.timestamp / 1000) % 1000;
        char time_str[16];
        strftime(time_str, sizeof(time_str), "%H:%M:%S", localtime(&time_sec));
        ImGui::Text("%s.%03lu", time_str, static_cast<unsigned long>(millis));
      } else {
        ImGui::Text("-");
      }

      // Price column
      ImGui::TableSetColumnIndex(1);
      const auto& colors = ThemeManager::getInstance().getColors();
      ImVec4 price_color = trade.is_buy ? colors.accent_green : colors.accent_red;
      ImGui::TextColored(price_color, "%.4f", trade.price);

      // Size column
      ImGui::TableSetColumnIndex(2);
      ImGui::Text("%.4f", trade.size);

      // Side column
      ImGui::TableSetColumnIndex(3);
      if (trade.is_buy) {
        ImGui::TextColored(colors.accent_green, "BUY");
      } else {
        ImGui::TextColored(colors.accent_red, "SELL");
      }

      ImGui::PopID();
    }

    // Auto-scroll to bottom (newest trades)
    if (auto_scroll_ && !cached_trades_.empty()) {
      ImGui::SetScrollHereY(0.0f);
    }

    ImGui::EndTable();
  }
}

}  // namespace BTQuant
