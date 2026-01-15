#include "../../include/vulkan_dashboard_advanced.hpp"
#include <imgui.h>
#include <imgui_internal.h>

namespace BTQuant {

// ============================================================================
// StrategyControlComponent
// ============================================================================

StrategyControlComponent::StrategyControlComponent(const glm::vec2 &position,
                                                   const glm::vec2 &size)
    : UIComponent(position, size) {
  strategies_ = {{"Grid Scalper", true, 1250.45f, 0.02f, 156, "RUNNING"},
                 {"Basis Arb", false, -45.20f, 0.00f, 0, "STOPPED"},
                 {"Trend Follower", true, 4500.22f, 0.04f, 12, "RUNNING"}};
}

void StrategyControlComponent::render_gui() {
  ImGui::SetNextWindowPos(ImVec2(position_.x, position_.y), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(size_.x, size_.y), ImGuiCond_Always);

  ImGui::SetNextWindowCollapsed(minimized_, ImGuiCond_Appearing);
  if (ImGui::Begin("Strategy Control", &visible_)) {
    minimized_ = false;
    if (ImGui::BeginTable("StrategyTable", 4,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
      ImGui::TableSetupColumn("Strategy");
      ImGui::TableSetupColumn("PnL");
      ImGui::TableSetupColumn("Status");
      ImGui::TableSetupColumn("Action");
      ImGui::TableHeadersRow();

      for (auto &strat : strategies_) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("%s", strat.name.c_str());

        ImGui::TableSetColumnIndex(1);
        ImGui::TextColored(strat.pnl >= 0 ? ImVec4(0, 1, 0, 1)
                                          : ImVec4(1, 0, 0, 1),
                           "$%.2f", strat.pnl);

        ImGui::TableSetColumnIndex(2);
        ImGui::Text("%s", strat.status.c_str());

        ImGui::TableSetColumnIndex(3);
        if (ImGui::SmallButton(strat.active ? "Stop" : "Start")) {
          strat.active = !strat.active;
          strat.status = strat.active ? "RUNNING" : "STOPPED";
        }
      }
      ImGui::EndTable();
    }
  } else {
    minimized_ = true;
  }
  ImGui::End();
}

// ============================================================================
// RiskManagerComponent
// ============================================================================

RiskManagerComponent::RiskManagerComponent(const glm::vec2 &position,
                                           const glm::vec2 &size)
    : UIComponent(position, size) {}

void RiskManagerComponent::render_gui() {
  ImGui::SetNextWindowPos(ImVec2(position_.x, position_.y), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(size_.x, size_.y), ImGuiCond_Always);

  ImGui::SetNextWindowCollapsed(minimized_, ImGuiCond_Appearing);
  if (ImGui::Begin("Risk Manager", &visible_)) {
    minimized_ = false;
    ImGui::Text("Portfolio Health");
    ImGui::ProgressBar(0.85f, ImVec2(-1, 0), "85% MARGIN OK");

    ImGui::Separator();

    ImGui::Columns(2, "RiskColumns");
    ImGui::Text("Daily Loss");
    ImGui::NextColumn();
    ImGui::TextColored(ImVec4(0, 1, 0, 1), "$1,250.00 / $5,000");
    ImGui::NextColumn();

    ImGui::Text("Max DD");
    ImGui::NextColumn();
    ImGui::TextColored(ImVec4(1, 1, 0, 1), "4.2%% / 10.0%%");
    ImGui::NextColumn();
    ImGui::Columns(1);

    ImGui::Spacing();
    if (ImGui::Button("EMERGENCY STOP (DE-LEVERAGE)", ImVec2(-1, 40))) {
      // Panic logic
    }
  } else {
    minimized_ = true;
  }
  ImGui::End();
}

// ============================================================================
// TradingInterfaceComponent
// ============================================================================

TradingInterfaceComponent::TradingInterfaceComponent(const glm::vec2 &position,
                                                     const glm::vec2 &size)
    : UIComponent(position, size) {}

void TradingInterfaceComponent::render_gui() {
  ImGui::SetNextWindowPos(ImVec2(position_.x, position_.y), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(size_.x, size_.y), ImGuiCond_Always);

  ImGui::SetNextWindowCollapsed(minimized_, ImGuiCond_Appearing);
  if (ImGui::Begin("Execution", &visible_)) {
    minimized_ = false;
    ImGui::PushFont(dashboard_ ? dashboard_->get_monospace_font() : nullptr);

    ImGui::InputFloat("Quantity", &quantity_, 0.1f, 1.0f, "%.2f");
    ImGui::InputFloat("Price", &price_, 0.5f, 10.0f, "%.2f");
    if (ImGui::Button("Sync crosshair price")) {
      if (dashboard_) {
        price_ = (float)dashboard_->get_crosshair_state().price;
      }
    }

    const char *types[] = {"Limit", "Market", "Post-Only"};
    if (ImGui::BeginCombo("Type", order_type_.c_str())) {
      for (int n = 0; n < IM_ARRAYSIZE(types); n++) {
        bool is_selected = (order_type_ == types[n]);
        if (ImGui::Selectable(types[n], is_selected))
          order_type_ = types[n];
        if (is_selected)
          ImGui::SetItemDefaultFocus();
      }
      ImGui::EndCombo();
    }

    ImGui::Spacing();
    float btn_w =
        (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) *
        0.5f;

    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.6f, 0.1f, 1.0f));
    if (ImGui::Button("BUY", ImVec2(btn_w, 50))) {
      // Execute Buy
    }
    ImGui::PopStyleColor();

    ImGui::SameLine();

    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.6f, 0.1f, 0.1f, 1.0f));
    if (ImGui::Button("SELL", ImVec2(btn_w, 50))) {
      // Execute Sell
    }
    ImGui::PopStyleColor();

    ImGui::PopFont();
  } else {
    minimized_ = true;
  }
  ImGui::End();
}

// ============================================================================
// TapeComponent (Time & Sales)
// ============================================================================

TapeComponent::TapeComponent(const glm::vec2 &position, const glm::vec2 &size)
    : UIComponent(position, size) {}

TapeComponent::~TapeComponent() {}

void TapeComponent::handle_trade(const RenderEngine::TradeData &trade) {
  TapeEntry entry;
  entry.timestamp_us = trade.timestamp_us;
  entry.price = trade.price;
  entry.size = trade.size;
  entry.is_buy = trade.is_buy;
  entry.is_large_trade = trade.size >= large_trade_threshold_;

  if (!entries_.empty()) {
    entry.delta = (float)(trade.price - entries_.front().price);
  } else {
    entry.delta = 0;
  }

  entries_.push_front(entry);
  if (entries_.size() > 100)
    entries_.pop_back();

  if (trade.is_buy)
    cumulative_delta_ += (float)trade.size;
  else
    cumulative_delta_ -= (float)trade.size;

  mark_dirty();
}

void TapeComponent::update(float delta_time) {}

void TapeComponent::clear_data() {
  entries_.clear();
  cumulative_delta_ = 0.0f;
  mark_dirty();
}

void TapeComponent::render_gui() {
  ImGui::SetNextWindowPos(ImVec2(position_.x, position_.y), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(size_.x, size_.y), ImGuiCond_Always);

  if (ImGui::Begin("Time & Sales", &visible_)) {
    ImGui::Text("Delta: %.2f", cumulative_delta_);
    ImGui::Separator();

    if (ImGui::BeginTable("TapeTable", 3, ImGuiTableFlags_ScrollY)) {
      ImGui::TableSetupColumn("Time");
      ImGui::TableSetupColumn("Price");
      ImGui::TableSetupColumn("Size");
      ImGui::TableHeadersRow();

      for (const auto &entry : entries_) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("%lu", entry.timestamp_us % 1000000);

        ImGui::TableSetColumnIndex(1);
        ImGui::TextColored(entry.is_buy ? ImVec4(0, 1, 0, 1)
                                        : ImVec4(1, 0, 0, 1),
                           "%.2f", entry.price);

        ImGui::TableSetColumnIndex(2);
        if (entry.is_large_trade) {
          ImGui::TextColored(ImVec4(1, 1, 0, 1), "%.4f !!", entry.size);
        } else {
          ImGui::Text("%.4f", entry.size);
        }
      }
      ImGui::EndTable();
    }
  }
  ImGui::End();
}

// ============================================================================
// OrderManagementComponent
// ============================================================================

OrderManagementComponent::OrderManagementComponent(const glm::vec2 &position,
                                                   const glm::vec2 &size)
    : UIComponent(position, size) {}

OrderManagementComponent::~OrderManagementComponent() {}

void OrderManagementComponent::update(float delta_time) {}

void OrderManagementComponent::render_gui() {
  ImGui::SetNextWindowPos(ImVec2(position_.x, position_.y), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(size_.x, size_.y), ImGuiCond_Always);

  if (ImGui::Begin("Order Management", &visible_)) {
    if (ImGui::BeginTabBar("OrderTabs")) {
      if (ImGui::BeginTabItem("New Order")) {
        ImGui::InputText("Symbol", &symbol_[0], symbol_.size());
        ImGui::InputFloat("Qty", &quantity_);
        ImGui::InputFloat("Price", &price_);
        if (ImGui::Button("PLACE BUY", ImVec2(120, 40))) {
        }
        ImGui::SameLine();
        if (ImGui::Button("PLACE SELL", ImVec2(120, 40))) {
        }
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Active Orders")) {
        ImGui::Text("No active orders");
        ImGui::EndTabItem();
      }
      ImGui::EndTabBar();
    }
  }
  ImGui::End();
}

// ============================================================================
// PositionPanelComponent
// ============================================================================

PositionPanelComponent::PositionPanelComponent(const glm::vec2 &position,
                                               const glm::vec2 &size)
    : UIComponent(position, size) {
  positions_ = {{"BTC-USDT", 42000.0f, 42500.0f, 0.5f, 250.0f, 1.19f}};
}

PositionPanelComponent::~PositionPanelComponent() {}

void PositionPanelComponent::update(float delta_time) {}

void PositionPanelComponent::render_gui() {
  ImGui::SetNextWindowPos(ImVec2(position_.x, position_.y), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(size_.x, size_.y), ImGuiCond_Always);

  if (ImGui::Begin("Positions & P&L", &visible_)) {
    ImGui::Text("Equity: $%.2f", total_equity_);
    ImGui::SameLine();
    ImGui::Text("Balance: $%.2f", available_balance_);
    ImGui::Separator();

    if (ImGui::BeginTable("PosTable", 5, ImGuiTableFlags_Borders)) {
      ImGui::TableSetupColumn("Symbol");
      ImGui::TableSetupColumn("Size");
      ImGui::TableSetupColumn("Entry");
      ImGui::TableSetupColumn("PnL");
      ImGui::TableSetupColumn("PnL %");
      ImGui::TableHeadersRow();

      for (const auto &pos : positions_) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("%s", pos.symbol.c_str());
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%.4f", pos.quantity);
        ImGui::TableSetColumnIndex(2);
        ImGui::Text("%.2f", pos.entry_price);
        ImGui::TableSetColumnIndex(3);
        ImGui::TextColored(pos.pnl >= 0 ? ImVec4(0, 1, 0, 1)
                                        : ImVec4(1, 0, 0, 1),
                           "$%.2f", pos.pnl);
        ImGui::TableSetColumnIndex(4);
        ImGui::TextColored(pos.pnl >= 0 ? ImVec4(0, 1, 0, 1)
                                        : ImVec4(1, 0, 0, 1),
                           "%.2f%%", pos.pnl_percent);
      }
      ImGui::EndTable();
    }
  }
  ImGui::End();
}

// ============================================================================
// MarketOverviewPanel
// ============================================================================

MarketOverviewPanel::MarketOverviewPanel(const glm::vec2 &position,
                                         const glm::vec2 &size)
    : UIComponent(position, size) {
  tickers_ = {{"BTC/USDT", 42500.50f, 1.25f},
              {"ETH/USDT", 2250.20f, -0.85f},
              {"SOL/USDT", 98.45f, 5.12f},
              {"DOT/USDT", 7.20f, 0.15f}};
}

MarketOverviewPanel::~MarketOverviewPanel() {}

void MarketOverviewPanel::update(float delta_time) {}

void MarketOverviewPanel::render_gui() {
  ImGui::SetNextWindowPos(ImVec2(position_.x, position_.y), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(size_.x, size_.y), ImGuiCond_Always);

  ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar |
                           ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                           ImGuiWindowFlags_NoScrollbar;

  if (ImGui::Begin("Market Overview", &visible_, flags)) {
    // Render tickers in a row
    for (size_t i = 0; i < tickers_.size(); ++i) {
      const auto &t = tickers_[i];
      ImGui::Text("%s", t.symbol.c_str());
      ImGui::SameLine();
      ImGui::TextColored(t.change_pct >= 0 ? ImVec4(0, 1, 0, 1)
                                           : ImVec4(1, 0, 0, 1),
                         "%.2f (%.2f%%)", t.price, t.change_pct);

      if (i < tickers_.size() - 1) {
        ImGui::SameLine();
        ImGui::Text(" | ");
        ImGui::SameLine();
      }
    }

    ImGui::SameLine(ImGui::GetWindowWidth() - 250);
    ImGui::Text("Vol: $%.1fB", global_volume_ / 1e9f);
    ImGui::SameLine();
    ImGui::Text("| Latency: %.2fms", system_latency_ms_);
  }
  ImGui::End();
}
} // namespace BTQuant
