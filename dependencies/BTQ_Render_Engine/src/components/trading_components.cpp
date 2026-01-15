#include "../../include/vulkan_dashboard_advanced.hpp"
#include <cfloat>
#include <cstdlib>
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

  ImGuiWindowFlags flags = ImGuiWindowFlags_NoScrollbar;
  ImGui::SetNextWindowCollapsed(minimized_, ImGuiCond_Appearing);
  if (ImGui::Begin("Trading Terminal", &visible_, flags)) {
    minimized_ = false;
    ImGui::PushFont(dashboard_ ? dashboard_->get_monospace_font() : nullptr);

    // Advanced Order Selector
    const char *types[] = {"Limit",         "Market",  "Stop-Limit",
                           "Trailing Stop", "Iceberg", "TWAP"};
    ImGui::PushItemWidth(-1);
    if (ImGui::BeginCombo("##OrderType", order_type_.c_str())) {
      for (int n = 0; n < IM_ARRAYSIZE(types); n++) {
        bool is_selected = (order_type_ == types[n]);
        if (ImGui::Selectable(types[n], is_selected))
          order_type_ = types[n];
      }
      ImGui::EndCombo();
    }
    ImGui::PopItemWidth();
    ImGui::Spacing();

    // Context-sensitive inputs
    ImGui::TextDisabled("QUANTITY");
    ImGui::InputFloat("##Qty", &quantity_, 0.1f, 1.0f, "%.2f");

    if (order_type_ != "Market") {
      ImGui::TextDisabled("LIMIT PRICE");
      ImGui::InputFloat("##Price", &price_, 0.5f, 10.0f, "%.2f");
    }

    if (order_type_ == "Stop-Limit") {
      ImGui::TextDisabled("STOP PRICE");
      ImGui::InputFloat("##StopPrice", &stop_price_, 0.5f, 10.0f, "%.2f");
    } else if (order_type_ == "Trailing Stop") {
      ImGui::TextDisabled("TRAILING PERCENT (%)");
      ImGui::SliderFloat("##Trail", &trailing_pct_, 0.1f, 5.0f, "%.2f%%");
    } else if (order_type_ == "Iceberg") {
      ImGui::TextDisabled("DISPLAY QTY");
      ImGui::InputFloat("##DisplayQty", &iceberg_display_qty_, 0.01f, 0.1f,
                        "%.4f");
    } else if (order_type_ == "TWAP") {
      ImGui::TextDisabled("DURATION (MINS)");
      ImGui::SliderInt("##TWAPDur", &twap_duration_mins_, 1, 480);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    float btn_w =
        (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) *
        0.5f;

    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.6f, 0.3f, 1.0f));
    if (ImGui::Button("BUY / LONG", ImVec2(btn_w, 45))) {
      // EXECUTE BUY
    }
    ImGui::PopStyleColor();

    ImGui::SameLine();

    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.1f, 0.1f, 1.0f));
    if (ImGui::Button("SELL / SHORT", ImVec2(btn_w, 45))) {
      // EXECUTE SELL
    }
    ImGui::PopStyleColor();

    ImGui::Spacing();

    // Emergency Controls
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.25f, 1.0f));
    if (ImGui::Button("CANCEL ALL", ImVec2(btn_w, 30))) {
      // CANCEL ALL
    }
    ImGui::PopStyleColor();

    ImGui::SameLine();

    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.3f, 0.0f, 1.0f));
    if (ImGui::Button("FLATTEN", ImVec2(btn_w, 30))) {
      // FLATTEN
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
  entry.is_large_trade = (float)trade.size >= large_trade_threshold_;
  entry.is_whale_trade = (float)trade.size >= whale_trade_threshold_;

  if (!entries_.empty()) {
    entry.delta = (float)(trade.price - entries_.front().price);
  } else {
    entry.delta = 0;
  }

  entries_.push_front(entry);
  if (entries_.size() > 200)
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

  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(4, 4));
  if (ImGui::Begin("Time & Sales", &visible_)) {
    // Delta Header
    ImGui::TextColored(cumulative_delta_ >= 0 ? ImVec4(0, 1, 0.4f, 1)
                                              : ImVec4(1, 0.2f, 0.2f, 1),
                       "CUMULATIVE DELTA: %.2f", cumulative_delta_);
    ImGui::Separator();

    if (ImGui::BeginTable("TapeTable", 3,
                          ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg |
                              ImGuiTableFlags_BordersOuter)) {
      ImGui::TableSetupColumn("Time", ImGuiTableColumnFlags_WidthFixed, 60.0f);
      ImGui::TableSetupColumn("Price", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableHeadersRow();

      for (const auto &entry : entries_) {
        ImGui::TableNextRow();

        // Coloring logic for the whole row if it's a whale trade
        if (entry.is_whale_trade) {
          ImGui::TableSetBgColor(
              ImGuiTableBgTarget_RowBg0,
              ImGui::GetColorU32(ImVec4(0.8f, 0.5f, 0.0f, 0.3f)));
        } else if (entry.is_large_trade) {
          ImGui::TableSetBgColor(
              ImGuiTableBgTarget_RowBg0,
              ImGui::GetColorU32(ImVec4(0.4f, 0.4f, 0.1f, 0.2f)));
        }

        ImGui::TableSetColumnIndex(0);
        ImGui::Text("%lu", (entry.timestamp_us / 1000000) %
                               86400); // Simple HH:MM:SS placeholder

        ImGui::TableSetColumnIndex(1);
        ImGui::TextColored(entry.is_buy ? ImVec4(0, 1, 0.2f, 1)
                                        : ImVec4(1, 0.1f, 0.1f, 1),
                           "%.2f", entry.price);

        ImGui::TableSetColumnIndex(2);
        if (entry.is_whale_trade) {
          ImGui::TextColored(ImVec4(1, 0.7f, 0, 1), "%.4f (WHALE)", entry.size);
        } else if (entry.is_large_trade) {
          ImGui::TextColored(ImVec4(1, 1, 0.3f, 1), "%.4f !!", entry.size);
        } else {
          ImGui::Text("%.4f", entry.size);
        }
      }
      ImGui::EndTable();
    }
  }
  ImGui::End();
  ImGui::PopStyleVar();
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
  positions_ = {{"BTC-USDT", 42000.0f, 42500.0f, 0.5f, 250.0f, 1.19f},
                {"ETH-USDT", 2450.0f, 2480.0f, 10.0f, 300.0f, 1.22f}};

  // Seed initial equity history
  float base = 100000.0f;
  for (int i = 0; i < 50; ++i) {
    base += (float)(rand() % 1000 - 500);
    equity_history_.push_back(base);
  }
}

PositionPanelComponent::~PositionPanelComponent() {
  if (vulkan_core_ && equity_vertex_buffer_.buffer != VK_NULL_HANDLE) {
    vulkan_core_->get_memory_manager().deallocate_buffer(equity_vertex_buffer_);
  }
}

void PositionPanelComponent::initialize_vulkan_resources(
    VulkanCore *vulkan_core) {
  vulkan_core_ = vulkan_core;
  // TODO: Initialize equity curve rendering pipeline
}

void PositionPanelComponent::rebuild_equity_geometry() {
  // TODO: Build line strip vertices for equity history
}

void PositionPanelComponent::update(float delta_time) {
  static float timer = 0;
  timer += delta_time;
  if (timer > 2.0f) { // Update equity mock data
    float last = equity_history_.back();
    equity_history_.push_back(last + (float)(rand() % 200 - 100));
    if (equity_history_.size() > 100)
      equity_history_.erase(equity_history_.begin());
    timer = 0;
    mark_dirty();
  }
}

void PositionPanelComponent::render_gui() {
  ImGui::SetNextWindowPos(ImVec2(position_.x, position_.y), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(size_.x, size_.y), ImGuiCond_Always);

  if (ImGui::Begin("Portfolio Analytics", &visible_)) {
    // Top Stats Bar
    ImGui::Columns(3, "EquityStats");
    ImGui::TextDisabled("TOTAL EQUITY");
    ImGui::TextColored(ImVec4(1, 1, 1, 1), "$%.2f", total_equity_);
    ImGui::NextColumn();
    ImGui::TextDisabled("AVAIL. MARGIN");
    ImGui::TextColored(ImVec4(0, 1, 0.8f, 1), "$%.2f", available_balance_);
    ImGui::NextColumn();
    ImGui::TextDisabled("UNREALIZED P&L");
    ImGui::TextColored(ImVec4(0, 1, 0, 1), "+$550.00");
    ImGui::Columns(1);

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Mini Equity Curve (ImGui Fallback)
    ImGui::TextDisabled("PERFORMANCE (EQUITY CURVE)");
    ImGui::PushStyleColor(ImGuiCol_PlotLines, ImVec4(0, 1, 0.5f, 1));
    ImGui::PlotLines("##EquityCurve", equity_history_.data(),
                     (int)equity_history_.size(), 0, nullptr, FLT_MAX, FLT_MAX,
                     ImVec2(-1, 80));
    ImGui::PopStyleColor();

    ImGui::Spacing();

    if (ImGui::BeginTable("PosTable", 5,
                          ImGuiTableFlags_RowBg |
                              ImGuiTableFlags_BordersOuter)) {
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
