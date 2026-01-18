#include "../../include/vulkan_dashboard_advanced.hpp"
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

  if (ImGui::Begin("Strategy Control", &visible_)) {
    if (ImGui::BeginTable("StrategyTable", 3,
                          ImGuiTableFlags_RowBg |
                              ImGuiTableFlags_SizingFixedFit)) {
      ImGui::TableSetupColumn("Strategy", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("PnL", ImGuiTableColumnFlags_WidthFixed, 60.0f);
      ImGui::TableSetupColumn("Status", ImGuiTableColumnFlags_WidthFixed,
                              60.0f);
      ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
      ImGui::TableSetColumnIndex(0);
      ImGui::Text("STRATEGY");
      ImGui::TableSetColumnIndex(1);
      ImGui::Text("PnL");
      ImGui::TableSetColumnIndex(2);
      ImGui::Text("STATUS");

      for (auto &s : strategies_) {
        ImGui::TableNextRow(ImGuiTableRowFlags_None, 18.0f);
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("%s", s.name.c_str());

        ImGui::TableSetColumnIndex(1);
        ImColor pnl_color =
            s.pnl >= 0 ? ImColor(0.0f, 1.0f, 0.6f) : ImColor(1.0f, 0.2f, 0.3f);
        ImGui::TextColored(pnl_color, "$%.1f", s.pnl);

        ImGui::TableSetColumnIndex(2);
        if (s.active)
          ImGui::TextColored(ImVec4(0, 1, 0.5f, 1), "RUNNING");
        else
          ImGui::TextDisabled("STOPPED");

        ImGui::SameLine();
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 0));
        if (ImGui::SmallButton(s.active ? "Stop" : "Start")) {
          s.active = !s.active;
        }
        ImGui::PopStyleVar();
      }
      ImGui::EndTable();
    }
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

  if (ImGui::Begin("Risk Manager", &visible_)) {
    ImGui::TextColored(theme_.accent_primary, "PORTFOLIO HEALTH");

    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, theme_.price_up);
    ImGui::ProgressBar(0.72f, ImVec2(-1, 14), "OK");
    ImGui::PopStyleColor();

    ImGui::Spacing();

    if (ImGui::BeginTable("RiskStats", 2, ImGuiTableFlags_NoBordersInBody)) {
      ImGui::TableNextRow(ImGuiTableRowFlags_None, 18.0f);
      ImGui::TableSetColumnIndex(0);
      ImGui::TextDisabled("Daily Loss");
      ImGui::TableSetColumnIndex(1);
      ImGui::TextColored(theme_.price_up, "$1,250.00 / $5,000");

      ImGui::TableNextRow(ImGuiTableRowFlags_None, 18.0f);
      ImGui::TableSetColumnIndex(0);
      ImGui::TextDisabled("Max DD");
      ImGui::TableSetColumnIndex(1);
      ImGui::TextColored(theme_.price_down, "4.2%% / 10.0%%");

      ImGui::EndTable();
    }

    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Button, theme_.price_down);
    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32_WHITE);
    if (ImGui::Button("EMERGENCY STOP (DE-LEVERAGE)", ImVec2(-1, 32))) {
      // Panic logic
    }
    ImGui::PopStyleColor(2);
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

  if (ImGui::Begin("Trading Terminal", &visible_,
                   ImGuiWindowFlags_NoScrollbar)) {
    // Advanced Order Selector
    const char *types[] = {"Limit",         "Market",  "Stop-Limit",
                           "Trailing Stop", "Iceberg", "TWAP"};
    ImGui::PushItemWidth(-1);
    if (ImGui::BeginCombo("##OrderType", order_type_.c_str())) {
      for (int n = 0; n < IM_ARRAYSIZE(types); n++) {
        if (ImGui::Selectable(types[n], order_type_ == types[n]))
          order_type_ = types[n];
      }
      ImGui::EndCombo();
    }
    ImGui::PopItemWidth();
    ImGui::Spacing();

    // Context-sensitive inputs - Tighter spacing
    auto input_labeled = [&](const char *label, float *val, float step,
                             float step_fast, const char *fmt) {
      ImGui::TextDisabled("%s", label);
      ImGui::SetNextItemWidth(-1.0f);
      ImGui::InputFloat((std::string("##") + label).c_str(), val, step,
                        step_fast, fmt);
    };

    input_labeled("QUANTITY", &quantity_, 0.1f, 1.0f, "%.2f");

    if (order_type_ != "Market") {
      input_labeled("LIMIT PRICE", &price_, 0.5f, 10.0f, "%.2f");
    }

    if (order_type_ == "Stop-Limit") {
      input_labeled("STOP PRICE", &stop_price_, 0.5f, 10.0f, "%.2f");
    } else if (order_type_ == "Trailing Stop") {
      ImGui::TextDisabled("TRAILING PERCENT (%%)");
      ImGui::SetNextItemWidth(-1.0f);
      ImGui::SliderFloat("##Trail", &trailing_pct_, 0.1f, 5.0f, "%.2f%%");
    } else if (order_type_ == "Iceberg") {
      input_labeled("DISPLAY QTY", &iceberg_display_qty_, 0.01f, 0.1f, "%.4f");
    } else if (order_type_ == "TWAP") {
      ImGui::TextDisabled("DURATION (MINS)");
      ImGui::SetNextItemWidth(-1.0f);
      ImGui::SliderInt("##TWAPDur", &twap_duration_mins_, 1, 480);
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    float btn_w =
        (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) *
        0.5f;

    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0.0f);

    ImGui::PushStyleColor(ImGuiCol_Button, theme_.price_up);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                          ImVec4(theme_.price_up.x, theme_.price_up.y,
                                 theme_.price_up.z,
                                 1.0f)); // Could be brightened
    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32_WHITE);
    if (ImGui::Button("BUY LMT", ImVec2(btn_w, 28))) { /* EXECUTE BUY */
    }
    ImGui::PopStyleColor(3);

    ImGui::SameLine();

    ImGui::PushStyleColor(ImGuiCol_Button,
                          ImVec4(theme_.price_down.x, theme_.price_down.y,
                                 theme_.price_down.z, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                          ImVec4(theme_.price_down.x, theme_.price_down.y,
                                 theme_.price_down.z,
                                 1.0f)); // Could be brightened
    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32_WHITE);
    if (ImGui::Button("SELL / SHORT", ImVec2(btn_w, 36))) { /* EXECUTE SELL */
    }
    ImGui::PopStyleColor(3);

    ImGui::Spacing();

    // Low priority controls
    ImGui::PushStyleColor(ImGuiCol_Button, theme_.background_secondary);
    if (ImGui::Button("CANCEL ALL", ImVec2(btn_w, 24))) { /* CANCEL ALL */
    }
    ImGui::SameLine();
    if (ImGui::Button("FLATTEN", ImVec2(btn_w, 24))) { /* FLATTEN */
    }
    ImGui::PopStyleColor();

    ImGui::PopStyleVar();
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
  if (trade.symbol != target_symbol_)
    return;

  std::lock_guard lock(data_mutex_);
  TapeEntry entry;
  entry.timestamp = trade.timestamp;
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

void TapeComponent::update(float) {}

void TapeComponent::clear_data() {
  std::lock_guard lock(data_mutex_);
  entries_.clear();
  cumulative_delta_ = 0.0f;
  mark_dirty();
}

void TapeComponent::render_gui() {
  ImGui::SetNextWindowPos(ImVec2(position_.x, position_.y), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(size_.x, size_.y), ImGuiCond_Always);

  if (ImGui::Begin("Time & Sales", &visible_)) {
    if (ImGui::BeginTable("TapeTable", 3,
                          ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg |
                              ImGuiTableFlags_NoBordersInBody)) {
      ImGui::TableSetupColumn("Time", ImGuiTableColumnFlags_WidthFixed, 60.0f);
      ImGui::TableSetupColumn("Price", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, 60.0f);

      ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
      ImGui::TableSetColumnIndex(0);
      ImGui::Text("TIME");
      ImGui::TableSetColumnIndex(1);
      ImGui::Text("PRICE");
      ImGui::TableSetColumnIndex(2);
      ImGui::Text("SIZE");

      std::lock_guard lock(data_mutex_);
      for (const auto &e : entries_) {
        ImGui::TableNextRow(ImGuiTableRowFlags_None, 16.0f);

        // Display time
        char time_str[32];
        snprintf(time_str, sizeof(time_str), "%lu", e.timestamp);

        ImGui::TableSetColumnIndex(0);
        ImGui::TextDisabled("%s", time_str);

        ImGui::TableSetColumnIndex(1);
        ImVec4 color = e.is_buy
                           ? ImVec4(theme_.price_up.x, theme_.price_up.y,
                                    theme_.price_up.z, 1.0f)
                           : ImVec4(theme_.price_down.x, theme_.price_down.y,
                                    theme_.price_down.z, 1.0f);
        if (e.is_whale_trade)
          ImGui::TableSetBgColor(
              ImGuiTableBgTarget_RowBg0,
              ImGui::GetColorU32(ImVec4(color.x, color.y, color.z, 0.2f)));

        ImGui::TextColored(color, "%.2f %s", e.price,
                           e.is_whale_trade ? "!!!" : "");

        ImGui::TableSetColumnIndex(2);
        ImGui::Text("%.4f", e.size);
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

void OrderManagementComponent::update(float) {}

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

  if (ImGui::Begin("Positions & P&L", &visible_)) {
    ImGui::Columns(3, "AccountHeader", false);
    ImGui::TextDisabled("TOTAL EQUITY");
    ImGui::TextColored(ImVec4(theme_.accent_primary.x, theme_.accent_primary.y,
                              theme_.accent_primary.z, 1),
                       "$%.2f", total_equity_);
    ImGui::NextColumn();
    ImGui::TextDisabled("AVAIL. MARGIN");
    ImGui::Text("$%.2f", available_balance_);
    ImGui::NextColumn();
    ImGui::TextColored(ImVec4(0, 1, 0.5f, 1), "UNREALIZED P&L");
    ImGui::TextColored(ImVec4(0, 1, 0.5f, 1), "+$550.00");
    ImGui::Columns(1);

    ImGui::Separator();

    if (ImGui::BeginTable("PositionTable", 5,
                          ImGuiTableFlags_RowBg |
                              ImGuiTableFlags_NoBordersInBody)) {
      ImGui::TableSetupColumn("Symbol", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, 60.0f);
      ImGui::TableSetupColumn("Entry", ImGuiTableColumnFlags_WidthFixed, 70.0f);
      ImGui::TableSetupColumn("PnL", ImGuiTableColumnFlags_WidthFixed, 70.0f);
      ImGui::TableSetupColumn("PnL %", ImGuiTableColumnFlags_WidthFixed, 60.0f);

      ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
      ImGui::TableSetColumnIndex(0);
      ImGui::Text("SYMBOL");
      ImGui::TableSetColumnIndex(1);
      ImGui::Text("SIZE");
      ImGui::TableSetColumnIndex(2);
      ImGui::Text("ENTRY");
      ImGui::TableSetColumnIndex(3);
      ImGui::Text("PnL");
      ImGui::TableSetColumnIndex(4);
      ImGui::Text("PnL %%");

      for (const auto &p : positions_) {
        ImGui::TableNextRow(ImGuiTableRowFlags_None, 18.0f);
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("%s", p.symbol.c_str());
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%.4f", p.quantity);
        ImGui::TableSetColumnIndex(2);
        ImGui::Text("%.2f", p.entry_price);

        ImGui::TableSetColumnIndex(3);
        ImGui::TextColored(p.pnl >= 0 ? ImVec4(0, 1, 0.5f, 1)
                                      : ImVec4(1, 0.2f, 0.3f, 1),
                           "$%.2f", p.pnl);

        ImGui::TableSetColumnIndex(4);
        ImGui::TextColored(p.pnl_percent >= 0 ? ImVec4(0, 1, 0.5f, 1)
                                              : ImVec4(1, 0.2f, 0.3f, 1),
                           "%.2f%%", p.pnl_percent);
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

void MarketOverviewPanel::update(float) {}

void MarketOverviewPanel::render_gui() {
  ImGui::SetNextWindowPos(ImVec2(position_.x, position_.y), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(size_.x, size_.y), ImGuiCond_Always);

  ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar |
                           ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                           ImGuiWindowFlags_NoScrollbar |
                           ImGuiWindowFlags_MenuBar;

  if (ImGui::Begin("Market Overview", &visible_, flags)) {
    // Branding
    ImGui::TextColored(ImVec4(theme_.accent_primary.x, theme_.accent_primary.y,
                              theme_.accent_primary.z, 1),
                       "BTQUANT | INSTITUTIONAL");
    ImGui::SameLine();
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine();

    // Render tickers in a row with TealStreet style
    for (size_t i = 0; i < tickers_.size(); ++i) {
      const auto &t = tickers_[i];
      ImGui::TextDisabled("%s", t.symbol.c_str());
      ImGui::SameLine();
      ImColor color =
          t.change_pct >= 0
              ? ImColor(theme_.price_up.x, theme_.price_up.y, theme_.price_up.z)
              : ImColor(theme_.price_down.x, theme_.price_down.y,
                        theme_.price_down.z);
      ImGui::TextColored(color, "%.2f", t.price);
      ImGui::SameLine();
      ImGui::TextDisabled("(%+.2f%%)", t.change_pct);

      if (i < tickers_.size() - 1) {
        ImGui::SameLine();
        ImGui::Text(" ");
        ImGui::SameLine();
      }
    }

    ImGui::SameLine(ImGui::GetWindowWidth() - 320);
    ImGui::TextDisabled("VOL $%.1fB", global_volume_ / 1e9f);
    ImGui::SameLine();
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine();
    ImGui::TextDisabled("LATENCY %.2fms", system_latency_ms_);
    ImGui::SameLine();
    ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0, 1, 0.5f, 1), "CONNECTED");
    ImGui::SameLine();
    ImGui::TextDisabled("CPU %02d%%", 5);
  }
  ImGui::End();
}
} // namespace BTQuant
