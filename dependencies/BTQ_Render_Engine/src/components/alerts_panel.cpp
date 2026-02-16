#include "../../include/components/alerts_panel.hpp"

#include <ctime>
#include <format>

#include "imgui.h"

namespace BTQuant {

AlertsPanel::AlertsPanel(const PanelConfig& config) : PanelBase(config) {
  // Add some dummy data for testing
  rules_.push_back({.id = "rule_1",
                    .name = "BTC Breakout",
                    .expression = "price > 98000",
                    .target_symbol = "BTCUSDT",
                    .status = AlertStatus::ACTIVE,
                    .actions = {"log", "sound"},
                    .last_triggered = std::chrono::system_clock::now()});

  rules_.push_back({.id = "rule_2",
                    .name = "ETH Dip",
                    .expression = "price < 2800",
                    .target_symbol = "ETHUSDT",
                    .status = AlertStatus::DISABLED,
                    .actions = {"log"},
                    .last_triggered = std::chrono::system_clock::now()});

  // Dummy logs
  auto now = std::chrono::system_clock::now();
  logs_.push_back(
      {now - std::chrono::minutes(5), "BTC Breakout", "BTCUSDT", 98005.50, "Price crossed 98000"});
  logs_.push_back(
      {now - std::chrono::minutes(12), "High Volume", "SOLUSDT", 145.20, "Volume spike detected"});
}

void AlertsPanel::update(float /*dt*/) {
  // In a real implementation, we would evaluate rules here against market data
  // For now, it's just a UI shell
}

void AlertsPanel::render_content() {
  begin_panel_window();

  // Top Bar
  if (ImGui::Button("+ New Rule")) {
    show_create_modal_ = true;
  }
  ImGui::SameLine();
  if (ImGui::Button("Clear Logs")) {
    logs_.clear();
  }
  ImGui::SameLine();
  if (ImGui::Button("Clear Rejections")) {
    rejections_.clear();
  }

  ImGui::Separator();

  // Split view: Rules (Top), Rejections (Middle), and Logs (Bottom)
  // Using Child windows for scrolling

  ImGui::TextDisabled("Active Rules");
  ImGui::BeginChild("RulesList", ImVec2(0, 150), true);
  render_rules_table();
  ImGui::EndChild();

  ImGui::Separator();

  // Rejections Panel - Display risk invariant failures in ASK_RED
  ImGui::TextDisabled("Risk Rejections");
  ImGui::BeginChild("RejectionsList", ImVec2(0, 150), true);
  render_rejections();
  ImGui::EndChild();

  ImGui::Separator();

  ImGui::TextDisabled("Recent Alerts");
  ImGui::BeginChild("LogsList", ImVec2(0, 0), true);  // Remaining height
  render_alert_logs();
  ImGui::EndChild();

  render_create_rule_modal();

  end_panel_window();
}

void AlertsPanel::render_rules_table() {
  if (ImGui::BeginTable(
          "RulesTable", 5,
          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable)) {
    ImGui::TableSetupColumn("Name");
    ImGui::TableSetupColumn("Expression");
    ImGui::TableSetupColumn("Target");
    ImGui::TableSetupColumn("Status");
    ImGui::TableSetupColumn("Actions");
    ImGui::TableHeadersRow();

    for (size_t i = 0; i < rules_.size(); ++i) {
      auto& rule = rules_[i];
      ImGui::TableNextRow();
      ImGui::PushID(static_cast<int>(i));

      ImGui::TableSetColumnIndex(0);
      ImGui::TextUnformatted(rule.name.c_str());

      ImGui::TableSetColumnIndex(1);
      ImGui::TextUnformatted(rule.expression.c_str());

      ImGui::TableSetColumnIndex(2);
      ImGui::TextColored(ImVec4(1, 0.8f, 0, 1), "%s", rule.target_symbol.c_str());

      ImGui::TableSetColumnIndex(3);
      const char* status_str = "Unknown";
      ImVec4 status_col = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);

      switch (rule.status) {
        case AlertStatus::ACTIVE:
          status_str = "Active";
          status_col = ImVec4(0.2f, 0.8f, 0.2f, 1.0f);
          break;
        case AlertStatus::TRIGGERED:
          status_str = "Triggered";
          status_col = ImVec4(0.8f, 0.2f, 0.2f, 1.0f);
          break;
        case AlertStatus::DISABLED:
          status_str = "Disabled";
          status_col = ImVec4(0.4f, 0.4f, 0.4f, 1.0f);
          break;
        case AlertStatus::COOLDOWN:
          status_str = "Cooldown";
          status_col = ImVec4(0.8f, 0.8f, 0.2f, 1.0f);
          break;
      }
      ImGui::TextColored(status_col, "%s", status_str);

      ImGui::TableSetColumnIndex(4);
      if (ImGui::Button("Edit")) {
        // TODO: Edit logic
      }
      ImGui::SameLine();
      if (ImGui::Button("Del")) {
        // TODO: Delete logic (would need iterator handling)
      }
      ImGui::PopID();
    }
    ImGui::EndTable();
  }
}

void AlertsPanel::render_alert_logs() {
  if (ImGui::BeginTable(
          "LogsTable", 4,
          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable)) {
    ImGui::TableSetupColumn("Time");
    ImGui::TableSetupColumn("Rule");
    ImGui::TableSetupColumn("Symbol");
    ImGui::TableSetupColumn("Message");
    ImGui::TableHeadersRow();

    for (const auto& log : logs_) {
      ImGui::TableNextRow();

      ImGui::TableSetColumnIndex(0);
      std::time_t t = std::chrono::system_clock::to_time_t(log.time);
      // using C++26 std::format would be ideal, falling back to strftime for
      // safety if compiler is older
      char time_buf[64];
      std::strftime(time_buf, sizeof(time_buf), "%H:%M:%S", std::localtime(&t));
      ImGui::Text("%s", time_buf);

      ImGui::TableSetColumnIndex(1);
      ImGui::Text("%s", log.rule_name.c_str());

      ImGui::TableSetColumnIndex(2);
      ImGui::TextColored(ImVec4(1, 0.8f, 0, 1), "%s", log.symbol.c_str());

      ImGui::TableSetColumnIndex(3);
      ImGui::Text("%s", log.message.c_str());
    }
    ImGui::EndTable();
  }
}

void AlertsPanel::render_rejections() {
  if (rejections_.empty()) {
    ImGui::TextDisabled("No risk rejections");
    return;
  }

  if (ImGui::BeginTable(
          "RejectionsTable", 5,
          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable)) {
    ImGui::TableSetupColumn("Time");
    ImGui::TableSetupColumn("Symbol");
    ImGui::TableSetupColumn("Side");
    ImGui::TableSetupColumn("Size");
    ImGui::TableSetupColumn("Reason");
    ImGui::TableHeadersRow();

    // ASK_RED color for risk invariant failures
    const ImVec4 ask_red_color(1.0f, 0.3f, 0.3f, 1.0f);

    for (const auto& rej : rejections_) {
      ImGui::TableNextRow();

      ImGui::TableSetColumnIndex(0);
      std::time_t t = std::chrono::system_clock::to_time_t(rej.timestamp);
      char time_buf[64];
      std::strftime(time_buf, sizeof(time_buf), "%H:%M:%S", std::localtime(&t));
      ImGui::Text("%s", time_buf);

      ImGui::TableSetColumnIndex(1);
      ImGui::TextColored(ask_red_color, "%s", rej.symbol.c_str());

      ImGui::TableSetColumnIndex(2);
      ImGui::TextColored(ask_red_color, "%s", rej.is_buy ? "BUY" : "SELL");

      ImGui::TableSetColumnIndex(3);
      ImGui::TextColored(ask_red_color, "%.4f @ %.2f", rej.quantity, rej.price);

      ImGui::TableSetColumnIndex(4);
      ImGui::TextColored(ask_red_color, "%s", rej.reason.c_str());
    }
    ImGui::EndTable();
  }
}

void AlertsPanel::render_create_rule_modal() {
  if (show_create_modal_) {
    ImGui::OpenPopup("Create Alert Rule");
  }

  if (ImGui::BeginPopupModal("Create Alert Rule", &show_create_modal_,
                             ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::InputText("Name", new_rule_name_, sizeof(new_rule_name_));
    ImGui::InputText("Expression", new_rule_expr_, sizeof(new_rule_expr_));
    ImGui::InputText("Symbol", new_rule_symbol_, sizeof(new_rule_symbol_));

    // Simple DSL help
    ImGui::TextDisabled("Examples: price > 50000, rsi < 30");

    ImGui::Separator();

    if (ImGui::Button("Create", ImVec2(120, 0))) {
      rules_.push_back({.id = std::format("rule_{}", std::rand()),  // Simple ID generation
                        .name = new_rule_name_,
                        .expression = new_rule_expr_,
                        .target_symbol = new_rule_symbol_,
                        .status = AlertStatus::ACTIVE,
                        .actions = {"log"},
                        .last_triggered = std::chrono::system_clock::now()});
      show_create_modal_ = false;
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(120, 0))) {
      show_create_modal_ = false;
      ImGui::CloseCurrentPopup();
    }

    ImGui::EndPopup();
  }
}

}  // namespace BTQuant
