#include "vulkan_dashboard_advanced.hpp"
#include <imgui.h>
#include <imgui_internal.h>

namespace BTQuant {

AlertComponent::AlertComponent(const glm::vec2 &position, const glm::vec2 &size,
                               AlertManager &manager)
    : UIComponent(position, size), manager_(manager) {}

AlertComponent::~AlertComponent() {}

void AlertComponent::update(float) {}

void AlertComponent::initialize_vulkan_resources(VulkanCore *) {}

void AlertComponent::clear_data() {
  std::lock_guard lock(data_mutex_);
  alerts_.clear();
}

void AlertComponent::render_gui() {
  ImGui::SetNextWindowPos(ImVec2(position_.x, position_.y), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(size_.x, size_.y), ImGuiCond_Always);

  if (ImGui::Begin("Alert Center", &visible_)) {
    if (ImGui::BeginTabBar("AlertTabs")) {
      if (ImGui::BeginTabItem("Active Alerts")) {
        auto alerts = manager_.get_alerts(); // Returns a thread-safe copy
        if (ImGui::BeginTable("AlertsTable", 5,
                              ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg |
                                  ImGuiTableFlags_NoBordersInBody)) {
          ImGui::TableSetupColumn("Symbol", ImGuiTableColumnFlags_WidthFixed,
                                  70.0f);
          ImGui::TableSetupColumn("Condition",
                                  ImGuiTableColumnFlags_WidthStretch);
          ImGui::TableSetupColumn("Target", ImGuiTableColumnFlags_WidthFixed,
                                  70.0f);
          ImGui::TableSetupColumn("Status", ImGuiTableColumnFlags_WidthFixed,
                                  65.0f);
          ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed,
                                  50.0f);

          ImGui::TableNextRow(ImGuiTableRowFlags_Headers, 18.0f);
          ImGui::TableSetColumnIndex(0);
          ImGui::Text("SYMBOL");
          ImGui::TableSetColumnIndex(1);
          ImGui::Text("CONDITION");
          ImGui::TableSetColumnIndex(2);
          ImGui::Text("TARGET");
          ImGui::TableSetColumnIndex(3);
          ImGui::Text("STATUS");
          ImGui::TableSetColumnIndex(4);
          ImGui::Text("CMD");

          std::lock_guard lock(data_mutex_);
          for (size_t i = 0; i < alerts.size(); ++i) {
            const auto &alert = alerts[i];
            ImGui::TableNextRow(ImGuiTableRowFlags_None, 16.0f);

            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%s", alert.symbol.c_str());

            ImGui::TableSetColumnIndex(1);
            const char *cond_str = "???";
            if (alert.condition == AlertCondition::PRICE_ABOVE)
              cond_str = "Price >=";
            else if (alert.condition == AlertCondition::PRICE_BELOW)
              cond_str = "Price <=";
            else if (alert.condition == AlertCondition::VOLUME_ABOVE)
              cond_str = "Vol >=";
            ImGui::TextDisabled("%s", cond_str);

            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%.4f", alert.target_value);

            ImGui::TableSetColumnIndex(3);
            if (alert.is_triggered) {
              ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "HIT");
            } else {
              ImGui::TextColored(ImVec4(0, 1, 0.5f, 1), "Watching");
            }

            ImGui::TableSetColumnIndex(4);
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 0));
            if (ImGui::SmallButton(("X##" + std::to_string(i)).c_str())) {
              manager_.remove_alert(i);
            }
            ImGui::PopStyleVar();
          }
          ImGui::EndTable();
        }
        ImGui::EndTabItem();
      }

      if (ImGui::BeginTabItem("New")) {
        ImGui::InputText("Symbol", symbol_buffer_, sizeof(symbol_buffer_));

        const char *conditions[] = {"Price >=", "Price <=", "Volume >="};
        ImGui::Combo("Check", &selected_condition_, conditions,
                     IM_ARRAYSIZE(conditions));
        ImGui::InputFloat("Value", &target_value_);

        ImGui::Spacing();
        if (ImGui::Button("SET ALERT", ImVec2(-FLT_MIN, 28))) {
          AlertRule rule;
          rule.symbol = symbol_buffer_;
          rule.target_value = target_value_;
          if (selected_condition_ == 0)
            rule.condition = AlertCondition::PRICE_ABOVE;
          else if (selected_condition_ == 1)
            rule.condition = AlertCondition::PRICE_BELOW;
          else if (selected_condition_ == 2)
            rule.condition = AlertCondition::VOLUME_ABOVE;

          manager_.add_alert(rule);
          symbol_buffer_[0] = '\0';
        }
        ImGui::EndTabItem();
      }
      ImGui::EndTabBar();
    }
  }
  ImGui::End();
}

void AlertManager::add_alert(const AlertRule &rule) {
  (void)rule;
  // Implementation
}

void AlertManager::check_alerts(const std::string &symbol, double price) {
  (void)symbol;
  (void)price;
  // Implementation
}

std::vector<AlertRule> AlertManager::get_alerts() {
  return {}; // Placeholder
}

void AlertManager::remove_alert(size_t index) {
  (void)index;
  // Implementation
}

} // namespace BTQuant
