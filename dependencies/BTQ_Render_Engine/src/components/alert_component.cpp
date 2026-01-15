#include "vulkan_dashboard_advanced.hpp"
#include <algorithm>
#include <imgui.h>
#include <imgui_internal.h>

AlertComponent::AlertComponent(const glm::vec2 &position, const glm::vec2 &size,
                               AlertManager &manager)
    : UIComponent(position, size), manager_(manager) {}

AlertComponent::~AlertComponent() {}

void AlertComponent::update(float delta_time) {}

void AlertComponent::render_gui() {
  ImGui::SetNextWindowPos(ImVec2(position_.x, position_.y), ImGuiCond_Always);
  ImGui::SetNextWindowSize(ImVec2(size_.x, size_.y), ImGuiCond_Always);

  if (ImGui::Begin("Alert Center", &visible_)) {
    if (ImGui::BeginTabBar("AlertTabs")) {
      if (ImGui::BeginTabItem("Active Alerts")) {
        const auto &alerts = manager_.get_alerts();
        if (ImGui::BeginTable("AlertsTable", 5,
                              ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg |
                                  ImGuiTableFlags_BordersInnerV)) {
          ImGui::TableSetupColumn("Symbol", ImGuiTableColumnFlags_WidthFixed,
                                  80.0f);
          ImGui::TableSetupColumn("Condition",
                                  ImGuiTableColumnFlags_WidthStretch);
          ImGui::TableSetupColumn("Target", ImGuiTableColumnFlags_WidthFixed,
                                  80.0f);
          ImGui::TableSetupColumn("Status", ImGuiTableColumnFlags_WidthFixed,
                                  70.0f);
          ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed,
                                  60.0f);
          ImGui::TableHeadersRow();

          for (size_t i = 0; i < alerts.size(); ++i) {
            const auto &alert = alerts[i];
            ImGui::TableNextRow();

            ImGui::TableSetColumnIndex(0);
            ImGui::Text("%s", alert.symbol.c_str());

            ImGui::TableSetColumnIndex(1);
            const char *cond_str = "Unknown";
            if (alert.condition == AlertCondition::PRICE_ABOVE)
              cond_str = "Price >=";
            else if (alert.condition == AlertCondition::PRICE_BELOW)
              cond_str = "Price <=";
            else if (alert.condition == AlertCondition::VOLUME_ABOVE)
              cond_str = "Vol >=";
            ImGui::Text("%s", cond_str);

            ImGui::TableSetColumnIndex(2);
            ImGui::Text("%.4f", alert.target_value);

            ImGui::TableSetColumnIndex(3);
            if (alert.is_triggered) {
              ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "TRIGGERED");
            } else {
              ImGui::TextColored(ImVec4(0, 1, 0, 1), "Active");
            }

            ImGui::TableSetColumnIndex(4);
            if (ImGui::SmallButton("Delete")) {
              manager_.remove_alert(i);
            }
          }
          ImGui::EndTable();
        }
        ImGui::EndTabItem();
      }

      if (ImGui::BeginTabItem("New Alert")) {
        ImGui::InputText("Symbol", symbol_buffer_, sizeof(symbol_buffer_));

        const char *conditions[] = {"Price Above", "Price Below",
                                    "Volume Above"};
        ImGui::Combo("Condition", &selected_condition_, conditions,
                     IM_ARRAYSIZE(conditions));

        ImGui::InputFloat("Target Value", &target_value_);

        if (ImGui::Button("CREATE ALERT", ImVec2(-FLT_MIN, 40))) {
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
