#include "../../include/components/metrics_panel.hpp"
#include "imgui.h"
#include <cmath>

namespace BTQuant {

MetricsPanel::MetricsPanel(const PanelConfig &config,
                           std::shared_ptr<PositionManager> position_manager,
                           std::shared_ptr<RiskAssessment> risk_assessment)
    : PanelBase(config), position_manager_(position_manager),
      risk_assessment_(risk_assessment) {
  update_metrics();
}

void MetricsPanel::update(float dt) {
  update_timer_ += dt;
  if (update_timer_ >= UPDATE_INTERVAL) {
    update_metrics();
    update_timer_ = 0.0f;
  }
}

void MetricsPanel::render() {
  begin_panel_window();

  if (!is_visible()) {
    end_panel_window();
    return;
  }

  render_metric_grid();

  end_panel_window();
}

void MetricsPanel::update_metrics() {
  metrics_.clear();

  // Get portfolio summary
  auto summary = position_manager_->get_portfolio_summary();

  // Portfolio Value
  Metric m1;
  m1.name = "Portfolio Value";
  m1.value =
      std::string("$") + std::to_string(static_cast<int>(summary.total_value));
  m1.unit = "";
  m1.color = get_metric_color(summary.total_value > 0);
  m1.has_progress = false;
  m1.progress = 0.0f;
  metrics_.push_back(m1);

  // Unrealized P&L
  Metric m2;
  m2.name = "Unrealized P&L";
  m2.value =
      std::string(summary.total_unrealized_pnl >= 0 ? "+$" : "-$") +
      std::to_string(static_cast<int>(std::abs(summary.total_unrealized_pnl)));
  m2.unit = "";
  m2.color = get_metric_color(summary.total_unrealized_pnl >= 0);
  m2.has_progress = false;
  m2.progress = 0.0f;
  metrics_.push_back(m2);

  // Realized P&L
  Metric m3;
  m3.name = "Realized P&L";
  m3.value =
      std::string(summary.total_realized_pnl >= 0 ? "+$" : "-$") +
      std::to_string(static_cast<int>(std::abs(summary.total_realized_pnl)));
  m3.unit = "";
  m3.color = get_metric_color(summary.total_realized_pnl >= 0);
  m3.has_progress = false;
  m3.progress = 0.0f;
  metrics_.push_back(m3);

  // Position Count
  Metric m4;
  m4.name = "Active Positions";
  m4.value = std::to_string(summary.position_count);
  m4.unit = "";
  m4.color = ImVec4(0.7f, 0.7f, 1.0f, 1.0f);
  m4.has_progress = false;
  m4.progress = 0.0f;
  metrics_.push_back(m4);

  // Sharpe Ratio placeholder
  Metric m5;
  m5.name = "Sharpe Ratio";
  m5.value = "-0.10";
  m5.unit = "";
  m5.color = ImVec4(1.0f, 0.4f, 0.4f, 1.0f);
  m5.has_progress = false;
  m5.progress = 0.0f;
  metrics_.push_back(m5);

  // Portfolio Beta placeholder
  Metric m6;
  m6.name = "Portfolio Beta";
  m6.value = "N/A";
  m6.unit = "";
  m6.color = ImVec4(0.7f, 0.7f, 0.7f, 1.0f);
  m6.has_progress = false;
  m6.progress = 0.0f;
  metrics_.push_back(m6);

  // VaR placeholder
  Metric m7;
  m7.name = "VaR (95%)";
  m7.value = "$0";
  m7.unit = "";
  m7.color = ImVec4(0.7f, 0.7f, 0.7f, 1.0f);
  m7.has_progress = false;
  m7.progress = 0.0f;
  metrics_.push_back(m7);

  // Buying Power placeholder
  Metric m8;
  m8.name = "Buying Power";
  m8.value = "$0.000K";
  m8.unit = "";
  m8.color = ImVec4(0.7f, 0.7f, 0.7f, 1.0f);
  m8.has_progress = false;
  m8.progress = 0.0f;
  metrics_.push_back(m8);
}

void MetricsPanel::render_metric_grid() {
  ImVec2 content_size = ImGui::GetContentRegionAvail();
  float card_width = (content_size.x - 16) / 2; // 2 columns with spacing
  float card_height = 80.0f;

  int cols = 2;
  int current_col = 0;

  for (const auto &metric : metrics_) {
    if (current_col > 0) {
      ImGui::SameLine();
    }

    render_metric_card(metric, card_width, card_height);

    current_col++;
    if (current_col >= cols) {
      current_col = 0;
    }
  }
}

void MetricsPanel::render_metric_card(const Metric &metric, float width,
                                      float height) {
  ImGui::BeginGroup();

  // Card background
  ImVec2 pos = ImGui::GetCursorScreenPos();
  ImGui::GetWindowDrawList()->AddRectFilled(
      pos, ImVec2(pos.x + width, pos.y + height),
      ImGui::GetColorU32(ImVec4(0.12f, 0.12f, 0.12f, 1.0f)), 8.0f);

  // Content padding
  ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 8);
  ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 8);

  // Metric name
  ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "%s", metric.name.c_str());

  // Metric value
  ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 8);
  ImGui::TextColored(metric.color, "%s%s", metric.value.c_str(),
                     metric.unit.c_str());

  // Progress bar (if applicable)
  if (metric.has_progress) {
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 8);
    ImGui::PushStyleColor(
        ImGuiCol_PlotHistogram,
        ImVec4(metric.color.x, metric.color.y, metric.color.z, 0.8f));
    ImGui::ProgressBar(metric.progress / 100.0f, ImVec2(width - 24, 4), "");
    ImGui::PopStyleColor();
  }

  // Dummy to reserve space
  ImGui::Dummy(ImVec2(width, height - ImGui::GetCursorPosY() + pos.y));

  ImGui::EndGroup();
}

ImVec4 MetricsPanel::get_metric_color(bool is_positive, bool) {
  if (is_positive) {
    return ImVec4(0.2f, 0.9f, 0.4f, 1.0f); // Green
  } else {
    return ImVec4(1.0f, 0.3f, 0.3f, 1.0f); // Red
  }
}

} // namespace BTQuant