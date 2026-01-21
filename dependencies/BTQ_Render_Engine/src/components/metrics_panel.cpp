#include "../../include/components/metrics_panel.hpp"
#include "imgui.h"
#include <cmath>
#include <iomanip>
#include <sstream>

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
  Metric portfolio_value;
  portfolio_value.name = "Portfolio Value";
  portfolio_value.value =
      "$" + std::to_string(static_cast<int>(summary.total_value));
  portfolio_value.unit = "";
  portfolio_value.color = get_metric_color(summary.total_value > 0);
  portfolio_value.is_percentage = false;
  portfolio_value.change = 0.0f;
  metrics_.push_back(portfolio_value);

  // Unrealized P&L
  Metric unrealized_pnl;
  unrealized_pnl.name = "Unrealized P&L";
  unrealized_pnl.value =
      std::string(summary.total_unrealized_pnl >= 0 ? "+$" : "-$") +
      std::to_string(static_cast<int>(std::abs(summary.total_unrealized_pnl)));
  unrealized_pnl.unit = "";
  unrealized_pnl.color = get_metric_color(summary.total_unrealized_pnl >= 0);
  unrealized_pnl.is_percentage = false;
  unrealized_pnl.change = 0.0f;
  metrics_.push_back(unrealized_pnl);

  // Realized P&L
  Metric realized_pnl;
  realized_pnl.name = "Realized P&L";
  realized_pnl.value =
      std::string(summary.total_realized_pnl >= 0 ? "+$" : "-$") +
      std::to_string(static_cast<int>(std::abs(summary.total_realized_pnl)));
  realized_pnl.unit = "";
  realized_pnl.color = get_metric_color(summary.total_realized_pnl >= 0);
  realized_pnl.is_percentage = false;
  realized_pnl.change = 0.0f;
  metrics_.push_back(realized_pnl);

  // Position Count
  Metric position_count;
  position_count.name = "Active Positions";
  position_count.value = std::to_string(summary.position_count);
  position_count.unit = "";
  position_count.color = ImVec4(0.7f, 0.7f, 1.0f, 1.0f);
  position_count.is_percentage = false;
  position_count.change = 0.0f;
  metrics_.push_back(position_count);

  // Sharpe Ratio (from portfolio summary)
  std::stringstream ss;
  ss << std::fixed << std::setprecision(2) << summary.sharpe_ratio;
  Metric sharpe;
  sharpe.name = "Sharpe Ratio";
  sharpe.value = ss.str();
  sharpe.unit = "";
  sharpe.color = get_metric_color(summary.sharpe_ratio > 1.0);
  sharpe.is_percentage = false;
  sharpe.change = 0.0f;
  metrics_.push_back(sharpe);

  // Portfolio Beta
  ss.str("");
  ss << std::fixed << std::setprecision(2) << summary.portfolio_beta;
  Metric beta;
  beta.name = "Portfolio Beta";
  beta.value = ss.str();
  beta.unit = "";
  beta.color = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
  beta.is_percentage = false;
  beta.change = 0.0f;
  metrics_.push_back(beta);

  // VaR (95%)
  ss.str("");
  ss << std::fixed << std::setprecision(0) << summary.portfolio_var;
  Metric var;
  var.name = "VaR (95%)";
  var.value = "$" + ss.str();
  var.unit = "";
  var.color = get_metric_color(summary.portfolio_var < 5000, false);
  var.is_percentage = false;
  var.change = 0.0f;
  metrics_.push_back(var);

  // Buying Power
  ss.str("");
  ss << std::fixed << std::setprecision(0) << summary.buying_power;
  Metric buying_power;
  buying_power.name = "Buying Power";
  buying_power.value = "$" + ss.str();
  buying_power.unit = "";
  buying_power.color = ImVec4(0.5f, 0.8f, 1.0f, 1.0f);
  buying_power.is_percentage = false;
  buying_power.change = 0.0f;
  metrics_.push_back(buying_power);
}

void MetricsPanel::render_metric_grid() {
  ImVec2 content_size = ImGui::GetContentRegionAvail();
  float card_width = (content_size.x - 16) / 2; // 2 columns with spacing

  int column = 0;
  for (const auto &metric : metrics_) {
    if (column > 0)
      ImGui::SameLine();

    render_metric_card(metric, card_width);

    column = (column + 1) % 2;
    if (column == 0) {
      ImGui::Dummy(ImVec2(0, 8)); // Row spacing
    }
  }
}

void MetricsPanel::render_metric_card(const Metric &metric, float width) {
  ImGui::BeginGroup();
  ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 8.0f);
  ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.12f, 0.12f, 0.12f, 1.0f));
  ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));

  ImGui::BeginChild(("metric_" + metric.name).c_str(), ImVec2(width, 70), true,
                    ImGuiWindowFlags_NoScrollbar |
                        ImGuiWindowFlags_NoScrollWithMouse);

  // Metric name
  ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "%s", metric.name.c_str());

  // Metric value
  ImGui::TextColored(metric.color, "%s%s", metric.value.c_str(),
                     metric.unit.c_str());

  // Change indicator (if applicable)
  if (metric.change != 0.0f) {
    ImVec4 change_color = metric.change >= 0 ? ImVec4(0.0f, 1.0f, 0.0f, 1.0f)
                                             : ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
    ImGui::TextColored(change_color, "%+.1f%%", metric.change);
  }

  ImGui::EndChild();
  ImGui::PopStyleColor(2);
  ImGui::PopStyleVar();

  ImGui::EndGroup();
}

ImVec4 MetricsPanel::get_metric_color(float value, bool is_positive_good) {
  if (is_positive_good) {
    if (value > 0)
      return ImVec4(0.0f, 1.0f, 0.0f, 1.0f); // Green
    else if (value < 0)
      return ImVec4(1.0f, 0.0f, 0.0f, 1.0f); // Red
  } else {
    if (value < 0.5f)
      return ImVec4(0.0f, 1.0f, 0.0f, 1.0f); // Green for low values
    else
      return ImVec4(1.0f, 0.5f, 0.0f, 1.0f); // Orange for high values
  }
  return ImVec4(1.0f, 1.0f, 1.0f, 1.0f); // White for neutral
}

} // namespace BTQuant