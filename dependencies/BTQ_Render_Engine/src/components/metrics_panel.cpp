#include "../../include/components/metrics_panel.hpp"
#include "imgui.h"
#include <iomanip>
#include <sstream>

namespace BTQuant {

MetricsPanel::MetricsPanel(const PanelConfig& config,
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
  metrics_.push_back({
    "Portfolio Value",
    "$" + std::to_string(static_cast<int>(summary.total_value)),
    "",
    get_metric_color(summary.total_value > 0),
    false,
    0.0f
  });

  // Unrealized P&L
  metrics_.push_back({
    "Unrealized P&L",
    (summary.total_unrealized_pnl >= 0 ? "+" : "") +
    "$" + std::to_string(static_cast<int>(summary.total_unrealized_pnl)),
    "",
    get_metric_color(summary.total_unrealized_pnl >= 0),
    false,
    0.0f
  });

  // Realized P&L
  metrics_.push_back({
    "Realized P&L",
    (summary.total_realized_pnl >= 0 ? "+" : "") +
    "$" + std::to_string(static_cast<int>(summary.total_realized_pnl)),
    "",
    get_metric_color(summary.total_realized_pnl >= 0),
    false,
    0.0f
  });

  // Position Count
  metrics_.push_back({
    "Active Positions",
    std::to_string(summary.position_count),
    "",
    ImVec4(0.7f, 0.7f, 1.0f, 1.0f),
    false,
    0.0f
  });

  // Risk Metrics
  auto risk_metrics = risk_assessment_->get_risk_metrics();

  // Sharpe Ratio
  std::stringstream ss;
  ss << std::fixed << std::setprecision(2) << risk_metrics.sharpe_ratio;
  metrics_.push_back({
    "Sharpe Ratio",
    ss.str(),
    "",
    get_metric_color(risk_metrics.sharpe_ratio > 1.0),
    false,
    0.0f
  });

  // Max Drawdown
  ss.str("");
  ss << std::fixed << std::setprecision(1) << (risk_metrics.max_drawdown * 100.0);
  metrics_.push_back({
    "Max Drawdown",
    ss.str(),
    "%",
    get_metric_color(risk_metrics.max_drawdown < 0.1, false), // Lower is better
    true,
    risk_metrics.max_drawdown * 100.0f
  });

  // Volatility
  ss.str("");
  ss << std::fixed << std::setprecision(1) << (risk_metrics.volatility * 100.0);
  metrics_.push_back({
    "Volatility",
    ss.str(),
    "%",
    get_metric_color(risk_metrics.volatility < 0.2, false), // Lower is better
    true,
    risk_metrics.volatility * 100.0f
  });

  // Win Rate
  ss.str("");
  ss << std::fixed << std::setprecision(1) << (risk_metrics.win_rate * 100.0);
  metrics_.push_back({
    "Win Rate",
    ss.str(),
    "%",
    get_metric_color(risk_metrics.win_rate > 0.5),
    true,
    risk_metrics.win_rate * 100.0f
  });
}

void MetricsPanel::render_metric_grid() {
  ImVec2 content_size = ImGui::GetContentRegionAvail();
  float card_width = (content_size.x - 16) / 2; // 2 columns with spacing
  float card_height = 80.0f;

  int column = 0;
  for (const auto& metric : metrics_) {
    if (column > 0) ImGui::SameLine();

    render_metric_card(metric, card_width);

    column = (column + 1) % 2;
    if (column == 0) {
      ImGui::Dummy(ImVec2(0, 8)); // Row spacing
    }
  }
}

void MetricsPanel::render_metric_card(const Metric& metric, float width) {
  ImGui::BeginGroup();
  ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 8.0f);
  ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.12f, 0.12f, 0.12f, 1.0f));
  ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));

  ImGui::BeginChild(("metric_" + metric.name).c_str(),
                     ImVec2(width, 70), true,
                     ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

  // Metric name
  ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[1]); // Smaller font
  ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "%s", metric.name.c_str());
  ImGui::PopFont();

  // Metric value
  ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[0]); // Larger font
  ImGui::TextColored(metric.color, "%s%s", metric.value.c_str(), metric.unit.c_str());
  ImGui::PopFont();

  // Change indicator (if applicable)
  if (metric.change != 0.0f) {
    ImVec4 change_color = metric.change >= 0 ?
                         ImVec4(0.0f, 1.0f, 0.0f, 1.0f) :
                         ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
    ImGui::TextColored(change_color, "%+.1f%%", metric.change);
  }

  ImGui::EndChild();
  ImGui::PopStyleColor(2);
  ImGui::PopStyleVar();

  ImGui::EndGroup();
}

ImVec4 MetricsPanel::get_metric_color(float value, bool is_positive_good) {
  if (is_positive_good) {
    if (value > 0) return ImVec4(0.0f, 1.0f, 0.0f, 1.0f); // Green
    else if (value < 0) return ImVec4(1.0f, 0.0f, 0.0f, 1.0f); // Red
  } else {
    if (value < 0.5f) return ImVec4(0.0f, 1.0f, 0.0f, 1.0f); // Green for low values
    else return ImVec4(1.0f, 0.5f, 0.0f, 1.0f); // Orange for high values
  }
  return ImVec4(1.0f, 1.0f, 1.0f, 1.0f); // White for neutral
}

} // namespace BTQuant