#pragma once

#include <chrono>
#include <string>
#include <vector>

#include "panel_base.hpp"
#include "alert_common.hpp"

namespace BTQuant {

struct AlertRule {
  std::string id;
  std::string name;
  std::string expression;  // e.g. "price > 100000"
  std::string target_symbol;
  AlertStatus status = AlertStatus::ACTIVE;
  std::vector<std::string> actions;  // e.g. "log", "sound"

  // Runtime state
  std::chrono::system_clock::time_point last_triggered;
  double last_value = 0.0;
};

class AlertsPanel : public PanelBase {
 public:
  AlertsPanel(const PanelConfig& config);
  ~AlertsPanel() override = default;

  void update(float dt) override;
  void render_content() override;

 public:
  // Public method to add logs from external sources (e.g., watchlist alerts)
  void add_alert_log(const AlertLog& log) {
    logs_.push_back(log);
  }

 private:
  void render_rules_table();
  void render_alert_logs();
  void render_create_rule_modal();

  std::vector<AlertRule> rules_;
  std::vector<AlertLog> logs_;

  // UI State
  bool show_create_modal_ = false;
  char new_rule_name_[64] = "";
  char new_rule_expr_[128] = "";
  char new_rule_symbol_[32] = "";
};

}  // namespace BTQuant
