#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <optional>

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

// Rejection message for display in alerts panel
struct RejectionMessage {
  std::string symbol;
  std::string reason;
  double quantity;
  double price;
  bool is_buy;
  std::chrono::system_clock::time_point timestamp;
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

  // Public method to add rejection messages from risk assessment
  void add_rejection(const RejectionMessage& msg) {
    rejections_.push_back(msg);
  }

  // Process rejections from a queue (template to avoid circular dependency)
  template<typename QueueType>
  void process_rejection_queue(const QueueType& queue) {
    while (true) {
      auto report = queue.try_pop();
      if (!report.has_value()) break;
      
      std::string reason_str;
      switch (report->reason) {
        case decltype(report)::type::RejectReason::DAILY_LOSS_LIMIT_EXCEEDED:
          reason_str = "Daily Loss Limit Exceeded";
          break;
        case decltype(report)::type::RejectReason::MAX_POSITION_SIZE_EXCEEDED:
          reason_str = "Max Position Size Exceeded";
          break;
        default:
          reason_str = "Unknown Risk Violation";
          break;
      }
      
      rejections_.push_back({
        report->symbol,
        reason_str,
        report->quantity,
        report->price,
        report->is_buy,
        report->timestamp
      });
    }
  }

 private:
  void render_rules_table();
  void render_alert_logs();
  void render_rejections();
  void render_create_rule_modal();

  std::vector<AlertRule> rules_;
  std::vector<AlertLog> logs_;
  std::vector<RejectionMessage> rejections_;

  // UI State
  bool show_create_modal_ = false;
  char new_rule_name_[64] = "";
  char new_rule_expr_[128] = "";
  char new_rule_symbol_[32] = "";
};

}  // namespace BTQuant
