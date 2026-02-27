#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "order_manager.hpp"
#include "position_manager.hpp"

namespace BTQuant {

class RiskAssessment {
 public:
  struct RiskLimits {
    double max_position_size;
    double max_portfolio_value;
    double max_daily_loss;
    double max_drawdown;
    double max_leverage;
    double max_concentration;
    double var_limit;
    std::unordered_map<std::string, double> symbol_limits;
    std::unordered_map<std::string, double> sector_limits;
  };

  struct RiskMetrics {
    double current_var = 0;
    double portfolio_beta = 1.0;
    double sharpe_ratio = 0;
    double max_drawdown = 0;
    double current_leverage = 0;
    double largest_position_pct = 0;
    double daily_pnl = 0;
    double unrealized_pnl = 0;
    double overall_risk_score = 0;
    double concentration_risk = 0;
    double leverage_risk = 0;
    double volatility_risk = 0;
    double liquidity_risk = 0;
    double daily_loss = 0;
    double drawdown = 0;
    double leverage = 0;
  };

  struct RiskAlert {
    enum class Severity { Info, Warning, Critical };
    Severity severity;
    std::string message;
    std::string symbol;
    std::string metric_name;
    double current_value;
    double threshold_value;
    uint64_t timestamp;
    bool acknowledged;
  };

  struct RiskReport {
    RiskMetrics metrics;
    std::vector<RiskAlert> alerts;
    std::vector<std::string> recommendations;
    double risk_adjusted_return;
    double maximum_trade_size;
    std::unordered_map<std::string, double> symbol_risk_scores;
  };

  RiskAssessment();
  void set_risk_limits(const RiskLimits& limits);
  RiskLimits get_risk_limits() const;
  RiskMetrics get_risk_metrics() const;
  RiskMetrics calculate_risk_metrics(const PositionManager::PortfolioSummary& summary,
                                     const std::vector<PositionManager::Position>& positions);
  std::vector<RiskAlert> check_risk_limits(const RiskMetrics& metrics,
                                           const PositionManager::PortfolioSummary& portfolio,
                                           const std::vector<PositionManager::Position>& positions);
  bool validate_order_risk(const OrderManager::Order& order,
                           const PositionManager::PortfolioSummary& portfolio,
                           const std::vector<PositionManager::Position>& positions);
  bool validate_order(const OrderManager::Order& order, const PositionManager::Position& position,
                      const RiskMetrics& metrics);
  bool is_risk_compliant(const RiskMetrics& metrics) const;
  RiskReport generate_risk_report(const PositionManager::PortfolioSummary& portfolio,
                                  const std::vector<PositionManager::Position>& positions);

 private:
  RiskLimits risk_limits_;
  void initialize_default_limits();
  double calculate_portfolio_var(const std::vector<PositionManager::Position>& positions);
  double calculate_concentration_risk(const std::vector<PositionManager::Position>& positions,
                                      double total_value);
  double calculate_leverage_risk(double leverage);
  double calculate_volatility_risk(const std::vector<PositionManager::Position>& positions);
  double calculate_liquidity_risk(const std::vector<PositionManager::Position>& positions);
  PositionManager::Position simulate_order_impact(
      const OrderManager::Order& order, const std::vector<PositionManager::Position>& positions);
  std::vector<std::string> generate_recommendations(const RiskMetrics& metrics,
                                                    const std::vector<RiskAlert>& alerts);
  double calculate_max_trade_size(const PositionManager::PortfolioSummary& portfolio,
                                  const RiskMetrics& metrics);
  double calculate_symbol_risk_score(const PositionManager::Position& position);
  uint64_t get_current_timestamp();
};

}  // namespace BTQuant
