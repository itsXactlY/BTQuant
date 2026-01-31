#include "trading/risk_assessment.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace BTQuant {

RiskAssessment::RiskAssessment() { initialize_default_limits(); }

void RiskAssessment::set_risk_limits(const RiskLimits& limits) { risk_limits_ = limits; }

RiskAssessment::RiskLimits RiskAssessment::get_risk_limits() const { return risk_limits_; }

RiskAssessment::RiskMetrics RiskAssessment::get_risk_metrics() const { return RiskMetrics(); }

RiskAssessment::RiskMetrics RiskAssessment::calculate_risk_metrics(
    const PositionManager::PortfolioSummary& summary,
    const std::vector<PositionManager::Position>& positions) {
  RiskMetrics metrics;

  // Calculate VaR (simplified)
  metrics.current_var = calculate_portfolio_var(positions);

  // Calculate portfolio beta
  metrics.portfolio_beta = summary.portfolio_beta;

  // Calculate Sharpe ratio
  metrics.sharpe_ratio = summary.sharpe_ratio;

  // Calculate drawdown
  if (summary.total_value > 0) {
    metrics.drawdown = (summary.total_value - summary.total_value) / summary.total_value;
  }

  // Calculate leverage
  metrics.current_leverage =
      summary.total_value / (summary.cash_balance + summary.total_unrealized_pnl);

  // Calculate concentration risk
  metrics.concentration_risk = calculate_concentration_risk(positions, summary.total_value);

  // Calculate leverage risk
  metrics.leverage_risk = calculate_leverage_risk(metrics.current_leverage);

  // Calculate volatility risk
  metrics.volatility_risk = calculate_volatility_risk(positions);

  // Calculate liquidity risk
  metrics.liquidity_risk = calculate_liquidity_risk(positions);

  // Calculate largest position percentage
  double max_position_pct = 0;
  for (const auto& pos : positions) {
    if (summary.total_value > 0) {
      double pct = pos.market_value / summary.total_value;
      if (pct > max_position_pct) {
        max_position_pct = pct;
      }
    }
  }
  metrics.largest_position_pct = max_position_pct;

  // Calculate daily P&L
  metrics.daily_pnl = summary.total_unrealized_pnl;
  metrics.unrealized_pnl = summary.total_unrealized_pnl;

  // Calculate overall risk score (0-100)
  double risk_score = 0;
  risk_score += std::min(metrics.concentration_risk * 25, 25.0);
  risk_score += std::min(metrics.leverage_risk * 25, 25.0);
  risk_score += std::min(metrics.volatility_risk * 25, 25.0);
  risk_score += std::min(metrics.liquidity_risk * 25, 25.0);
  metrics.overall_risk_score = std::min(risk_score, 100.0);

  return metrics;
}

std::vector<RiskAssessment::RiskAlert> RiskAssessment::check_risk_limits(
    const RiskMetrics& metrics, const PositionManager::PortfolioSummary& /*portfolio*/,
    const std::vector<PositionManager::Position>& positions) {
  std::vector<RiskAlert> alerts;

  // Check position size limits
  for (const auto& pos : positions) {
    auto symbol_limit_it = risk_limits_.symbol_limits.find(pos.symbol);
    if (symbol_limit_it != risk_limits_.symbol_limits.end()) {
      if (pos.market_value > symbol_limit_it->second) {
        RiskAlert alert;
        alert.severity = RiskAlert::Severity::Warning;
        alert.message = "Position size exceeds symbol limit";
        alert.symbol = pos.symbol;
        alert.metric_name = "position_size";
        alert.current_value = pos.market_value;
        alert.threshold_value = symbol_limit_it->second;
        alert.timestamp = get_current_timestamp();
        alert.acknowledged = false;
        alerts.push_back(alert);
      }
    }
  }

  // Check concentration limits
  if (metrics.largest_position_pct > risk_limits_.max_concentration) {
    RiskAlert alert;
    alert.severity = RiskAlert::Severity::Warning;
    alert.message = "Position concentration exceeds limit";
    alert.symbol = "";
    alert.metric_name = "concentration";
    alert.current_value = metrics.largest_position_pct;
    alert.threshold_value = risk_limits_.max_concentration;
    alert.timestamp = get_current_timestamp();
    alert.acknowledged = false;
    alerts.push_back(alert);
  }

  // Check leverage limits
  if (metrics.current_leverage > risk_limits_.max_leverage) {
    RiskAlert alert;
    alert.severity = RiskAlert::Severity::Critical;
    alert.message = "Leverage exceeds limit";
    alert.symbol = "";
    alert.metric_name = "leverage";
    alert.current_value = metrics.current_leverage;
    alert.threshold_value = risk_limits_.max_leverage;
    alert.timestamp = get_current_timestamp();
    alert.acknowledged = false;
    alerts.push_back(alert);
  }

  // Check VaR limits
  if (metrics.current_var > risk_limits_.var_limit) {
    RiskAlert alert;
    alert.severity = RiskAlert::Severity::Warning;
    alert.message = "VaR exceeds limit";
    alert.symbol = "";
    alert.metric_name = "var";
    alert.current_value = metrics.current_var;
    alert.threshold_value = risk_limits_.var_limit;
    alert.timestamp = get_current_timestamp();
    alert.acknowledged = false;
    alerts.push_back(alert);
  }

  // Check daily loss limit
  if (std::abs(metrics.daily_pnl) > risk_limits_.max_daily_loss) {
    RiskAlert alert;
    alert.severity = RiskAlert::Severity::Critical;
    alert.message = "Daily loss limit exceeded";
    alert.symbol = "";
    alert.metric_name = "daily_loss";
    alert.current_value = metrics.daily_pnl;
    alert.threshold_value = -risk_limits_.max_daily_loss;
    alert.timestamp = get_current_timestamp();
    alert.acknowledged = false;
    alerts.push_back(alert);
  }

  return alerts;
}

bool RiskAssessment::validate_order_risk(const OrderManager::Order& order,
                                         const PositionManager::PortfolioSummary& portfolio,
                                         const std::vector<PositionManager::Position>& positions) {
  PositionManager::Position simulated = simulate_order_impact(order, positions);
  RiskMetrics metrics = calculate_risk_metrics(portfolio, positions);
  return validate_order(order, simulated, metrics);
}

bool RiskAssessment::validate_order(const OrderManager::Order& /*order*/,
                                    const PositionManager::Position& /*position*/,
                                    const RiskMetrics& metrics) {
  // Check leverage impact
  if (metrics.current_leverage > risk_limits_.max_leverage * 0.8) {
    // Near leverage limit, reject additional orders
    return false;
  }

  // Check concentration impact
  if (metrics.largest_position_pct > risk_limits_.max_concentration * 0.8) {
    // Near concentration limit
    return false;
  }

  return true;
}

bool RiskAssessment::is_risk_compliant(const RiskMetrics& metrics) const {
  if (metrics.concentration_risk > 0.5) {
    return false;
  }
  if (metrics.leverage_risk > 0.8) {
    return false;
  }
  if (metrics.overall_risk_score > 75.0) {
    return false;
  }
  return true;
}

RiskAssessment::RiskReport RiskAssessment::generate_risk_report(
    const PositionManager::PortfolioSummary& portfolio,
    const std::vector<PositionManager::Position>& positions) {
  RiskReport report;

  report.metrics = calculate_risk_metrics(portfolio, positions);
  report.alerts = check_risk_limits(report.metrics, portfolio, positions);
  report.recommendations = generate_recommendations(report.metrics, report.alerts);

  // Calculate risk-adjusted return
  if (report.metrics.sharpe_ratio > 0) {
    report.risk_adjusted_return =
        report.metrics.sharpe_ratio * (100 - report.metrics.overall_risk_score) / 100;
  } else {
    report.risk_adjusted_return = report.metrics.sharpe_ratio;
  }

  // Calculate maximum trade size
  report.maximum_trade_size = calculate_max_trade_size(portfolio, report.metrics);

  // Calculate symbol risk scores
  for (const auto& pos : positions) {
    report.symbol_risk_scores[pos.symbol] = calculate_symbol_risk_score(pos);
  }

  return report;
}

void RiskAssessment::initialize_default_limits() {
  risk_limits_.max_position_size = 50000.0;
  risk_limits_.max_portfolio_value = 500000.0;
  risk_limits_.max_daily_loss = 10000.0;
  risk_limits_.max_drawdown = 0.15;
  risk_limits_.max_leverage = 2.0;
  risk_limits_.max_concentration = 0.25;
  risk_limits_.var_limit = 5000.0;
}

double RiskAssessment::calculate_portfolio_var(
    const std::vector<PositionManager::Position>& positions) {
  if (positions.empty()) {
    return 0.0;
  }

  double total_var = 0.0;
  for (const auto& pos : positions) {
    total_var += pos.var_95 * pos.var_95;
  }

  return std::sqrt(total_var);
}

double RiskAssessment::calculate_concentration_risk(
    const std::vector<PositionManager::Position>& positions, double total_value) {
  if (total_value <= 0 || positions.empty()) {
    return 0.0;
  }

  double max_concentration = 0.0;
  for (const auto& pos : positions) {
    double concentration = pos.market_value / total_value;
    if (concentration > max_concentration) {
      max_concentration = concentration;
    }
  }

  // Normalize to 0-1 scale
  return std::min(max_concentration / 0.5, 1.0);
}

double RiskAssessment::calculate_leverage_risk(double leverage) {
  if (leverage <= 1.0) {
    return 0.0;
  }

  // Normalize leverage to 0-1 risk scale
  return std::min((leverage - 1.0) / 2.0, 1.0);
}

double RiskAssessment::calculate_volatility_risk(
    const std::vector<PositionManager::Position>& positions) {
  // Simplified volatility risk calculation
  if (positions.empty()) {
    return 0.0;
  }

  double avg_volatility = 0.0;
  int count = 0;

  for (const auto& pos : positions) {
    // Use beta as a proxy for volatility
    avg_volatility += pos.beta;
    count++;
  }

  if (count > 0) {
    avg_volatility /= count;
  }

  return std::min(avg_volatility - 1.0, 1.0);
}

double RiskAssessment::calculate_liquidity_risk(
    const std::vector<PositionManager::Position>& positions) {
  // Simplified liquidity risk calculation
  // In a real implementation, this would use actual volume data
  double liquidity_risk = 0.0;

  for (const auto& pos : positions) {
    if (pos.market_value > 100000) {
      liquidity_risk += 0.1;
    }
  }

  return std::min(liquidity_risk, 1.0);
}

PositionManager::Position RiskAssessment::simulate_order_impact(
    const OrderManager::Order& order, const std::vector<PositionManager::Position>& positions) {
  PositionManager::Position simulated;

  for (const auto& pos : positions) {
    if (pos.symbol == order.symbol) {
      simulated = pos;
      if (order.side == OrderManager::OrderSide::Buy) {
        simulated.quantity += order.quantity;
      } else {
        simulated.quantity -= order.quantity;
      }
      return simulated;
    }
  }

  // New position
  simulated.symbol = order.symbol;
  simulated.quantity =
      order.side == OrderManager::OrderSide::Buy ? order.quantity : -order.quantity;
  simulated.average_price = order.price;

  return simulated;
}

std::vector<std::string> RiskAssessment::generate_recommendations(
    const RiskMetrics& metrics, const std::vector<RiskAlert>& alerts) {
  std::vector<std::string> recommendations;

  // Generate recommendations based on alerts
  for (const auto& alert : alerts) {
    if (alert.severity == RiskAlert::Severity::Critical) {
      recommendations.push_back("URGENT: " + alert.message);
    } else if (alert.severity == RiskAlert::Severity::Warning) {
      recommendations.push_back("Consider addressing: " + alert.message);
    }
  }

  // General recommendations
  if (metrics.sharpe_ratio < 0.5) {
    recommendations.push_back("Sharpe ratio is low, consider improving risk-adjusted returns");
  }

  if (metrics.concentration_risk > 0.3) {
    recommendations.push_back("Portfolio concentration is high, consider diversifying");
  }

  if (metrics.leverage_risk > 0.5) {
    recommendations.push_back("Leverage usage is high, consider reducing positions");
  }

  return recommendations;
}

double RiskAssessment::calculate_max_trade_size(const PositionManager::PortfolioSummary& portfolio,
                                                const RiskMetrics& metrics) {
  double max_size = risk_limits_.max_position_size;

  // Adjust based on available buying power
  double buying_power = portfolio.buying_power;
  max_size = std::min(max_size, buying_power * 0.5);

  // Adjust based on concentration limits
  if (portfolio.total_value > 0) {
    double remaining_concentration = risk_limits_.max_concentration - metrics.largest_position_pct;
    if (remaining_concentration > 0) {
      double concentration_limit = portfolio.total_value * remaining_concentration;
      max_size = std::min(max_size, concentration_limit);
    }
  }

  // Adjust based on leverage limits
  double remaining_leverage = risk_limits_.max_leverage - metrics.current_leverage;
  if (remaining_leverage > 0) {
    double leverage_limit = portfolio.total_value * remaining_leverage;
    max_size = std::min(max_size, leverage_limit);
  }

  return std::max(0.0, max_size);
}

double RiskAssessment::calculate_symbol_risk_score(const PositionManager::Position& position) {
  double risk_score = 0.0;

  // Beta contributes to risk
  risk_score += position.beta * 10.0;

  // VaR contribution
  if (position.market_value > 0) {
    risk_score += (position.var_95 / position.market_value) * 20.0;
  }

  // Drawdown contribution
  if (position.max_drawdown < 0) {
    risk_score += std::abs(position.max_drawdown) * 10.0;
  }

  return std::min(risk_score, 100.0);
}

uint64_t RiskAssessment::get_current_timestamp() {
  auto now = std::chrono::system_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
}

}  // namespace BTQuant
