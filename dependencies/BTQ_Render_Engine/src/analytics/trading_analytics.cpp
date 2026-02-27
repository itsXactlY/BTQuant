#include "analytics/trading_analytics.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace BTQuant {

// ============================================================================
// TradingAnalytics::Impl
// ============================================================================

struct TradingAnalytics::Impl {
  std::vector<Trade> trades;
  TradingAnalytics::TradeRecordedCallback trade_recorded_callback;
};

// ============================================================================
// TradingAnalytics Implementation
// ============================================================================

TradingAnalytics::TradingAnalytics() : impl_(std::make_unique<Impl>()) {}

TradingAnalytics::~TradingAnalytics() = default;

void TradingAnalytics::record_trade(const Trade& trade) {
  impl_->trades.push_back(trade);
  if (impl_->trade_recorded_callback) {
    impl_->trade_recorded_callback(trade);
  }
}

void TradingAnalytics::record_trades(const std::vector<Trade>& trades) {
  for (const auto& trade : trades) {
    record_trade(trade);
  }
}

std::vector<Trade> TradingAnalytics::get_trades() const { return impl_->trades; }

std::vector<Trade> TradingAnalytics::get_trades(const std::string& symbol) const {
  std::vector<Trade> result;
  for (const auto& trade : impl_->trades) {
    if (trade.symbol == symbol) {
      result.push_back(trade);
    }
  }
  return result;
}

void TradingAnalytics::clear_trades() { impl_->trades.clear(); }

PerformanceMetrics TradingAnalytics::calculate_performance_metrics() const {
  PerformanceMetrics metrics{};

  if (impl_->trades.empty()) {
    return metrics;
  }

  // Calculate basic statistics
  double total_profit = 0.0;
  double total_loss = 0.0;
  int winning_trades = 0;
  int losing_trades = 0;
  std::vector<double> returns;

  for (const auto& trade : impl_->trades) {
    double pnl = trade.pnl;

    returns.push_back(pnl);

    if (pnl > 0) {
      total_profit += pnl;
      winning_trades++;
    } else {
      total_loss += std::abs(pnl);
      losing_trades++;
    }
  }

  metrics.total_trades = static_cast<int>(impl_->trades.size());
  metrics.winning_trades = winning_trades;
  metrics.losing_trades = losing_trades;
  metrics.win_rate =
      metrics.total_trades > 0 ? static_cast<double>(winning_trades) / metrics.total_trades : 0.0;
  metrics.total_profit = total_profit;
  metrics.total_loss = total_loss;
  metrics.net_profit = total_profit - total_loss;
  metrics.average_win = winning_trades > 0 ? total_profit / winning_trades : 0.0;
  metrics.average_loss = losing_trades > 0 ? total_loss / losing_trades : 0.0;
  metrics.profit_factor = total_loss > 0 ? total_profit / total_loss : 0.0;
  metrics.largest_win = total_profit > 0 ? total_profit : 0.0;
  metrics.largest_loss = total_loss > 0 ? total_loss : 0.0;

  // Calculate Sharpe ratio (simplified)
  if (!returns.empty()) {
    double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double variance = 0.0;
    for (double r : returns) {
      variance += (r - mean) * (r - mean);
    }
    variance /= returns.size();
    double std_dev = std::sqrt(variance);
    metrics.sharpe_ratio = std_dev > 0 ? mean / std_dev : 0.0;
  }

  // Calculate max drawdown
  metrics.max_drawdown = calculate_max_drawdown();

  return metrics;
}

PerformanceMetrics TradingAnalytics::calculate_performance_metrics(
    const std::string& symbol) const {
  std::vector<Trade> symbol_trades = get_trades(symbol);
  if (symbol_trades.empty()) {
    return PerformanceMetrics{};
  }

  // Create a temporary TradingAnalytics for this symbol
  TradingAnalytics temp;
  temp.record_trades(symbol_trades);
  return temp.calculate_performance_metrics();
}

TradeStatistics TradingAnalytics::calculate_trade_statistics() const {
  TradeStatistics stats{};

  if (impl_->trades.empty()) {
    return stats;
  }

  double total_profit = 0.0;
  double total_loss = 0.0;
  int winning_trades = 0;
  int losing_trades = 0;
  double largest_win = 0.0;
  double largest_loss = 0.0;
  double total_duration = 0.0;

  for (const auto& trade : impl_->trades) {
    double pnl = trade.pnl;

    if (pnl > 0) {
      total_profit += pnl;
      winning_trades++;
      if (pnl > largest_win) {
        largest_win = pnl;
      }
    } else {
      total_loss += std::abs(pnl);
      losing_trades++;
      if (std::abs(pnl) > largest_loss) {
        largest_loss = std::abs(pnl);
      }
    }
  }

  stats.total_trades = static_cast<int>(impl_->trades.size());
  stats.winning_trades = winning_trades;
  stats.losing_trades = losing_trades;
  stats.win_rate =
      stats.total_trades > 0 ? static_cast<double>(winning_trades) / stats.total_trades : 0.0;
  stats.total_profit = total_profit;
  stats.total_loss = total_loss;
  stats.net_profit = total_profit - total_loss;
  stats.average_win = winning_trades > 0 ? total_profit / winning_trades : 0.0;
  stats.average_loss = losing_trades > 0 ? total_loss / losing_trades : 0.0;
  stats.profit_factor = total_loss > 0 ? total_profit / total_loss : 0.0;
  stats.largest_win = largest_win;
  stats.largest_loss = largest_loss;
  stats.average_trade_duration_seconds =
      stats.total_trades > 0 ? total_duration / stats.total_trades : 0.0;

  return stats;
}

TradeStatistics TradingAnalytics::calculate_trade_statistics(const std::string& symbol) const {
  std::vector<Trade> symbol_trades = get_trades(symbol);
  if (symbol_trades.empty()) {
    return TradeStatistics{};
  }

  TradingAnalytics temp;
  temp.record_trades(symbol_trades);
  return temp.calculate_trade_statistics();
}

EquityCurve TradingAnalytics::calculate_equity_curve(double initial_capital) const {
  EquityCurve curve;

  if (impl_->trades.empty()) {
    return curve;
  }

  // Sort trades by timestamp
  std::vector<Trade> sorted_trades = impl_->trades;
  std::sort(sorted_trades.begin(), sorted_trades.end(),
            [](const Trade& a, const Trade& b) { return a.timestamp < b.timestamp; });

  double equity = initial_capital;
  double peak_equity = initial_capital;

  for (const auto& trade : sorted_trades) {
    double pnl = trade.pnl;

    equity += pnl;

    if (equity > peak_equity) {
      peak_equity = equity;
    }

    double drawdown = peak_equity > 0 ? (peak_equity - equity) / peak_equity : 0.0;

    curve.timestamps.push_back(trade.timestamp);
    curve.equity_values.push_back(equity);
    curve.drawdowns.push_back(drawdown);
  }

  return curve;
}

EquityCurve TradingAnalytics::calculate_equity_curve(const std::string& symbol,
                                                     double initial_capital) const {
  std::vector<Trade> symbol_trades = get_trades(symbol);
  if (symbol_trades.empty()) {
    return EquityCurve{};
  }

  TradingAnalytics temp;
  temp.record_trades(symbol_trades);
  return temp.calculate_equity_curve(initial_capital);
}

double TradingAnalytics::calculate_max_drawdown() const {
  EquityCurve curve = calculate_equity_curve(100000.0);  // Default initial capital

  if (curve.drawdowns.empty()) {
    return 0.0;
  }

  return *std::max_element(curve.drawdowns.begin(), curve.drawdowns.end());
}

double TradingAnalytics::calculate_max_drawdown(const std::string& symbol) const {
  EquityCurve curve = calculate_equity_curve(symbol, 100000.0);

  if (curve.drawdowns.empty()) {
    return 0.0;
  }

  return *std::max_element(curve.drawdowns.begin(), curve.drawdowns.end());
}

std::vector<std::pair<std::chrono::system_clock::time_point, double>>
TradingAnalytics::get_drawdown_periods() const {
  std::vector<std::pair<std::chrono::system_clock::time_point, double>> result;

  EquityCurve curve = calculate_equity_curve(100000.0);

  for (size_t i = 0; i < curve.timestamps.size(); ++i) {
    result.emplace_back(curve.timestamps[i], curve.drawdowns[i]);
  }

  return result;
}

double TradingAnalytics::calculate_volatility(int period_days) const {
  if (impl_->trades.empty()) {
    return 0.0;
  }

  // Calculate daily returns
  std::vector<double> daily_returns;
  std::unordered_map<std::string, double> daily_pnl;

  for (const auto& trade : impl_->trades) {
    auto time_t = std::chrono::system_clock::to_time_t(trade.timestamp);
    std::tm tm = *std::localtime(&time_t);
    std::string day_key = std::to_string(tm.tm_year) + "-" + std::to_string(tm.tm_mon) + "-" +
                          std::to_string(tm.tm_mday);

    double pnl = trade.pnl;

    daily_pnl[day_key] += pnl;
  }

  for (const auto& [day, pnl] : daily_pnl) {
    daily_returns.push_back(pnl);
  }

  if (daily_returns.empty()) {
    return 0.0;
  }

  // Calculate standard deviation
  double mean =
      std::accumulate(daily_returns.begin(), daily_returns.end(), 0.0) / daily_returns.size();
  double variance = 0.0;
  for (double r : daily_returns) {
    variance += (r - mean) * (r - mean);
  }
  variance /= daily_returns.size();

  return std::sqrt(variance);
}

double TradingAnalytics::calculate_var(double confidence_level) const {
  if (impl_->trades.empty()) {
    return 0.0;
  }

  std::vector<double> returns;
  for (const auto& trade : impl_->trades) {
    double pnl = trade.pnl;
    returns.push_back(pnl);
  }

  if (returns.empty()) {
    return 0.0;
  }

  std::sort(returns.begin(), returns.end());

  size_t index = static_cast<size_t>((1.0 - confidence_level) * returns.size());
  index = std::min(index, returns.size() - 1);

  return returns[index];
}

double TradingAnalytics::calculate_cvar(double confidence_level) const {
  if (impl_->trades.empty()) {
    return 0.0;
  }

  std::vector<double> returns;
  for (const auto& trade : impl_->trades) {
    double pnl = trade.pnl;
    returns.push_back(pnl);
  }

  if (returns.empty()) {
    return 0.0;
  }

  std::sort(returns.begin(), returns.end());

  size_t cutoff_index = static_cast<size_t>((1.0 - confidence_level) * returns.size());
  cutoff_index = std::min(cutoff_index, returns.size());

  double sum = 0.0;
  for (size_t i = 0; i < cutoff_index; ++i) {
    sum += returns[i];
  }

  return cutoff_index > 0 ? sum / cutoff_index : 0.0;
}

std::vector<Trade> TradingAnalytics::get_winning_trades() const {
  std::vector<Trade> result;
  for (const auto& trade : impl_->trades) {
    double pnl = trade.pnl;
    if (pnl > 0) {
      result.push_back(trade);
    }
  }
  return result;
}

std::vector<Trade> TradingAnalytics::get_losing_trades() const {
  std::vector<Trade> result;
  for (const auto& trade : impl_->trades) {
    double pnl = trade.pnl;
    if (pnl < 0) {
      result.push_back(trade);
    }
  }
  return result;
}

std::vector<Trade> TradingAnalytics::get_trades_in_period(
    std::chrono::system_clock::time_point start, std::chrono::system_clock::time_point end) const {
  std::vector<Trade> result;
  for (const auto& trade : impl_->trades) {
    if (trade.timestamp >= start && trade.timestamp <= end) {
      result.push_back(trade);
    }
  }
  return result;
}

std::vector<std::string> TradingAnalytics::get_traded_symbols() const {
  std::unordered_set<std::string> symbols;
  for (const auto& trade : impl_->trades) {
    symbols.insert(trade.symbol);
  }
  return std::vector<std::string>(symbols.begin(), symbols.end());
}

std::unordered_map<std::string, TradeStatistics> TradingAnalytics::get_statistics_by_symbol()
    const {
  std::unordered_map<std::string, TradeStatistics> result;

  for (const auto& symbol : get_traded_symbols()) {
    result[symbol] = calculate_trade_statistics(symbol);
  }

  return result;
}

void TradingAnalytics::set_trade_recorded_callback(TradeRecordedCallback callback) {
  impl_->trade_recorded_callback = std::move(callback);
}

}  // namespace BTQuant
