#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <functional>

namespace BTQuant {

// ============================================================================
// Trading Analytics Types
// ============================================================================

struct Trade {
    std::string trade_id;
    std::string symbol;
    std::string order_id;
    double quantity;
    double price;
    double commission;
    double pnl;  // Profit and Loss for this trade
    std::chrono::system_clock::time_point timestamp;
    bool is_buy;
};

struct PerformanceMetrics {
    double total_return;
    double annualized_return;
    double sharpe_ratio;
    double sortino_ratio;
    double max_drawdown;
    double win_rate;
    double profit_factor;
    double average_win;
    double average_loss;
    int total_trades;
    int winning_trades;
    int losing_trades;
    double total_profit;
    double total_loss;
    double net_profit;
    double largest_win;
    double largest_loss;
};

struct TradeStatistics {
    int total_trades;
    int winning_trades;
    int losing_trades;
    double win_rate;
    double total_profit;
    double total_loss;
    double net_profit;
    double average_win;
    double average_loss;
    double profit_factor;
    double largest_win;
    double largest_loss;
    double average_trade_duration_seconds;
};

struct EquityCurve {
    std::vector<std::chrono::system_clock::time_point> timestamps;
    std::vector<double> equity_values;
    std::vector<double> drawdowns;
};

// ============================================================================
// TradingAnalytics Class
// ============================================================================

class TradingAnalytics {
public:
    TradingAnalytics();
    ~TradingAnalytics();

    // Trade Recording
    void record_trade(const Trade& trade);
    void record_trades(const std::vector<Trade>& trades);
    std::vector<Trade> get_trades() const;
    std::vector<Trade> get_trades(const std::string& symbol) const;
    void clear_trades();

    // Performance Metrics
    PerformanceMetrics calculate_performance_metrics() const;
    PerformanceMetrics calculate_performance_metrics(const std::string& symbol) const;
    TradeStatistics calculate_trade_statistics() const;
    TradeStatistics calculate_trade_statistics(const std::string& symbol) const;

    // Equity Curve
    EquityCurve calculate_equity_curve(double initial_capital) const;
    EquityCurve calculate_equity_curve(const std::string& symbol, double initial_capital) const;

    // Drawdown Analysis
    double calculate_max_drawdown() const;
    double calculate_max_drawdown(const std::string& symbol) const;
    std::vector<std::pair<std::chrono::system_clock::time_point, double>> get_drawdown_periods() const;

    // Risk Metrics
    double calculate_volatility(int period_days = 30) const;
    double calculate_var(double confidence_level = 0.95) const;
    double calculate_cvar(double confidence_level = 0.95) const;

    // Trade Analysis
    std::vector<Trade> get_winning_trades() const;
    std::vector<Trade> get_losing_trades() const;
    std::vector<Trade> get_trades_in_period(
        std::chrono::system_clock::time_point start,
        std::chrono::system_clock::time_point end) const;

    // Symbol Analysis
    std::vector<std::string> get_traded_symbols() const;
    std::unordered_map<std::string, TradeStatistics> get_statistics_by_symbol() const;

    // Callbacks
    using TradeRecordedCallback = std::function<void(const Trade&)>;
    void set_trade_recorded_callback(TradeRecordedCallback callback);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace BTQuant
