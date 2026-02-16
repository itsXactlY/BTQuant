#ifndef PUBBTQUANT_RISK_ASSESSMENT_H
#define PUBBTQUANT_RISK_ASSESSMENT_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>

#include "../dependencies/BTQ_Render_Engine/include/threading/lockfree_queue.hpp"

// Structure to represent an order
struct Order {
    double quantity;
    double price;
    bool is_buy;
    std::string symbol;

    Order() : quantity(0.0), price(0.0), is_buy(true), symbol("") {}
    Order(double qty, double p, bool buy, const std::string& sym)
        : quantity(qty), price(p), is_buy(buy), symbol(sym) {}
};

// Result of risk assessment
enum class RiskAssessmentResult {
    APPROVED,
    DAILY_LOSS_LIMIT_EXCEEDED,
    MAX_POSITION_SIZE_EXCEEDED
};

// Execution report for rejected orders
struct ExecutionReport {
    enum class RejectReason {
        NONE = 0,
        DAILY_LOSS_LIMIT_EXCEEDED,
        MAX_POSITION_SIZE_EXCEEDED,
        UNKNOWN
    };

    std::string symbol;
    double quantity;
    double price;
    bool is_buy;
    RejectReason reason;
    std::chrono::system_clock::time_point timestamp;

    ExecutionReport()
        : symbol(""), quantity(0.0), price(0.0), is_buy(true),
          reason(RejectReason::NONE), timestamp(std::chrono::system_clock::now()) {}

    ExecutionReport(const std::string& sym, double qty, double p, bool buy, RejectReason r)
        : symbol(sym), quantity(qty), price(p), is_buy(buy), reason(r),
          timestamp(std::chrono::system_clock::now()) {}
};

class RiskAssessment {
private:
    // Daily loss limit (atomic for lock-free reads)
    std::atomic<double> daily_loss_limit_;

    // Current daily P&L (atomic for lock-free updates)
    std::atomic<double> current_daily_pnl_;

    // Maximum position size per symbol (atomic for lock-free reads)
    std::atomic<double> max_position_size_;

    // Current position size (atomic for lock-free updates)
    std::atomic<double> current_position_size_;

    // Track the date for daily reset (store as seconds since epoch)
    std::atomic<int64_t> last_reset_date_;

    // Lock-free queue for rejected orders (reverse queue for error codes)
    mutable btq::threading::LockFreeQueue<ExecutionReport> rejection_queue_;

public:
    explicit RiskAssessment(double daily_loss_limit = 10000.0,
                           double max_position_size = 1000.0);

    // Check if order passes daily loss limit (lock-free)
    RiskAssessmentResult check_daily_loss_limit(const Order& order) const;

    // Check if order passes max position size (lock-free)
    RiskAssessmentResult check_max_position_size(const Order& order) const;

    // Combined check - runs both atomic checks
    RiskAssessmentResult assess_order(const Order& order) const;

    // Update daily P&L (thread-safe via atomic)
    void update_daily_pnl(double pnl_change);

    // Update position size (thread-safe via atomic)
    void update_position_size(double quantity_change);

    // Reset daily P&L if a new day has started (lock-free check)
    void maybe_reset_daily_pnl();

    // Getters for current state (lock-free reads)
    double get_daily_loss_limit() const;
    double get_current_daily_pnl() const;
    double get_max_position_size() const;
    double get_current_position_size() const;

    // Setters for configuration (thread-safe via atomic store)
    void set_daily_loss_limit(double limit);
    void set_max_position_size(double size);

    // Manual reset functions
    void reset_daily_pnl();
    void reset_position_size();

    // Get rejection queue for UI rendering
    const btq::threading::LockFreeQueue<ExecutionReport>& get_rejection_queue() const {
        return rejection_queue_;
    }

    // Pop a rejection from the queue (for UI consumption)
    std::optional<ExecutionReport> pop_rejection() const {
        return rejection_queue_.try_pop();
    }
};

#endif // PUBBTQUANT_RISK_ASSESSMENT_H
