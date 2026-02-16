#include "risk_assessment.h"
#include <ctime>

RiskAssessment::RiskAssessment(double daily_loss_limit, double max_position_size)
    : daily_loss_limit_(daily_loss_limit),
      current_daily_pnl_(0.0),
      max_position_size_(max_position_size),
      current_position_size_(0.0) {
    // Initialize last_reset_date_ to current date (midnight as seconds since epoch)
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm* tm_now = std::localtime(&time_t_now);
    tm_now->tm_hour = 0;
    tm_now->tm_min = 0;
    tm_now->tm_sec = 0;
    last_reset_date_.store(static_cast<int64_t>(std::mktime(tm_now)));
}

RiskAssessmentResult RiskAssessment::check_daily_loss_limit(const Order& order) const {
    // Lock-free read of current daily P&L and loss limit
    double current_pnl = current_daily_pnl_.load(std::memory_order_acquire);
    double limit = daily_loss_limit_.load(std::memory_order_acquire);

    // Calculate potential P&L impact of this order
    // For a sell order, estimate potential loss based on order value
    double potential_impact = order.quantity * order.price;

    // Check if current P&L minus potential impact exceeds the loss limit
    // Negative P&L represents losses
    if (current_pnl - potential_impact < -limit) {
        return RiskAssessmentResult::DAILY_LOSS_LIMIT_EXCEEDED;
    }

    return RiskAssessmentResult::APPROVED;
}

RiskAssessmentResult RiskAssessment::check_max_position_size(const Order& order) const {
    // Lock-free read of current position size and max limit
    double current_size = current_position_size_.load(std::memory_order_acquire);
    double max_size = max_position_size_.load(std::memory_order_acquire);

    // Calculate new position size after this order
    double new_position_size;
    if (order.is_buy) {
        new_position_size = current_size + order.quantity;
    } else {
        new_position_size = current_size - order.quantity;
    }

    // Check absolute position size against limit
    if (std::abs(new_position_size) > max_size) {
        return RiskAssessmentResult::MAX_POSITION_SIZE_EXCEEDED;
    }

    return RiskAssessmentResult::APPROVED;
}

RiskAssessmentResult RiskAssessment::assess_order(const Order& order) const {
    // Run both atomic checks without acquiring any locks
    // Order matters: check daily loss first, then position size

    RiskAssessmentResult loss_check = check_daily_loss_limit(order);
    if (loss_check != RiskAssessmentResult::APPROVED) {
        return loss_check;
    }

    RiskAssessmentResult position_check = check_max_position_size(order);
    if (position_check != RiskAssessmentResult::APPROVED) {
        return position_check;
    }

    return RiskAssessmentResult::APPROVED;
}

void RiskAssessment::update_daily_pnl(double pnl_change) {
    // Thread-safe update using atomic fetch_add
    current_daily_pnl_.fetch_add(pnl_change, std::memory_order_acq_rel);
}

void RiskAssessment::update_position_size(double quantity_change) {
    // Thread-safe update using atomic fetch_add
    current_position_size_.fetch_add(quantity_change, std::memory_order_acq_rel);
}

void RiskAssessment::maybe_reset_daily_pnl() {
    // Get current date (midnight)
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm* tm_now = std::localtime(&time_t_now);
    tm_now->tm_hour = 0;
    tm_now->tm_min = 0;
    tm_now->tm_sec = 0;
    int64_t today_midnight = static_cast<int64_t>(std::mktime(tm_now));

    // Lock-free compare: if stored date is different from today, reset
    int64_t stored_date = last_reset_date_.load(std::memory_order_acquire);
    if (stored_date != today_midnight) {
        // Attempt to update the date atomically
        // Only one thread will succeed in resetting
        if (last_reset_date_.compare_exchange_strong(stored_date, today_midnight,
                                                      std::memory_order_acq_rel)) {
            // Successfully updated date, now reset P&L
            current_daily_pnl_.store(0.0, std::memory_order_release);
        }
        // If compare_exchange failed, another thread already reset
    }
}

double RiskAssessment::get_daily_loss_limit() const {
    return daily_loss_limit_.load(std::memory_order_acquire);
}

double RiskAssessment::get_current_daily_pnl() const {
    return current_daily_pnl_.load(std::memory_order_acquire);
}

double RiskAssessment::get_max_position_size() const {
    return max_position_size_.load(std::memory_order_acquire);
}

double RiskAssessment::get_current_position_size() const {
    return current_position_size_.load(std::memory_order_acquire);
}

void RiskAssessment::set_daily_loss_limit(double limit) {
    daily_loss_limit_.store(limit, std::memory_order_release);
}

void RiskAssessment::set_max_position_size(double size) {
    max_position_size_.store(size, std::memory_order_release);
}

void RiskAssessment::reset_daily_pnl() {
    current_daily_pnl_.store(0.0, std::memory_order_release);
}

void RiskAssessment::reset_position_size() {
    current_position_size_.store(0.0, std::memory_order_release);
}
