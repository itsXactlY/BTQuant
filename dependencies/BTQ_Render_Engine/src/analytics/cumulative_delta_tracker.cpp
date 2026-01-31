#include "analytics/cumulative_delta_tracker.hpp"
#include <chrono>
#include <memory>

namespace BTQuant {

// ============================================================================
// CumulativeDeltaTracker::Impl
// ============================================================================

struct CumulativeDeltaTracker::Impl {
    double cumulative_delta = 0.0;
    bool has_been_reset = false;
    size_t delta_count_since_reset = 0;
    std::optional<std::chrono::system_clock::time_point> last_update_time;
};

// ============================================================================
// CumulativeDeltaTracker Implementation
// ============================================================================

CumulativeDeltaTracker::CumulativeDeltaTracker()
    : impl_(std::make_unique<Impl>()) {
}

CumulativeDeltaTracker::~CumulativeDeltaTracker() = default;

void CumulativeDeltaTracker::add_delta(double delta_value) {
    impl_->cumulative_delta += delta_value;
    impl_->has_been_reset = false;
    impl_->delta_count_since_reset++;
    impl_->last_update_time = std::chrono::system_clock::now();
}

double CumulativeDeltaTracker::get_cumulative_delta() const {
    return impl_->cumulative_delta;
}

void CumulativeDeltaTracker::reset() {
    impl_->cumulative_delta = 0.0;
    impl_->has_been_reset = true;
    impl_->delta_count_since_reset = 0;
    // Note: We don't reset the last_update_time here as the reset itself isn't an update event
}

bool CumulativeDeltaTracker::is_reset() const {
    return impl_->has_been_reset;
}

void CumulativeDeltaTracker::set_session_boundary() {
    reset();
}

std::optional<std::chrono::system_clock::time_point> 
CumulativeDeltaTracker::get_last_update_time() const {
    return impl_->last_update_time;
}

size_t CumulativeDeltaTracker::get_delta_count_since_reset() const {
    return impl_->delta_count_since_reset;
}

} // namespace BTQuant