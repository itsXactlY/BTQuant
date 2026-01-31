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

    // Session boundary tracking
    std::optional<std::chrono::system_clock::time_point> session_start_time;
    std::chrono::minutes session_duration{60}; // Default to 60-minute sessions
};

// ============================================================================
// CumulativeDeltaTracker Implementation
// ============================================================================

CumulativeDeltaTracker::CumulativeDeltaTracker()
    : impl_(std::make_unique<Impl>()) {
}

CumulativeDeltaTracker::~CumulativeDeltaTracker() = default;

void CumulativeDeltaTracker::add_delta(double delta_value) {
    auto current_time = std::chrono::system_clock::now();

    // Check if we need to reset due to session boundary
    if (impl_->session_start_time.has_value()) {
        auto elapsed = current_time - impl_->session_start_time.value();
        if (elapsed >= impl_->session_duration) {
            // Session boundary crossed, reset the tracker
            reset();
            impl_->session_start_time = current_time;
        }
    } else {
        // Initialize session start time if not set
        impl_->session_start_time = current_time;
    }

    impl_->cumulative_delta += delta_value;
    impl_->has_been_reset = false;
    impl_->delta_count_since_reset++;
    impl_->last_update_time = current_time;
}

double CumulativeDeltaTracker::get_cumulative_delta() const {
    return impl_->cumulative_delta;
}

void CumulativeDeltaTracker::reset() {
    impl_->cumulative_delta = 0.0;
    impl_->has_been_reset = true;
    impl_->delta_count_since_reset = 0;
    impl_->session_start_time.reset(); // Reset session tracking as well
    // Note: We don't reset the last_update_time here as the reset itself isn't an update event
}

bool CumulativeDeltaTracker::is_reset() const {
    return impl_->has_been_reset;
}

void CumulativeDeltaTracker::set_session_boundary() {
    reset();
    // Also record the current time as the start of a new session
    impl_->session_start_time = std::chrono::system_clock::now();
}

std::optional<std::chrono::system_clock::time_point> 
CumulativeDeltaTracker::get_last_update_time() const {
    return impl_->last_update_time;
}

size_t CumulativeDeltaTracker::get_delta_count_since_reset() const {
    return impl_->delta_count_since_reset;
}

void CumulativeDeltaTracker::set_session_duration(std::chrono::minutes duration) {
    impl_->session_duration = duration;
}

std::chrono::minutes CumulativeDeltaTracker::get_session_duration() const {
    return impl_->session_duration;
}

bool CumulativeDeltaTracker::is_session_active() const {
    if (!impl_->session_start_time.has_value()) {
        return false; // No session has started yet
    }

    auto current_time = std::chrono::system_clock::now();
    auto elapsed = current_time - impl_->session_start_time.value();
    return elapsed < impl_->session_duration;
}

std::optional<std::chrono::system_clock::time_point>
CumulativeDeltaTracker::get_session_start_time() const {
    return impl_->session_start_time;
}

} // namespace BTQuant