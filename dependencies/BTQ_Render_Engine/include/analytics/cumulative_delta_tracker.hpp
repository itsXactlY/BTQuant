#pragma once

#include <chrono>
#include <optional>
#include <memory>

namespace BTQuant {

// ============================================================================
// Cumulative Delta Tracker
// ============================================================================

class CumulativeDeltaTracker {
public:
    CumulativeDeltaTracker();
    ~CumulativeDeltaTracker();

    // Add delta value for current time bar
    void add_delta(double delta_value);

    // Get current cumulative delta
    double get_cumulative_delta() const;

    // Reset cumulative delta to zero (for session boundaries)
    void reset();

    // Check if tracker has been reset since last addition
    bool is_reset() const;

    // Set session boundary - resets cumulative delta
    void set_session_boundary();

    // Get timestamp of last delta addition
    std::optional<std::chrono::system_clock::time_point> get_last_update_time() const;

    // Get count of delta values added since last reset
    size_t get_delta_count_since_reset() const;

    // Set the session duration for automatic reset
    void set_session_duration(std::chrono::minutes duration);

    // Get the current session duration
    std::chrono::minutes get_session_duration() const;

    // Check if current session is still active (hasn't exceeded session duration)
    bool is_session_active() const;

    // Get the start time of the current session
    std::optional<std::chrono::system_clock::time_point> get_session_start_time() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace BTQuant