#pragma once

#include "shm_health_monitor.hpp"
#include "alert_manager.hpp"
#include <atomic>
#include <chrono>
#include <string>
#include <functional>
#include <optional>

namespace BTQuant {

/**
 * RecoveryController - Handles automatic recovery from shared memory failures
 * 
 * This class monitors system health and automatically attempts recovery
 * actions when failures are detected.
 */
class RecoveryController {
public:
    enum class RecoveryAction {
        NONE = 0,              // No action needed
        RETRY_ATTACH = 1,      // Retry attaching to shared memory
        RECREATE_SHM = 2,      // Recreate shared memory segment
        RESTART_COLLECTOR = 3, // Restart the market data collector
        FAIL_OVER = 4,         // Activate fail-over mode
        EXIT = 5               // Exit the application
    };

    struct RecoveryConfig {
        uint32_t max_retries = 3;                    // Maximum retry attempts
        uint64_t retry_delay_us = 1'000'000;         // 1 second between retries
        uint64_t timeout_us = 30'000'000;            // 30 second overall timeout
        bool auto_recovery = true;                   // Enable automatic recovery
        bool alert_on_recovery = true;               // Send alerts during recovery
        std::string collector_command = "";          // Command to restart collector
        
        // Thresholds
        uint32_t consecutive_failure_threshold = 3;
        uint64_t data_staleness_threshold_us = 10'000'000; // 10 seconds
    };

    struct RecoveryState {
        RecoveryAction last_action;
        RecoveryAction current_action;
        uint32_t attempt_count;
        uint32_t success_count;
        uint32_t failure_count;
        uint64_t last_attempt_us;
        uint64_t last_success_us;
        bool is_recovering;
        std::string status_message;
        
        RecoveryState()
            : last_action(RecoveryAction::NONE)
            , current_action(RecoveryAction::NONE)
            , attempt_count(0)
            , success_count(0)
            , failure_count(0)
            , last_attempt_us(0)
            , last_success_us(0)
            , is_recovering(false)
        {}
    };

    explicit RecoveryController(const RecoveryConfig& config);
    ~RecoveryController();

    // Non-copyable
    RecoveryController(const RecoveryController&) = delete;
    RecoveryController& operator=(const RecoveryController&) = delete;

    // Configuration
    const RecoveryConfig& get_config() const { return config_; }
    void set_config(const RecoveryConfig& config);
    
    // Recovery operations
    RecoveryAction determine_action(const Monitoring::ShmHealthMonitor::HealthReport& report);
    bool execute_recovery(RecoveryAction action);
    bool attempt_recovery(const Monitoring::ShmHealthMonitor::HealthReport& report);
    
    // State management
    const RecoveryState& get_state() const { return state_; }
    void reset_state();
    void record_attempt(RecoveryAction action, bool success, const std::string& message = "");
    
    // Status
    bool is_recovering() const { return state_.is_recovering; }
    uint32_t get_attempt_count() const { return state_.attempt_count; }
    uint32_t get_success_count() const { return state_.success_count; }
    std::string get_status_message() const { return state_.status_message; }
    
    // Recovery action helpers
    static std::string action_to_string(RecoveryAction action);
    
private:
    RecoveryConfig config_;
    RecoveryState state_;
    std::atomic<bool> running_{false};
    
    // Recovery action implementations
    bool retry_attach();
    bool recreate_shared_memory();
    bool restart_collector();
    bool activate_failover();
    bool exit_application();
    
    // Helper methods
    void set_status(const std::string& message);
    void raise_recovery_alert(RecoveryAction action, bool success, const std::string& details);
    bool should_abort_recovery() const;
};

/**
 * Recovery callback types for custom recovery actions
 */
using RecoveryCallback = std::function<bool(RecoveryController::RecoveryAction)>;

/**
 * RecoveryOrchestrator - Coordinates complex recovery scenarios
 */
class RecoveryOrchestrator {
public:
    static RecoveryOrchestrator& instance();
    
    void register_recovery_handler(RecoveryController::RecoveryAction action, 
                                   RecoveryCallback callback);
    
    bool execute_handlers(RecoveryController::RecoveryAction action);
    
    void set_default_callback(RecoveryCallback callback);
    
private:
    RecoveryOrchestrator() = default;
    ~RecoveryOrchestrator() = default;
    
    std::map<RecoveryController::RecoveryAction, RecoveryCallback> handlers_;
    RecoveryCallback default_callback_;
};

/**
 * Recovery statistics
 */
struct RecoveryStats {
    uint64_t total_recovery_attempts;
    uint64_t successful_recoveries;
    uint64_t failed_recoveries;
    uint64_t total_downtime_us;
    uint64_t longest_downtime_us;
    uint32_t consecutive_failures;
    
    RecoveryStats()
        : total_recovery_attempts(0)
        , successful_recoveries(0)
        , failed_recoveries(0)
        , total_downtime_us(0)
        , longest_downtime_us(0)
        , consecutive_failures(0)
    {}
    
    double success_rate() const {
        if (total_recovery_attempts == 0) return 0.0;
        return static_cast<double>(successful_recoveries) / total_recovery_attempts * 100.0;
    }
};

} // namespace BTQuant
