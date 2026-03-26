#include "utils/recovery_controller.hpp"
#include "utils/dynamic_logger.hpp"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

namespace BTQuant {

// ============================================================================
// RecoveryController Implementation
// ============================================================================

RecoveryController::RecoveryController(const RecoveryConfig &config)
    : config_(config) {
  Logging::DynamicLogger::instance().info("RECOVERY",
                                          "RecoveryController initialized");
}

RecoveryController::~RecoveryController() {
  Logging::DynamicLogger::instance().info("RECOVERY",
                                          "RecoveryController destroyed");
}

void RecoveryController::set_config(const RecoveryConfig &config) {
  config_ = config;
  Logging::DynamicLogger::instance().info("RECOVERY", "Configuration updated");
}

RecoveryController::RecoveryAction RecoveryController::determine_action(
    const Monitoring::ShmHealthMonitor::HealthReport &report) {

  // If system is healthy, no action needed
  if (report.status == Monitoring::ShmHealthMonitor::HealthStatus::HEALTHY) {
    return RecoveryAction::NONE;
  }

  // Check if we should abort recovery
  if (should_abort_recovery()) {
    Logging::DynamicLogger::instance().error(
        "RECOVERY", "Recovery aborted - maximum retries exceeded");
    return RecoveryAction::EXIT;
  }

  // Determine action based on health status
  switch (report.status) {
  case Monitoring::ShmHealthMonitor::HealthStatus::DEGRADED:
    // Try to retry attach first
    return RecoveryAction::RETRY_ATTACH;

  case Monitoring::ShmHealthMonitor::HealthStatus::CRITICAL:
    // Check if file was deleted
    if (!report.file_exists) {
      return RecoveryAction::RECREATE_SHM;
    }
    // Check if file is corrupted
    if (!report.has_valid_header) {
      return RecoveryAction::RECREATE_SHM;
    }
    // Permission issues
    if (!report.permissions_ok) {
      return RecoveryAction::RETRY_ATTACH;
    }
    return RecoveryAction::RETRY_ATTACH;

  case Monitoring::ShmHealthMonitor::HealthStatus::FAILED:
    // Complete failure - try to restart collector
    return RecoveryAction::RESTART_COLLECTOR;

  default:
    return RecoveryAction::NONE;
  }
}

bool RecoveryController::execute_recovery(RecoveryAction action) {
  if (action == RecoveryAction::NONE) {
    return true;
  }

  state_.current_action = action;
  state_.is_recovering = true;

  bool success = false;

  switch (action) {
  case RecoveryAction::RETRY_ATTACH:
    success = retry_attach();
    break;
  case RecoveryAction::RECREATE_SHM:
    success = recreate_shared_memory();
    break;
  case RecoveryAction::RESTART_COLLECTOR:
    success = restart_collector();
    break;
  case RecoveryAction::FAIL_OVER:
    success = activate_failover();
    break;
  case RecoveryAction::EXIT:
    success = exit_application();
    break;
  default:
    set_status("Unknown recovery action");
    return false;
  }

  record_attempt(action, success);
  state_.is_recovering = false;

  return success;
}

bool RecoveryController::attempt_recovery(
    const Monitoring::ShmHealthMonitor::HealthReport &report) {

  RecoveryAction action = determine_action(report);

  if (action == RecoveryAction::NONE) {
    return true;
  }

  if (action == RecoveryAction::EXIT) {
    set_status("Maximum recovery attempts exceeded");
    return false;
  }

  return execute_recovery(action);
}

void RecoveryController::reset_state() {
  state_ = RecoveryState();
  Logging::DynamicLogger::instance().info("RECOVERY", "State reset");
}

void RecoveryController::record_attempt(RecoveryAction action, bool success,
                                        const std::string &message) {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  state_.last_attempt_us =
      static_cast<uint64_t>(tv.tv_sec) * 1000000ULL + tv.tv_usec;

  state_.attempt_count++;
  state_.last_action = action;

  if (success) {
    state_.success_count++;
    state_.last_success_us = state_.last_attempt_us;
    set_status("Recovery successful: " + action_to_string(action));
    raise_recovery_alert(action, true, message);
  } else {
    state_.failure_count++;
    set_status("Recovery failed: " + action_to_string(action));
    raise_recovery_alert(action, false, message);
  }
}

std::string RecoveryController::action_to_string(RecoveryAction action) {
  switch (action) {
  case RecoveryAction::NONE:
    return "NONE";
  case RecoveryAction::RETRY_ATTACH:
    return "RETRY_ATTACH";
  case RecoveryAction::RECREATE_SHM:
    return "RECREATE_SHM";
  case RecoveryAction::RESTART_COLLECTOR:
    return "RESTART_COLLECTOR";
  case RecoveryAction::FAIL_OVER:
    return "FAIL_OVER";
  case RecoveryAction::EXIT:
    return "EXIT";
  default:
    return "UNKNOWN";
  }
}

bool RecoveryController::retry_attach() {
  Logging::DynamicLogger::instance().warning("RECOVERY",
                                             "Attempting to retry SHM attach");

  // Wait before retry
  std::this_thread::sleep_for(
      std::chrono::microseconds(config_.retry_delay_us));

  // The actual reattach will be handled by the HotSpineReader
  // This method just signals that we attempted recovery
  set_status("Retry attach attempted");

  // Check if file now exists
  struct stat st;
  if (stat("/dev/shm/btq_hotspine", &st) == 0) {
    Logging::DynamicLogger::instance().info("RECOVERY",
                                            "SHM file now accessible");
    return true;
  }

  Logging::DynamicLogger::instance().warning("RECOVERY",
                                             "SHM file still not accessible");
  return false;
}

bool RecoveryController::recreate_shared_memory() {
  Logging::DynamicLogger::instance().warning("RECOVERY",
                                             "Attempting to recreate SHM");

  // Note: In a real implementation, this would signal the market data collector
  // to recreate the shared memory segment

  // For now, we just log the attempt
  if (!config_.collector_command.empty()) {
    int result = system(config_.collector_command.c_str());
    if (result == 0) {
      Logging::DynamicLogger::instance().info(
          "RECOVERY", "Collector restart command executed");
      return true;
    }
  }

  set_status("SHM recreation attempted (may require collector restart)");
  return false;
}

bool RecoveryController::restart_collector() {
  Logging::DynamicLogger::instance().warning("RECOVERY",
                                             "Attempting to restart collector");

  if (config_.collector_command.empty()) {
    Logging::DynamicLogger::instance().warning(
        "RECOVERY", "No collector command configured");
    return false;
  }

  int result = system(config_.collector_command.c_str());
  if (result == 0) {
    Logging::DynamicLogger::instance().info("RECOVERY",
                                            "Collector restarted successfully");
    return true;
  }

  Logging::DynamicLogger::instance().error(
      "RECOVERY", "Failed to restart collector: " + std::to_string(result));
  return false;
}

bool RecoveryController::activate_failover() {
  Logging::DynamicLogger::instance().warning("RECOVERY",
                                             "Activating fail-over mode");

  // In a real implementation, this would:
  // 1. Switch to backup data source
  // 2. Update configuration to use alternative paths
  // 3. Notify downstream systems

  set_status("Fail-over mode activated");
  return true;
}

bool RecoveryController::exit_application() {
  Logging::DynamicLogger::instance().error(
      "RECOVERY", "Unrecoverable failure - application will exit");
  set_status("Unrecoverable failure");

  // Signal the main application to exit
  // This is done by returning false to the caller

  return false;
}

void RecoveryController::set_status(const std::string &message) {
  state_.status_message = message;
  Logging::DynamicLogger::instance().info("RECOVERY", message);
}

void RecoveryController::raise_recovery_alert(RecoveryAction action,
                                              bool success,
                                              const std::string &details) {
  if (!config_.alert_on_recovery) {
    return;
  }

  std::string message = "Recovery action " + action_to_string(action);
  if (!details.empty()) {
    message += ": " + details;
  }

  if (success) {
    Alerting::AlertManager::instance().raise_alert(
        Alerting::AlertType::SYSTEM_RECOVERY, Alerting::AlertSeverity::INFO,
        message, "RECOVERY");
  } else {
    Alerting::AlertManager::instance().raise_alert(
        Alerting::AlertType::SYSTEM_DEGRADED, Alerting::AlertSeverity::ERROR,
        message, "RECOVERY");
  }
}

bool RecoveryController::should_abort_recovery() const {
  // Check if we've exceeded max retries
  if (state_.attempt_count >= config_.max_retries) {
    return true;
  }

  // Check timeout
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  uint64_t now = static_cast<uint64_t>(tv.tv_sec) * 1000000ULL + tv.tv_usec;

  if (state_.last_attempt_us > 0 &&
      (now - state_.last_attempt_us) > config_.timeout_us) {
    return true;
  }

  return false;
}

// ============================================================================
// RecoveryOrchestrator Implementation
// ============================================================================

RecoveryOrchestrator &RecoveryOrchestrator::instance() {
  static RecoveryOrchestrator instance;
  return instance;
}

void RecoveryOrchestrator::register_recovery_handler(
    RecoveryController::RecoveryAction action, RecoveryCallback callback) {
  handlers_[action] = callback;
}

bool RecoveryOrchestrator::execute_handlers(
    RecoveryController::RecoveryAction action) {
  auto it = handlers_.find(action);
  if (it != handlers_.end()) {
    return it->second(action);
  }

  if (default_callback_) {
    return default_callback_(action);
  }

  return false;
}

void RecoveryOrchestrator::set_default_callback(RecoveryCallback callback) {
  default_callback_ = callback;
}

} // namespace BTQuant
