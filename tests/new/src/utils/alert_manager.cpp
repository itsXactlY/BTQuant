#include "utils/alert_manager.hpp"
#include "utils/dynamic_logger.hpp"
#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/time.h>

namespace BTQuant::Alerting {

// ============================================================================
// Alert Implementation
// ============================================================================

std::string Alert::type_string() const {
  return AlertUtils::type_to_string(type);
}

std::string Alert::severity_string() const {
  return AlertUtils::severity_to_string(severity);
}

std::string Alert::to_json() const {
  std::ostringstream oss;
  oss << "{";
  oss << "\"type\":\"" << type_string() << "\",";
  oss << "\"severity\":\"" << severity_string() << "\",";
  oss << "\"message\":\"" << message << "\",";
  oss << "\"timestamp\":" << timestamp_us << ",";
  oss << "\"component\":\"" << component << "\"";
  oss << "}";
  return oss.str();
}

// ============================================================================
// AlertUtils Implementation
// ============================================================================

std::string AlertUtils::severity_to_string(AlertSeverity severity) {
  switch (severity) {
  case AlertSeverity::INFO:
    return "INFO";
  case AlertSeverity::WARNING:
    return "WARNING";
  case AlertSeverity::ERROR:
    return "ERROR";
  case AlertSeverity::CRITICAL:
    return "CRITICAL";
  default:
    return "UNKNOWN";
  }
}

std::string AlertUtils::type_to_string(AlertType type) {
  switch (type) {
  case AlertType::SHM_FILE_DELETED:
    return "SHM_FILE_DELETED";
  case AlertType::SHM_FILE_CORRUPTED:
    return "SHM_FILE_CORRUPTED";
  case AlertType::SHM_PERMISSION_DENIED:
    return "SHM_PERMISSION_DENIED";
  case AlertType::SHM_ATTACH_FAILED:
    return "SHM_ATTACH_FAILED";
  case AlertType::SHM_HEADER_INVALID:
    return "SHM_HEADER_INVALID";
  case AlertType::DETECTOR_INIT_FAILED:
    return "DETECTOR_INIT_FAILED";
  case AlertType::DETECTOR_EXECUTION_ERROR:
    return "DETECTOR_EXECUTION_ERROR";
  case AlertType::DETECTOR_TIMEOUT:
    return "DETECTOR_TIMEOUT";
  case AlertType::DATA_STREAM_INTERRUPTED:
    return "DATA_STREAM_INTERRUPTED";
  case AlertType::DATA_LOSS_DETECTED:
    return "DATA_LOSS_DETECTED";
  case AlertType::BUFFER_OVERFLOW:
    return "BUFFER_OVERFLOW";
  case AlertType::SYSTEM_DEGRADED:
    return "SYSTEM_DEGRADED";
  case AlertType::SYSTEM_RECOVERY:
    return "SYSTEM_RECOVERY";
  case AlertType::SYSTEM_SHUTDOWN:
    return "SYSTEM_SHUTDOWN";
  case AlertType::CUSTOM:
    return "CUSTOM";
  default:
    return "UNKNOWN";
  }
}

AlertSeverity AlertUtils::severity_from_string(const std::string &str) {
  if (str == "INFO")
    return AlertSeverity::INFO;
  if (str == "WARNING")
    return AlertSeverity::WARNING;
  if (str == "ERROR")
    return AlertSeverity::ERROR;
  if (str == "CRITICAL")
    return AlertSeverity::CRITICAL;
  return AlertSeverity::INFO;
}

AlertType AlertUtils::type_from_string(const std::string &str) {
  if (str == "SHM_FILE_DELETED")
    return AlertType::SHM_FILE_DELETED;
  if (str == "SHM_FILE_CORRUPTED")
    return AlertType::SHM_FILE_CORRUPTED;
  if (str == "SHM_PERMISSION_DENIED")
    return AlertType::SHM_PERMISSION_DENIED;
  if (str == "SHM_ATTACH_FAILED")
    return AlertType::SHM_ATTACH_FAILED;
  if (str == "SHM_HEADER_INVALID")
    return AlertType::SHM_HEADER_INVALID;
  if (str == "DETECTOR_INIT_FAILED")
    return AlertType::DETECTOR_INIT_FAILED;
  if (str == "DETECTOR_EXECUTION_ERROR")
    return AlertType::DETECTOR_EXECUTION_ERROR;
  if (str == "DETECTOR_TIMEOUT")
    return AlertType::DETECTOR_TIMEOUT;
  if (str == "DATA_STREAM_INTERRUPTED")
    return AlertType::DATA_STREAM_INTERRUPTED;
  if (str == "DATA_LOSS_DETECTED")
    return AlertType::DATA_LOSS_DETECTED;
  if (str == "BUFFER_OVERFLOW")
    return AlertType::BUFFER_OVERFLOW;
  if (str == "SYSTEM_DEGRADED")
    return AlertType::SYSTEM_DEGRADED;
  if (str == "SYSTEM_RECOVERY")
    return AlertType::SYSTEM_RECOVERY;
  if (str == "SYSTEM_SHUTDOWN")
    return AlertType::SYSTEM_SHUTDOWN;
  return AlertType::CUSTOM;
}

Alert AlertUtils::create_shm_deleted_alert(const std::string &path) {
  return Alert(AlertType::SHM_FILE_DELETED, AlertSeverity::CRITICAL,
               "Shared memory file has been deleted: " + path, "SHM-MONITOR");
}

Alert AlertUtils::create_shm_corrupted_alert(const std::string &reason) {
  return Alert(AlertType::SHM_FILE_CORRUPTED, AlertSeverity::CRITICAL,
               "Shared memory corruption detected: " + reason, "SHM-MONITOR");
}

Alert AlertUtils::create_detector_error_alert(const std::string &detector,
                                              const std::string &error) {
  return Alert(AlertType::DETECTOR_EXECUTION_ERROR, AlertSeverity::ERROR,
               "Detector error in " + detector + ": " + error, detector);
}

Alert AlertUtils::create_recovery_success_alert(const std::string &action) {
  return Alert(AlertType::SYSTEM_RECOVERY, AlertSeverity::INFO,
               "Recovery successful: " + action, "RECOVERY");
}

Alert AlertUtils::create_recovery_failed_alert(const std::string &action,
                                               const std::string &reason) {
  return Alert(AlertType::SYSTEM_DEGRADED, AlertSeverity::ERROR,
               "Recovery failed for " + action + ": " + reason, "RECOVERY");
}

// ============================================================================
// ConsoleAlertSink Implementation
// ============================================================================

ConsoleAlertSink::ConsoleAlertSink(AlertSeverity min_level)
    : min_level_(min_level) {}

void ConsoleAlertSink::send(const Alert &alert) {
  if (alert.severity < min_level_)
    return;

  std::ostringstream oss;
  oss << "[ALERT][" << alert.severity_string() << "][" << alert.type_string()
      << "] ";

  if (!alert.component.empty()) {
    oss << "[" << alert.component << "] ";
  }

  oss << alert.message;

  if (alert.severity == AlertSeverity::CRITICAL) {
    std::cerr << oss.str() << std::endl;
  } else {
    std::cout << oss.str() << std::endl;
  }
}

void ConsoleAlertSink::flush() {
  std::cout.flush();
  std::cerr.flush();
}

bool ConsoleAlertSink::is_enabled(AlertSeverity level) const {
  return level >= min_level_;
}

// ============================================================================
// FileAlertSink Implementation
// ============================================================================

FileAlertSink::FileAlertSink(const std::string &path, AlertSeverity min_level)
    : path_(path), min_level_(min_level) {
  open(path);
}

FileAlertSink::~FileAlertSink() { close(); }

bool FileAlertSink::open(const std::string &path) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  file_ = fopen(path.c_str(), "a");
  return file_ != nullptr;
}

void FileAlertSink::close() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (file_) {
    fflush(file_);
    fclose(file_);
    file_ = nullptr;
  }
}

void FileAlertSink::send(const Alert &alert) {
  if (alert.severity < min_level_)
    return;

  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!file_)
    return;

  // Get timestamp
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  time_t now = tv.tv_sec;
  struct tm *tm_info = localtime(&now);

  char timestamp[64];
  strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", tm_info);

  fprintf(file_, "[%s.%06ld][%s][%s][%s] %s\n", timestamp, tv.tv_usec,
          alert.severity_string().c_str(), alert.type_string().c_str(),
          alert.component.c_str(), alert.message.c_str());

  fflush(file_);
}

void FileAlertSink::flush() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (file_) {
    fflush(file_);
  }
}

bool FileAlertSink::is_enabled(AlertSeverity level) const {
  return level >= min_level_;
}

// ============================================================================
// AlertManager Implementation
// ============================================================================

AlertManager::~AlertManager() { shutdown(); }

AlertManager &AlertManager::instance() {
  static AlertManager instance;
  return instance;
}

bool AlertManager::initialize() {
  if (initialized_.exchange(true)) {
    return true; // Already initialized
  }

  // Register default console sink
  auto console_sink =
      std::make_shared<ConsoleAlertSink>(AlertSeverity::WARNING);
  register_sink(console_sink);

  Logging::DynamicLogger::instance().info("ALERT-MGR",
                                          "AlertManager initialized");
  return true;
}

void AlertManager::shutdown() {
  if (!initialized_.exchange(false)) {
    return;
  }

  // Flush and clear sinks
  clear_sinks();

  Logging::DynamicLogger::instance().info("ALERT-MGR", "AlertManager shutdown");
}

void AlertManager::register_sink(std::shared_ptr<IAlertSink> sink) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  sinks_.push_back(sink);
}

void AlertManager::unregister_sink(const std::string &name) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  sinks_.erase(std::remove_if(sinks_.begin(), sinks_.end(),
                              [&name](const auto &sink) {
                                return sink->get_name() == name;
                              }),
               sinks_.end());
}

void AlertManager::clear_sinks() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  for (auto &sink : sinks_) {
    sink->flush();
  }
  sinks_.clear();
}

void AlertManager::raise_alert(const Alert &alert) {
  if (alert.severity < min_severity_.load()) {
    return;
  }

  // Check for duplicates
  if (suppress_duplicates_.load() && is_duplicate(alert)) {
    return;
  }

  std::lock_guard<std::recursive_mutex> lock(mutex_);

  // Add to history
  alert_history_.push_back(alert);
  if (alert_history_.size() > MAX_HISTORY) {
    alert_history_.pop_front();
  }

  // Update deduplication tracking
  last_alert_time_[alert.type] = alert.timestamp_us;

  // Send to all sinks
  send_to_sinks(alert);
  total_alerts_sent_++;
}

void AlertManager::raise_alert(
    AlertType type, AlertSeverity severity, const std::string &message,
    const std::string &component,
    const std::map<std::string, std::string> &metadata) {
  Alert alert(type, severity, message, component);
  alert.metadata = metadata;
  raise_alert(alert);
}

void AlertManager::send_to_sinks(const Alert &alert) {
  for (auto &sink : sinks_) {
    if (sink->is_enabled(alert.severity)) {
      sink->send(alert);
    }
  }
}

bool AlertManager::is_duplicate(const Alert &alert) const {
  auto it = last_alert_time_.find(alert.type);
  if (it == last_alert_time_.end()) {
    return false;
  }

  uint64_t window = deduplication_window_us_.load();
  return (alert.timestamp_us - it->second) < window;
}

std::vector<Alert> AlertManager::get_recent_alerts(size_t count) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  std::vector<Alert> result;

  size_t start =
      (alert_history_.size() > count) ? (alert_history_.size() - count) : 0;
  for (size_t i = start; i < alert_history_.size(); ++i) {
    result.push_back(alert_history_[i]);
  }

  return result;
}

std::vector<Alert> AlertManager::get_alerts_by_type(AlertType type) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  std::vector<Alert> result;

  for (const auto &alert : alert_history_) {
    if (alert.type == type) {
      result.push_back(alert);
    }
  }

  return result;
}

std::vector<Alert>
AlertManager::get_alerts_by_severity(AlertSeverity severity) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  std::vector<Alert> result;

  for (const auto &alert : alert_history_) {
    if (alert.severity == severity) {
      result.push_back(alert);
    }
  }

  return result;
}

uint64_t
AlertManager::get_alerts_by_severity_count(AlertSeverity severity) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  uint64_t count = 0;

  for (const auto &alert : alert_history_) {
    if (alert.severity == severity) {
      count++;
    }
  }

  return count;
}

void AlertManager::clear_expired_alerts() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  // Keep only recent alerts (older than 1 hour)
  uint64_t one_hour_ago = []() -> uint64_t {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return static_cast<uint64_t>(tv.tv_sec) * 1000000ULL + tv.tv_usec -
           3600000000ULL;
  }();

  while (!alert_history_.empty() &&
         alert_history_.front().timestamp_us < one_hour_ago) {
    alert_history_.pop_front();
  }
}

void AlertManager::set_deduplication_window_us(uint64_t window_us) {
  deduplication_window_us_.store(window_us);
}

} // namespace BTQuant::Alerting
