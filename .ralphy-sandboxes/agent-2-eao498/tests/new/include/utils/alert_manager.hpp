#pragma once

#include <string>
#include <vector>
#include <memory>
#include <deque>
#include <mutex>
#include <atomic>
#include <map>
#include <functional>
#include <chrono>

namespace BTQuant::Alerting {

/**
 * Alert severity levels
 */
enum class AlertSeverity {
    INFO = 0,       // Informational
    WARNING = 1,    // Warning conditions
    ERROR = 2,      // Error conditions
    CRITICAL = 3    // Critical conditions requiring immediate attention
};

/**
 * Alert types for the manipulation detector system
 */
enum class AlertType {
    // Shared memory related alerts
    SHM_FILE_DELETED = 0,
    SHM_FILE_CORRUPTED = 1,
    SHM_PERMISSION_DENIED = 2,
    SHM_ATTACH_FAILED = 3,
    SHM_HEADER_INVALID = 4,
    
    // Detector related alerts
    DETECTOR_INIT_FAILED = 10,
    DETECTOR_EXECUTION_ERROR = 11,
    DETECTOR_TIMEOUT = 12,
    
    // Data related alerts
    DATA_STREAM_INTERRUPTED = 20,
    DATA_LOSS_DETECTED = 21,
    BUFFER_OVERFLOW = 22,
    
    // System related alerts
    SYSTEM_DEGRADED = 30,
    SYSTEM_RECOVERY = 31,
    SYSTEM_SHUTDOWN = 32,
    
    // Custom alerts
    CUSTOM = 100
};

/**
 * Alert structure containing all alert information
 */
struct Alert {
    AlertType type;
    AlertSeverity severity;
    std::string message;
    uint64_t timestamp_us;
    std::string component;
    std::map<std::string, std::string> metadata;
    
    // Helper methods
    std::string type_string() const;
    std::string severity_string() const;
    std::string to_json() const;
    
    Alert()
        : type(AlertType::CUSTOM)
        , severity(AlertSeverity::INFO)
        , timestamp_us(0)
    {}
    
    Alert(AlertType t, AlertSeverity s, const std::string& msg, const std::string& comp = "")
        : type(t)
        , severity(s)
        , message(msg)
        , component(comp)
    {
        auto now = std::chrono::system_clock::now();
        timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count();
    }
};

/**
 * Alert sink interface - implementations handle alert delivery
 */
class IAlertSink {
public:
    virtual ~IAlertSink() = default;
    virtual void send(const Alert& alert) = 0;
    virtual void flush() = 0;
    virtual bool is_enabled(AlertSeverity level) const = 0;
    virtual std::string get_name() const = 0;
};

/**
 * Console alert sink - writes alerts to stdout/stderr
 */
class ConsoleAlertSink : public IAlertSink {
public:
    explicit ConsoleAlertSink(AlertSeverity min_level = AlertSeverity::WARNING);
    void send(const Alert& alert) override;
    void flush() override;
    bool is_enabled(AlertSeverity level) const override;
    std::string get_name() const override { return "console"; }
    
private:
    AlertSeverity min_level_;
};

/**
 * File alert sink - writes alerts to a log file
 */
class FileAlertSink : public IAlertSink {
public:
    explicit FileAlertSink(const std::string& path, AlertSeverity min_level = AlertSeverity::INFO);
    ~FileAlertSink() override;
    
    void send(const Alert& alert) override;
    void flush() override;
    bool is_enabled(AlertSeverity level) const override;
    std::string get_name() const override { return "file"; }
    
    bool open(const std::string& path);
    void close();
    
private:
    std::string path_;
    FILE* file_ = nullptr;
    AlertSeverity min_level_;
    std::mutex mutex_;
};

/**
 * AlertManager - central alert management system
 */
class AlertManager {
public:
    static AlertManager& instance();
    
    // Initialization
    bool initialize();
    void shutdown();
    
    // Alert creation and raising
    void raise_alert(const Alert& alert);
    void raise_alert(AlertType type, AlertSeverity severity, const std::string& message,
                     const std::string& component = "", 
                     const std::map<std::string, std::string>& metadata = {});
    
    // Sink management
    void register_sink(std::shared_ptr<IAlertSink> sink);
    void unregister_sink(const std::string& name);
    void clear_sinks();
    
    // Alert history
    std::vector<Alert> get_recent_alerts(size_t count = 100) const;
    std::vector<Alert> get_alerts_by_type(AlertType type) const;
    std::vector<Alert> get_alerts_by_severity(AlertSeverity severity) const;
    size_t get_alert_count() const { return alert_history_.size(); }
    
    // Alert filtering and deduplication
    void set_deduplication_window_us(uint64_t window_us);
    void clear_expired_alerts();
    void suppress_duplicates(bool suppress) { suppress_duplicates_.store(suppress); }
    
    // Statistics
    uint64_t get_total_alerts_sent() const { return total_alerts_sent_.load(); }
    uint64_t get_alerts_by_severity_count(AlertSeverity severity) const;
    
    // Configuration
    void set_min_severity(AlertSeverity level) { min_severity_.store(level); }
    AlertSeverity get_min_severity() const { return min_severity_.load(); }
    
private:
    AlertManager() = default;
    ~AlertManager();
    
    AlertManager(const AlertManager&) = delete;
    AlertManager& operator=(const AlertManager&) = delete;
    
    void send_to_sinks(const Alert& alert);
    bool is_duplicate(const Alert& alert) const;
    
    std::vector<std::shared_ptr<IAlertSink>> sinks_;
    std::deque<Alert> alert_history_;
    mutable std::mutex mutex_;
    
    static constexpr size_t MAX_HISTORY = 1000;
    
    std::atomic<bool> initialized_{false};
    std::atomic<uint64_t> total_alerts_sent_{0};
    std::atomic<AlertSeverity> min_severity_{AlertSeverity::INFO};
    std::atomic<bool> suppress_duplicates_{false};
    std::atomic<uint64_t> deduplication_window_us_{5'000'000}; // 5 seconds
    
    // Last alert tracking for deduplication
    mutable std::map<AlertType, uint64_t> last_alert_time_;
};

/**
 * Alert utilities
 */
class AlertUtils {
public:
    static std::string severity_to_string(AlertSeverity severity);
    static std::string type_to_string(AlertType type);
    static AlertSeverity severity_from_string(const std::string& str);
    static AlertType type_from_string(const std::string& str);
    
    // Create standard alerts
    static Alert create_shm_deleted_alert(const std::string& path);
    static Alert create_shm_corrupted_alert(const std::string& reason);
    static Alert create_detector_error_alert(const std::string& detector, const std::string& error);
    static Alert create_recovery_success_alert(const std::string& action);
    static Alert create_recovery_failed_alert(const std::string& action, const std::string& reason);
};

// Convenience macro for raising alerts
#define BTQ_ALERT(type, severity, message, ...) \
    BTQuant::Alerting::AlertManager::instance().raise_alert( \
        type, severity, message, __VA_ARGS__)

} // namespace BTQuant::Alerting
