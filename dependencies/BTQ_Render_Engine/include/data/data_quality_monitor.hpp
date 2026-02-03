#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <mutex>
#include <memory>
#include <functional>
#include <unordered_set>

#include "TradeData.h"

namespace BTQuant {
namespace Data {

// Data quality issue types
enum class DataQualityIssueType {
  MISSING_DATA,
  DUPLICATE_TRADE,
  OUT_OF_ORDER_TIMESTAMP,
  LATENCY_ISSUE,
  INVALID_PRICE,
  INVALID_VOLUME,
  MISSING_FIELD
};

// Data quality issue structure
struct DataQualityIssue {
  DataQualityIssueType type;
  std::string symbol;
  uint64_t timestamp;
  std::string description;
  double severity; // 0.0 to 1.0, where 1.0 is most severe
  std::chrono::high_resolution_clock::time_point detection_time;
  
  DataQualityIssue(DataQualityIssueType t, const std::string& sym, uint64_t ts, 
                   const std::string& desc, double sev = 0.5)
      : type(t), symbol(sym), timestamp(ts), description(desc), severity(sev),
        detection_time(std::chrono::high_resolution_clock::now()) {}
};

// Data quality metrics
struct DataQualityMetrics {
  size_t total_trades_processed = 0;
  size_t missing_data_issues = 0;
  size_t duplicate_trade_issues = 0;
  size_t out_of_order_timestamp_issues = 0;
  size_t latency_issues = 0;
  size_t invalid_price_issues = 0;
  size_t invalid_volume_issues = 0;
  size_t missing_field_issues = 0;
  double average_latency_ms = 0.0;
  uint64_t last_timestamp = 0;
  std::chrono::high_resolution_clock::time_point last_update_time;
  
  DataQualityMetrics() : last_update_time(std::chrono::high_resolution_clock::now()) {}
};

// Data quality monitor class
class DataQualityMonitor {
public:
  DataQualityMonitor();
  ~DataQualityMonitor() = default;

  // Process a single trade and check for quality issues
  std::vector<DataQualityIssue> process_trade(const TradeData& trade, const std::string& symbol);

  // Process a batch of trades and check for quality issues
  std::vector<DataQualityIssue> process_trades_batch(const std::vector<TradeData>& trades, 
                                                     const std::string& symbol);

  // Check for missing data based on expected intervals
  void check_missing_data(const std::string& symbol, uint64_t current_timestamp, 
                          uint64_t expected_interval_ms);

  // Check for duplicate trades
  bool is_duplicate_trade(const TradeData& trade, const std::string& symbol);

  // Check for out-of-order timestamps
  bool is_out_of_order_timestamp(const TradeData& trade, const std::string& symbol);

  // Check for latency issues
  void check_latency_issue(const TradeData& trade, const std::string& symbol);

  // Validate price and volume values
  bool validate_trade_values(const TradeData& trade);

  // Check for missing or invalid fields
  void check_missing_fields(const TradeData& trade, const std::string& symbol, uint64_t timestamp);

  // Get current data quality metrics
  DataQualityMetrics get_metrics() const;

  // Get recent data quality issues
  std::vector<DataQualityIssue> get_recent_issues(size_t limit = 100) const;

  // Reset all metrics and issues
  void reset();

  // Alert callback function type
  using AlertCallback = std::function<void(const DataQualityIssue&)>;

  // Set callback function for when issues are detected
  void set_alert_callback(AlertCallback callback);

  // Generate a summary of current data quality status
  std::string get_quality_summary() const;

  // Get recent high-severity issues
  std::vector<DataQualityIssue> get_high_severity_issues(double min_severity_threshold = 0.7) const;

  // Configuration methods for adjusting thresholds
  void set_missing_data_threshold(uint64_t threshold_ms) { missing_data_threshold_ms_ = threshold_ms; }
  uint64_t get_missing_data_threshold() const { return missing_data_threshold_ms_; }

  void set_duplicate_check_window(uint64_t window_ms) { duplicate_check_window_ms_ = window_ms; }
  uint64_t get_duplicate_check_window() const { return duplicate_check_window_ms_; }

  void set_latency_alert_threshold(uint64_t threshold_ms) { latency_alert_threshold_ms_ = threshold_ms; }
  uint64_t get_latency_alert_threshold() const { return latency_alert_threshold_ms_; }

  void set_out_of_order_tolerance(uint64_t tolerance_ms) { out_of_order_tolerance_ms_ = tolerance_ms; }
  uint64_t get_out_of_order_tolerance() const { return out_of_order_tolerance_ms_; }

  // Enhanced alerting methods
  void trigger_alert(const std::string& symbol, DataQualityIssueType issue_type, const std::string& description, double severity = 0.5);
  void trigger_data_quality_alerts();  // Triggers alerts based on current metrics

  // Methods for configuring alert destinations
  void set_console_alerts_enabled(bool enabled) { console_alerts_enabled_ = enabled; }
  void set_file_logging_enabled(bool enabled) { file_logging_enabled_ = enabled; }
  void set_external_alert_callback(std::function<void(const DataQualityIssue&)> callback) { external_alert_callback_ = callback; }

  // Method to send alerts to external monitoring systems
  void send_external_alert(const DataQualityIssue& issue);

  // Method to alert users to data problems
  void alert_user_to_data_problems(const std::string& symbol, const std::string& problem_description, double severity = 0.5);

  // NEW: Method to specifically notify users of data problems in a clear way
  void notify_users_of_data_problem(const std::string& symbol, const std::string& problem_description,
                                   double severity, DataQualityIssueType issue_type);

  // NEW: Enhanced method to alert users with additional context
  void alert_user_to_data_problems_with_context(const std::string& symbol,
                                              const std::string& problem_description,
                                              double severity,
                                              const std::string& source_component = "",
                                              const std::string& additional_context = "");

  // Enhanced alerting methods for specific data quality issues
  void alert_on_missing_data(const std::string& symbol, uint64_t expected_time, uint64_t actual_time);
  void alert_on_duplicate_trade(const TradeData& trade, const std::string& symbol);
  void alert_on_out_of_order_timestamp(const TradeData& trade, const std::string& symbol, uint64_t last_timestamp);
  void alert_on_latency_issue(const TradeData& trade, const std::string& symbol, int64_t latency_ms);

  // Method to trigger visual alerts for high severity issues
  void trigger_visual_alert(const std::string& symbol, const std::string& problem_description);

  // Method to get alert counts by type
  std::unordered_map<DataQualityIssueType, size_t> get_alert_counts_by_type() const;

  // Generate a comprehensive alert report
  void generate_comprehensive_alert_report();

  // Monitor the health of a specific data stream
  void monitor_data_stream_health(const std::string& symbol);

  // NEW: Enhanced monitoring methods for specific data quality patterns
  void check_missing_data_patterns(const std::string& symbol);
  void check_duplicate_trade_patterns(const std::string& symbol);
  void check_out_of_order_timestamp_patterns(const std::string& symbol);
  void check_latency_issue_patterns(const std::string& symbol);

  // Send notifications to UI components
  void send_ui_notification(const DataQualityIssue& issue);

  // Get a user-friendly summary of data quality issues
  std::string get_user_friendly_summary() const;

  // Get a comprehensive summary of data quality issues
  std::string get_comprehensive_summary() const;

  // Public methods for enhanced data quality monitoring
  uint64_t calculate_safe_time_diff(uint64_t current, uint64_t previous) const;
  bool are_trades_equivalent(const TradeData& trade1, const TradeData& trade2,
                           double price_tolerance = 0.000001,
                           double volume_tolerance = 0.0001f) const;

private:
  // Structure to track statistics per symbol for advanced data quality checks
  struct SymbolStats {
    uint64_t last_timestamp = 0;
    size_t trade_count = 0;
    uint64_t total_interval_sum = 0;
    std::vector<uint64_t> recent_intervals;  // Track recent intervals for pattern analysis

    SymbolStats() : last_timestamp(0), trade_count(0), total_interval_sum(0), recent_intervals() {}
  };
  mutable std::mutex mutex_;
  DataQualityMetrics metrics_;
  std::vector<DataQualityIssue> recent_issues_;
  std::unordered_map<std::string, uint64_t> last_timestamps_;  // Last timestamp per symbol
  std::unordered_map<std::string, std::vector<TradeData>> recent_trades_;  // Recent trades per symbol for duplicate detection
  std::unordered_map<std::string, std::unordered_set<size_t>> recent_trade_hashes_;  // Hashes of recent trades for faster duplicate detection
  std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> last_received_times_;  // For latency tracking
  std::unordered_map<std::string, SymbolStats> symbol_stats_;  // Statistics per symbol for advanced analysis
  std::unordered_map<std::string, std::vector<uint64_t>> recent_delays_;  // Recent delays for latency trend analysis

  // Alert burst tracking
  std::vector<std::chrono::high_resolution_clock::time_point> recent_alert_times_;
  std::unordered_map<DataQualityIssueType, size_t> alert_counts_by_type_;  // Counts of alerts by type

  // Thresholds for data quality monitoring
  uint64_t missing_data_threshold_ms_;
  uint64_t duplicate_check_window_ms_;
  uint64_t latency_alert_threshold_ms_;
  uint64_t out_of_order_tolerance_ms_;

  static constexpr size_t MAX_RECENT_ISSUES = 1000;
  static constexpr size_t MAX_RECENT_TRADES = 1000;  // Number of recent trades to track for duplicate detection
  static constexpr uint64_t MAX_LATENCY_THRESHOLD_MS = 1000;  // Maximum acceptable latency in milliseconds
  static constexpr double MIN_VALID_PRICE = 0.00000001;  // Minimum valid price
  static constexpr double MAX_VALID_PRICE = 100000000.0;  // Maximum valid price
  static constexpr float MIN_VALID_VOLUME = 0.0f;  // Minimum valid volume

  AlertCallback alert_callback_;

  // Alert configuration
  bool console_alerts_enabled_ = true;
  bool file_logging_enabled_ = false;
  std::function<void(const DataQualityIssue&)> external_alert_callback_ = nullptr;

  // Helper method to add an issue
  void add_issue(const DataQualityIssue& issue);

  // Advanced missing data detection
  void check_missing_data_for_symbol(const std::string& symbol, uint64_t current_timestamp);

  // Helper method to calculate median interval
  uint64_t get_median_interval(const std::vector<uint64_t>& intervals) const;

  // Helper method to send critical alerts
  void send_critical_alert(const DataQualityIssue& issue);

  // Method to check for alert bursts (many alerts in a short time period)
  void check_alert_bursts(const DataQualityIssue& issue);
};

// Global data quality monitor instance
extern DataQualityMonitor g_data_quality_monitor;

} // namespace Data
} // namespace BTQuant