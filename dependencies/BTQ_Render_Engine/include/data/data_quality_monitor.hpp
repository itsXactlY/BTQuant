#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <mutex>
#include <memory>
#include <functional>

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

// Structure to track statistics per symbol for advanced data quality checks
struct SymbolStats {
  uint64_t last_timestamp = 0;
  size_t trade_count = 0;
  uint64_t total_interval_sum = 0;
  std::vector<uint64_t> recent_intervals;  // Track recent intervals for pattern analysis

  SymbolStats() : last_timestamp(0), trade_count(0), total_interval_sum(0), recent_intervals() {}
};

private:
  mutable std::mutex mutex_;
  DataQualityMetrics metrics_;
  std::vector<DataQualityIssue> recent_issues_;
  std::unordered_map<std::string, uint64_t> last_timestamps_;  // Last timestamp per symbol
  std::unordered_map<std::string, std::vector<TradeData>> recent_trades_;  // Recent trades per symbol for duplicate detection
  std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> last_received_times_;  // For latency tracking
  std::unordered_map<std::string, SymbolStats> symbol_stats_;  // Statistics per symbol for advanced analysis

  static constexpr size_t MAX_RECENT_ISSUES = 1000;
  static constexpr size_t MAX_RECENT_TRADES = 1000;  // Number of recent trades to track for duplicate detection
  static constexpr uint64_t MAX_LATENCY_THRESHOLD_MS = 1000;  // Maximum acceptable latency in milliseconds
  static constexpr double MIN_VALID_PRICE = 0.00000001;  // Minimum valid price
  static constexpr double MAX_VALID_PRICE = 100000000.0;  // Maximum valid price
  static constexpr float MIN_VALID_VOLUME = 0.0f;  // Minimum valid volume

  AlertCallback alert_callback_;

  // Helper method to add an issue
  void add_issue(const DataQualityIssue& issue);

  // Advanced missing data detection
  void check_missing_data_for_symbol(const std::string& symbol, uint64_t current_timestamp);
};

// Global data quality monitor instance
extern DataQualityMonitor g_data_quality_monitor;

} // namespace Data
} // namespace BTQuant