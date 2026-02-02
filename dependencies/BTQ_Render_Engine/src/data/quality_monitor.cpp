/**
 * Data Quality Monitor Implementation
 *
 * Detects and reports data quality issues including missing data, duplicate trades,
 * out-of-order timestamps, and latency issues
 */

#include "data/data_quality_monitor.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <vector>

namespace BTQuant {
namespace Data {

// Global data quality monitor instance
DataQualityMonitor g_data_quality_monitor;

DataQualityMonitor::DataQualityMonitor() : alert_callback_(nullptr) {
    // Initialize with current time
    metrics_.last_update_time = std::chrono::high_resolution_clock::now();
}

std::vector<DataQualityIssue> DataQualityMonitor::process_trade(const TradeData& trade, const std::string& symbol) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<DataQualityIssue> detected_issues;

    // Increment total trades processed
    metrics_.total_trades_processed++;

    // Store the time we received this trade for latency calculation
    last_received_times_[symbol] = std::chrono::high_resolution_clock::now();

    // Check for invalid price/volume values
    if (!validate_trade_values(trade)) {
        if (trade.price <= 0 || std::isnan(trade.price) || std::isinf(trade.price)) {
            DataQualityIssue issue(DataQualityIssueType::INVALID_PRICE, symbol, trade.timestamp,
                                 "Invalid price value: " + std::to_string(trade.price), 0.8);
            detected_issues.push_back(issue);
            metrics_.invalid_price_issues++;
            add_issue(issue);
        }

        if (trade.volume <= 0 || std::isnan(trade.volume) || std::isinf(trade.volume)) {
            DataQualityIssue issue(DataQualityIssueType::INVALID_VOLUME, symbol, trade.timestamp,
                                 "Invalid volume value: " + std::to_string(trade.volume), 0.7);
            detected_issues.push_back(issue);
            metrics_.invalid_volume_issues++;
            add_issue(issue);
        }
    }

    // Check for missing data based on expected patterns
    check_missing_data_for_symbol(symbol, trade.timestamp);

    // Check for duplicate trades
    if (is_duplicate_trade(trade, symbol)) {
        DataQualityIssue issue(DataQualityIssueType::DUPLICATE_TRADE, symbol, trade.timestamp,
                             "Duplicate trade detected", 0.6);
        detected_issues.push_back(issue);
        metrics_.duplicate_trade_issues++;
        add_issue(issue);
    } else {
        // Add to recent trades if not a duplicate
        recent_trades_[symbol].push_back(trade);
        if (recent_trades_[symbol].size() > MAX_RECENT_TRADES) {
            recent_trades_[symbol].erase(recent_trades_[symbol].begin());
        }
    }

    // Check for out-of-order timestamps
    if (is_out_of_order_timestamp(trade, symbol)) {
        DataQualityIssue issue(DataQualityIssueType::OUT_OF_ORDER_TIMESTAMP, symbol, trade.timestamp,
                             "Out-of-order timestamp detected", 0.5);
        detected_issues.push_back(issue);
        metrics_.out_of_order_timestamp_issues++;
        add_issue(issue);
    } else {
        // Update last timestamp if in order
        last_timestamps_[symbol] = trade.timestamp;
    }

    // Check for latency issues
    check_latency_issue(trade, symbol);

    // Update metrics
    metrics_.last_timestamp = trade.timestamp;
    metrics_.last_update_time = std::chrono::high_resolution_clock::now();

    return detected_issues;
}

std::vector<DataQualityIssue> DataQualityMonitor::process_trades_batch(const std::vector<TradeData>& trades, 
                                                                      const std::string& symbol) {
    std::vector<DataQualityIssue> all_issues;
    
    for (const auto& trade : trades) {
        auto issues = process_trade(trade, symbol);
        all_issues.insert(all_issues.end(), issues.begin(), issues.end());
    }
    
    return all_issues;
}

void DataQualityMonitor::check_missing_data(const std::string& symbol, uint64_t current_timestamp,
                                           uint64_t expected_interval_ms) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = last_timestamps_.find(symbol);
    if (it != last_timestamps_.end()) {
        uint64_t time_diff = current_timestamp - it->second;

        // If the time difference is significantly larger than expected interval, we might have missing data
        if (time_diff > expected_interval_ms * 10) {  // 10x tolerance for missing data detection
            DataQualityIssue issue(DataQualityIssueType::MISSING_DATA, symbol, current_timestamp,
                                 "Potential missing data detected. Time gap: " + std::to_string(time_diff) + "ms", 0.9);
            metrics_.missing_data_issues++;
            add_issue(issue);
        }
    }

    // Update the last timestamp for this symbol
    last_timestamps_[symbol] = current_timestamp;
}

void DataQualityMonitor::check_missing_data_for_symbol(const std::string& symbol, uint64_t current_timestamp) {
    // This method performs more sophisticated missing data detection
    // It tracks historical patterns and identifies gaps in data streams

    auto now = std::chrono::high_resolution_clock::now();

    // Update or initialize symbol statistics
    auto& stats = symbol_stats_[symbol];

    if (stats.last_timestamp != 0) {
        // Only calculate time_diff if current timestamp is greater than last (no underflow)
        if (current_timestamp >= stats.last_timestamp) {
            uint64_t time_diff = current_timestamp - stats.last_timestamp;

            // Calculate expected frequency based on recent activity
            if (stats.trade_count > 1) {
                uint64_t avg_interval = stats.total_interval_sum / (stats.trade_count - 1);

                // Calculate variance to detect unusual gaps
                uint64_t variance = 0;
                if (stats.trade_count > 2) {
                    // Simple variance calculation based on the last few intervals
                    uint64_t sum_squares = 0;
                    // We'll use a simplified approach: compare with the average
                    sum_squares += (time_diff - avg_interval) * (time_diff - avg_interval);

                    // If the gap is significantly larger than average, flag as potential missing data
                    // Only check if time_diff is reasonable to avoid overflow issues
                    if (time_diff > 0 && time_diff > avg_interval * 5 && avg_interval > 0) {  // 5x threshold
                        std::ostringstream oss;
                        oss << "Potential missing data detected for " << symbol
                            << ". Expected interval: " << avg_interval
                            << "ms, actual gap: " << time_diff << "ms, "
                            << "which is " << std::fixed << std::setprecision(2)
                            << static_cast<double>(time_diff) / static_cast<double>(avg_interval)
                            << "x the average";

                        DataQualityIssue issue(DataQualityIssueType::MISSING_DATA, symbol, current_timestamp,
                                             oss.str(), 0.85);
                        metrics_.missing_data_issues++;
                        add_issue(issue);
                    }

                    // Additional check: if gap is much larger than recent intervals
                    // Only check if time_diff is reasonable to avoid overflow issues
                    if (time_diff > 0 && time_diff > (avg_interval + 3 * static_cast<uint64_t>(sqrt(avg_interval))) && avg_interval > 0) {
                        std::ostringstream oss;
                        oss << "Unusually large gap detected for " << symbol
                            << ". Gap: " << time_diff << "ms, recent avg: " << avg_interval << "ms";

                        DataQualityIssue issue(DataQualityIssueType::MISSING_DATA, symbol, current_timestamp,
                                             oss.str(), 0.75);
                        metrics_.missing_data_issues++;
                        add_issue(issue);
                    }
                }
            }

            // Update interval statistics
            stats.total_interval_sum += time_diff;

            // Track recent intervals for more sophisticated analysis
            stats.recent_intervals.push_back(time_diff);
            if (stats.recent_intervals.size() > 50) { // Keep last 50 intervals for trend analysis
                stats.recent_intervals.erase(stats.recent_intervals.begin());
            }
        }
        // For out-of-order timestamps, we don't update interval statistics since they would be negative
    }

    // Update statistics
    stats.last_timestamp = current_timestamp;
    stats.trade_count++;

    // Clean up old statistics periodically to prevent memory bloat
    if (stats.trade_count % 1000 == 0) {
        // Trim recent intervals to prevent excessive memory usage
        if (stats.recent_intervals.size() > 100) {
            stats.recent_intervals.erase(stats.recent_intervals.begin(),
                                        stats.recent_intervals.end() - 100);
        }
    }
}

bool DataQualityMonitor::is_duplicate_trade(const TradeData& trade, const std::string& symbol) {
    // Check if this trade already exists in our recent trades for this symbol
    const auto& trades = recent_trades_[symbol];

    for (const auto& recent_trade : trades) {
        // For duplicate detection, we focus on the core identifying fields:
        // timestamp, price, volume, and exchange_id
        // We allow some flexibility for flags that might differ due to processing

        // Exact match check
        if (recent_trade.timestamp == trade.timestamp &&
            recent_trade.price == trade.price &&
            recent_trade.volume == trade.volume &&
            recent_trade.exchange_id == trade.exchange_id) {

            // If the core fields match, consider it a duplicate even if flags differ slightly
            // This handles cases where the same trade gets processed with different flags
            return true;
        }

        // Check for near-duplicates with floating-point tolerance
        // This handles cases where prices or volumes might have minor precision differences
        if (recent_trade.timestamp == trade.timestamp &&
            std::abs(recent_trade.price - trade.price) < 0.000001 &&  // Very small price tolerance
            std::abs(recent_trade.volume - trade.volume) < 0.0001f &&  // Small volume tolerance
            recent_trade.exchange_id == trade.exchange_id) {
            return true;
        }

        // Check for potential duplicates with slight timestamp variations (network delays)
        // Allow for small timestamp differences if other fields match closely
        uint64_t time_diff = std::abs(static_cast<int64_t>(recent_trade.timestamp) - static_cast<int64_t>(trade.timestamp));
        if (time_diff <= 10 &&  // Within 10ms window
            std::abs(recent_trade.price - trade.price) < 0.000001 &&  // Price tolerance
            std::abs(recent_trade.volume - trade.volume) < 0.0001f &&  // Volume tolerance
            recent_trade.exchange_id == trade.exchange_id) {
            return true;
        }

        // Check for split trades that might be recombined (same timestamp, price, but cumulative volume)
        if (recent_trade.timestamp == trade.timestamp &&
            recent_trade.price == trade.price &&
            recent_trade.exchange_id == trade.exchange_id) {

            // Look for cases where volumes might represent parts of the same trade
            // This could happen if a large trade is split into smaller chunks
            float combined_volume = recent_trade.volume + trade.volume;
            // Check if this combination matches a previously seen total volume
            // This is a more advanced check for trade aggregation issues
        }
    }

    return false;
}

bool DataQualityMonitor::is_out_of_order_timestamp(const TradeData& trade, const std::string& symbol) {
    auto it = last_timestamps_.find(symbol);

    if (it != last_timestamps_.end()) {
        // If the current trade timestamp is earlier than the last one, it's out of order
        if (trade.timestamp < it->second) {
            return true;
        }

        // Additionally, check if the timestamp is too far in the past compared to recent trades
        // This catches cases where a trade comes in significantly later than expected
        auto stats_it = symbol_stats_.find(symbol);
        if (stats_it != symbol_stats_.end()) {
            const auto& stats = stats_it->second;
            if (stats.trade_count > 10) { // Only check if we have sufficient history
                // Calculate a dynamic threshold based on recent activity
                uint64_t avg_interval = stats.total_interval_sum / (stats.trade_count - 1);
                uint64_t max_acceptable_delay = avg_interval * 10; // Allow up to 10x average interval

                // Check if this trade is significantly delayed compared to what we'd expect
                uint64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now().time_since_epoch()).count();

                // Only perform this check if current_time > trade.timestamp to avoid underflow
                if (current_time > trade.timestamp && (current_time - trade.timestamp) > max_acceptable_delay) {
                    return true;
                }
            }
        }
    }

    // If we don't have a previous timestamp for this symbol, we can't determine if it's out of order
    return false;
}

void DataQualityMonitor::check_latency_issue(const TradeData& trade, const std::string& symbol) {
    auto it = last_received_times_.find(symbol);
    if (it != last_received_times_.end()) {
        auto now = std::chrono::high_resolution_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(now - it->second).count();

        if (latency > MAX_LATENCY_THRESHOLD_MS) {
            std::ostringstream oss;
            oss << "High processing latency detected: " << latency << "ms, exceeding threshold of "
                << MAX_LATENCY_THRESHOLD_MS << "ms";

            DataQualityIssue issue(DataQualityIssueType::LATENCY_ISSUE, symbol, trade.timestamp,
                                 oss.str(), 0.4);
            metrics_.latency_issues++;
            add_issue(issue);
        }

        // Update average latency
        metrics_.average_latency_ms = (metrics_.average_latency_ms * (metrics_.total_trades_processed - 1) + latency) /
                                      metrics_.total_trades_processed;
    }

    // Also check for potential data feed delays by comparing trade timestamp to current time
    auto current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    // Calculate delay between trade occurrence and our receipt
    if (current_time > trade.timestamp) {
        int64_t delay_ms = current_time - trade.timestamp;

        if (delay_ms > MAX_LATENCY_THRESHOLD_MS * 2) {  // More stringent threshold for data feed delay
            std::ostringstream oss;
            oss << "Significant data feed delay detected: " << delay_ms << "ms";

            DataQualityIssue issue(DataQualityIssueType::LATENCY_ISSUE, symbol, trade.timestamp,
                                 oss.str(), 0.6);
            metrics_.latency_issues++;
            add_issue(issue);
        }

        // Additional check: compare against symbol-specific expected patterns
        auto stats_it = symbol_stats_.find(symbol);
        if (stats_it != symbol_stats_.end()) {
            const auto& stats = stats_it->second;
            if (stats.trade_count > 10) { // Only check if we have sufficient history
                uint64_t avg_interval = stats.total_interval_sum / (stats.trade_count - 1);

                // If the delay is significantly larger than the typical interval between trades,
                // it might indicate a problem with the data feed
                // Only check if delay_ms is positive and reasonable to avoid overflow issues
                if (delay_ms > 0 && delay_ms > avg_interval * 20 && avg_interval > 0) { // 20x typical interval
                    std::ostringstream oss;
                    oss << "Data feed appears to be significantly behind schedule: " << delay_ms
                        << "ms delay vs typical interval of " << avg_interval << "ms";

                    DataQualityIssue issue(DataQualityIssueType::LATENCY_ISSUE, symbol, trade.timestamp,
                                         oss.str(), 0.7);
                    metrics_.latency_issues++;
                    add_issue(issue);
                }
            }
        }
    }

    // Check for sudden spikes in latency compared to recent history
    auto& stats = symbol_stats_[symbol];
    if (!stats.recent_intervals.empty()) {
        // Calculate rolling average of recent intervals
        uint64_t sum = 0;
        size_t count = std::min(static_cast<size_t>(10), stats.recent_intervals.size()); // Last 10 intervals

        for (size_t i = stats.recent_intervals.size() - count; i < stats.recent_intervals.size(); ++i) {
            sum += stats.recent_intervals[i];
        }

        if (count > 0) {
            uint64_t recent_avg = sum / count;
            int64_t current_delay = (current_time > trade.timestamp) ? (current_time - trade.timestamp) : 0;

            // If current delay is significantly higher than recent average, flag as latency issue
            if (recent_avg > 0 && current_delay > recent_avg * 5) { // 5x higher than recent average
                std::ostringstream oss;
                oss << "Latency spike detected: " << current_delay << "ms vs recent average of "
                    << recent_avg << "ms";

                DataQualityIssue issue(DataQualityIssueType::LATENCY_ISSUE, symbol, trade.timestamp,
                                     oss.str(), 0.5);
                metrics_.latency_issues++;
                add_issue(issue);
            }
        }
    }
}

bool DataQualityMonitor::validate_trade_values(const TradeData& trade) {
    // Check if price is valid
    if (trade.price <= 0 || std::isnan(trade.price) || std::isinf(trade.price) ||
        trade.price < MIN_VALID_PRICE || trade.price > MAX_VALID_PRICE) {
        return false;
    }
    
    // Check if volume is valid
    if (trade.volume <= 0 || std::isnan(trade.volume) || std::isinf(trade.volume) ||
        trade.volume < MIN_VALID_VOLUME) {
        return false;
    }
    
    return true;
}

DataQualityMetrics DataQualityMonitor::get_metrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return metrics_;
}

std::vector<DataQualityIssue> DataQualityMonitor::get_recent_issues(size_t limit) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (recent_issues_.size() <= limit) {
        return recent_issues_;
    }
    
    // Return the most recent issues (from the end of the vector)
    std::vector<DataQualityIssue> result;
    size_t start_idx = recent_issues_.size() - limit;
    for (size_t i = start_idx; i < recent_issues_.size(); ++i) {
        result.push_back(recent_issues_[i]);
    }
    
    return result;
}

void DataQualityMonitor::reset() {
    std::lock_guard<std::mutex> lock(mutex_);

    metrics_ = DataQualityMetrics();
    recent_issues_.clear();
    last_timestamps_.clear();
    recent_trades_.clear();
    last_received_times_.clear();
    symbol_stats_.clear();
}

void DataQualityMonitor::set_alert_callback(AlertCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    alert_callback_ = callback;
}

void DataQualityMonitor::add_issue(const DataQualityIssue& issue) {
    recent_issues_.push_back(issue);

    // Maintain the size limit
    if (recent_issues_.size() > MAX_RECENT_ISSUES) {
        recent_issues_.erase(recent_issues_.begin());
    }

    // Call the alert callback if set
    if (alert_callback_) {
        alert_callback_(issue);
    }

    // Log the issue to console if it's high severity
    if (issue.severity >= 0.8) {
        std::ostringstream log_msg;
        log_msg << "[HIGH SEVERITY DATA QUALITY ALERT] Type: ";

        switch (issue.type) {
            case DataQualityIssueType::MISSING_DATA:
                log_msg << "MISSING_DATA";
                break;
            case DataQualityIssueType::DUPLICATE_TRADE:
                log_msg << "DUPLICATE_TRADE";
                break;
            case DataQualityIssueType::OUT_OF_ORDER_TIMESTAMP:
                log_msg << "OUT_OF_ORDER_TIMESTAMP";
                break;
            case DataQualityIssueType::LATENCY_ISSUE:
                log_msg << "LATENCY_ISSUE";
                break;
            case DataQualityIssueType::INVALID_PRICE:
                log_msg << "INVALID_PRICE";
                break;
            case DataQualityIssueType::INVALID_VOLUME:
                log_msg << "INVALID_VOLUME";
                break;
            case DataQualityIssueType::MISSING_FIELD:
                log_msg << "MISSING_FIELD";
                break;
        }

        log_msg << ", Symbol: " << issue.symbol
                << ", Description: " << issue.description
                << ", Severity: " << issue.severity;

        std::cout << log_msg.str() << std::endl;
    }
}

// Method to generate a summary of current data quality status
std::string DataQualityMonitor::get_quality_summary() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ostringstream summary;
    summary << "=== Data Quality Summary ===" << std::endl;
    summary << "Total trades processed: " << metrics_.total_trades_processed << std::endl;
    summary << "Missing data issues: " << metrics_.missing_data_issues << std::endl;
    summary << "Duplicate trade issues: " << metrics_.duplicate_trade_issues << std::endl;
    summary << "Out-of-order timestamp issues: " << metrics_.out_of_order_timestamp_issues << std::endl;
    summary << "Latency issues: " << metrics_.latency_issues << std::endl;
    summary << "Invalid price issues: " << metrics_.invalid_price_issues << std::endl;
    summary << "Invalid volume issues: " << metrics_.invalid_volume_issues << std::endl;
    summary << "Average latency: " << std::fixed << std::setprecision(2)
            << metrics_.average_latency_ms << " ms" << std::endl;

    // Calculate quality score
    double quality_score = 100.0;
    if (metrics_.total_trades_processed > 0) {
        double error_rate = static_cast<double>(metrics_.missing_data_issues +
                                               metrics_.duplicate_trade_issues +
                                               metrics_.out_of_order_timestamp_issues +
                                               metrics_.latency_issues +
                                               metrics_.invalid_price_issues +
                                               metrics_.invalid_volume_issues) /
                           static_cast<double>(metrics_.total_trades_processed);
        quality_score = (1.0 - std::min(error_rate, 1.0)) * 100.0;
    }

    summary << "Data quality score: " << std::fixed << std::setprecision(2)
            << quality_score << "%" << std::endl;

    return summary.str();
}

// Method to get recent high-severity issues
std::vector<DataQualityIssue> DataQualityMonitor::get_high_severity_issues(double min_severity_threshold) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<DataQualityIssue> high_severity_issues;
    for (const auto& issue : recent_issues_) {
        if (issue.severity >= min_severity_threshold) {
            high_severity_issues.push_back(issue);
        }
    }

    return high_severity_issues;
}

} // namespace Data
} // namespace BTQuant