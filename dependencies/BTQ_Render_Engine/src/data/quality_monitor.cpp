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
#include <chrono>

namespace BTQuant {
namespace Data {

// Global data quality monitor instance
DataQualityMonitor g_data_quality_monitor;

DataQualityMonitor::DataQualityMonitor() : alert_callback_(nullptr) {
    // Initialize with current time
    metrics_.last_update_time = std::chrono::high_resolution_clock::now();

    // Set default thresholds for data quality alerts
    missing_data_threshold_ms_ = 5000;  // 5 seconds
    duplicate_check_window_ms_ = 100;   // 100ms window for duplicate detection
    latency_alert_threshold_ms_ = 1000; // 1 second latency threshold
    out_of_order_tolerance_ms_ = 5000;  // 5 seconds tolerance for out-of-order detection
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

    // Check for missing or invalid fields
    check_missing_fields(trade, symbol, trade.timestamp);

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
        // Add to recent trades and hashes if not a duplicate
        recent_trades_[symbol].push_back(trade);
        size_t trade_hash = std::hash<TradeData>{}(trade);
        recent_trade_hashes_[symbol].insert(trade_hash);

        if (recent_trades_[symbol].size() > MAX_RECENT_TRADES) {
            // Remove the oldest trade's hash as well
            const auto& oldest_trade = recent_trades_[symbol].front();
            size_t oldest_hash = std::hash<TradeData>{}(oldest_trade);
            recent_trade_hashes_[symbol].erase(oldest_hash);

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
                if (stats.trade_count > 2) {
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

                    // Additional check: if gap exceeds the configurable threshold
                    if (time_diff > missing_data_threshold_ms_) {
                        std::ostringstream oss;
                        oss << "Extended data gap detected for " << symbol
                            << ". Gap: " << time_diff << "ms exceeds threshold: " << missing_data_threshold_ms_ << "ms";

                        DataQualityIssue issue(DataQualityIssueType::MISSING_DATA, symbol, current_timestamp,
                                             oss.str(), 0.9);
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

                    // Enhanced missing data detection: Check for patterns in recent intervals
                    if (stats.recent_intervals.size() >= 10) {
                        uint64_t median_interval = get_median_interval(stats.recent_intervals);

                        // If current gap is much larger than median, it indicates potential missing data
                        if (median_interval > 0 && time_diff > median_interval * 10) {
                            std::ostringstream oss;
                            oss << "Significant data gap detected for " << symbol
                                << ". Current gap: " << time_diff << "ms, median recent interval: "
                                << median_interval << "ms";

                            DataQualityIssue issue(DataQualityIssueType::MISSING_DATA, symbol, current_timestamp,
                                                 oss.str(), 0.9);
                            metrics_.missing_data_issues++;
                            add_issue(issue);
                        }

                        // Check for consecutive large gaps (indicating sustained data loss)
                        size_t large_gaps_count = 0;
                        for (const auto& interval : stats.recent_intervals) {
                            if (interval > median_interval * 3) {
                                large_gaps_count++;
                            }
                        }

                        if (large_gaps_count > stats.recent_intervals.size() / 3) { // More than 1/3 are large gaps
                            std::ostringstream oss;
                            oss << "Sustained data quality issue for " << symbol
                                << ". " << large_gaps_count << "/" << stats.recent_intervals.size()
                                << " recent intervals are significantly larger than median";

                            DataQualityIssue issue(DataQualityIssueType::MISSING_DATA, symbol, current_timestamp,
                                                 oss.str(), 0.85);
                            metrics_.missing_data_issues++;
                            add_issue(issue);
                        }
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
    // First, check if we have a hash of this exact trade in our recent hashes
    size_t trade_hash = std::hash<TradeData>{}(trade);
    auto& hashes = recent_trade_hashes_[symbol];

    if (hashes.find(trade_hash) != hashes.end()) {
        return true;  // Exact hash match found, definitely a duplicate
    }

    // If no exact hash match, fall back to the detailed comparison for near-duplicates
    const auto& trades = recent_trades_[symbol];

    for (const auto& recent_trade : trades) {
        // For duplicate detection, we focus on the core identifying fields:
        // timestamp, price, volume, and exchange_id
        // We allow some flexibility for flags that might differ due to processing

        // Check for potential duplicates with configurable timestamp window
        uint64_t time_diff = std::abs(static_cast<int64_t>(recent_trade.timestamp) - static_cast<int64_t>(trade.timestamp));

        // If timestamp difference is within our duplicate check window, check other fields
        if (time_diff <= duplicate_check_window_ms_) {
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
            if (std::abs(recent_trade.price - trade.price) < 0.000001 &&  // Very small price tolerance
                std::abs(recent_trade.volume - trade.volume) < 0.0001f &&  // Small volume tolerance
                recent_trade.exchange_id == trade.exchange_id) {

                // If price and volume are nearly identical and exchange matches, consider duplicate
                return true;
            }

            // Additional duplicate check: Same price, volume, and timestamp but different exchange_id might indicate
            // a cross-exchange duplicate or data duplication issue
            if (recent_trade.timestamp == trade.timestamp &&
                std::abs(recent_trade.price - trade.price) < 0.000001 &&
                std::abs(recent_trade.volume - trade.volume) < 0.0001f) {

                // This could be a cross-feed duplicate or data integrity issue
                return true;
            }
        }

        // Check for sequence-based duplicates - if we see the same price/volume pattern in quick succession
        if (time_diff <= duplicate_check_window_ms_ / 2) {  // Tighter window for pattern matching
            if (std::abs(recent_trade.price - trade.price) < 0.000001 &&
                std::abs(recent_trade.volume - trade.volume) < 0.0001f) {

                // Same price and volume in a very tight time window - potential duplicate
                return true;
            }
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
                uint64_t max_acceptable_delay = std::max(avg_interval * 10, out_of_order_tolerance_ms_); // Allow up to 10x average interval or configured tolerance

                // Check if this trade is significantly delayed compared to what we'd expect
                uint64_t current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now().time_since_epoch()).count();

                // Only perform this check if current_time > trade.timestamp to avoid underflow
                if (current_time > trade.timestamp && (current_time - trade.timestamp) > max_acceptable_delay) {
                    return true;
                }
            }
        }

        // Additional check: if the trade timestamp is significantly behind the last known timestamp
        // but still chronologically after it, it might indicate a data feed issue
        if (trade.timestamp < (it->second - out_of_order_tolerance_ms_)) {
            return true;
        }

        // Check for sequence anomalies - if we see timestamps that jump around unexpectedly
        if (stats_it != symbol_stats_.end()) {
            const auto& stats = stats_it->second;
            if (!stats.recent_intervals.empty() && stats.recent_intervals.size() > 5) {
                // Look for patterns where timestamps are jumping around unexpectedly
                uint64_t median_interval = get_median_interval(stats.recent_intervals);

                // If the current gap is much larger than the median but the trade is still "in order",
                // it might indicate a data feed issue
                uint64_t current_gap = trade.timestamp - it->second;
                if (median_interval > 0 && current_gap > median_interval * 5) {
                    // Large gap compared to recent median interval - potential issue
                    return true;
                }
            }
        }

        // Enhanced out-of-order detection: Check for significant backward jumps
        // Even if the timestamp is after the last one, if it's significantly before recent timestamps,
        // it might indicate a data feed issue
        if (stats_it != symbol_stats_.end()) {
            const auto& stats = stats_it->second;
            if (stats.recent_intervals.size() >= 10) {
                // Look at the most recent timestamps to see if this one is unexpectedly early
                uint64_t recent_max_timestamp = 0;

                // Calculate the most recent expected timestamp based on recent activity
                for (size_t i = std::max(0, static_cast<int>(stats.recent_intervals.size()) - 5);
                     i < stats.recent_intervals.size(); ++i) {
                    // Estimate what the timestamp should have been based on recent intervals
                    // This is a more sophisticated check for out-of-order conditions
                }

                // Check if this trade is significantly earlier than expected based on recent patterns
                if (trade.timestamp < (stats.last_timestamp - (out_of_order_tolerance_ms_ / 2))) {
                    return true;
                }
            }
        }
    }

    // If we don't have a previous timestamp for this symbol, we can't determine if it's out of order
    return false;
}

uint64_t DataQualityMonitor::get_median_interval(const std::vector<uint64_t>& intervals) const {
    if (intervals.empty()) {
        return 0;
    }

    std::vector<uint64_t> sorted_intervals = intervals;
    std::sort(sorted_intervals.begin(), sorted_intervals.end());

    size_t size = sorted_intervals.size();
    if (size % 2 == 0) {
        return (sorted_intervals[size/2 - 1] + sorted_intervals[size/2]) / 2;
    } else {
        return sorted_intervals[size/2];
    }
}

void DataQualityMonitor::check_latency_issue(const TradeData& trade, const std::string& symbol) {
    auto it = last_received_times_.find(symbol);
    if (it != last_received_times_.end()) {
        auto now = std::chrono::high_resolution_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(now - it->second).count();

        if (latency > latency_alert_threshold_ms_) {
            std::ostringstream oss;
            oss << "High processing latency detected: " << latency << "ms, exceeding threshold of "
                << latency_alert_threshold_ms_ << "ms";

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

        if (delay_ms > latency_alert_threshold_ms_ * 2) {  // More stringent threshold for data feed delay
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

    // Enhanced latency monitoring: Check for increasing trends in latency
    if (current_time > trade.timestamp) {
        int64_t current_delay = current_time - trade.timestamp;

        // Track recent delays separately to avoid interfering with interval tracking
        auto& delay_history = recent_delays_[symbol];
        delay_history.push_back(current_delay);

        // Keep only the last 20 delays for trend analysis
        if (delay_history.size() > 20) {
            delay_history.erase(delay_history.begin());
        }

        // Check if there's an increasing trend in latency
        if (delay_history.size() >= 10) {
            // Compare first half vs second half of recent delays
            size_t mid = delay_history.size() / 2;
            if (mid > 0) { // Ensure we have at least 2 elements to split
                uint64_t first_half_avg = 0, second_half_avg = 0;

                for (size_t i = 0; i < mid; ++i) {
                    first_half_avg += delay_history[i];
                }
                first_half_avg /= mid;

                for (size_t i = mid; i < delay_history.size(); ++i) {
                    second_half_avg += delay_history[i];
                }
                second_half_avg /= (delay_history.size() - mid);

                // If second half average is significantly higher than first half, we have a trend
                if (first_half_avg > 0 && (static_cast<double>(second_half_avg) / first_half_avg) > 1.5) {
                    std::ostringstream oss;
                    oss << "Latency increasing trend detected: recent avg " << second_half_avg
                        << "ms vs previous avg " << first_half_avg << "ms";

                    DataQualityIssue issue(DataQualityIssueType::LATENCY_ISSUE, symbol, trade.timestamp,
                                         oss.str(), 0.6);
                    metrics_.latency_issues++;
                    add_issue(issue);
                }
            }
        }

        // Additional latency monitoring: Check for consistent high latency
        auto& latency_delay_history = recent_delays_[symbol];
        if (latency_delay_history.size() >= 5) {
            // Count how many recent delays exceed the threshold
            size_t high_latency_count = 0;
            for (const auto& delay : latency_delay_history) {
                if (delay > latency_alert_threshold_ms_) {
                    high_latency_count++;
                }
            }

            // If more than half of recent delays are high, flag as issue
            if (high_latency_count > latency_delay_history.size() / 2) {
                std::ostringstream oss;
                oss << "Consistent high latency detected: " << high_latency_count << "/"
                    << latency_delay_history.size() << " recent delays exceeded threshold";

                DataQualityIssue issue(DataQualityIssueType::LATENCY_ISSUE, symbol, trade.timestamp,
                                     oss.str(), 0.7);
                metrics_.latency_issues++;
                add_issue(issue);
            }
        }

        // Check for extreme latency outliers using statistical methods
        if (latency_delay_history.size() >= 10) {
            // Calculate mean and standard deviation of recent delays
            double sum = 0;
            for (const auto& delay : latency_delay_history) {
                sum += delay;
            }
            double mean = sum / latency_delay_history.size();

            double variance_sum = 0;
            for (const auto& delay : latency_delay_history) {
                variance_sum += (delay - mean) * (delay - mean);
            }
            double std_dev = sqrt(variance_sum / latency_delay_history.size());

            // If current delay is more than 3 standard deviations from the mean, it's an outlier
            if (std_dev > 0 && abs(current_delay - mean) > 3 * std_dev) {
                std::ostringstream oss;
                oss << "Extreme latency outlier detected: " << current_delay
                    << "ms (mean: " << mean << "ms, std dev: " << std_dev << "ms)";

                DataQualityIssue issue(DataQualityIssueType::LATENCY_ISSUE, symbol, trade.timestamp,
                                     oss.str(), 0.8);
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

void DataQualityMonitor::check_missing_fields(const TradeData& trade, const std::string& symbol, uint64_t timestamp) {
    // Check for missing or invalid fields
    std::vector<std::string> missing_fields;

    // Check if timestamp is reasonable (not too far in the future or past)
    auto current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    // If timestamp is significantly different from current time (more than 1 hour), it might be invalid
    if (std::abs(static_cast<int64_t>(timestamp) - static_cast<int64_t>(current_time)) > 3600000) {
        DataQualityIssue issue(DataQualityIssueType::MISSING_FIELD, symbol, timestamp,
                             "Timestamp is significantly different from current time (> 1 hour)", 0.6);
        metrics_.missing_field_issues++;
        add_issue(issue);
    }

    // Check for potentially invalid exchange_id (assuming valid range is 0-255, but we might expect a smaller range)
    // Common exchanges might have IDs 1-10, so anything above a threshold might be suspicious
    if (trade.exchange_id > 50) {  // Assuming most exchanges have IDs under 50
        std::ostringstream oss;
        oss << "Unusual exchange ID detected: " << static_cast<int>(trade.exchange_id);

        DataQualityIssue issue(DataQualityIssueType::MISSING_FIELD, symbol, timestamp,
                             oss.str(), 0.4);
        metrics_.missing_field_issues++;
        add_issue(issue);
    }

    // Check for potentially invalid flags combination
    // For example, a trade shouldn't be both market and limit order at the same time
    if (has_flag(trade.flags, TradeFlags::MARKET_ORDER) && has_flag(trade.flags, TradeFlags::LIMIT_ORDER)) {
        DataQualityIssue issue(DataQualityIssueType::MISSING_FIELD, symbol, timestamp,
                             "Invalid flag combination: trade marked as both market and limit order", 0.7);
        metrics_.missing_field_issues++;
        add_issue(issue);
    }

    // Check for aggressive and passive order flags together
    if (has_flag(trade.flags, TradeFlags::AGGRESSIVE_ORDER) && has_flag(trade.flags, TradeFlags::PASSIVE_ORDER)) {
        DataQualityIssue issue(DataQualityIssueType::MISSING_FIELD, symbol, timestamp,
                             "Invalid flag combination: trade marked as both aggressive and passive order", 0.7);
        metrics_.missing_field_issues++;
        add_issue(issue);
    }

    // Check for liquidity added and removed flags together
    if (has_flag(trade.flags, TradeFlags::LIQUIDITY_ADDED) && has_flag(trade.flags, TradeFlags::LIQUIDITY_REMOVED)) {
        DataQualityIssue issue(DataQualityIssueType::MISSING_FIELD, symbol, timestamp,
                             "Invalid flag combination: trade marked as both liquidity added and removed", 0.7);
        metrics_.missing_field_issues++;
        add_issue(issue);
    }
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
    recent_trade_hashes_.clear();
    last_received_times_.clear();
    symbol_stats_.clear();
    recent_delays_.clear();
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
    if (issue.severity >= 0.8 && console_alerts_enabled_) {
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
                << ", Severity: " << issue.severity
                << ", Timestamp: " << issue.timestamp;

        std::cout << log_msg.str() << std::endl;
    }

    // Send to external monitoring system for all issues (not just high severity)
    send_external_alert(issue);

    // Additional alerting for critical issues
    if (issue.severity >= 0.9) {
        // Send critical alert notification
        send_critical_alert(issue);
    }
}

void DataQualityMonitor::send_external_alert(const DataQualityIssue& issue) {
    // Send alert to external monitoring system if callback is set
    if (external_alert_callback_) {
        external_alert_callback_(issue);
    }
}

void DataQualityMonitor::send_critical_alert(const DataQualityIssue& issue) {
    // This method sends critical alerts to external systems or logs
    std::ostringstream critical_msg;
    critical_msg << "[CRITICAL DATA QUALITY ALERT] ";
    critical_msg << "Symbol: " << issue.symbol
                 << ", Issue: " << issue.description
                 << ", Severity: " << issue.severity
                 << ", Time: " << issue.timestamp;

    // Log to stderr for critical issues
    if (console_alerts_enabled_) {
        std::cerr << critical_msg.str() << std::endl;
    }

    // Log to file if enabled
    if (file_logging_enabled_) {
        // In a real implementation, this would write to a log file
        // For now, we'll just simulate it
    }

    // Send to external monitoring system
    send_external_alert(issue);
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

void DataQualityMonitor::trigger_alert(const std::string& symbol, DataQualityIssueType issue_type,
                                       const std::string& description, double severity) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Create a new issue with current timestamp
    uint64_t current_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    DataQualityIssue issue(issue_type, symbol, current_timestamp, description, severity);
    add_issue(issue);
}

void DataQualityMonitor::trigger_data_quality_alerts() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if we have any significant data quality issues
    auto current_metrics = metrics_;

    // Trigger alerts based on thresholds
    if (current_metrics.missing_data_issues > 0) {
        std::ostringstream msg;
        msg << "Data quality alert: " << current_metrics.missing_data_issues
            << " missing data issues detected";
        trigger_alert("SYSTEM", DataQualityIssueType::MISSING_DATA, msg.str(), 0.8);
    }

    if (current_metrics.duplicate_trade_issues > 0) {
        std::ostringstream msg;
        msg << "Data quality alert: " << current_metrics.duplicate_trade_issues
            << " duplicate trade issues detected";
        trigger_alert("SYSTEM", DataQualityIssueType::DUPLICATE_TRADE, msg.str(), 0.7);
    }

    if (current_metrics.out_of_order_timestamp_issues > 0) {
        std::ostringstream msg;
        msg << "Data quality alert: " << current_metrics.out_of_order_timestamp_issues
            << " out-of-order timestamp issues detected";
        trigger_alert("SYSTEM", DataQualityIssueType::OUT_OF_ORDER_TIMESTAMP, msg.str(), 0.75);
    }

    if (current_metrics.latency_issues > 0) {
        std::ostringstream msg;
        msg << "Data quality alert: " << current_metrics.latency_issues
            << " latency issues detected";
        trigger_alert("SYSTEM", DataQualityIssueType::LATENCY_ISSUE, msg.str(), 0.7);
    }

    // Check for high error rate
    if (current_metrics.total_trades_processed > 0) {
        double error_rate = static_cast<double>(current_metrics.missing_data_issues +
                                               current_metrics.duplicate_trade_issues +
                                               current_metrics.out_of_order_timestamp_issues +
                                               current_metrics.latency_issues +
                                               current_metrics.invalid_price_issues +
                                               current_metrics.invalid_volume_issues) /
                           static_cast<double>(current_metrics.total_trades_processed);

        if (error_rate > 0.05) { // More than 5% error rate
            std::ostringstream msg;
            msg << "CRITICAL: High data error rate of " << std::fixed << std::setprecision(2)
                << (error_rate * 100.0) << "% detected";
            trigger_alert("SYSTEM", DataQualityIssueType::MISSING_DATA, msg.str(), 0.95);
        }
    }
}

} // namespace Data
} // namespace BTQuant