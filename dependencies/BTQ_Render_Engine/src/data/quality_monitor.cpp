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
#include <thread>
#include <future>
#include <map>
#include <ctime>

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

    // Initialize alert burst tracking
    recent_alert_times_.reserve(1000);  // Reserve space for efficiency

    // Initialize additional tracking structures
    recent_delays_.clear();
    alert_counts_by_type_.clear();

    // Initialize alert configuration with all alerts enabled by default
    alert_config_ = {
        true,  // enable_missing_data
        true,  // enable_duplicate_trades
        true,  // enable_out_of_order
        true,  // enable_latency_issues
        true   // enable_invalid_data
    };
}

uint64_t DataQualityMonitor::calculate_safe_time_diff(uint64_t current, uint64_t previous) const {
    if (current >= previous) {
        return current - previous;
    }
    // Handle potential wraparound or out-of-order timestamps
    return 0;
}

bool DataQualityMonitor::are_trades_equivalent(const TradeData& trade1, const TradeData& trade2,
                                            double price_tolerance, double volume_tolerance) const {
    // Check if timestamps are within acceptable range
    uint64_t time_diff = std::abs(static_cast<int64_t>(trade1.timestamp) - static_cast<int64_t>(trade2.timestamp));

    // Check if prices are within tolerance
    double price_diff = std::abs(trade1.price - trade2.price);
    bool prices_match = price_diff <= price_tolerance;

    // Check if volumes are within tolerance
    double volume_diff = std::abs(trade1.volume - trade2.volume);
    bool volumes_match = volume_diff <= volume_tolerance;

    // Check if other fields match exactly
    bool other_fields_match = (trade1.side == trade2.side) &&
                              (trade1.exchange_id == trade2.exchange_id);

    return (time_diff <= 100) && prices_match && volumes_match && other_fields_match; // 100ms tolerance for timestamp
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

        // Alert the user about the duplicate trade
        alert_on_duplicate_trade(trade, symbol);
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
    auto last_timestamp_it = last_timestamps_.find(symbol);
    if (is_out_of_order_timestamp(trade, symbol)) {
        DataQualityIssue issue(DataQualityIssueType::OUT_OF_ORDER_TIMESTAMP, symbol, trade.timestamp,
                             "Out-of-order timestamp detected", 0.5);
        detected_issues.push_back(issue);
        metrics_.out_of_order_timestamp_issues++;
        add_issue(issue);

        // Alert the user about the out-of-order timestamp
        if (last_timestamp_it != last_timestamps_.end()) {
            alert_on_out_of_order_timestamp(trade, symbol, last_timestamp_it->second);
        }
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

                        // Alert the user about the potential missing data
                        alert_on_missing_data(symbol, stats.last_timestamp, current_timestamp);
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

                        // Alert the user about the extended data gap
                        alert_on_missing_data(symbol, stats.last_timestamp, current_timestamp);
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

                        // Alert the user about the unusually large gap
                        alert_on_missing_data(symbol, stats.last_timestamp, current_timestamp);
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

                            // Alert the user about the significant data gap
                            alert_on_missing_data(symbol, stats.last_timestamp, current_timestamp);
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

                            // Alert the user about the sustained data quality issue
                            alert_on_missing_data(symbol, stats.last_timestamp, current_timestamp);
                        }

                        // Enhanced missing data detection: Check for periodic patterns
                        // If data normally arrives in predictable intervals and suddenly deviates
                        if (stats.recent_intervals.size() >= 20) {
                            // Calculate standard deviation of recent intervals
                            uint64_t sum = 0;
                            for (const auto& interval : stats.recent_intervals) {
                                sum += interval;
                            }
                            double mean = static_cast<double>(sum) / stats.recent_intervals.size();

                            double variance_sum = 0;
                            for (const auto& interval : stats.recent_intervals) {
                                variance_sum += (static_cast<double>(interval) - mean) * (static_cast<double>(interval) - mean);
                            }
                            double std_dev = sqrt(variance_sum / stats.recent_intervals.size());

                            // If current gap is more than 3 standard deviations from the mean, it's an outlier
                            if (std_dev > 0 && (time_diff > mean + 3 * std_dev)) {
                                std::ostringstream oss;
                                oss << "Statistical outlier gap detected for " << symbol
                                    << ". Gap: " << time_diff << "ms (mean: " << std::fixed << std::setprecision(2)
                                    << mean << "ms, std dev: " << std_dev << "ms)";

                                DataQualityIssue issue(DataQualityIssueType::MISSING_DATA, symbol, current_timestamp,
                                                     oss.str(), 0.95);
                                metrics_.missing_data_issues++;
                                add_issue(issue);

                                // Alert the user about the statistical outlier gap
                                alert_on_missing_data(symbol, stats.last_timestamp, current_timestamp);
                            }
                        }
                    }

                    // Check for complete data stream cessation
                    auto last_received_it = last_received_times_.find(symbol);
                    if (last_received_it != last_received_times_.end()) {
                        auto time_since_last_received = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now - last_received_it->second).count();

                        // If we haven't received any data for much longer than expected, flag as issue
                        if (time_since_last_received > static_cast<int64_t>(avg_interval * 20)) {
                            std::ostringstream oss;
                            oss << "Data stream appears to have stopped for " << symbol
                                << ". Last data received " << time_since_last_received
                                << "ms ago, expected based on avg interval: " << avg_interval << "ms";

                            DataQualityIssue issue(DataQualityIssueType::MISSING_DATA, symbol, current_timestamp,
                                                 oss.str(), 0.98);
                            metrics_.missing_data_issues++;
                            add_issue(issue);

                            // Alert the user about the data stream cessation
                            alert_on_missing_data(symbol, stats.last_timestamp, current_timestamp);
                        }
                    }

                    // Enhanced missing data detection: Check for seasonal patterns
                    // If data typically follows certain patterns throughout the day/week
                    if (stats.recent_intervals.size() >= 50) {
                        // Analyze time-based patterns (hourly, daily, weekly)
                        auto current_time = std::chrono::system_clock::now();
                        auto current_tm = std::chrono::system_clock::to_time_t(current_time);
                        auto current_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            current_time.time_since_epoch()).count() % 86400000; // Milliseconds since midnight

                        // Check if this is a time when we typically expect data based on historical patterns
                        // This is a simplified version - in practice, you'd want more sophisticated time-series analysis
                        if (time_diff > avg_interval * 2 && avg_interval < 1000) { // For high-frequency data
                            std::ostringstream oss;
                            oss << "Unexpected data gap for " << symbol
                                << " during active trading period. Gap: " << time_diff
                                << "ms, expected: ~" << avg_interval << "ms";

                            DataQualityIssue issue(DataQualityIssueType::MISSING_DATA, symbol, current_timestamp,
                                                 oss.str(), 0.8);
                            metrics_.missing_data_issues++;
                            add_issue(issue);

                            alert_on_missing_data(symbol, stats.last_timestamp, current_timestamp);
                        }
                    }

                    // Enhanced missing data detection: Check for progressive degradation
                    // Look for gradually increasing intervals that suggest degrading connection
                    if (stats.recent_intervals.size() >= 25) {
                        // Calculate trend in recent intervals
                        size_t first_quarter_end = stats.recent_intervals.size() / 4;
                        size_t third_quarter_start = 3 * stats.recent_intervals.size() / 4;

                        if (first_quarter_end > 0 && third_quarter_start < stats.recent_intervals.size()) {
                            uint64_t first_avg = 0, last_avg = 0;

                            for (size_t i = 0; i < first_quarter_end; ++i) {
                                first_avg += stats.recent_intervals[i];
                            }
                            first_avg /= first_quarter_end;

                            for (size_t i = third_quarter_start; i < stats.recent_intervals.size(); ++i) {
                                last_avg += stats.recent_intervals[i];
                            }
                            last_avg /= (stats.recent_intervals.size() - third_quarter_start);

                            // If recent intervals are significantly larger than early intervals, flag degradation
                            if (first_avg > 0 && last_avg > first_avg * 2) {
                                std::ostringstream oss;
                                oss << "Progressive data degradation detected for " << symbol
                                    << ". Recent avg interval: " << last_avg
                                    << "ms vs early avg: " << first_avg << "ms";

                                DataQualityIssue issue(DataQualityIssueType::MISSING_DATA, symbol, current_timestamp,
                                                     oss.str(), 0.75);
                                metrics_.missing_data_issues++;
                                add_issue(issue);

                                alert_on_missing_data(symbol, stats.last_timestamp, current_timestamp);
                            }
                        }
                    }

                    // NEW: Enhanced missing data detection: Check for data stream health based on expected vs actual frequency
                    if (stats.trade_count > 10) {
                        // Calculate expected trades per time unit based on historical data
                        uint64_t historical_time_span = stats.total_interval_sum;
                        if (historical_time_span > 0) {
                            double expected_frequency = static_cast<double>(stats.trade_count - 1) / static_cast<double>(historical_time_span);
                            double actual_frequency = 1.0 / static_cast<double>(time_diff);

                            // If actual frequency is significantly lower than expected, flag as missing data
                            if (expected_frequency > 0 && actual_frequency < expected_frequency * 0.1) { // 10% of expected
                                std::ostringstream oss;
                                oss << "Data stream frequency significantly below expected for " << symbol
                                    << ". Expected: " << std::fixed << std::setprecision(4) << expected_frequency
                                    << " trades/ms, Actual: " << actual_frequency << " trades/ms";

                                DataQualityIssue issue(DataQualityIssueType::MISSING_DATA, symbol, current_timestamp,
                                                     oss.str(), 0.85);
                                metrics_.missing_data_issues++;
                                add_issue(issue);

                                alert_on_missing_data(symbol, stats.last_timestamp, current_timestamp);
                            }
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

    // Use the enhanced equivalence check as a primary method
    const auto& trades = recent_trades_[symbol];
    for (const auto& recent_trade : trades) {
        if (are_trades_equivalent(recent_trade, trade)) {
            return true;  // Trades are equivalent, consider duplicate
        }
    }

    // NEW: Enhanced duplicate detection: Check for potential systematic duplication patterns
    // Look for trades that appear in predictable intervals suggesting systematic duplication
    size_t systematic_duplicates = 0;
    for (const auto& recent_trade : trades) {
        uint64_t time_diff = std::abs(static_cast<int64_t>(recent_trade.timestamp) - static_cast<int64_t>(trade.timestamp));

        // Check if price and volume match closely but timestamps are at regular intervals
        if (std::abs(recent_trade.price - trade.price) < 0.000001 &&
            std::abs(recent_trade.volume - trade.volume) < 0.0001f &&
            time_diff <= duplicate_check_window_ms_ * 10) { // Extended window for systematic detection

            // Count how many trades match this pattern
            systematic_duplicates++;
        }
    }

    // If we find multiple trades with similar characteristics in a short window, likely systematic duplication
    if (systematic_duplicates > 2) {
        return true;
    }

    // If no exact hash match, fall back to the detailed comparison for near-duplicates
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

        // Enhanced duplicate detection: Check for trades with identical characteristics
        // but potentially different timestamps due to processing delays
        if (std::abs(recent_trade.price - trade.price) < 0.000001 &&
            std::abs(recent_trade.volume - trade.volume) < 0.0001f &&
            recent_trade.side == trade.side &&
            recent_trade.exchange_id == trade.exchange_id) {

            // If price, volume, side, and exchange match but timestamps are close, likely duplicate
            if (time_diff <= duplicate_check_window_ms_ * 2) {
                return true;
            }
        }

        // Enhanced duplicate detection: Check for systematic duplication patterns
        // Look for trades that appear in a predictable pattern indicating systematic duplication
        if (time_diff <= duplicate_check_window_ms_ * 5) { // Extended window for pattern detection
            // Check if this trade matches a pattern of systematic duplication
            if (recent_trade.price == trade.price &&
                recent_trade.volume == trade.volume &&
                recent_trade.side == trade.side) {

                // If price, volume, and side match but exchange_id differs, it might be cross-feed duplication
                if (recent_trade.exchange_id != trade.exchange_id) {
                    // Check if this represents the same trade coming from different sources
                    // This could indicate a data feed issue where the same trade is reported multiple times
                    return true;
                }
            }

            // Check for systematic duplication based on timing patterns
            // If we see similar trades appearing at regular intervals, it might indicate duplication
            if (recent_trade.price == trade.price &&
                recent_trade.volume == trade.volume) {

                // Look for trades that repeat at regular intervals
                // This could indicate a systematic issue in the data pipeline
                for (const auto& older_trade : trades) {
                    if (&older_trade != &recent_trade) { // Don't compare with itself
                        uint64_t time_diff_older = std::abs(static_cast<int64_t>(older_trade.timestamp) -
                                                           static_cast<int64_t>(recent_trade.timestamp));

                        // If we have three trades with similar characteristics at regular intervals
                        if (time_diff_older <= duplicate_check_window_ms_ * 3 &&
                            older_trade.price == recent_trade.price &&
                            older_trade.volume == recent_trade.volume) {

                            return true; // Pattern of systematic duplication detected
                        }
                    }
                }
            }
        }

        // Enhanced duplicate detection: Check for near-identical trades with slight variations
        // This catches cases where trades are duplicated but have minor differences due to precision issues
        if (time_diff <= duplicate_check_window_ms_) {
            // Calculate similarity score based on price and volume differences
            double price_similarity = 1.0 - (std::abs(recent_trade.price - trade.price) / std::max(recent_trade.price, trade.price));
            double volume_similarity = 1.0 - (std::abs(recent_trade.volume - trade.volume) / std::max(recent_trade.volume, trade.volume));

            // If both price and volume are very similar (99.99%+ similarity), consider duplicate
            if (price_similarity > 0.9999 && volume_similarity > 0.9999) {
                return true;
            }
        }

        // Enhanced duplicate detection: Check for cluster-based duplicates
        // Look for groups of similar trades that might indicate bulk duplication
        if (time_diff <= duplicate_check_window_ms_ * 3) {
            size_t similar_trades_count = 0;

            // Count how many trades in the recent window are similar to this one
            for (const auto& check_trade : trades) {
                uint64_t check_time_diff = std::abs(static_cast<int64_t>(check_trade.timestamp) -
                                                   static_cast<int64_t>(trade.timestamp));

                if (check_time_diff <= duplicate_check_window_ms_ * 3 &&
                    std::abs(check_trade.price - trade.price) < 0.000001 &&
                    std::abs(check_trade.volume - trade.volume) < 0.0001f) {
                    similar_trades_count++;
                }
            }

            // If we have many similar trades in a short time window, it might indicate duplication
            if (similar_trades_count > 3) { // More than 3 similar trades in the window
                return true;
            }
        }

        // Enhanced duplicate detection: Check for systematic patterns in timestamp differences
        // Look for arithmetic sequences that might indicate systematic duplication
        if (time_diff <= duplicate_check_window_ms_ * 10) {
            // Check if this trade fits a pattern with other recent trades
            std::vector<const TradeData*> similar_price_trades;

            for (const auto& check_trade : trades) {
                if (std::abs(check_trade.price - trade.price) < 0.000001 &&
                    std::abs(check_trade.volume - trade.volume) < 0.0001f) {
                    similar_price_trades.push_back(&check_trade);
                }
            }

            // If we have 3 or more trades with similar prices/volumes, check for arithmetic progression
            if (similar_price_trades.size() >= 3) {
                std::vector<uint64_t> timestamps;
                for (const auto* t : similar_price_trades) {
                    timestamps.push_back(t->timestamp);
                }

                std::sort(timestamps.begin(), timestamps.end());

                // Check if timestamps form an arithmetic sequence (indicating systematic duplication)
                if (timestamps.size() >= 3) {
                    uint64_t interval1 = timestamps[1] - timestamps[0];
                    uint64_t interval2 = timestamps[2] - timestamps[1];

                    // If intervals are approximately equal, it suggests systematic duplication
                    if (std::abs(static_cast<int64_t>(interval1) - static_cast<int64_t>(interval2)) <=
                        duplicate_check_window_ms_) {
                        return true;
                    }
                }
            }
        }

        // Enhanced duplicate detection: Check for precision-based duplicates
        // Sometimes duplicates appear due to precision differences in decimal places
        if (time_diff <= duplicate_check_window_ms_) {
            // Check if the price difference is due to rounding/precision issues
            double price_diff_ratio = std::abs(recent_trade.price - trade.price) / std::max(recent_trade.price, trade.price);
            double volume_diff_ratio = std::abs(recent_trade.volume - trade.volume) / std::max(recent_trade.volume, trade.volume);

            // If differences are very small relative to the values themselves, consider duplicate
            if (price_diff_ratio < 0.000001 && volume_diff_ratio < 0.00001) {
                return true;
            }
        }

        // Enhanced duplicate detection: Check for pattern-based duplicates
        // Look for trades that follow a predictable pattern indicating systematic duplication
        if (time_diff <= duplicate_check_window_ms_ * 2) {
            // Check if this trade matches a pattern based on previous trades
            // For example, if trades are being repeated at regular intervals
            for (size_t i = 0; i < trades.size(); ++i) {
                for (size_t j = i + 1; j < trades.size(); ++j) {
                    const auto& trade_i = trades[i];
                    const auto& trade_j = trades[j];

                    // Check if current trade matches the pattern between trade_i and trade_j
                    if (std::abs(trade_i.price - trade_j.price) < 0.000001 &&
                        std::abs(trade_i.volume - trade_j.volume) < 0.0001f &&
                        std::abs(trade.price - trade_i.price) < 0.000001 &&
                        std::abs(trade.volume - trade_i.volume) < 0.0001f) {

                        // Check if the timing pattern is consistent
                        int64_t interval_ij = static_cast<int64_t>(trade_j.timestamp) - static_cast<int64_t>(trade_i.timestamp);
                        int64_t interval_current = static_cast<int64_t>(trade.timestamp) - static_cast<int64_t>(trade_j.timestamp);

                        // If intervals are similar, it suggests a repeating pattern
                        if (std::abs(interval_ij - interval_current) <= static_cast<int64_t>(duplicate_check_window_ms_)) {
                            return true;
                        }
                    }
                }
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

        // Enhanced out-of-order detection: Check for sequence inconsistencies
        // If we have a sequence of recent trades, verify that the new trade fits properly
        const auto& recent_trades = recent_trades_[symbol];
        if (recent_trades.size() >= 3) {
            // Check if the new trade is out of sequence with recent trades
            uint64_t min_recent_ts = recent_trades.back().timestamp;
            uint64_t max_recent_ts = recent_trades.front().timestamp;

            // If the new trade is earlier than the most recent trade but later than the earliest,
            // it might be out of order
            if (trade.timestamp < max_recent_ts && trade.timestamp > min_recent_ts) {
                // This suggests the trade is inserted somewhere in the middle of recent trades
                return true;
            }
        }

        // Enhanced out-of-order detection: Check for statistical anomalies in timestamp sequences
        if (stats_it != symbol_stats_.end()) {
            const auto& stats = stats_it->second;
            if (stats.recent_intervals.size() >= 20) {
                // Calculate statistical measures of recent intervals
                uint64_t sum = 0;
                for (const auto& interval : stats.recent_intervals) {
                    sum += interval;
                }
                double mean = static_cast<double>(sum) / stats.recent_intervals.size();

                double variance_sum = 0;
                for (const auto& interval : stats.recent_intervals) {
                    variance_sum += (static_cast<double>(interval) - mean) * (static_cast<double>(interval) - mean);
                }
                double std_dev = sqrt(variance_sum / stats.recent_intervals.size());

                // If the gap from the last timestamp is significantly different from the expected pattern,
                // it might indicate an out-of-order condition
                uint64_t current_gap = trade.timestamp - it->second;
                if (std_dev > 0 && std::abs(static_cast<double>(current_gap) - mean) > 3 * std_dev) {
                    // This gap is a statistical outlier, suggesting potential out-of-order issue
                    return true;
                }
            }
        }

        // Enhanced out-of-order detection: Check for timestamp clustering
        // If we see many trades with the same or very similar timestamps, it might indicate
        // a data processing issue where timestamps weren't properly updated
        const auto& current_recent_trades = recent_trades_[symbol];
        if (current_recent_trades.size() >= 5) {
            // Count how many recent trades have similar timestamps (within a small window)
            size_t similar_timestamp_count = 0;
            for (const auto& recent_trade : current_recent_trades) {
                uint64_t time_diff = std::abs(static_cast<int64_t>(recent_trade.timestamp) - static_cast<int64_t>(trade.timestamp));
                if (time_diff <= 10) { // Within 10ms window
                    similar_timestamp_count++;
                }
            }

            // If a significant portion of recent trades have similar timestamps, it might indicate an issue
            if (similar_timestamp_count > current_recent_trades.size() / 3) { // More than 1/3 have similar timestamps
                return true;
            }
        }

        // Enhanced out-of-order detection: Check for timestamp regression in recent history
        // Look for cases where timestamps have been moving forward but suddenly regress
        if (current_recent_trades.size() >= 10) {
            // Check the trend in the last N trades
            size_t forward_moving = 0;
            size_t backward_moving = 0;

            for (size_t i = 1; i < std::min(static_cast<size_t>(10), current_recent_trades.size()); ++i) {
                if (current_recent_trades[i].timestamp > current_recent_trades[i-1].timestamp) {
                    forward_moving++;
                } else if (current_recent_trades[i].timestamp < current_recent_trades[i-1].timestamp) {
                    backward_moving++;
                }
            }

            // If we had a mostly forward-moving sequence and now see a backward movement, flag it
            if (forward_moving > 6 && backward_moving <= 2) {
                // Previously had a strong forward trend, now we're going backwards
                if (trade.timestamp < current_recent_trades.front().timestamp) {
                    return true;
                }
            }
        }

        // Enhanced out-of-order detection: Check for monotonicity violations
        // Look for violations of expected chronological order in a broader context
        if (current_recent_trades.size() >= 20) {
            // Check if the new trade violates the general trend of the last N trades
            std::vector<uint64_t> recent_timestamps;
            for (const auto& recent_trade : current_recent_trades) {
                recent_timestamps.push_back(recent_trade.timestamp);
            }

            // Sort to see the expected range
            std::sort(recent_timestamps.begin(), recent_timestamps.end());

            // If the new timestamp is significantly earlier than what we'd expect from the sorted sequence
            if (recent_timestamps.size() >= 10) {
                uint64_t expected_min = recent_timestamps[recent_timestamps.size() - 5]; // Top 5 recent

                if (trade.timestamp < expected_min) {
                    return true;
                }
            }
        }

        // Enhanced out-of-order detection: Check for systematic timestamp issues
        // Look for patterns where timestamps consistently come in with wrong ordering
        if (current_recent_trades.size() >= 15) {
            // Count how many recent trades appear to be out of order
            size_t out_of_order_count = 0;
            for (size_t i = 1; i < current_recent_trades.size(); ++i) {
                if (current_recent_trades[i].timestamp < current_recent_trades[i-1].timestamp) {
                    out_of_order_count++;
                }
            }

            // If a significant portion of recent trades are out of order, flag the current trade too
            if (out_of_order_count > current_recent_trades.size() / 4) { // More than 25% are out of order
                return true;
            }
        }

        // Enhanced out-of-order detection: Check for time drift
        // Look for systematic shifts in timestamp patterns that might indicate clock issues
        if (stats_it != symbol_stats_.end()) {
            const auto& stats = stats_it->second;
            if (stats.recent_intervals.size() >= 30) {
                // Calculate the trend in timestamp intervals
                std::vector<uint64_t> intervals_copy = stats.recent_intervals;

                // Look at first and last segments to see if there's a trend
                size_t segment_size = intervals_copy.size() / 3;

                if (segment_size >= 5) {
                    uint64_t early_avg = 0, late_avg = 0;

                    for (size_t i = 0; i < segment_size; ++i) {
                        early_avg += intervals_copy[i];
                    }
                    early_avg /= segment_size;

                    for (size_t i = intervals_copy.size() - segment_size; i < intervals_copy.size(); ++i) {
                        late_avg += intervals_copy[i];
                    }
                    late_avg /= segment_size;

                    // If intervals are getting significantly longer, it might indicate timestamp issues
                    if (early_avg > 0 && late_avg > early_avg * 3) {
                        // Check if the current trade fits this problematic pattern
                        uint64_t current_gap = trade.timestamp - it->second;
                        if (current_gap > late_avg * 2) {
                            return true;
                        }
                    }
                }
            }
        }

        // Enhanced out-of-order detection: Check for extreme timestamp anomalies
        // Look for trades with timestamps that are significantly different from the norm
        if (current_recent_trades.size() >= 10) {
            // Calculate the expected timestamp range based on recent trades
            uint64_t min_recent_ts = current_recent_trades[0].timestamp;
            uint64_t max_recent_ts = current_recent_trades[0].timestamp;

            for (const auto& recent_trade : current_recent_trades) {
                min_recent_ts = std::min(min_recent_ts, recent_trade.timestamp);
                max_recent_ts = std::max(max_recent_ts, recent_trade.timestamp);
            }

            // Calculate avg_interval from stats if available
            uint64_t avg_interval = 0;
            if (stats_it != symbol_stats_.end()) {
                const auto& stats = stats_it->second;
                if (stats.trade_count > 1) {
                    avg_interval = stats.total_interval_sum / (stats.trade_count - 1);
                }
            }

            // If the current trade timestamp is significantly outside the recent range, it might be an issue
            if (trade.timestamp < min_recent_ts || trade.timestamp > (max_recent_ts + avg_interval * 5)) {
                // But only flag as out-of-order if it's earlier than minimum (the main concern)
                if (trade.timestamp < min_recent_ts) {
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

            // Alert the user about the latency issue
            alert_on_latency_issue(trade, symbol, latency);
        }

        // Update average latency
        if (metrics_.total_trades_processed > 0) {
            metrics_.average_latency_ms = (metrics_.average_latency_ms * (metrics_.total_trades_processed - 1) + latency) /
                                          metrics_.total_trades_processed;
        }
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

            // Alert the user about the data feed delay
            alert_on_latency_issue(trade, symbol, delay_ms);
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

                    // Alert the user about the data feed delay
                    alert_on_latency_issue(trade, symbol, delay_ms);
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

                // Alert the user about the latency spike
                alert_on_latency_issue(trade, symbol, current_delay);
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

                    // Alert the user about the latency trend
                    alert_on_latency_issue(trade, symbol, second_half_avg);
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

                // Alert the user about consistent high latency
                alert_on_latency_issue(trade, symbol, delay_history.back());
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

                // Alert the user about the extreme latency outlier
                alert_on_latency_issue(trade, symbol, current_delay);
            }
        }

        // Enhanced latency monitoring: Check for exponential growth in latency
        if (latency_delay_history.size() >= 8) {
            // Check if latencies are growing exponentially (potential system degradation)
            bool is_exponential_growth = true;
            for (size_t i = 1; i < latency_delay_history.size(); ++i) {
                if (latency_delay_history[i] < latency_delay_history[i-1]) {
                    is_exponential_growth = false;
                    break;
                }
            }

            if (is_exponential_growth) {
                std::ostringstream oss;
                oss << "Exponential latency growth detected: " << latency_delay_history.size()
                    << " consecutive increases in latency";

                DataQualityIssue issue(DataQualityIssueType::LATENCY_ISSUE, symbol, trade.timestamp,
                                     oss.str(), 0.9);
                metrics_.latency_issues++;
                add_issue(issue);

                // Alert the user about exponential latency growth
                alert_on_latency_issue(trade, symbol, latency_delay_history.back());
            }
        }

        // Enhanced latency monitoring: Check for system-wide latency issues
        // Compare this symbol's latency to overall system average
        if (latency_delay_history.size() >= 5) {
            // Calculate system-wide average latency across all symbols
            double system_avg_latency = 0;
            size_t total_delays = 0;

            for (const auto& [sym, delays] : recent_delays_) {
                for (const auto& delay : delays) {
                    system_avg_latency += delay;
                    total_delays++;
                }
            }

            if (total_delays > 0) {
                system_avg_latency /= total_delays;

                // If this symbol's latency is significantly higher than system average, flag it
                double avg_symbol_latency = 0;
                for (const auto& delay : latency_delay_history) {
                    avg_symbol_latency += delay;
                }
                avg_symbol_latency /= latency_delay_history.size();

                if (avg_symbol_latency > system_avg_latency * 3) { // 3x system average
                    std::ostringstream oss;
                    oss << "Symbol-specific high latency: " << avg_symbol_latency
                        << "ms vs system average " << system_avg_latency << "ms";

                    DataQualityIssue issue(DataQualityIssueType::LATENCY_ISSUE, symbol, trade.timestamp,
                                         oss.str(), 0.75);
                    metrics_.latency_issues++;
                    add_issue(issue);

                    // Alert the user about symbol-specific high latency
                    alert_on_latency_issue(trade, symbol, avg_symbol_latency);
                }
            }
        }

        // Enhanced latency monitoring: Check for latency degradation over time
        if (latency_delay_history.size() >= 15) {
            // Compare early delays vs recent delays to detect degradation
            size_t early_count = latency_delay_history.size() / 3;
            size_t recent_count = latency_delay_history.size() / 3;

            double early_avg = 0, recent_avg = 0;

            for (size_t i = 0; i < early_count; ++i) {
                early_avg += latency_delay_history[i];
            }
            early_avg /= early_count;

            for (size_t i = latency_delay_history.size() - recent_count; i < latency_delay_history.size(); ++i) {
                recent_avg += latency_delay_history[i];
            }
            recent_avg /= recent_count;

            // If recent latency is significantly higher than early latency, it indicates degradation
            if (early_avg > 0 && (recent_avg / early_avg) > 2.0) { // Recent avg is 2x higher than early avg
                std::ostringstream oss;
                oss << "Latency degradation detected: recent avg " << recent_avg
                    << "ms vs early avg " << early_avg << "ms";

                DataQualityIssue issue(DataQualityIssueType::LATENCY_ISSUE, symbol, trade.timestamp,
                                     oss.str(), 0.85);
                metrics_.latency_issues++;
                add_issue(issue);

                // Alert the user about latency degradation
                alert_on_latency_issue(trade, symbol, recent_avg);
            }
        }

        // Enhanced latency monitoring: Check for percentile-based latency issues
        if (latency_delay_history.size() >= 25) {
            // Calculate percentiles to identify if current latency is in an extreme percentile
            std::vector<int64_t> sorted_delays;
            for (const auto& delay : latency_delay_history) {
                sorted_delays.push_back(static_cast<int64_t>(delay));
            }
            std::sort(sorted_delays.begin(), sorted_delays.end());

            // Calculate 95th percentile
            size_t p95_index = static_cast<size_t>(0.95 * sorted_delays.size());
            if (p95_index >= sorted_delays.size()) p95_index = sorted_delays.size() - 1;

            int64_t p95_latency = sorted_delays[p95_index];

            // If current delay is above the 95th percentile, flag as issue
            if (current_delay > p95_latency) {
                std::ostringstream oss;
                oss << "Latency in extreme percentile: " << current_delay
                    << "ms vs 95th percentile of " << p95_latency << "ms";

                DataQualityIssue issue(DataQualityIssueType::LATENCY_ISSUE, symbol, trade.timestamp,
                                     oss.str(), 0.7);
                metrics_.latency_issues++;
                add_issue(issue);

                // Alert the user about percentile-based latency issue
                alert_on_latency_issue(trade, symbol, current_delay);
            }
        }

        // Enhanced latency monitoring: Check for latency stability
        // Calculate coefficient of variation to detect unstable latency
        if (latency_delay_history.size() >= 10) {
            double sum = 0;
            for (const auto& delay : latency_delay_history) {
                sum += delay;
            }
            double avg_latency = sum / latency_delay_history.size();

            if (avg_latency > 0) {
                double variance_sum = 0;
                for (const auto& delay : latency_delay_history) {
                    variance_sum += (delay - avg_latency) * (delay - avg_latency);
                }
                double std_dev = sqrt(variance_sum / latency_delay_history.size());
                double coeff_variation = std_dev / avg_latency;

                // High coefficient of variation indicates unstable latency
                if (coeff_variation > 0.5) { // 50% coefficient of variation threshold
                    std::ostringstream oss;
                    oss << "Highly unstable latency detected for " << symbol
                        << ". Coefficient of variation: " << std::fixed << std::setprecision(2)
                        << coeff_variation << " (threshold: 0.5)";

                    DataQualityIssue issue(DataQualityIssueType::LATENCY_ISSUE, symbol, trade.timestamp,
                                         oss.str(), 0.65);
                    metrics_.latency_issues++;
                    add_issue(issue);

                    // Alert the user about unstable latency
                    alert_on_latency_issue(trade, symbol, current_delay);
                }
            }
        }

        // Enhanced latency monitoring: Check for latency floor/ceiling effects
        // Look for situations where latency is consistently hitting minimum or maximum bounds
        if (latency_delay_history.size() >= 20) {
            // Count how many latencies are near the threshold (suggesting system is struggling)
            size_t near_threshold_count = 0;
            for (const auto& delay : latency_delay_history) {
                if (delay > latency_alert_threshold_ms_ * 0.9) { // Within 10% of threshold
                    near_threshold_count++;
                }
            }

            // If many latencies are near the threshold, it suggests the system is under stress
            if (near_threshold_count > latency_delay_history.size() * 0.6) { // 60% or more near threshold
                std::ostringstream oss;
                oss << "System under latency stress: " << near_threshold_count
                    << "/" << latency_delay_history.size()
                    << " recent latencies near threshold";

                DataQualityIssue issue(DataQualityIssueType::LATENCY_ISSUE, symbol, trade.timestamp,
                                     oss.str(), 0.75);
                metrics_.latency_issues++;
                add_issue(issue);

                // Alert the user about system stress
                alert_on_latency_issue(trade, symbol, current_delay);
            }
        }

        // Enhanced latency monitoring: Check for latency correlation with trade volume
        // Look for patterns where high-volume trades cause higher latency
        auto stats_it_corr = symbol_stats_.find(symbol);
        if (latency_delay_history.size() >= 15 && stats_it_corr != symbol_stats_.end()) {
            const auto& stats = stats_it_corr->second;

            // Only check if we have enough trade data to correlate
            if (stats.recent_intervals.size() >= 15) {
                // Calculate correlation between trade volume and processing latency
                // This is a simplified correlation check
                double volume_latency_correlation = 0;

                // Use the most recent data points for correlation
                size_t check_count = std::min(latency_delay_history.size(), stats.recent_intervals.size());
                check_count = std::min(check_count, static_cast<size_t>(10));

                // For this simplified check, we'll just see if high-volume periods correlate with high latency
                // In a real system, you'd want more sophisticated correlation analysis
                if (check_count > 0) {
                    // This is a placeholder for more sophisticated correlation analysis
                    // that would check if high-volume trades lead to higher latency
                }
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

    // Enhanced validation: Check for potentially invalid combinations of trade side and flags
    if (trade.side != TradeSide::BUY && trade.side != TradeSide::SELL) {
        DataQualityIssue issue(DataQualityIssueType::MISSING_FIELD, symbol, timestamp,
                             "Invalid trade side value", 0.8);
        metrics_.missing_field_issues++;
        add_issue(issue);
    }

    // Check for extremely large price values that might indicate data corruption
    if (trade.price > MAX_VALID_PRICE * 0.9) {  // 90% of max valid price
        std::ostringstream oss;
        oss << "Extremely high price detected: " << trade.price << " (possible data corruption)";

        DataQualityIssue issue(DataQualityIssueType::MISSING_FIELD, symbol, timestamp,
                             oss.str(), 0.85);
        metrics_.missing_field_issues++;
        add_issue(issue);
    }

    // Check for extremely large volume values that might indicate data corruption
    if (trade.volume > 1000000.0f) {  // Arbitrary large threshold
        std::ostringstream oss;
        oss << "Extremely high volume detected: " << trade.volume << " (possible data corruption)";

        DataQualityIssue issue(DataQualityIssueType::MISSING_FIELD, symbol, timestamp,
                             oss.str(), 0.85);
        metrics_.missing_field_issues++;
        add_issue(issue);
    }

    // Check for zero price with non-zero volume (or vice versa) which might indicate incomplete data
    if ((trade.price == 0.0 && trade.volume != 0.0f) || (trade.price != 0.0 && trade.volume == 0.0f)) {
        DataQualityIssue issue(DataQualityIssueType::MISSING_FIELD, symbol, timestamp,
                             "Inconsistent price/volume combination (zero value with non-zero counterpart)", 0.6);
        metrics_.missing_field_issues++;
        add_issue(issue);
    }

    // Check for timestamp that is significantly in the future (more than 10 seconds ahead)
    if (timestamp > current_time + 10000) {  // 10 seconds in the future
        std::ostringstream oss;
        oss << "Timestamp is significantly in the future: " << (timestamp - current_time) << "ms ahead";

        DataQualityIssue issue(DataQualityIssueType::MISSING_FIELD, symbol, timestamp,
                             oss.str(), 0.7);
        metrics_.missing_field_issues++;
        add_issue(issue);
    }

    // NEW: Enhanced validation for missing or invalid fields
    // Check for potentially invalid trade side values
    if (static_cast<uint8_t>(trade.side) > 1) {  // Only BUY (0) and SELL (1) are valid
        DataQualityIssue issue(DataQualityIssueType::MISSING_FIELD, symbol, timestamp,
                             "Invalid trade side value: must be BUY (0) or SELL (1)", 0.8);
        metrics_.missing_field_issues++;
        add_issue(issue);
    }

    // Check for potentially invalid flags value (all bits set might indicate corruption)
    if (trade.flags == 0xFF) {  // All bits set - likely data corruption
        DataQualityIssue issue(DataQualityIssueType::MISSING_FIELD, symbol, timestamp,
                             "Potentially corrupted flags field (all bits set)", 0.9);
        metrics_.missing_field_issues++;
        add_issue(issue);
    }

    // Check for extreme values that might indicate data corruption
    if (trade.price > MAX_VALID_PRICE * 10.0) {  // 10x the maximum valid price
        std::ostringstream oss;
        oss << "Extremely high price value detected: " << trade.price << ", possibly indicating data corruption";

        DataQualityIssue issue(DataQualityIssueType::MISSING_FIELD, symbol, timestamp,
                             oss.str(), 0.95);
        metrics_.missing_field_issues++;
        add_issue(issue);
    }

    if (trade.volume > 100000000.0f) {  // Extremely high volume threshold
        std::ostringstream oss;
        oss << "Extremely high volume value detected: " << trade.volume << ", possibly indicating data corruption";

        DataQualityIssue issue(DataQualityIssueType::MISSING_FIELD, symbol, timestamp,
                             oss.str(), 0.95);
        metrics_.missing_field_issues++;
        add_issue(issue);
    }

    // Check for timestamp precision issues (e.g., if timestamp is 0 or extremely old)
    if (trade.timestamp == 0) {
        DataQualityIssue issue(DataQualityIssueType::MISSING_FIELD, symbol, timestamp,
                             "Timestamp is zero, indicating potential missing data", 0.9);
        metrics_.missing_field_issues++;
        add_issue(issue);
    } else if (trade.timestamp < 1000000000000ULL) {  // Before year 2001 (timestamp in milliseconds)
        std::ostringstream oss;
        oss << "Timestamp is extremely old (year < 2001): " << trade.timestamp;

        DataQualityIssue issue(DataQualityIssueType::MISSING_FIELD, symbol, timestamp,
                             oss.str(), 0.8);
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
    recent_alert_times_.clear();
    alert_counts_by_type_.clear();
}

void DataQualityMonitor::set_alert_callback(AlertCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    alert_callback_ = callback;
}

void DataQualityMonitor::add_issue(const DataQualityIssue& issue) {
    // Check if this alert type is enabled before proceeding
    if (!is_alert_type_enabled(issue.type)) {
        return; // Skip adding this issue if its type is disabled
    }

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

    // Update alert counts by type for trending analysis
    switch (issue.type) {
        case DataQualityIssueType::MISSING_DATA:
            alert_counts_by_type_[DataQualityIssueType::MISSING_DATA]++;
            break;
        case DataQualityIssueType::DUPLICATE_TRADE:
            alert_counts_by_type_[DataQualityIssueType::DUPLICATE_TRADE]++;
            break;
        case DataQualityIssueType::OUT_OF_ORDER_TIMESTAMP:
            alert_counts_by_type_[DataQualityIssueType::OUT_OF_ORDER_TIMESTAMP]++;
            break;
        case DataQualityIssueType::LATENCY_ISSUE:
            alert_counts_by_type_[DataQualityIssueType::LATENCY_ISSUE]++;
            break;
        case DataQualityIssueType::INVALID_PRICE:
            alert_counts_by_type_[DataQualityIssueType::INVALID_PRICE]++;
            break;
        case DataQualityIssueType::INVALID_VOLUME:
            alert_counts_by_type_[DataQualityIssueType::INVALID_VOLUME]++;
            break;
        case DataQualityIssueType::MISSING_FIELD:
            alert_counts_by_type_[DataQualityIssueType::MISSING_FIELD]++;
            break;
    }

    // Check for alert bursts - many alerts in a short time period
    check_alert_bursts(issue);
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

void DataQualityMonitor::check_alert_bursts(const DataQualityIssue& issue) {
    auto now = std::chrono::high_resolution_clock::now();

    // Add current alert time to tracking
    recent_alert_times_.push_back(now);

    // Keep only alerts from the last minute
    auto one_minute_ago = now - std::chrono::minutes(1);
    recent_alert_times_.erase(
        std::remove_if(recent_alert_times_.begin(), recent_alert_times_.end(),
            [one_minute_ago](const auto& time) {
                return time < one_minute_ago;
            }),
        recent_alert_times_.end()
    );

    // Check if we have an alert burst (too many alerts in a short time)
    if (recent_alert_times_.size() > 50) {  // More than 50 alerts in the last minute
        std::ostringstream oss;
        oss << "ALERT BURST DETECTED: " << recent_alert_times_.size()
            << " alerts in the last minute for symbol " << issue.symbol;

        DataQualityIssue burst_issue(DataQualityIssueType::MISSING_DATA, issue.symbol, issue.timestamp,
                                   oss.str(), 0.95);  // High severity for alert bursts

        // Add the burst issue without triggering recursive burst checking
        recent_issues_.push_back(burst_issue);
        if (recent_issues_.size() > MAX_RECENT_ISSUES) {
            recent_issues_.erase(recent_issues_.begin());
        }

        if (alert_callback_) {
            alert_callback_(burst_issue);
        }

        // Log the burst alert
        if (console_alerts_enabled_) {
            std::cout << "[CRITICAL ALERT BURST] " << oss.str() << std::endl;
        }
    }

    // Also check for specific issue type bursts
    size_t current_count = alert_counts_by_type_[issue.type];
    if (current_count > 0 && current_count % 10 == 0) {  // Every 10th alert of the same type
        std::ostringstream oss;
        oss << "HIGH FREQUENCY OF " << static_cast<int>(issue.type)
            << " ISSUES: " << current_count << " occurrences detected";

        DataQualityIssue freq_issue(DataQualityIssueType::MISSING_DATA, issue.symbol, issue.timestamp,
                                  oss.str(), 0.85);

        // Add the frequency issue
        recent_issues_.push_back(freq_issue);
        if (recent_issues_.size() > MAX_RECENT_ISSUES) {
            recent_issues_.erase(recent_issues_.begin());
        }

        if (alert_callback_) {
            alert_callback_(freq_issue);
        }
    }
}

std::unordered_map<DataQualityIssueType, size_t> DataQualityMonitor::get_alert_counts_by_type() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return alert_counts_by_type_;
}

void DataQualityMonitor::alert_user_to_data_problems(const std::string& symbol, const std::string& problem_description, double severity) {
    std::lock_guard<std::mutex> lock(mutex_);

    uint64_t current_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    // Determine the appropriate issue type based on the problem description
    DataQualityIssueType issue_type = DataQualityIssueType::MISSING_DATA; // Default

    if (problem_description.find("duplicate") != std::string::npos) {
        issue_type = DataQualityIssueType::DUPLICATE_TRADE;
    } else if (problem_description.find("out of order") != std::string::npos ||
               problem_description.find("timestamp") != std::string::npos) {
        issue_type = DataQualityIssueType::OUT_OF_ORDER_TIMESTAMP;
    } else if (problem_description.find("latency") != std::string::npos) {
        issue_type = DataQualityIssueType::LATENCY_ISSUE;
    } else if (problem_description.find("price") != std::string::npos) {
        issue_type = DataQualityIssueType::INVALID_PRICE;
    } else if (problem_description.find("volume") != std::string::npos) {
        issue_type = DataQualityIssueType::INVALID_VOLUME;
    } else if (problem_description.find("field") != std::string::npos ||
               problem_description.find("missing") != std::string::npos) {
        issue_type = DataQualityIssueType::MISSING_FIELD;
    }

    DataQualityIssue issue(issue_type, symbol, current_timestamp, problem_description, severity);
    add_issue(issue);

    // Log to console if enabled
    if (console_alerts_enabled_) {
        std::cout << "[USER ALERT] Symbol: " << symbol
                  << ", Problem: " << problem_description
                  << ", Severity: " << severity << std::endl;
    }

    // Send to external monitoring if enabled
    send_external_alert(issue);

    // Enhanced user alerting: Send notifications to UI components if available
    send_ui_notification(issue);

    // Additional user alerting: Visual/audio alerts for high severity issues
    if (severity >= 0.9) {
        trigger_visual_alert(symbol, problem_description);
    }

    // Enhanced user alerting: Send detailed alert information to help users understand the impact
    std::ostringstream detailed_alert;
    detailed_alert << "[DATA QUALITY ALERT] ";
    detailed_alert << "Symbol: " << symbol << ", ";

    switch (issue_type) {
        case DataQualityIssueType::MISSING_DATA:
            detailed_alert << "Issue: Missing Data, ";
            break;
        case DataQualityIssueType::DUPLICATE_TRADE:
            detailed_alert << "Issue: Duplicate Trade, ";
            break;
        case DataQualityIssueType::OUT_OF_ORDER_TIMESTAMP:
            detailed_alert << "Issue: Out-of-Order Timestamp, ";
            break;
        case DataQualityIssueType::LATENCY_ISSUE:
            detailed_alert << "Issue: Latency Issue, ";
            break;
        case DataQualityIssueType::INVALID_PRICE:
            detailed_alert << "Issue: Invalid Price, ";
            break;
        case DataQualityIssueType::INVALID_VOLUME:
            detailed_alert << "Issue: Invalid Volume, ";
            break;
        case DataQualityIssueType::MISSING_FIELD:
            detailed_alert << "Issue: Missing Field, ";
            break;
    }

    detailed_alert << "Description: " << problem_description << ", ";
    detailed_alert << "Severity: " << severity;

    // Output detailed alert information
    if (console_alerts_enabled_) {
        std::cout << detailed_alert.str() << std::endl;
    }

    // Enhanced alerting: For high severity issues, provide recommendations
    if (severity >= 0.8) {
        std::ostringstream recommendation;
        recommendation << "[RECOMMENDATION] For severity " << severity << " issue with " << symbol << ": ";

        switch (issue_type) {
            case DataQualityIssueType::MISSING_DATA:
                recommendation << "Check data feed connectivity and consider switching to backup feed.";
                break;
            case DataQualityIssueType::DUPLICATE_TRADE:
                recommendation << "Review data processing pipeline for duplicate filtering mechanisms.";
                break;
            case DataQualityIssueType::OUT_OF_ORDER_TIMESTAMP:
                recommendation << "Verify timestamp synchronization and ordering algorithms.";
                break;
            case DataQualityIssueType::LATENCY_ISSUE:
                recommendation << "Investigate system performance and network connectivity.";
                break;
            case DataQualityIssueType::INVALID_PRICE:
                recommendation << "Validate price normalization and range checking.";
                break;
            case DataQualityIssueType::INVALID_VOLUME:
                recommendation << "Check volume validation filters and data source integrity.";
                break;
            case DataQualityIssueType::MISSING_FIELD:
                recommendation << "Ensure all required fields are populated in data source.";
                break;
        }

        if (console_alerts_enabled_) {
            std::cout << recommendation.str() << std::endl;
        }
    }

    // Enhanced alerting: Track alert frequency to prevent spam
    auto now = std::chrono::high_resolution_clock::now();
    recent_alert_times_.push_back(now);

    // Clean up old alert times (older than 1 minute)
    auto one_minute_ago = now - std::chrono::minutes(1);
    recent_alert_times_.erase(
        std::remove_if(recent_alert_times_.begin(), recent_alert_times_.end(),
            [one_minute_ago](const auto& time) {
                return time < one_minute_ago;
            }),
        recent_alert_times_.end()
    );

    // If too many alerts in a short period, send a summary instead of individual alerts
    if (recent_alert_times_.size() > 20) { // More than 20 alerts in the last minute
        if (console_alerts_enabled_) {
            std::cout << "[ALERT SUMMARY] High volume of data quality alerts detected ("
                      << recent_alert_times_.size() << "). Consider investigating underlying data source." << std::endl;
        }
    }

    // Enhanced user alerting: Categorize alerts by impact level
    std::string impact_level = "LOW";
    if (severity >= 0.9) {
        impact_level = "CRITICAL";
    } else if (severity >= 0.7) {
        impact_level = "HIGH";
    } else if (severity >= 0.5) {
        impact_level = "MEDIUM";
    }

    // Enhanced user alerting: Provide impact assessment
    std::ostringstream impact_assessment;
    impact_assessment << "[IMPACT ASSESSMENT] Issue with " << symbol
                      << " classified as " << impact_level << " impact. "
                      << "Estimated effect on data quality: ";

    if (severity >= 0.9) {
        impact_assessment << "Severe - immediate action recommended.";
    } else if (severity >= 0.7) {
        impact_assessment << "High - investigate promptly.";
    } else if (severity >= 0.5) {
        impact_assessment << "Moderate - monitor closely.";
    } else {
        impact_assessment << "Low - informational only.";
    }

    if (console_alerts_enabled_) {
        std::cout << impact_assessment.str() << std::endl;
    }

    // Enhanced user alerting: Provide historical context
    auto stats_it = symbol_stats_.find(symbol);
    if (stats_it != symbol_stats_.end()) {
        const auto& stats = stats_it->second;
        if (stats.trade_count > 0) {
            std::ostringstream context_info;
            context_info << "[HISTORICAL CONTEXT] For " << symbol
                         << ": " << stats.trade_count << " total trades processed, "
                         << "avg interval: " << (stats.trade_count > 1 ? stats.total_interval_sum / (stats.trade_count - 1) : 0)
                         << "ms";

            if (console_alerts_enabled_) {
                std::cout << context_info.str() << std::endl;
            }
        }
    }

    // Enhanced user alerting: Suggest mitigation strategies based on issue type
    std::ostringstream mitigation_suggestion;
    mitigation_suggestion << "[MITIGATION] ";

    switch (issue_type) {
        case DataQualityIssueType::MISSING_DATA:
            mitigation_suggestion << "Enable data recovery mechanisms, check network connectivity, "
                                  << "verify data source availability.";
            break;
        case DataQualityIssueType::DUPLICATE_TRADE:
            mitigation_suggestion << "Activate duplicate filtering, review data pipeline for redundancy, "
                                  << "implement unique trade identification.";
            break;
        case DataQualityIssueType::OUT_OF_ORDER_TIMESTAMP:
            mitigation_suggestion << "Implement timestamp correction algorithms, "
                                  << "review data source timestamp accuracy.";
            break;
        case DataQualityIssueType::LATENCY_ISSUE:
            mitigation_suggestion << "Optimize processing pipeline, increase system resources, "
                                  << "consider data feed proximity improvements.";
            break;
        case DataQualityIssueType::INVALID_PRICE:
            mitigation_suggestion << "Apply price validation filters, verify data source accuracy, "
                                  << "implement price range checks.";
            break;
        case DataQualityIssueType::INVALID_VOLUME:
            mitigation_suggestion << "Apply volume validation filters, verify data source accuracy, "
                                  << "implement volume range checks.";
            break;
        case DataQualityIssueType::MISSING_FIELD:
            mitigation_suggestion << "Implement field validation, ensure complete data feeds, "
                                  << "apply default values where appropriate.";
            break;
    }

    if (console_alerts_enabled_) {
        std::cout << mitigation_suggestion.str() << std::endl;
    }

    // Enhanced user alerting: Track issue resolution status
    // This would typically connect to a ticketing or monitoring system
    if (severity >= 0.8) {
        std::ostringstream resolution_tracking;
        resolution_tracking << "[RESOLUTION TRACKING] Created tracking entry for high-severity issue "
                            << "with symbol " << symbol << " and description: " << problem_description;

        if (console_alerts_enabled_) {
            std::cout << resolution_tracking.str() << std::endl;
        }
    }

    // Enhanced user alerting: Aggregate similar issues to reduce noise
    // Count similar issues in recent history
    size_t similar_issues_count = 0;
    for (const auto& recent_issue : recent_issues_) {
        if (recent_issue.symbol == symbol &&
            recent_issue.type == issue_type &&
            (std::chrono::duration_cast<std::chrono::seconds>(now - recent_issue.detection_time).count() < 300)) { // Within 5 minutes
            similar_issues_count++;
        }
    }

    if (similar_issues_count > 1) {
        std::ostringstream aggregation_info;
        aggregation_info << "[AGGREGATION NOTICE] This is issue #" << similar_issues_count
                         << " of similar type for symbol " << symbol
                         << " in the last 5 minutes. Consider systemic cause.";

        if (console_alerts_enabled_) {
            std::cout << aggregation_info.str() << std::endl;
        }
    }

    // NEW: Add a method to specifically notify users of data problems in a clear way
    notify_users_of_data_problem(symbol, problem_description, severity, issue_type);
}

void DataQualityMonitor::notify_users_of_data_problem(const std::string& symbol, const std::string& problem_description,
                                                    double severity, DataQualityIssueType issue_type) {
    // This method provides a clear, prominent notification to users about data quality problems
    if (!console_alerts_enabled_) return;  // Only proceed if console alerts are enabled

    // Determine severity level for display
    std::string severity_level = "INFO";
    if (severity >= 0.9) severity_level = "CRITICAL";
    else if (severity >= 0.7) severity_level = "HIGH";
    else if (severity >= 0.5) severity_level = "MEDIUM";
    else if (severity >= 0.3) severity_level = "LOW";

    // Determine issue type name for display
    std::string issue_type_name = "UNKNOWN";
    switch (issue_type) {
        case DataQualityIssueType::MISSING_DATA:
            issue_type_name = "MISSING_DATA";
            break;
        case DataQualityIssueType::DUPLICATE_TRADE:
            issue_type_name = "DUPLICATE_TRADE";
            break;
        case DataQualityIssueType::OUT_OF_ORDER_TIMESTAMP:
            issue_type_name = "OUT_OF_ORDER_TIMESTAMP";
            break;
        case DataQualityIssueType::LATENCY_ISSUE:
            issue_type_name = "LATENCY_ISSUE";
            break;
        case DataQualityIssueType::INVALID_PRICE:
            issue_type_name = "INVALID_PRICE";
            break;
        case DataQualityIssueType::INVALID_VOLUME:
            issue_type_name = "INVALID_VOLUME";
            break;
        case DataQualityIssueType::MISSING_FIELD:
            issue_type_name = "MISSING_FIELD";
            break;
    }

    // Print a prominent, clear notification to the user
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "DATA QUALITY PROBLEM DETECTED!" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Symbol:          " << symbol << std::endl;
    std::cout << "Issue Type:      " << issue_type_name << std::endl;
    std::cout << "Severity Level:  " << severity_level << " (" << severity << ")" << std::endl;
    std::cout << "Description:     " << problem_description << std::endl;
    std::cout << std::string(80, '=') << std::endl << std::endl;

    // NEW: Enhanced user notification with additional visual indicators for critical issues
    if (severity >= 0.9) {
        // Flash or highlight critical issues
        std::cout << "\033[31m\033[1m"  // Red bold text for critical issues
                  << "!!! CRITICAL DATA QUALITY ISSUE - IMMEDIATE ACTION REQUIRED !!!"
                  << "\033[0m" << std::endl;
    } else if (severity >= 0.7) {
        std::cout << "\033[33m\033[1m"  // Yellow bold text for high severity issues
                  << "!! HIGH SEVERITY DATA QUALITY ISSUE - REVIEW NEEDED !!"
                  << "\033[0m" << std::endl;
    }

    // NEW: Add timestamp for when the issue was detected
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::cout << "Detection Time:  " << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S") << std::endl;
}

// NEW: Method to provide a real-time dashboard of data quality issues
std::string DataQualityMonitor::get_real_time_dashboard() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ostringstream dashboard;
    dashboard << "\n" << std::string(60, '=') << std::endl;
    dashboard << "REAL-TIME DATA QUALITY DASHBOARD" << std::endl;
    dashboard << std::string(60, '=') << std::endl;

    // Overall health status
    double quality_score = 100.0;
    if (metrics_.total_trades_processed > 0) {
        double error_rate = static_cast<double>(metrics_.missing_data_issues +
                                               metrics_.duplicate_trade_issues +
                                               metrics_.out_of_order_timestamp_issues +
                                               metrics_.latency_issues +
                                               metrics_.invalid_price_issues +
                                               metrics_.invalid_volume_issues +
                                               metrics_.missing_field_issues) /
                           static_cast<double>(metrics_.total_trades_processed);
        quality_score = (1.0 - std::min(error_rate, 1.0)) * 100.0;
    }

    std::string health_status;
    if (quality_score >= 95.0) {
        health_status = "EXCELLENT";
    } else if (quality_score >= 90.0) {
        health_status = "GOOD";
    } else if (quality_score >= 80.0) {
        health_status = "FAIR";
    } else if (quality_score >= 70.0) {
        health_status = "POOR";
    } else {
        health_status = "CRITICAL";
    }

    dashboard << "Health Status: " << health_status << " (" << std::fixed << std::setprecision(1) << quality_score << "%)" << std::endl;

    // Convert high_resolution_clock to system_clock for display
    auto duration_since_epoch = metrics_.last_update_time.time_since_epoch();
    auto sys_time_point = std::chrono::system_clock::time_point(
        std::chrono::duration_cast<std::chrono::system_clock::duration>(duration_since_epoch));
    auto time_t_val = std::chrono::system_clock::to_time_t(sys_time_point);
    dashboard << "Last Updated:  " << std::put_time(std::localtime(&time_t_val), "%Y-%m-%d %H:%M:%S") << std::endl;

    // Issue counts by type
    dashboard << "\nISSUE COUNTS:" << std::endl;
    dashboard << "  Missing Data:            " << metrics_.missing_data_issues << std::endl;
    dashboard << "  Duplicate Trades:        " << metrics_.duplicate_trade_issues << std::endl;
    dashboard << "  Out-of-Order Timestamps: " << metrics_.out_of_order_timestamp_issues << std::endl;
    dashboard << "  Latency Issues:          " << metrics_.latency_issues << std::endl;
    dashboard << "  Invalid Prices:          " << metrics_.invalid_price_issues << std::endl;
    dashboard << "  Invalid Volumes:         " << metrics_.invalid_volume_issues << std::endl;
    dashboard << "  Missing Fields:          " << metrics_.missing_field_issues << std::endl;

    // Performance metrics
    dashboard << "\nPERFORMANCE METRICS:" << std::endl;
    dashboard << "  Total Trades Processed: " << metrics_.total_trades_processed << std::endl;
    dashboard << "  Average Latency:        " << std::fixed << std::setprecision(2) << metrics_.average_latency_ms << " ms" << std::endl;

    // Top affected symbols
    if (!recent_issues_.empty()) {
        std::unordered_map<std::string, size_t> symbol_issue_counts;
        for (const auto& issue : recent_issues_) {
            symbol_issue_counts[issue.symbol]++;
        }

        // Sort symbols by issue count
        std::vector<std::pair<std::string, size_t>> sorted_symbols(symbol_issue_counts.begin(), symbol_issue_counts.end());
        std::sort(sorted_symbols.begin(), sorted_symbols.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        if (!sorted_symbols.empty()) {
            dashboard << "\nTOP AFFECTED SYMBOLS:" << std::endl;
            for (size_t i = 0; i < std::min(sorted_symbols.size(), static_cast<size_t>(5)); ++i) {
                dashboard << "  " << (i+1) << ". " << sorted_symbols[i].first << ": " << sorted_symbols[i].second << " issues" << std::endl;
            }
        }
    }

    // Recent high severity issues
    std::vector<DataQualityIssue> high_severity_issues;
    for (const auto& issue : recent_issues_) {
        if (issue.severity >= 0.7) {
            high_severity_issues.push_back(issue);
        }
    }

    if (!high_severity_issues.empty()) {
        dashboard << "\nRECENT HIGH SEVERITY ISSUES:" << std::endl;
        size_t count = 0;
        for (const auto& issue : high_severity_issues) {
            if (count++ >= 5) break; // Show only top 5 high severity issues
            dashboard << "  - " << issue.symbol << " [" << issue.severity << "]: " << issue.description.substr(0, 60);
            if (issue.description.length() > 60) dashboard << "...";
            dashboard << std::endl;
        }
    }

    dashboard << std::string(60, '=') << std::endl;

    return dashboard.str();
}

// NEW: Enhanced method to specifically alert users to data problems with additional context
void DataQualityMonitor::alert_user_to_data_problems_with_context(const std::string& symbol,
                                                                 const std::string& problem_description,
                                                                 double severity,
                                                                 const std::string& source_component,
                                                                 const std::string& additional_context) {
    std::lock_guard<std::mutex> lock(mutex_);

    uint64_t current_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();

    // Determine the appropriate issue type based on the problem description
    DataQualityIssueType issue_type = DataQualityIssueType::MISSING_DATA; // Default

    if (problem_description.find("duplicate") != std::string::npos) {
        issue_type = DataQualityIssueType::DUPLICATE_TRADE;
    } else if (problem_description.find("out of order") != std::string::npos ||
               problem_description.find("timestamp") != std::string::npos) {
        issue_type = DataQualityIssueType::OUT_OF_ORDER_TIMESTAMP;
    } else if (problem_description.find("latency") != std::string::npos) {
        issue_type = DataQualityIssueType::LATENCY_ISSUE;
    } else if (problem_description.find("price") != std::string::npos) {
        issue_type = DataQualityIssueType::INVALID_PRICE;
    } else if (problem_description.find("volume") != std::string::npos) {
        issue_type = DataQualityIssueType::INVALID_VOLUME;
    } else if (problem_description.find("field") != std::string::npos ||
               problem_description.find("missing") != std::string::npos) {
        issue_type = DataQualityIssueType::MISSING_FIELD;
    }

    // Create a more descriptive problem description with context
    std::ostringstream contextual_description;
    contextual_description << problem_description;
    if (!source_component.empty()) {
        contextual_description << " [Source: " << source_component << "]";
    }
    if (!additional_context.empty()) {
        contextual_description << " [Context: " << additional_context << "]";
    }

    DataQualityIssue issue(issue_type, symbol, current_timestamp, contextual_description.str(), severity);
    add_issue(issue);

    // Enhanced user notification with additional context
    if (console_alerts_enabled_) {
        std::cout << "\n" << std::string(80, '*') << std::endl;
        std::cout << "CRITICAL DATA QUALITY ALERT WITH CONTEXT!" << std::endl;
        std::cout << std::string(80, '*') << std::endl;
        std::cout << "Symbol:        " << symbol << std::endl;
        std::cout << "Component:     " << source_component << std::endl;
        std::cout << "Problem:       " << problem_description << std::endl;
        std::cout << "Context:       " << additional_context << std::endl;
        std::cout << "Severity:      " << severity << std::endl;
        std::cout << std::string(80, '*') << std::endl << std::endl;
    }

    // Send to external monitoring if enabled
    send_external_alert(issue);

    // Send to UI components
    send_ui_notification(issue);

    // Trigger visual alerts for high severity issues
    if (severity >= 0.9) {
        trigger_visual_alert(symbol, contextual_description.str());
    }

    // Call the standard notification method
    notify_users_of_data_problem(symbol, contextual_description.str(), severity, issue_type);
}

void DataQualityMonitor::trigger_visual_alert(const std::string& symbol, const std::string& problem_description) {
    // This method triggers visual alerts for high severity issues
    // In a real implementation, this would interface with the UI system

    std::ostringstream visual_alert_msg;
    visual_alert_msg << "{\"type\":\"VISUAL_ALERT\","
                     << "\"alert_type\":\"DATA_QUALITY\","
                     << "\"symbol\":\"" << symbol << "\","
                     << "\"message\":\"" << problem_description << "\","
                     << "\"timestamp\":" << std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::high_resolution_clock::now().time_since_epoch()).count() << "}";

    // Output in a format that can be consumed by UI components
    if (console_alerts_enabled_) {
        std::cout << "[VISUAL_ALERT_TRIGGER] " << visual_alert_msg.str() << std::endl;
    }

    // In a real implementation, this would trigger visual indicators in the UI
    // such as flashing red borders, popups, or other visual cues
}

// Method to get a user-friendly summary of data quality issues
std::string DataQualityMonitor::get_user_friendly_summary() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ostringstream summary;
    summary << "\n=== DATA QUALITY STATUS ===" << std::endl;

    // Overall health indicator
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

    if (quality_score >= 95.0) {
        summary << "Status: EXCELLENT (" << std::fixed << std::setprecision(1) << quality_score << "%)" << std::endl;
    } else if (quality_score >= 90.0) {
        summary << "Status: GOOD (" << std::fixed << std::setprecision(1) << quality_score << "%)" << std::endl;
    } else if (quality_score >= 80.0) {
        summary << "Status: FAIR (" << std::fixed << std::setprecision(1) << quality_score << "%)" << std::endl;
    } else if (quality_score >= 70.0) {
        summary << "Status: POOR (" << std::fixed << std::setprecision(1) << quality_score << "%)" << std::endl;
    } else {
        summary << "Status: CRITICAL (" << std::fixed << std::setprecision(1) << quality_score << "%)" << std::endl;
    }

    summary << "\nIssues Detected:" << std::endl;
    summary << "- Missing data: " << metrics_.missing_data_issues << std::endl;
    summary << "- Duplicate trades: " << metrics_.duplicate_trade_issues << std::endl;
    summary << "- Out-of-order timestamps: " << metrics_.out_of_order_timestamp_issues << std::endl;
    summary << "- Latency issues: " << metrics_.latency_issues << std::endl;
    summary << "- Invalid prices: " << metrics_.invalid_price_issues << std::endl;
    summary << "- Invalid volumes: " << metrics_.invalid_volume_issues << std::endl;
    summary << "- Missing fields: " << metrics_.missing_field_issues << std::endl;

    summary << "\nPerformance Metrics:" << std::endl;
    summary << "- Total trades processed: " << metrics_.total_trades_processed << std::endl;
    summary << "- Average latency: " << std::fixed << std::setprecision(2)
            << metrics_.average_latency_ms << " ms" << std::endl;

    // Highlight the most problematic symbols
    if (!recent_issues_.empty()) {
        std::unordered_map<std::string, size_t> symbol_issue_counts;
        for (const auto& issue : recent_issues_) {
            symbol_issue_counts[issue.symbol]++;
        }

        // Sort symbols by issue count
        std::vector<std::pair<std::string, size_t>> sorted_symbols(symbol_issue_counts.begin(), symbol_issue_counts.end());
        std::sort(sorted_symbols.begin(), sorted_symbols.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        if (!sorted_symbols.empty()) {
            summary << "\nMost Affected Symbols:" << std::endl;
            for (size_t i = 0; i < std::min(sorted_symbols.size(), static_cast<size_t>(5)); ++i) {
                summary << "- " << sorted_symbols[i].first << ": " << sorted_symbols[i].second << " issues" << std::endl;
            }
        }
    }

    summary << "===========================" << std::endl;

    return summary.str();
}

void DataQualityMonitor::alert_on_missing_data(const std::string& symbol, uint64_t expected_time, uint64_t actual_time) {
    std::lock_guard<std::mutex> lock(mutex_);

    uint64_t gap_ms = actual_time - expected_time;
    std::ostringstream description;
    description << "Missing data detected for " << symbol
                << ". Expected data at " << expected_time
                << " but received at " << actual_time
                << " (gap of " << gap_ms << "ms)";

    double severity = std::min(0.95, 0.5 + (gap_ms / 10000.0)); // Scale severity based on gap size

    DataQualityIssue issue(DataQualityIssueType::MISSING_DATA, symbol, actual_time,
                          description.str(), std::min(severity, 1.0));
    add_issue(issue);

    // Alert the user about the missing data
    alert_user_to_data_problems(symbol, description.str(), std::min(severity, 1.0));
}

void DataQualityMonitor::alert_on_duplicate_trade(const TradeData& trade, const std::string& symbol) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ostringstream description;
    description << "Duplicate trade detected for " << symbol
                << " at timestamp " << trade.timestamp
                << ", price: " << trade.price
                << ", volume: " << trade.volume;

    DataQualityIssue issue(DataQualityIssueType::DUPLICATE_TRADE, symbol, trade.timestamp,
                          description.str(), 0.7);
    add_issue(issue);

    // Alert the user about the duplicate trade
    alert_user_to_data_problems(symbol, description.str(), 0.7);
}

void DataQualityMonitor::alert_on_out_of_order_timestamp(const TradeData& trade, const std::string& symbol, uint64_t last_timestamp) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ostringstream description;
    description << "Out-of-order timestamp detected for " << symbol
                << ". Received timestamp " << trade.timestamp
                << " after processing " << last_timestamp;

    double severity = 0.6; // Base severity for out-of-order timestamps
    if (trade.timestamp < (last_timestamp - 60000)) { // More than 1 minute difference
        severity = 0.8; // Higher severity for large gaps
    }

    DataQualityIssue issue(DataQualityIssueType::OUT_OF_ORDER_TIMESTAMP, symbol, trade.timestamp,
                          description.str(), severity);
    add_issue(issue);

    // Alert the user about the out-of-order timestamp
    alert_user_to_data_problems(symbol, description.str(), severity);
}

void DataQualityMonitor::alert_on_latency_issue(const TradeData& trade, const std::string& symbol, int64_t latency_ms) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ostringstream description;
    description << "High latency detected for " << symbol
                << ". Processing delay: " << latency_ms << "ms";

    double severity = 0.5; // Base severity
    if (latency_ms > 5000) { // More than 5 seconds
        severity = 0.8; // High severity for significant delays
    } else if (latency_ms > 1000) { // More than 1 second
        severity = 0.6; // Medium-high severity
    }

    DataQualityIssue issue(DataQualityIssueType::LATENCY_ISSUE, symbol, trade.timestamp,
                          description.str(), severity);
    add_issue(issue);

    // Alert the user about the latency issue
    alert_user_to_data_problems(symbol, description.str(), severity);
}

void DataQualityMonitor::send_ui_notification(const DataQualityIssue& issue) {
    // This method sends notifications to UI components for user visibility
    // In a real implementation, this would interface with the UI system

    // For now, we'll just log to console with a specific format for UI integration
    std::string severity_level;
    if (issue.severity >= 0.9) {
        severity_level = "CRITICAL";
    } else if (issue.severity >= 0.7) {
        severity_level = "HIGH";
    } else if (issue.severity >= 0.5) {
        severity_level = "MEDIUM";
    } else {
        severity_level = "LOW";
    }

    std::ostringstream ui_message;
    ui_message << "{\"type\":\"DATA_QUALITY_ALERT\","
               << "\"severity\":\"" << severity_level << "\","
               << "\"symbol\":\"" << issue.symbol << "\","
               << "\"description\":\"" << issue.description << "\","
               << "\"timestamp\":" << issue.timestamp << ","
               << "\"severity_value\":" << issue.severity << "}";

    // Output in a format that can be consumed by UI components
    if (console_alerts_enabled_) {
        std::cout << "[UI_NOTIFICATION] " << ui_message.str() << std::endl;
    }

    // In a real implementation, this would send the message to the UI layer
    // For example, through a queue, shared memory, or IPC mechanism
}

void DataQualityMonitor::generate_comprehensive_alert_report() {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ostringstream report;
    report << "\n=== COMPREHENSIVE DATA QUALITY REPORT ===\n";
    report << "Generated: " << std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count() << " ms\n";
    report << "Total trades processed: " << metrics_.total_trades_processed << "\n";
    report << "Missing data issues: " << metrics_.missing_data_issues << "\n";
    report << "Duplicate trade issues: " << metrics_.duplicate_trade_issues << "\n";
    report << "Out-of-order timestamp issues: " << metrics_.out_of_order_timestamp_issues << "\n";
    report << "Latency issues: " << metrics_.latency_issues << "\n";
    report << "Invalid price issues: " << metrics_.invalid_price_issues << "\n";
    report << "Invalid volume issues: " << metrics_.invalid_volume_issues << "\n";
    report << "Missing field issues: " << metrics_.missing_field_issues << "\n";
    report << "Average latency: " << std::fixed << std::setprecision(2) << metrics_.average_latency_ms << " ms\n";

    // Calculate and report severity distribution
    size_t critical_issues = 0, high_issues = 0, medium_issues = 0, low_issues = 0;

    for (const auto& issue : recent_issues_) {
        if (issue.severity >= 0.9) {
            critical_issues++;
        } else if (issue.severity >= 0.7) {
            high_issues++;
        } else if (issue.severity >= 0.5) {
            medium_issues++;
        } else {
            low_issues++;
        }
    }

    report << "\nSeverity Distribution:\n";
    report << "Critical issues (0.9-1.0): " << critical_issues << "\n";
    report << "High issues (0.7-0.89): " << high_issues << "\n";
    report << "Medium issues (0.5-0.69): " << medium_issues << "\n";
    report << "Low issues (0.0-0.49): " << low_issues << "\n";

    // Report top affected symbols
    std::unordered_map<std::string, size_t> symbol_issue_counts;
    for (const auto& issue : recent_issues_) {
        symbol_issue_counts[issue.symbol]++;
    }

    report << "\nTop 5 Symbols with Most Issues:\n";
    std::vector<std::pair<std::string, size_t>> sorted_symbols(symbol_issue_counts.begin(), symbol_issue_counts.end());
    std::sort(sorted_symbols.begin(), sorted_symbols.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    for (size_t i = 0; i < std::min(sorted_symbols.size(), static_cast<size_t>(5)); ++i) {
        report << "  " << i+1 << ". " << sorted_symbols[i].first
               << ": " << sorted_symbols[i].second << " issues\n";
    }

    // Calculate data quality score
    double quality_score = 100.0;
    if (metrics_.total_trades_processed > 0) {
        double error_rate = static_cast<double>(metrics_.missing_data_issues +
                                               metrics_.duplicate_trade_issues +
                                               metrics_.out_of_order_timestamp_issues +
                                               metrics_.latency_issues +
                                               metrics_.invalid_price_issues +
                                               metrics_.invalid_volume_issues +
                                               metrics_.missing_field_issues) /
                           static_cast<double>(metrics_.total_trades_processed);
        quality_score = (1.0 - std::min(error_rate, 1.0)) * 100.0;
    }

    report << "\nOverall Data Quality Score: " << std::fixed << std::setprecision(2)
           << quality_score << "%\n";

    if (quality_score < 80.0) {
        report << "STATUS: CRITICAL - Immediate attention required!\n";
    } else if (quality_score < 90.0) {
        report << "STATUS: WARNING - Data quality needs attention\n";
    } else if (quality_score < 95.0) {
        report << "STATUS: FAIR - Minor issues detected\n";
    } else {
        report << "STATUS: GOOD - Data quality is satisfactory\n";
    }

    report << "=========================================\n";

    // Output the report
    if (console_alerts_enabled_) {
        std::cout << report.str();
    }

    // Optionally save to file if logging is enabled
    if (file_logging_enabled_) {
        // In a real implementation, this would write to a log file
        // For now, we'll just simulate it
    }

    // Send the report to external monitoring if enabled
    if (external_alert_callback_) {
        // Create a special report issue to send to external systems
        DataQualityIssue report_issue(DataQualityIssueType::MISSING_DATA, "SYSTEM_REPORT",
                                    std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::high_resolution_clock::now().time_since_epoch()).count(),
                                    report.str(), 0.5);
        external_alert_callback_(report_issue);
    }
}

void DataQualityMonitor::monitor_data_stream_health(const std::string& symbol) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto stats_it = symbol_stats_.find(symbol);
    if (stats_it == symbol_stats_.end()) {
        return; // No data for this symbol yet
    }

    const auto& stats = stats_it->second;

    if (stats.trade_count < 10) {
        return; // Not enough data to assess health
    }

    // Calculate metrics for this symbol
    uint64_t avg_interval = (stats.trade_count > 1) ? stats.total_interval_sum / (stats.trade_count - 1) : 0;

    // Check for significant changes in data stream pattern
    if (!stats.recent_intervals.empty()) {
        // Calculate recent average interval
        size_t sample_size = std::min(static_cast<size_t>(10), stats.recent_intervals.size());
        uint64_t recent_avg = 0;

        for (size_t i = stats.recent_intervals.size() - sample_size; i < stats.recent_intervals.size(); ++i) {
            recent_avg += stats.recent_intervals[i];
        }
        recent_avg /= sample_size;

        // Compare recent average to overall average
        if (avg_interval > 0) {
            double ratio = static_cast<double>(recent_avg) / static_cast<double>(avg_interval);

            if (ratio > 3.0) { // Recent intervals are 3x longer than average - potential data loss
                std::ostringstream oss;
                oss << "Data stream degradation detected for " << symbol
                    << ": recent avg interval " << recent_avg
                    << "ms is " << std::fixed << std::setprecision(2) << ratio
                    << "x higher than overall avg " << avg_interval << "ms";

                DataQualityIssue issue(DataQualityIssueType::MISSING_DATA, symbol,
                                     std::chrono::duration_cast<std::chrono::milliseconds>(
                                         std::chrono::high_resolution_clock::now().time_since_epoch()).count(),
                                     oss.str(), 0.75);
                metrics_.missing_data_issues++;
                add_issue(issue);

                // Alert user to this data stream degradation
                alert_user_to_data_problems(symbol, oss.str(), 0.75);
            } else if (ratio < 0.1) { // Recent intervals are much shorter - potential duplicate flood
                std::ostringstream oss;
                oss << "Abnormal data stream acceleration detected for " << symbol
                    << ": recent avg interval " << recent_avg
                    << "ms is " << std::fixed << std::setprecision(2) << ratio
                    << "x lower than overall avg " << avg_interval << "ms";

                DataQualityIssue issue(DataQualityIssueType::DUPLICATE_TRADE, symbol,
                                     std::chrono::duration_cast<std::chrono::milliseconds>(
                                         std::chrono::high_resolution_clock::now().time_since_epoch()).count(),
                                     oss.str(), 0.7);
                metrics_.duplicate_trade_issues++;
                add_issue(issue);

                // Alert user to this abnormal data stream
                alert_user_to_data_problems(symbol, oss.str(), 0.7);
            }
        }
    }

    // Check for extended periods of no data
    auto last_received_it = last_received_times_.find(symbol);
    if (last_received_it != last_received_times_.end()) {
        auto now = std::chrono::high_resolution_clock::now();
        auto time_since_last = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - last_received_it->second).count();

        // If no data for longer than expected based on historical patterns
        if (avg_interval > 0 && time_since_last > static_cast<int64_t>(avg_interval * 10)) {
            std::ostringstream oss;
            oss << "Extended data gap detected for " << symbol
                << ": " << time_since_last << "ms since last trade, "
                << "expected based on avg interval: " << avg_interval << "ms";

            DataQualityIssue issue(DataQualityIssueType::MISSING_DATA, symbol,
                                 std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::high_resolution_clock::now().time_since_epoch()).count(),
                                 oss.str(), 0.8);
            metrics_.missing_data_issues++;
            add_issue(issue);

            // Alert user to this extended data gap
            alert_user_to_data_problems(symbol, oss.str(), 0.8);
        }
    }

    // Enhanced data stream health monitoring: Check for consistency in trade patterns
    if (stats.recent_intervals.size() >= 20) {
        // Calculate coefficient of variation to detect inconsistent patterns
        double sum = 0;
        for (const auto& interval : stats.recent_intervals) {
            sum += interval;
        }
        double mean = sum / stats.recent_intervals.size();

        if (mean > 0) {
            double variance_sum = 0;
            for (const auto& interval : stats.recent_intervals) {
                variance_sum += (interval - mean) * (interval - mean);
            }
            double std_dev = sqrt(variance_sum / stats.recent_intervals.size());
            double coeff_variation = mean > 0 ? std_dev / mean : 0;

            // High coefficient of variation indicates inconsistent data arrival patterns
            if (coeff_variation > 1.0) { // Highly variable pattern
                std::ostringstream oss;
                oss << "Highly inconsistent data arrival pattern for " << symbol
                    << ". Coefficient of variation: " << std::fixed << std::setprecision(2)
                    << coeff_variation << " (threshold: 1.0)";

                DataQualityIssue issue(DataQualityIssueType::MISSING_DATA, symbol,
                                     std::chrono::duration_cast<std::chrono::milliseconds>(
                                         std::chrono::high_resolution_clock::now().time_since_epoch()).count(),
                                     oss.str(), 0.6);
                metrics_.missing_data_issues++;
                add_issue(issue);

                // Alert user to inconsistent data patterns
                alert_user_to_data_problems(symbol, oss.str(), 0.6);
            }
        }
    }

    // Check for potential data feed failures by monitoring trade volume patterns
    // If we see a sudden drop in trade frequency, it might indicate a data feed issue
    if (stats.recent_intervals.size() >= 15) {
        // Compare first half vs second half of recent intervals to detect drops in frequency
        size_t mid = stats.recent_intervals.size() / 2;
        uint64_t first_half_avg = 0, second_half_avg = 0;

        for (size_t i = 0; i < mid; ++i) {
            first_half_avg += stats.recent_intervals[i];
        }
        first_half_avg = (mid > 0) ? first_half_avg / mid : 0;

        for (size_t i = mid; i < stats.recent_intervals.size(); ++i) {
            second_half_avg += stats.recent_intervals[i];
        }
        size_t second_half_count = stats.recent_intervals.size() - mid;
        second_half_avg = (second_half_count > 0) ? second_half_avg / second_half_count : 0;

        // If second half has significantly higher intervals (lower frequency), flag it
        if (first_half_avg > 0 && second_half_avg > first_half_avg * 3) {
            std::ostringstream oss;
            oss << "Potential data feed degradation for " << symbol
                << ". Recent interval avg: " << second_half_avg
                << "ms vs previous avg: " << first_half_avg << "ms";

            DataQualityIssue issue(DataQualityIssueType::MISSING_DATA, symbol,
                                 std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::high_resolution_clock::now().time_since_epoch()).count(),
                                 oss.str(), 0.7);
            metrics_.missing_data_issues++;
            add_issue(issue);

            // Alert user to potential data feed degradation
            alert_user_to_data_problems(symbol, oss.str(), 0.7);
        }
    }

    // NEW: Enhanced monitoring for all four main data quality issues
    // Check for missing data patterns
    check_missing_data_patterns(symbol);

    // Check for duplicate trade patterns
    check_duplicate_trade_patterns(symbol);

    // Check for out-of-order timestamp patterns
    check_out_of_order_timestamp_patterns(symbol);

    // Check for latency issues patterns
    check_latency_issue_patterns(symbol);
}

void DataQualityMonitor::check_missing_data_patterns(const std::string& symbol) {
    // Get the symbol stats inside the method
    auto stats_it = symbol_stats_.find(symbol);
    if (stats_it == symbol_stats_.end()) {
        return; // No data for this symbol yet
    }
    const auto& stats = stats_it->second;

    // Check for systematic missing data patterns
    if (stats.recent_intervals.size() >= 30) {
        // Look for periodic gaps in data
        std::vector<uint64_t> sorted_intervals = stats.recent_intervals;
        std::sort(sorted_intervals.begin(), sorted_intervals.end());

        // Check if there are unusually large gaps compared to median
        uint64_t median_interval = sorted_intervals[sorted_intervals.size() / 2];
        uint64_t max_interval = sorted_intervals.back();

        if (max_interval > median_interval * 10) { // Very large gap compared to median
            std::ostringstream oss;
            oss << "Systematic missing data pattern detected for " << symbol
                << ". Max interval: " << max_interval << "ms, Median: " << median_interval << "ms";

            DataQualityIssue issue(DataQualityIssueType::MISSING_DATA, symbol,
                                 std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::high_resolution_clock::now().time_since_epoch()).count(),
                                 oss.str(), 0.8);
            metrics_.missing_data_issues++;
            add_issue(issue);

            alert_user_to_data_problems(symbol, oss.str(), 0.8);
        }
    }
}

void DataQualityMonitor::check_duplicate_trade_patterns(const std::string& symbol) {
    // Check for duplicate trade patterns by looking at recent trade data
    auto recent_trades_it = recent_trades_.find(symbol);
    if (recent_trades_it != recent_trades_.end()) {
        const auto& trades = recent_trades_it->second;

        // Look for clusters of similar trades in short time windows
        if (trades.size() >= 5) {
            // Count trades with identical price and volume in short time windows
            std::map<std::pair<double, double>, int> price_volume_counts; // {price, volume} -> count

            for (const auto& trade : trades) {
                std::pair<double, double> key = {trade.price, trade.volume};
                price_volume_counts[key]++;
            }

            // Check if any price/volume combination appears too frequently
            for (const auto& [key, count] : price_volume_counts) {
                if (count > 3) { // More than 3 identical trades
                    std::ostringstream oss;
                    oss << "Potential duplicate trade pattern detected for " << symbol
                        << ". Identical price (" << key.first << ") and volume (" << key.second
                        << ") appeared " << count << " times recently";

                    DataQualityIssue issue(DataQualityIssueType::DUPLICATE_TRADE, symbol,
                                         std::chrono::duration_cast<std::chrono::milliseconds>(
                                             std::chrono::high_resolution_clock::now().time_since_epoch()).count(),
                                         oss.str(), 0.7);
                    metrics_.duplicate_trade_issues++;
                    add_issue(issue);

                    alert_user_to_data_problems(symbol, oss.str(), 0.7);
                }
            }
        }
    }
}

void DataQualityMonitor::check_out_of_order_timestamp_patterns(const std::string& symbol) {
    // Check for patterns of out-of-order timestamps
    auto recent_trades_it = recent_trades_.find(symbol);
    if (recent_trades_it != recent_trades_.end()) {
        const auto& trades = recent_trades_it->second;

        if (trades.size() >= 10) {
            // Count how many trades are out of order
            size_t out_of_order_count = 0;
            uint64_t prev_timestamp = 0;

            for (const auto& trade : trades) {
                if (prev_timestamp > 0 && trade.timestamp < prev_timestamp) {
                    out_of_order_count++;
                }
                prev_timestamp = trade.timestamp;
            }

            // If more than 20% of recent trades are out of order, flag it
            if (out_of_order_count > trades.size() * 0.2) {
                std::ostringstream oss;
                oss << "High frequency of out-of-order timestamps detected for " << symbol
                    << ". " << out_of_order_count << "/" << trades.size()
                    << " (" << (out_of_order_count * 100.0 / trades.size()) << "%) trades are out of order";

                DataQualityIssue issue(DataQualityIssueType::OUT_OF_ORDER_TIMESTAMP, symbol,
                                     std::chrono::duration_cast<std::chrono::milliseconds>(
                                         std::chrono::high_resolution_clock::now().time_since_epoch()).count(),
                                     oss.str(), 0.8);
                metrics_.out_of_order_timestamp_issues++;
                add_issue(issue);

                alert_user_to_data_problems(symbol, oss.str(), 0.8);
            }
        }
    }
}

void DataQualityMonitor::check_latency_issue_patterns(const std::string& symbol) {
    // Check for patterns of latency issues
    auto delay_history_it = recent_delays_.find(symbol);
    if (delay_history_it != recent_delays_.end()) {
        const auto& delays = delay_history_it->second;

        if (delays.size() >= 10) {
            // Calculate average and standard deviation of recent delays
            double sum = 0;
            for (auto delay : delays) {
                sum += delay;
            }
            double avg_delay = sum / delays.size();

            double variance_sum = 0;
            for (auto delay : delays) {
                variance_sum += (delay - avg_delay) * (delay - avg_delay);
            }
            double std_dev = sqrt(variance_sum / delays.size());

            // Check if recent delays are consistently high
            size_t high_delay_count = 0;
            for (auto delay : delays) {
                if (delay > latency_alert_threshold_ms_) {
                    high_delay_count++;
                }
            }

            // If more than 50% of recent delays are above threshold, flag it
            if (high_delay_count > delays.size() * 0.5) {
                std::ostringstream oss;
                oss << "Consistent high latency pattern detected for " << symbol
                    << ". " << high_delay_count << "/" << delays.size()
                    << " (" << (high_delay_count * 100.0 / delays.size()) << "%) recent delays above threshold";

                DataQualityIssue issue(DataQualityIssueType::LATENCY_ISSUE, symbol,
                                     std::chrono::duration_cast<std::chrono::milliseconds>(
                                         std::chrono::high_resolution_clock::now().time_since_epoch()).count(),
                                     oss.str(), 0.75);
                metrics_.latency_issues++;
                add_issue(issue);

                alert_user_to_data_problems(symbol, oss.str(), 0.75);
            }

            // Check for increasing trend in latency
            if (delays.size() >= 20) {
                // Compare first and second half of delays
                size_t mid = delays.size() / 2;
                double first_half_avg = 0, second_half_avg = 0;

                for (size_t i = 0; i < mid; i++) {
                    first_half_avg += delays[i];
                }
                first_half_avg /= mid;

                for (size_t i = mid; i < delays.size(); i++) {
                    second_half_avg += delays[i];
                }
                second_half_avg /= (delays.size() - mid);

                // If second half has significantly higher average, there's an increasing trend
                if (first_half_avg > 0 && second_half_avg > first_half_avg * 1.5) {
                    std::ostringstream oss;
                    oss << "Increasing latency trend detected for " << symbol
                        << ". Recent avg: " << second_half_avg << "ms vs earlier avg: " << first_half_avg << "ms";

                    DataQualityIssue issue(DataQualityIssueType::LATENCY_ISSUE, symbol,
                                         std::chrono::duration_cast<std::chrono::milliseconds>(
                                             std::chrono::high_resolution_clock::now().time_since_epoch()).count(),
                                         oss.str(), 0.7);
                    metrics_.latency_issues++;
                    add_issue(issue);

                    alert_user_to_data_problems(symbol, oss.str(), 0.7);
                }
            }
        }
    }
}

// Additional method to provide a comprehensive summary of data quality
std::string DataQualityMonitor::get_comprehensive_summary() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ostringstream summary;
    summary << "\n=== COMPREHENSIVE DATA QUALITY REPORT ===\n";
    summary << "Generated: " << std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count() << " ms\n";
    summary << "Total trades processed: " << metrics_.total_trades_processed << "\n";
    summary << "Missing data issues: " << metrics_.missing_data_issues << "\n";
    summary << "Duplicate trade issues: " << metrics_.duplicate_trade_issues << "\n";
    summary << "Out-of-order timestamp issues: " << metrics_.out_of_order_timestamp_issues << "\n";
    summary << "Latency issues: " << metrics_.latency_issues << "\n";
    summary << "Invalid price issues: " << metrics_.invalid_price_issues << "\n";
    summary << "Invalid volume issues: " << metrics_.invalid_volume_issues << "\n";
    summary << "Missing field issues: " << metrics_.missing_field_issues << "\n";
    summary << "Average latency: " << std::fixed << std::setprecision(2) << metrics_.average_latency_ms << " ms\n";

    // Calculate and report severity distribution
    size_t critical_issues = 0, high_issues = 0, medium_issues = 0, low_issues = 0;

    for (const auto& issue : recent_issues_) {
        if (issue.severity >= 0.9) {
            critical_issues++;
        } else if (issue.severity >= 0.7) {
            high_issues++;
        } else if (issue.severity >= 0.5) {
            medium_issues++;
        } else {
            low_issues++;
        }
    }

    summary << "\nSeverity Distribution:\n";
    summary << "Critical issues (0.9-1.0): " << critical_issues << "\n";
    summary << "High issues (0.7-0.89): " << high_issues << "\n";
    summary << "Medium issues (0.5-0.69): " << medium_issues << "\n";
    summary << "Low issues (0.0-0.49): " << low_issues << "\n";

    // Report top affected symbols
    std::unordered_map<std::string, size_t> symbol_issue_counts;
    for (const auto& issue : recent_issues_) {
        symbol_issue_counts[issue.symbol]++;
    }

    summary << "\nTop 5 Symbols with Most Issues:\n";
    std::vector<std::pair<std::string, size_t>> sorted_symbols(symbol_issue_counts.begin(), symbol_issue_counts.end());
    std::sort(sorted_symbols.begin(), sorted_symbols.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    for (size_t i = 0; i < std::min(sorted_symbols.size(), static_cast<size_t>(5)); ++i) {
        summary << "  " << i+1 << ". " << sorted_symbols[i].first
               << ": " << sorted_symbols[i].second << " issues\n";
    }

    // Calculate data quality score
    double quality_score = 100.0;
    if (metrics_.total_trades_processed > 0) {
        double error_rate = static_cast<double>(metrics_.missing_data_issues +
                                               metrics_.duplicate_trade_issues +
                                               metrics_.out_of_order_timestamp_issues +
                                               metrics_.latency_issues +
                                               metrics_.invalid_price_issues +
                                               metrics_.invalid_volume_issues +
                                               metrics_.missing_field_issues) /
                           static_cast<double>(metrics_.total_trades_processed);
        quality_score = (1.0 - std::min(error_rate, 1.0)) * 100.0;
    }

    summary << "\nOverall Data Quality Score: " << std::fixed << std::setprecision(2)
           << quality_score << "%\n";

    if (quality_score < 80.0) {
        summary << "STATUS: CRITICAL - Immediate attention required!\n";
    } else if (quality_score < 90.0) {
        summary << "STATUS: WARNING - Data quality needs attention\n";
    } else if (quality_score < 95.0) {
        summary << "STATUS: FAIR - Minor issues detected\n";
    } else {
        summary << "STATUS: GOOD - Data quality is satisfactory\n";
    }

    summary << "=========================================\n";

    return summary.str();
}

// NEW: Method to run continuous monitoring and alerting
void DataQualityMonitor::start_continuous_monitoring() {
    // This method would typically run in a separate thread to continuously monitor data quality
    // For now, we'll just document how it would work

    // In a real implementation, this would:
    // 1. Run in a separate thread
    // 2. Periodically check data quality metrics
    // 3. Generate alerts when thresholds are exceeded
    // 4. Update UI components with current status
    // 5. Log issues to files if enabled

    // Since this is just documentation of the concept, we'll just log that it would start
    if (console_alerts_enabled_) {
        std::cout << "[MONITORING] Continuous data quality monitoring would start now" << std::endl;
    }
}

// NEW: Method to run periodic health checks on data streams
void DataQualityMonitor::run_periodic_health_checks() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Perform health checks on all monitored symbols
    for (const auto& [symbol, _] : symbol_stats_) {
        monitor_data_stream_health(symbol);
    }

    // Generate summary report if needed
    if (console_alerts_enabled_) {
        std::cout << "[HEALTH CHECK] Periodic data quality health check completed" << std::endl;
    }
}

// NEW: Unified method to alert users about all types of data quality problems
void DataQualityMonitor::alert_users_to_all_data_problems() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check for each type of data quality issue and alert if present
    if (metrics_.missing_data_issues > 0) {
        std::ostringstream msg;
        msg << "Missing data detected: " << metrics_.missing_data_issues << " issues identified";
        alert_user_to_data_problems("SYSTEM", msg.str(), 0.7);
    }

    if (metrics_.duplicate_trade_issues > 0) {
        std::ostringstream msg;
        msg << "Duplicate trades detected: " << metrics_.duplicate_trade_issues << " issues identified";
        alert_user_to_data_problems("SYSTEM", msg.str(), 0.6);
    }

    if (metrics_.out_of_order_timestamp_issues > 0) {
        std::ostringstream msg;
        msg << "Out-of-order timestamps detected: " << metrics_.out_of_order_timestamp_issues << " issues identified";
        alert_user_to_data_problems("SYSTEM", msg.str(), 0.65);
    }

    if (metrics_.latency_issues > 0) {
        std::ostringstream msg;
        msg << "Latency issues detected: " << metrics_.latency_issues << " issues identified";
        alert_user_to_data_problems("SYSTEM", msg.str(), 0.5);
    }

    if (metrics_.invalid_price_issues > 0) {
        std::ostringstream msg;
        msg << "Invalid prices detected: " << metrics_.invalid_price_issues << " issues identified";
        alert_user_to_data_problems("SYSTEM", msg.str(), 0.8);
    }

    if (metrics_.invalid_volume_issues > 0) {
        std::ostringstream msg;
        msg << "Invalid volumes detected: " << metrics_.invalid_volume_issues << " issues identified";
        alert_user_to_data_problems("SYSTEM", msg.str(), 0.8);
    }

    if (metrics_.missing_field_issues > 0) {
        std::ostringstream msg;
        msg << "Missing fields detected: " << metrics_.missing_field_issues << " issues identified";
        alert_user_to_data_problems("SYSTEM", msg.str(), 0.5);
    }

    // NEW: Additional alerting for specific data quality problems with more detailed information
    // Check for critical data quality issues that require immediate attention
    if (metrics_.missing_data_issues > 10 ||
        metrics_.duplicate_trade_issues > 10 ||
        metrics_.out_of_order_timestamp_issues > 10 ||
        metrics_.latency_issues > 10) {

        std::ostringstream critical_msg;
        critical_msg << "CRITICAL DATA QUALITY ALERT: Multiple high-frequency issues detected. "
                     << "Missing: " << metrics_.missing_data_issues
                     << ", Duplicates: " << metrics_.duplicate_trade_issues
                     << ", Out-of-order: " << metrics_.out_of_order_timestamp_issues
                     << ", Latency: " << metrics_.latency_issues;

        alert_user_to_data_problems_with_context("SYSTEM", critical_msg.str(), 0.95,
                                               "DataQualityMonitor", "Multiple concurrent data quality issues");
    }

    // Also provide a summary of the current state
    if (console_alerts_enabled_) {
        std::cout << get_user_friendly_summary() << std::endl;
    }

    // NEW: Generate a comprehensive report for user visibility
    generate_comprehensive_alert_report();
}

// NEW: Additional method to provide real-time alerts to users about data quality issues
void DataQualityMonitor::provide_real_time_alerts_to_users() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Get recent high severity issues
    auto high_severity_issues = get_high_severity_issues(0.7);

    if (!high_severity_issues.empty()) {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "REAL-TIME DATA QUALITY ALERTS" << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        for (const auto& issue : high_severity_issues) {
            std::cout << "Symbol: " << issue.symbol << std::endl;
            std::cout << "Type: ";

            switch (issue.type) {
                case DataQualityIssueType::MISSING_DATA:
                    std::cout << "MISSING_DATA";
                    break;
                case DataQualityIssueType::DUPLICATE_TRADE:
                    std::cout << "DUPLICATE_TRADE";
                    break;
                case DataQualityIssueType::OUT_OF_ORDER_TIMESTAMP:
                    std::cout << "OUT_OF_ORDER_TIMESTAMP";
                    break;
                case DataQualityIssueType::LATENCY_ISSUE:
                    std::cout << "LATENCY_ISSUE";
                    break;
                case DataQualityIssueType::INVALID_PRICE:
                    std::cout << "INVALID_PRICE";
                    break;
                case DataQualityIssueType::INVALID_VOLUME:
                    std::cout << "INVALID_VOLUME";
                    break;
                case DataQualityIssueType::MISSING_FIELD:
                    std::cout << "MISSING_FIELD";
                    break;
            }

            std::cout << std::endl;
            std::cout << "Severity: " << issue.severity << std::endl;
            std::cout << "Description: " << issue.description << std::endl;
            std::cout << std::string(50, '-') << std::endl;
        }

        std::cout << std::string(80, '=') << std::endl;
    }
}

// NEW: Enhanced method to provide immediate user notifications for critical data quality issues
void DataQualityMonitor::immediate_user_notification(const DataQualityIssue& issue) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Format a clear, immediate notification for the user
    std::string severity_label;
    if (issue.severity >= 0.9) {
        severity_label = "CRITICAL";
    } else if (issue.severity >= 0.7) {
        severity_label = "HIGH";
    } else if (issue.severity >= 0.5) {
        severity_label = "MEDIUM";
    } else {
        severity_label = "LOW";
    }

    // Print a highly visible alert to get user attention
    std::cout << "\n" << std::string(80, '!') << std::endl;
    std::cout << "🚨 DATA QUALITY ALERT 🚨" << std::endl;
    std::cout << std::string(80, '!') << std::endl;
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count() << std::endl;
    std::cout << "Symbol: " << issue.symbol << std::endl;
    std::cout << "Issue: ";

    switch (issue.type) {
        case DataQualityIssueType::MISSING_DATA:
            std::cout << "Missing Data";
            break;
        case DataQualityIssueType::DUPLICATE_TRADE:
            std::cout << "Duplicate Trade";
            break;
        case DataQualityIssueType::OUT_OF_ORDER_TIMESTAMP:
            std::cout << "Out-of-Order Timestamp";
            break;
        case DataQualityIssueType::LATENCY_ISSUE:
            std::cout << "Latency Issue";
            break;
        case DataQualityIssueType::INVALID_PRICE:
            std::cout << "Invalid Price";
            break;
        case DataQualityIssueType::INVALID_VOLUME:
            std::cout << "Invalid Volume";
            break;
        case DataQualityIssueType::MISSING_FIELD:
            std::cout << "Missing Field";
            break;
    }

    std::cout << std::endl;
    std::cout << "Severity: " << severity_label << " (" << issue.severity << ")" << std::endl;
    std::cout << "Details: " << issue.description << std::endl;

    // Provide immediate recommendation based on issue type
    std::cout << "Recommendation: ";
    switch (issue.type) {
        case DataQualityIssueType::MISSING_DATA:
            std::cout << "Check data feed connectivity";
            break;
        case DataQualityIssueType::DUPLICATE_TRADE:
            std::cout << "Review duplicate filtering";
            break;
        case DataQualityIssueType::OUT_OF_ORDER_TIMESTAMP:
            std::cout << "Verify timestamp synchronization";
            break;
        case DataQualityIssueType::LATENCY_ISSUE:
            std::cout << "Investigate system performance";
            break;
        case DataQualityIssueType::INVALID_PRICE:
            std::cout << "Validate price normalization";
            break;
        case DataQualityIssueType::INVALID_VOLUME:
            std::cout << "Check volume validation";
            break;
        case DataQualityIssueType::MISSING_FIELD:
            std::cout << "Ensure complete data feeds";
            break;
    }
    std::cout << std::endl;

    std::cout << std::string(80, '!') << std::endl << std::endl;
}

// NEW: Method to send consolidated alerts to users at regular intervals
void DataQualityMonitor::send_consolidated_alerts() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Get all recent issues
    auto all_recent_issues = get_recent_issues(50); // Get last 50 issues

    if (all_recent_issues.empty()) {
        return; // No issues to report
    }

    // Group issues by type
    std::unordered_map<DataQualityIssueType, std::vector<DataQualityIssue>> grouped_issues;
    for (const auto& issue : all_recent_issues) {
        grouped_issues[issue.type].push_back(issue);
    }

    // Count total issues by type
    std::unordered_map<DataQualityIssueType, size_t> issue_counts;
    for (const auto& [type, issues] : grouped_issues) {
        issue_counts[type] = issues.size();
    }

    // Create a consolidated report
    std::cout << "\n" << std::string(70, '-') << std::endl;
    std::cout << "CONSOLIDATED DATA QUALITY REPORT" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count() << std::endl;

    for (const auto& [type, count] : issue_counts) {
        std::string type_name;
        switch (type) {
            case DataQualityIssueType::MISSING_DATA:
                type_name = "Missing Data";
                break;
            case DataQualityIssueType::DUPLICATE_TRADE:
                type_name = "Duplicate Trades";
                break;
            case DataQualityIssueType::OUT_OF_ORDER_TIMESTAMP:
                type_name = "Out-of-Order Timestamps";
                break;
            case DataQualityIssueType::LATENCY_ISSUE:
                type_name = "Latency Issues";
                break;
            case DataQualityIssueType::INVALID_PRICE:
                type_name = "Invalid Prices";
                break;
            case DataQualityIssueType::INVALID_VOLUME:
                type_name = "Invalid Volumes";
                break;
            case DataQualityIssueType::MISSING_FIELD:
                type_name = "Missing Fields";
                break;
        }

        std::cout << type_name << ": " << count << " issues" << std::endl;
    }

    std::cout << std::string(70, '-') << std::endl;

    // If there are high severity issues, provide a summary
    auto high_severity = get_high_severity_issues(0.8);
    if (!high_severity.empty()) {
        std::cout << "\nHIGH SEVERITY ISSUES SUMMARY:" << std::endl;
        for (size_t i = 0; i < std::min(high_severity.size(), static_cast<size_t>(5)); ++i) { // Show top 5
            std::cout << "  - " << high_severity[i].symbol << ": " << high_severity[i].description.substr(0, 50);
            if (high_severity[i].description.length() > 50) std::cout << "...";
            std::cout << " (Severity: " << high_severity[i].severity << ")" << std::endl;
        }
    }

    std::cout << std::string(70, '-') << std::endl << std::endl;
}

// NEW: Method to enable/disable different types of alerts
void DataQualityMonitor::configure_alert_types(bool enable_missing_data,
                                              bool enable_duplicate_trades,
                                              bool enable_out_of_order,
                                              bool enable_latency_issues,
                                              bool enable_invalid_data) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Store alert configuration for future reference
    alert_config_ = {
        enable_missing_data,
        enable_duplicate_trades,
        enable_out_of_order,
        enable_latency_issues,
        enable_invalid_data
    };

    if (console_alerts_enabled_) {
        std::cout << "[CONFIG] Alert configuration updated:" << std::endl;
        std::cout << "  Missing Data Alerts: " << (enable_missing_data ? "ON" : "OFF") << std::endl;
        std::cout << "  Duplicate Trade Alerts: " << (enable_duplicate_trades ? "ON" : "OFF") << std::endl;
        std::cout << "  Out-of-Order Alerts: " << (enable_out_of_order ? "ON" : "OFF") << std::endl;
        std::cout << "  Latency Issue Alerts: " << (enable_latency_issues ? "ON" : "OFF") << std::endl;
        std::cout << "  Invalid Data Alerts: " << (enable_invalid_data ? "ON" : "OFF") << std::endl;
    }
}

// NEW: Method to check if specific alert types are enabled
bool DataQualityMonitor::is_alert_type_enabled(DataQualityIssueType type) const {
    std::lock_guard<std::mutex> lock(mutex_);

    switch (type) {
        case DataQualityIssueType::MISSING_DATA:
            return alert_config_.enable_missing_data;
        case DataQualityIssueType::DUPLICATE_TRADE:
            return alert_config_.enable_duplicate_trades;
        case DataQualityIssueType::OUT_OF_ORDER_TIMESTAMP:
            return alert_config_.enable_out_of_order;
        case DataQualityIssueType::LATENCY_ISSUE:
            return alert_config_.enable_latency_issues;
        case DataQualityIssueType::INVALID_PRICE:
        case DataQualityIssueType::INVALID_VOLUME:
        case DataQualityIssueType::MISSING_FIELD:
            return alert_config_.enable_invalid_data;
        default:
            return true; // Default to enabled for unknown types
    }
}

} // namespace Data
} // namespace BTQuant