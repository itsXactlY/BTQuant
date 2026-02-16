#include "analytics/tpoengine.h"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>

#include "data/data_types.hpp"

// Implementation of TPONode constructor
TPONode::TPONode(std::chrono::system_clock::time_point start,
                 std::chrono::system_clock::time_point end, double price)
    : time_start(start),
      time_end(end),
      price_level(price),
      count(0),
      total_volume(0.0),
      high_price(price),
      low_price(price) {}

// Implementation of TPOProfile methods
TPOProfile::TPOProfile() : letter_counter(0) {}

// Get the next letter in sequence (A-Z, then a-z)
std::string TPOProfile::get_next_letter() {
  if (letter_counter < 26) {
    // Uppercase A-Z
    return std::string(1, 'A' + letter_counter);
  } else if (letter_counter < 52) {
    // Lowercase a-z
    return std::string(1, 'a' + letter_counter - 26);
  } else {
    // If we exceed 52 time brackets, cycle back or return a special indicator
    return std::string(1, 'A' + (letter_counter % 26));
  }
}

// Assign a letter to a time bracket if not already assigned
std::string TPOProfile::assign_letter_to_time_bracket(
    const std::chrono::system_clock::time_point& time_bracket) {
  if (time_bracket_to_letter.find(time_bracket) == time_bracket_to_letter.end()) {
    std::string letter = get_next_letter();
    time_bracket_to_letter[time_bracket] = letter;
    letter_counter++;
    return letter;
  }
  return time_bracket_to_letter[time_bracket];
}

// Add a price level to a time bracket (assign letter to price level)
void TPOProfile::add_price_to_time_bracket(
    double price_level, const std::chrono::system_clock::time_point& time_bracket) {
  std::string letter = assign_letter_to_time_bracket(time_bracket);

  // Add the letter to the price level's sequence
  if (price_to_letters.find(price_level) == price_to_letters.end()) {
    price_to_letters[price_level] = letter;
  } else {
    // Append the letter to the existing sequence
    price_to_letters[price_level] += letter;
  }
}

// Get the letter sequence for a specific price level
std::string TPOProfile::get_letter_sequence_for_price(double price_level) const {
  auto it = price_to_letters.find(price_level);
  if (it != price_to_letters.end()) {
    return it->second;
  }
  return "";
}

// Clear all data
void TPOProfile::clear() {
  price_to_letters.clear();
  time_bracket_to_letter.clear();
  letter_counter = 0;
}

// Get all price levels that were touched in a specific time bracket
std::vector<double> TPOProfile::get_prices_for_time_bracket(
    const std::chrono::system_clock::time_point& time_bracket) const {
  std::vector<double> result;

  auto it = time_bracket_to_letter.find(time_bracket);
  if (it != time_bracket_to_letter.end()) {
    std::string letter = it->second;

    for (const auto& [price, letters] : price_to_letters) {
      if (letters.find(letter) != std::string::npos) {
        result.push_back(price);
      }
    }
  }
  return result;
}

// Get the time bracket for a specific letter
std::chrono::system_clock::time_point TPOProfile::get_time_bracket_for_letter(
    const std::string& letter) const {
  for (const auto& [time_bracket, time_letter] : time_bracket_to_letter) {
    if (time_letter == letter) {
      return time_bracket;
    }
  }
  // Return null time point if not found
  return std::chrono::system_clock::time_point();
}

// Get the count of how many times each price level was touched
std::map<double, int> TPOProfile::get_touch_counts() const {
  std::map<double, int> counts;
  for (const auto& [price, letters] : price_to_letters) {
    counts[price] = letters.length();
  }
  return counts;
}

// Get the total number of unique price levels in the profile
size_t TPOProfile::get_unique_price_count() const { return price_to_letters.size(); }

// Get the total number of unique time brackets in the profile
size_t TPOProfile::get_unique_time_bracket_count() const { return time_bracket_to_letter.size(); }

// Print the profile for debugging
void TPOProfile::print_profile() const {
  std::cout << "TPO Profile:\n";
  for (const auto& [price, letters] : price_to_letters) {
    std::cout << "Price: " << price << " -> Letters: " << letters << "\n";
  }
  std::cout << "\nTime Bracket Mappings:\n";
  for (const auto& [time_bracket, letter] : time_bracket_to_letter) {
    auto time_t = std::chrono::system_clock::to_time_t(time_bracket);
    std::cout << "Time: " << std::put_time(std::localtime(&time_t), "%F %T")
              << " -> Letter: " << letter << "\n";
  }

  // Print POC and Value Area info
  std::cout << "\nPOC: " << get_poc() << "\n";
  auto va = get_value_area();
  std::cout << "Value Area: " << va.first << " - " << va.second << "\n";
}

// Implementation of TPOEngine methods
TPOEngine::TPOEngine(double bucket_size) : price_bucket_size(bucket_size) {}

// Calculate the time bucket start time for a given timestamp (30-minute intervals)
std::chrono::system_clock::time_point TPOEngine::get_time_bucket_start(
    const std::chrono::system_clock::time_point& timestamp) const {
  // Convert to seconds since epoch
  auto timestamp_seconds =
      std::chrono::time_point_cast<std::chrono::seconds>(timestamp).time_since_epoch().count();

  // Calculate the start of the 30-minute bucket (1800 seconds = 30 minutes)
  auto bucket_start_seconds = (timestamp_seconds / 1800) * 1800;

  return std::chrono::system_clock::time_point{std::chrono::seconds(bucket_start_seconds)};
}

// Calculate the price bucket for a given price
double TPOEngine::get_price_bucket(double price) const {
  if (price_bucket_size <= 0) {
    return price;  // No bucketing if bucket size is invalid
  }
  return std::floor(price / price_bucket_size) * price_bucket_size;
}

// Process a single price tick and aggregate into TPO buckets
void TPOEngine::process_tick(const PriceTick& tick) {
  auto time_bucket_start = get_time_bucket_start(tick.timestamp);
  auto time_bucket_end = time_bucket_start + std::chrono::minutes(30);
  auto price_bucket = get_price_bucket(tick.price);

  // Create or update the TPONode for this time-price combination
  auto& node = tpo_data[time_bucket_start][price_bucket];

  // Initialize node if it's new
  if (node.count == 0) {
    node.time_start = time_bucket_start;
    node.time_end = time_bucket_end;
    node.price_level = price_bucket;
    node.high_price = tick.price;
    node.low_price = tick.price;
  } else {
    // Update high and low prices for this bucket
    node.high_price = std::max(node.high_price, tick.price);
    node.low_price = std::min(node.low_price, tick.price);
  }

  // Update statistics
  node.count++;
  node.total_volume += tick.volume;

  // Add the price level to the TPO profile with the corresponding time bracket
  tpo_profile.add_price_to_time_bracket(price_bucket, time_bucket_start);
}

// Process a vector of ticks
void TPOEngine::process_ticks(const std::vector<PriceTick>& ticks) {
  for (const auto& tick : ticks) {
    process_tick(tick);
  }
}

// Process a candle (OHLC) and assign blocks for the entire range
void TPOEngine::process_candle(const BTQuant::OHLCVCandle& candle) {
  // 1. Determine MMT 30-minute Block Letter (A-Z, a-z)
  // Assuming session starts at 00:00 UTC or purely based on timestamp
  // We reuse our time bucket logic which aligns to 30m
  auto time_bucket_start = get_time_bucket_start(
      std::chrono::system_clock::time_point(std::chrono::microseconds(candle.timestamp)));

  // 2. Assign block to all price levels traded during this 30m window
  // Iterate from Low to High in steps of tick size (price_bucket_size)
  if (price_bucket_size <= 0) return;

  double current_price = std::floor(candle.low / price_bucket_size) * price_bucket_size;
  double end_price = std::floor(candle.high / price_bucket_size) * price_bucket_size;

  while (current_price <= end_price + 0.00001) {  // Epsilon for float comparison
    double price_bucket = get_price_bucket(current_price);

    // Create a synthetic tick for this price level to reuse aggregation logic
    // We distribute volume evenly or just mark it. For TPO count, volume doesn't technically matter
    // per hit, but if we track volume per node, we might want to split it. For TPO Profile, we just
    // need the "hit".

    // Direct update to avoid overhead of creating full PriceTick struct if possible,
    // but process_tick handles all the logic. Let's use it for consistency.

    // Note: Volume is smeared across the range? Or attributed to POC of candle?
    // Standard TPO just cares about "touched".
    // We will add 0 volume for the range fill, and maybe real volume for the Close?
    // MMT spec implies purely range based.

    // Logic from MMT spec:
    // "Assign block to all price levels traded during this 30m window"

    // Direct TPONode update
    auto& node = tpo_data[time_bucket_start][price_bucket];
    if (node.count == 0) {
      node.time_start = time_bucket_start;
      node.time_end = time_bucket_start + std::chrono::minutes(30);
      node.price_level = price_bucket;
      node.high_price = current_price;
      node.low_price = current_price;
    }

    node.count++;
    // Volume? We don't have per-tick volume here.
    // We could assign candle.volume / num_levels?
    // For now, leave volume as 0 for filled levels.

    tpo_profile.add_price_to_time_bracket(price_bucket, time_bucket_start);

    current_price += price_bucket_size;
  }
}

// Get TPO data for a specific time range
std::map<std::chrono::system_clock::time_point, std::map<double, TPONode>>
TPOEngine::get_tpo_data_for_range(const std::chrono::system_clock::time_point& start_time,
                                  const std::chrono::system_clock::time_point& end_time) const {
  std::map<std::chrono::system_clock::time_point, std::map<double, TPONode>> result;

  for (auto it = tpo_data.lower_bound(start_time); it != tpo_data.upper_bound(end_time); ++it) {
    result[it->first] = it->second;
  }

  return result;
}

// Get all TPO data
const std::map<std::chrono::system_clock::time_point, std::map<double, TPONode>>&
TPOEngine::get_all_tpo_data() const {
  return tpo_data;
}

// Get TPO data for a specific time bucket
const std::map<double, TPONode>* TPOEngine::get_tpo_data_for_time_bucket(
    const std::chrono::system_clock::time_point& time_bucket) const {
  auto it = tpo_data.find(time_bucket);
  if (it != tpo_data.end()) {
    return &(it->second);
  }
  return nullptr;
}

// Get the highest price in a specific time-price bucket
double TPOEngine::get_high_price(const std::chrono::system_clock::time_point& time_bucket,
                                 double price_bucket) const {
  auto time_it = tpo_data.find(time_bucket);
  if (time_it != tpo_data.end()) {
    auto price_it = time_it->second.find(price_bucket);
    if (price_it != time_it->second.end()) {
      return price_it->second.high_price;
    }
  }
  return 0.0;
}

// Get the lowest price in a specific time-price bucket
double TPOEngine::get_low_price(const std::chrono::system_clock::time_point& time_bucket,
                                double price_bucket) const {
  auto time_it = tpo_data.find(time_bucket);
  if (time_it != tpo_data.end()) {
    auto price_it = time_it->second.find(price_bucket);
    if (price_it != time_it->second.end()) {
      return price_it->second.low_price;
    }
  }
  return 0.0;
}

// Clear all stored data
void TPOEngine::clear() {
  tpo_data.clear();
  tpo_profile.clear();
}

// Print TPO data for debugging purposes
void TPOEngine::print_tpo_data() const {
  for (const auto& [time_bucket, price_buckets] : tpo_data) {
    auto time_t = std::chrono::system_clock::to_time_t(time_bucket);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%d %H:%M:%S");

    std::cout << "Time Bucket Start: " << ss.str() << "\n";

    for (const auto& [price_level, node] : price_buckets) {
      std::cout << "  Price Level: " << std::fixed << std::setprecision(2) << price_level
                << ", Hits: " << node.count << ", Volume: " << node.total_volume
                << ", High: " << node.high_price << ", Low: " << node.low_price << "\n";
    }
    std::cout << "\n";
  }
}

// Print TPO profile for debugging purposes
void TPOEngine::print_tpo_profile() const { tpo_profile.print_profile(); }

// Get statistics for a specific time period
TPOStatistics TPOEngine::get_statistics_for_period(
    const std::chrono::system_clock::time_point& start_time,
    const std::chrono::system_clock::time_point& end_time) const {
  TPOStatistics stats;
  stats.total_ticks_processed = 0;
  stats.total_volume = 0.0;
  stats.unique_time_buckets = 0;
  stats.unique_price_levels = 0;

  auto data_in_range = get_tpo_data_for_range(start_time, end_time);

  std::set<double> all_price_levels;

  for (const auto& [time_bucket, price_buckets] : data_in_range) {
    stats.unique_time_buckets++;

    for (const auto& [price_level, node] : price_buckets) {
      stats.total_ticks_processed += node.count;
      stats.total_volume += node.total_volume;
      all_price_levels.insert(price_level);
    }
  }

  stats.unique_price_levels = all_price_levels.size();

  return stats;
}

// Calculate Point of Control (POC) - price level with highest TPO count
double TPOProfile::get_poc() const {
  if (price_to_letters.empty()) {
    return 0.0;
  }

  double poc_price = 0.0;
  int max_count = 0;

  for (const auto& [price, letters] : price_to_letters) {
    int count = static_cast<int>(letters.length());
    if (count > max_count) {
      max_count = count;
      poc_price = price;
    }
  }

  return poc_price;
}

// Calculate Value Area (70% of TPOs) centered around POC
std::pair<double, double> TPOProfile::get_value_area(double percent) const {
  if (price_to_letters.empty()) {
    return std::make_pair(0.0, 0.0);
  }

  // Get total TPO count
  int total_count = get_total_tpo_count();
  if (total_count == 0) {
    return std::make_pair(0.0, 0.0);
  }

  // Calculate target count for value area (percent of total)
  int target_count = static_cast<int>(total_count * (percent / 100.0));

  // Get POC as starting point
  double poc_price = get_poc();

  // Create a sorted vector of price levels with their TPO counts
  std::vector<std::pair<double, int>> price_counts;
  for (const auto& [price, letters] : price_to_letters) {
    price_counts.emplace_back(price, static_cast<int>(letters.length()));
  }

  // Sort by distance from POC
  std::sort(price_counts.begin(), price_counts.end(), [poc_price](const auto& a, const auto& b) {
    return std::abs(a.first - poc_price) < std::abs(b.first - poc_price);
  });

  // Start with POC and expand outward until reaching the target count
  int accumulated_count = 0;
  double min_price = poc_price;
  double max_price = poc_price;

  for (const auto& [price, count] : price_counts) {
    if (accumulated_count >= target_count) {
      break;
    }

    accumulated_count += count;
    min_price = std::min(min_price, price);
    max_price = std::max(max_price, price);
  }

  // If we didn't reach the target count, expand further
  if (accumulated_count < target_count) {
    // Get all prices sorted by value
    std::vector<double> all_prices;
    for (const auto& [price, letters] : price_to_letters) {
      all_prices.push_back(price);
    }
    std::sort(all_prices.begin(), all_prices.end());

    // Find the POC index
    auto poc_iter = std::lower_bound(all_prices.begin(), all_prices.end(), poc_price);
    if (poc_iter != all_prices.end()) {
      int poc_idx = std::distance(all_prices.begin(), poc_iter);

      // Expand outward from POC
      int left_idx = poc_idx;
      int right_idx = poc_idx;
      accumulated_count = 0;

      // Reset accumulated count with POC value
      auto poc_entry = price_to_letters.find(poc_price);
      if (poc_entry != price_to_letters.end()) {
        accumulated_count = static_cast<int>(poc_entry->second.length());
      }

      min_price = poc_price;
      max_price = poc_price;

      // Alternate expanding left and right from POC
      while (accumulated_count < target_count) {
        bool expand_left = false;

        // Decide which direction to expand based on which has more remaining TPOs
        int left_remaining = 0, right_remaining = 0;

        if (left_idx > 0) {
          auto left_it = price_to_letters.find(all_prices[left_idx - 1]);
          if (left_it != price_to_letters.end()) {
            left_remaining = static_cast<int>(left_it->second.length());
          }
        }

        if (right_idx < static_cast<int>(all_prices.size()) - 1) {
          auto right_it = price_to_letters.find(all_prices[right_idx + 1]);
          if (right_it != price_to_letters.end()) {
            right_remaining = static_cast<int>(right_it->second.length());
          }
        }

        if (left_idx > 0 && (right_idx >= static_cast<int>(all_prices.size()) - 1 ||
                             left_remaining >= right_remaining)) {
          expand_left = true;
        } else if (right_idx < static_cast<int>(all_prices.size()) - 1) {
          expand_left = false;
        } else {
          break;  // Can't expand further
        }

        if (expand_left) {
          left_idx--;
          auto it = price_to_letters.find(all_prices[left_idx]);
          if (it != price_to_letters.end()) {
            accumulated_count += static_cast<int>(it->second.length());
            min_price = std::min(min_price, it->first);
          }
        } else {
          right_idx++;
          auto it = price_to_letters.find(all_prices[right_idx]);
          if (it != price_to_letters.end()) {
            accumulated_count += static_cast<int>(it->second.length());
            max_price = std::max(max_price, it->first);
          }
        }

        if (left_idx <= 0 && right_idx >= static_cast<int>(all_prices.size()) - 1) {
          break;  // Reached both ends
        }
      }
    }
  }

  return std::make_pair(min_price, max_price);
}

// Get price levels with only one TPO letter (single prints)
std::vector<double> TPOProfile::get_single_print_levels() const {
  std::vector<double> single_prints;
  for (const auto& [price, letters] : price_to_letters) {
    if (letters.length() == 1) {
      single_prints.push_back(price);
    }
  }
  return single_prints;
}

// Get total TPO count across all price levels
int TPOProfile::get_total_tpo_count() const {
  int total = 0;
  for (const auto& [price, letters] : price_to_letters) {
    total += static_cast<int>(letters.length());
  }
  return total;
}

// Check if a price level is a single print (isolated TPO)
bool TPOProfile::is_single_print(double price) const {
  return get_letter_sequence_for_price(price).length() == 1;
}