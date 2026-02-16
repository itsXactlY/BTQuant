#ifndef PUBBTQUANT_TPOENGINE_H
#define PUBBTQUANT_TPOENGINE_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "data/core_types.hpp"

// Forward declaration for integration
class LiquiditySweepDetector;

// Structure to represent a single price tick
struct PriceTick {
  std::chrono::system_clock::time_point timestamp;
  double price;
  double volume;
};

// Structure to represent TPO statistics
struct TPOStatistics {
  int total_ticks_processed;
  double total_volume;
  int unique_time_buckets;
  int unique_price_levels;
};

// Structure to represent a TPO (Time-Price Opportunity) bucket
struct TPONode {
  std::chrono::system_clock::time_point time_start;
  std::chrono::system_clock::time_point time_end;
  double price_level;
  int count;  // Number of times this price level was hit in the time bucket
  double total_volume;
  double high_price;  // Highest price in this time-price bucket
  double low_price;   // Lowest price in this time-price bucket

  // Default constructor
  TPONode()
      : time_start(),
        time_end(),
        price_level(0.0),
        count(0),
        total_volume(0.0),
        high_price(0.0),
        low_price(0.0) {}

  TPONode(std::chrono::system_clock::time_point start, std::chrono::system_clock::time_point end,
          double price);
};

// Structure to represent a TPO Profile that maps price levels to letters (A-Z, a-z) based on time
// brackets
struct TPOProfile {
  // Map price levels to letters (A-Z, a-z) indicating the sequence of time brackets they were
  // touched
  std::map<double, std::string> price_to_letters;

  // Map time brackets to letters (A-Z, a-z) to maintain consistent letter assignment
  std::map<std::chrono::system_clock::time_point, std::string> time_bracket_to_letter;

  // Counter to assign letters A-Z, then a-z (total 52 unique letters)
  int letter_counter;

  // Cached POC and Value Area for performance
  mutable bool poc_cached;
  mutable double cached_poc;
  mutable bool value_area_cached;
  mutable std::pair<double, double> cached_value_area;
  mutable double cached_value_area_percent;

  TPOProfile();

  // Get the next letter in sequence (A-Z, then a-z)
  std::string get_next_letter();

  // Assign a letter to a time bracket if not already assigned
  std::string assign_letter_to_time_bracket(
      const std::chrono::system_clock::time_point& time_bracket);

  // Add a price level to a time bracket (assign letter to price level)
  void add_price_to_time_bracket(double price_level,
                                 const std::chrono::system_clock::time_point& time_bracket);

  // Get the letter sequence for a specific price level
  std::string get_letter_sequence_for_price(double price_level) const;

  // Clear all data
  void clear();

  // Get all price levels that were touched in a specific time bracket
  std::vector<double> get_prices_for_time_bracket(
      const std::chrono::system_clock::time_point& time_bracket) const;

  // Get the time bracket for a specific letter
  std::chrono::system_clock::time_point get_time_bracket_for_letter(
      const std::string& letter) const;

  // Get the count of how many times each price level was touched
  std::map<double, int> get_touch_counts() const;

  // Get the total number of unique price levels in the profile
  size_t get_unique_price_count() const;

  // Get the total number of unique time brackets in the profile
  size_t get_unique_time_bracket_count() const;

  // Calculate Point of Control (POC) - price level with highest TPO count
  double get_poc() const;

  // Calculate Value Area (70% of TPOs) centered around POC
  std::pair<double, double> get_value_area(double percent = 70.0) const;

  // Get total TPO count across all price levels
  int get_total_tpo_count() const;

  // Get price levels with only one TPO letter (single prints)
  // Get price levels with only one TPO letter (single prints)
  std::vector<double> get_single_print_levels() const;

  // Check if a price level is a single print (isolated TPO)
  bool is_single_print(double price) const;

  // Print the profile for debugging
  void print_profile() const;

  // Check if a price level is within the current value area
  bool is_in_value_area(double price_level, double percent = 70.0) const;

  // Get the current value area boundaries
  std::pair<double, double> get_current_value_area_bounds() const;

  // Get the current POC price
  double get_current_poc() const;
};

class TPOEngine {
 private:
  // Price bucket size - configurable based on instrument
  double price_bucket_size;

  // Map to store aggregated TPO data: time_bucket -> price_bucket -> TPONode
  std::map<std::chrono::system_clock::time_point, std::map<double, TPONode>> tpo_data;

  // TPO Profile to map price levels to letters based on time brackets
  TPOProfile tpo_profile;

 public:
  explicit TPOEngine(double bucket_size = 0.25);

  // Calculate the time bucket start time for a given timestamp
  std::chrono::system_clock::time_point get_time_bucket_start(
      const std::chrono::system_clock::time_point& timestamp) const;

  // Calculate the price bucket for a given price
  double get_price_bucket(double price) const;

  // Process a single price tick and aggregate into TPO buckets
  void process_tick(const PriceTick& tick);

  // Process a vector of ticks
  // Process a vector of ticks
  void process_ticks(const std::vector<PriceTick>& ticks);

  // Process a candle (OHLC) and assign blocks for the entire range
  void process_candle(const BTQuant::OHLCVCandle& candle);

  // Get TPO data for a specific time range
  std::map<std::chrono::system_clock::time_point, std::map<double, TPONode>> get_tpo_data_for_range(
      const std::chrono::system_clock::time_point& start_time,
      const std::chrono::system_clock::time_point& end_time) const;

  // Get all TPO data
  const std::map<std::chrono::system_clock::time_point, std::map<double, TPONode>>&
  get_all_tpo_data() const;

  // Get the TPO profile
  const TPOProfile& get_tpo_profile() const { return tpo_profile; }

  // Get mutable reference to TPO profile for modification
  TPOProfile& get_tpo_profile() { return tpo_profile; }

  // Check if a price level is within the current value area
  bool is_price_in_value_area(double price_level, double percent = 70.0) const;

  // Get the current Point of Control (POC)
  double get_current_poc() const;

  // Get the current Value Area boundaries
  std::pair<double, double> get_current_value_area_bounds(double percent = 70.0) const;

  // Get TPO data with opacity information for visualization
  // Returns pairs of (TPONode, opacity) where opacity is 1.0 inside VA and 0.3 outside
  std::vector<std::pair<TPONode, float>> get_tpo_data_with_opacity(double va_percent = 70.0) const;

  // Get POC line data for visualization (returns the POC price level)
  double get_poc_line_data() const;

  // Clear all stored data
  void clear();

  // Print TPO data for debugging purposes
  void print_tpo_data() const;

  // Get TPO data for a specific time bucket
  const std::map<double, TPONode>* get_tpo_data_for_time_bucket(
      const std::chrono::system_clock::time_point& time_bucket) const;

  // Get the highest price in a specific time-price bucket
  double get_high_price(const std::chrono::system_clock::time_point& time_bucket,
                        double price_bucket) const;

  // Get the lowest price in a specific time-price bucket
  double get_low_price(const std::chrono::system_clock::time_point& time_bucket,
                       double price_bucket) const;

  // Get statistics for a specific time period
  TPOStatistics get_statistics_for_period(
      const std::chrono::system_clock::time_point& start_time,
      const std::chrono::system_clock::time_point& end_time) const;

  // Print TPO profile for debugging purposes
  void print_tpo_profile() const;

  // Integration with LiquiditySweepDetector
  // Methods to support liquidity sweep detection alongside TPO analysis
  void integrate_with_liquidity_detector(LiquiditySweepDetector& detector);

  // Get price-volume data that can be used by liquidity detectors
  std::vector<std::pair<double, double>> get_price_volume_profile() const;
};

#endif  // PUBBTQUANT_TPOENGINE_H