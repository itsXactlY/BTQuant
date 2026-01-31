#pragma once

#include "config/config_loader.hpp"
#include "hotspine_extended_reader.hpp"
#include <deque>
#include <map>
#include <optional>
#include <string>

namespace BTQuant {

struct WhaleSignal {
  std::string symbol;
  std::string whale_exchange;
  double whale_price;
  double whale_size;
  double whale_size_usd;
  bool is_buy;
  std::vector<std::string> lagging_exchanges;
  double expected_move_bps;
  uint64_t timestamp_us;
  uint64_t reaction_window_us;

  std::string to_string() const;
};

class WhaleFrontRunDetector {
public:
  explicit WhaleFrontRunDetector(HotSpineExtendedReader &reader);

  // Initialize with configuration
  bool initialize(const BTQuant::Config::ConfigLoader &config,
                  const std::string &detector_name = "whale_frontrun");

  // Detect whale movements
  std::optional<WhaleSignal> detect(const std::string &symbol);

  // Configuration accessors
  double get_threshold_usd() const { return whale_threshold_usd_; }
  void set_threshold_usd(double threshold) { whale_threshold_usd_ = threshold; }
  size_t get_min_lagging_exchanges() const { return min_lagging_exchanges_; }
  double get_reaction_threshold_bps() const { return reaction_threshold_bps_; }
  uint64_t get_reaction_window_us() const { return reaction_window_us_; }

  // Statistics
  uint64_t get_detections() const { return detections_; }
  void reset_statistics() { detections_ = 0; }

private:
  HotSpineExtendedReader &reader_;
  double whale_threshold_usd_;
  size_t min_lagging_exchanges_;
  double reaction_threshold_bps_;
  uint64_t reaction_window_us_;
  uint64_t detections_;

  struct PriceHistory {
    std::deque<std::pair<uint64_t, double>> prices;
    size_t max_history;
    static const size_t MAX_HISTORY = 100;

    void add(uint64_t timestamp_us, double price);
    double get_volatility_bps(uint64_t lookback_us);
  };

  std::map<std::string, PriceHistory> price_histories_;

  std::string make_key(const std::string &exchange,
                       const std::string &symbol) const;
};

} // namespace BTQuant
