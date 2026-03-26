#pragma once

#include "config/config_loader.hpp"
#include "hotspine_extended_reader.hpp"
#include <map>
#include <optional>
#include <set>
#include <string>

namespace BTQuant {

enum class SpoofConfidence { LOW, MEDIUM, HIGH };

struct SpoofingSignal {
  std::string symbol;
  std::string exchange;
  bool is_bid_spoof;
  double spoof_price;
  double spoof_size;
  double spoof_duration_ms;
  uint64_t cancel_count;
  SpoofConfidence confidence;
  uint64_t timestamp_us;

  std::string to_string() const;
};

class SpoofingDetector {
public:
  explicit SpoofingDetector(HotSpineExtendedReader &reader);

  // Initialize with configuration
  bool initialize(const BTQuant::Config::ConfigLoader &config,
                  const std::string &detector_name = "spoofing");

  // Update orderbook state for an exchange
  void update_orderbook(const std::string &exchange, const std::string &symbol);

  // Detect spoofing
  std::optional<SpoofingSignal> detect(const std::string &exchange,
                                       const std::string &symbol);

  // Configuration accessors
  size_t get_min_cancel_count() const { return min_cancel_count_; }
  void set_min_cancel_count(size_t count) { min_cancel_count_ = count; }
  double get_max_duration_ms() const { return max_duration_ms_; }
  double get_min_size() const { return min_size_; }

  // Statistics
  uint64_t get_detections() const { return detections_; }
  void reset_statistics() { detections_ = 0; }

private:
  HotSpineExtendedReader &reader_;
  size_t min_cancel_count_;
  double max_duration_ms_;
  double min_size_;
  uint64_t detections_;

  struct OrderbookLevel {
    double price;
    double size;
    uint64_t first_seen_us;
    uint64_t last_seen_us;
    size_t cancel_count;
  };

  std::map<std::string, std::map<double, OrderbookLevel>> bid_history_;
  std::map<std::string, std::map<double, OrderbookLevel>> ask_history_;

  std::string make_key(const std::string &exchange,
                       const std::string &symbol) const;
};

} // namespace BTQuant
