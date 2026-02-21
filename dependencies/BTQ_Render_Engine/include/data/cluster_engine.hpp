#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "data/core_types.hpp"

namespace BTQuant {

// ============================================================================
// Tick Cluster — volume aggregated at a single price level
// ============================================================================
struct TickCluster {
  double price = 0.0;
  double buy_vol = 0.0;
  double sell_vol = 0.0;
  uint32_t buy_count = 0;
  uint32_t sell_count = 0;

  double total_vol() const { return buy_vol + sell_vol; }
  double delta() const { return buy_vol - sell_vol; }
  uint32_t total_count() const { return buy_count + sell_count; }
};

// ============================================================================
// Footprint Column — one candle's footprint data
// ============================================================================
struct FootprintColumn {
  uint64_t open_time_us = 0;
  uint64_t close_time_us = 0;
  double open = 0.0;
  double high = 0.0;
  double low = 0.0;
  double close = 0.0;
  double delta = 0.0;  // cumulative buy_vol - sell_vol

  std::vector<TickCluster> ticks;  // sorted by price ascending

  double total_volume() const {
    double v = 0.0;
    for (const auto& t : ticks) v += t.total_vol();
    return v;
  }
};

// ============================================================================
// Volume Profile — session-level aggregation
// ============================================================================
struct VolumeProfile {
  std::vector<TickCluster> levels;  // sorted by price ascending
  double poc_price = 0.0;           // Point of Control (highest volume price)
  double vah = 0.0;                 // Value Area High
  double val = 0.0;                 // Value Area Low
  double max_volume = 0.0;          // Volume at POC
};

// ============================================================================
// ClusterEngine — aggregates trades into footprint columns and volume profiles
// ============================================================================
class ClusterEngine {
 public:
  // Candle duration for footprint columns (default 1 minute)
  explicit ClusterEngine(uint64_t candle_duration_us = 60'000'000);

  // Ingest a single trade (called from MarketDataProcessor notification)
  void ingest_trade(const TradeData& trade);

  // ---- Read API (called from render thread) ----

  // Get footprint columns for a symbol (most recent N)
  // Copies data under lock to avoid contention
  std::vector<FootprintColumn> get_footprint_columns(uint32_t symbol_id,
                                                     size_t max_columns = 60) const;

  // Get session volume profile for a symbol
  VolumeProfile get_volume_profile(uint32_t symbol_id) const;

  // Get cumulative volume delta (CVD) for a symbol
  double get_cvd(uint32_t symbol_id) const;

  // Configuration
  void set_tick_size(uint32_t symbol_id, double tick_size);
  void set_candle_duration(uint64_t duration_us);

 private:
  static constexpr size_t MAX_SYMBOLS = 100;
  static constexpr size_t MAX_COLUMNS = 120;  // Keep last 120 candles

  uint64_t candle_duration_us_;

  // Per-symbol state
  struct SymbolState {
    mutable std::mutex mu;  // Protects all fields below

    double tick_size = 0.0;  // 0 = auto-detect
    bool tick_detected = false;

    // Footprint columns (ring buffer, newest at back)
    std::vector<FootprintColumn> columns;

    // Current (in-progress) column
    FootprintColumn current_column;
    bool has_current = false;

    // Session volume profile (cumulative)
    std::unordered_map<int64_t, TickCluster> profile_map;  // tick_index → cluster

    // Cumulative Volume Delta
    double cvd = 0.0;
  };

  SymbolState symbols_[MAX_SYMBOLS];

  // Helpers
  int64_t price_to_tick(double price, double tick_size) const;
  double tick_to_price(int64_t tick, double tick_size) const;
  double detect_tick_size(double price) const;
  void finalize_column(SymbolState& state);
  void compute_value_area(VolumeProfile& profile) const;
};

}  // namespace BTQuant
