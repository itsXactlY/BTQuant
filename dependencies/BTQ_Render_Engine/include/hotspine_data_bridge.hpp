#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <expected>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <vector>

#pragma once

namespace BTQuant {
namespace RenderEngine {
class MarketDataProcessor;
}  // namespace RenderEngine

// ============================================================================
// Zero-Copy Shared Memory Data Structures
// Must match Python ctypes structure exactly
// ============================================================================

struct HotTrade {
  uint64_t ts_exchange;
  uint64_t ts_local;
  double price;
  double size;
  uint32_t symbol_id;
  uint8_t side;  // 0=Buy, 1=Sell
  uint8_t padding[3];
};

struct HotOrderbookLevel {
  double price;
  double size;
};

struct alignas(64) OrderBookSnapshot {
  uint64_t ts_exchange;
  uint64_t ts_local;
  uint32_t symbol_id;
  uint8_t bids_count;
  uint8_t asks_count;
  uint8_t padding[2];
  std::array<HotOrderbookLevel, 200> bids;
  std::array<HotOrderbookLevel, 200> asks;
};

static_assert(sizeof(OrderBookSnapshot) % 64 == 0, "OrderBookSnapshot size must be multiple of 64 bytes for alignas(64)");
static_assert(std::is_trivial_v<OrderBookSnapshot>, "OrderBookSnapshot MUST be trivial");
static_assert(std::is_standard_layout_v<OrderBookSnapshot>, "OrderBookSnapshot MUST be standard layout");

struct SharedMemoryHeader {
  uint32_t magic;    // "BTQU"
  uint32_t version;  // 2
  uint64_t capacity;
  uint64_t write_index;  // Atomic access via intrinsics
  uint64_t read_index;
  uint64_t lost_count;

  uint64_t orderbook_write_index;  // Atomic access via intrinsics
  uint64_t orderbook_read_index;
  uint64_t orderbook_lost_count;
  uint64_t orderbook_capacity;
  uint8_t padding[8];
};

struct InstrumentStore {
  std::string symbol;
  std::string exchange;
  std::atomic<uint32_t> symbol_id{0};

  // Structure of Arrays (SoA) for ImPlot compatibility & high-throughput
  // Pre-allocated capacity to prevent runtime growth
  static constexpr size_t INITIAL_CAPACITY = 10000;
  
  std::vector<double> timestamps;
  std::vector<double> opens;
  std::vector<double> highs;
  std::vector<double> lows;
  std::vector<double> closes;
  std::vector<double> volumes;

  // Volume Profile using flat arrays instead of std::map
  // Price bins and volumes stored in parallel vectors for O(1) access
  static constexpr size_t VOL_PROFILE_BINS = 1000;
  std::array<double, VOL_PROFILE_BINS> vol_profile_prices_;
  std::array<double, VOL_PROFILE_BINS> vol_profile_volumes_;
  std::atomic<size_t> vol_profile_count_{0};
  double vol_profile_min_price_ = 0.0;
  double vol_profile_max_price_ = 0.0;
  double vol_profile_bin_size_ = 0.0;

  // Latest Snapshot for Heatmap/Orderbook - Atomic for thread-safety
  std::atomic<OrderBookSnapshot*> latest_snapshot{nullptr};

  InstrumentStore() {
    // Pre-allocate capacity to prevent runtime reallocation
    timestamps.reserve(INITIAL_CAPACITY);
    opens.reserve(INITIAL_CAPACITY);
    highs.reserve(INITIAL_CAPACITY);
    lows.reserve(INITIAL_CAPACITY);
    closes.reserve(INITIAL_CAPACITY);
    volumes.reserve(INITIAL_CAPACITY);
    
    // Initialize volume profile arrays
    vol_profile_prices_.fill(0.0);
    vol_profile_volumes_.fill(0.0);
  }
  
  ~InstrumentStore() {
    if (latest_snapshot.load()) {
      delete latest_snapshot.load();
    }
  }

  // Copy constructor for lock-free duplication
  InstrumentStore(const InstrumentStore& other) {
    symbol = other.symbol;
    exchange = other.exchange;
    symbol_id.store(other.symbol_id.load());
    timestamps = other.timestamps;
    opens = other.opens;
    highs = other.highs;
    lows = other.lows;
    closes = other.closes;
    volumes = other.volumes;
    vol_profile_prices_ = other.vol_profile_prices_;
    vol_profile_volumes_ = other.vol_profile_volumes_;
    vol_profile_count_.store(other.vol_profile_count_.load());
    vol_profile_min_price_ = other.vol_profile_min_price_;
    vol_profile_max_price_ = other.vol_profile_max_price_;
    vol_profile_bin_size_ = other.vol_profile_bin_size_;
    
    OrderBookSnapshot* snap = other.latest_snapshot.load();
    if (snap) {
      latest_snapshot.store(new OrderBookSnapshot(*snap));
    }
  }
  
  // O(1) volume profile update using binning
  void updateVolumeProfile(double price, double volume) {
    if (vol_profile_bin_size_ <= 0.0) return;
    
    size_t bin = static_cast<size_t>((price - vol_profile_min_price_) / vol_profile_bin_size_);
    if (bin < VOL_PROFILE_BINS) {
      vol_profile_volumes_[bin] += volume;
    }
  }
  
  // Reset volume profile for new price range
  void resetVolumeProfile(double min_price, double max_price) {
    vol_profile_min_price_ = min_price;
    vol_profile_max_price_ = max_price;
    vol_profile_bin_size_ = (max_price - min_price) / VOL_PROFILE_BINS;
    vol_profile_volumes_.fill(0.0);
    vol_profile_count_.store(0);
  }
};
}  // namespace BTQuant