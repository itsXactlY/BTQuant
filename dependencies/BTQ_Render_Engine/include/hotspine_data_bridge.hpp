#pragma once

#include <array>
#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <chrono>

namespace BTQuant {
namespace RenderEngine {
class MarketDataProcessor;
} // namespace RenderEngine

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
  uint8_t side; // 0=Buy, 1=Sell
  uint8_t padding[3];
};

struct HotOrderbookLevel {
  double price;
  double size;
};

struct HotOrderbookSnapshot {
  uint64_t ts_exchange;
  uint64_t ts_local;
  uint32_t symbol_id;
  uint8_t bids_count;
  uint8_t asks_count;
  uint8_t padding[2];
  std::array<HotOrderbookLevel, 20> bids;
  std::array<HotOrderbookLevel, 20> asks;
};

struct SharedMemoryHeader {
  uint32_t magic;   // "BTQU"
  uint32_t version; // 2
  uint64_t capacity;
  uint64_t write_index; // Atomic access via intrinsics
  uint64_t read_index;
  uint64_t lost_count;

  uint64_t orderbook_write_index; // Atomic access via intrinsics
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
  // Lock-free access assuming single-writer, multiple-reader pattern
  std::vector<double> timestamps;
  std::vector<double> opens;
  std::vector<double> highs;
  std::vector<double> lows;
  std::vector<double> closes;
  std::vector<double> volumes;

  // Volume Profile (Price -> Cumulative Volume) - Lock-free updates
  std::map<double, double> m_vol_profile;

  // Latest Snapshot for Heatmap/Orderbook - Atomic for thread-safety
  std::atomic<HotOrderbookSnapshot*> latest_snapshot{nullptr};

  InstrumentStore() = default;
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
    m_vol_profile = other.m_vol_profile;
    HotOrderbookSnapshot* snap = other.latest_snapshot.load();
    if (snap) {
      latest_snapshot.store(new HotOrderbookSnapshot(*snap));
    }
  }
};

// ============================================================================
// Lock-Free Shared Memory Reader
// ============================================================================

class HotSpineDataBridge {
public:
  HotSpineDataBridge(const std::string &shm_path = "/btquant");
  ~HotSpineDataBridge();

  bool start();
  void stop();
  void sync(); // Performs real-time synchronization

  // Direkter Zugriff auf MarketDataProcessor für alle Datenoperationen
  void setMarketDataProcessor(
      std::shared_ptr<RenderEngine::MarketDataProcessor> processor) {
    m_data_processor = processor;
  }

  // Get active symbols from MarketDataProcessor (direct)
  std::vector<uint32_t> getActiveSymbols() const;

  // Get symbol information from registry
  std::string getSymbolName(uint32_t symbol_id) const;
  std::string getExchangeName(uint32_t symbol_id) const;

private:
  std::shared_ptr<RenderEngine::MarketDataProcessor> m_data_processor;
  std::string m_shm_path;
  int m_shm_fd = -1;
  void *m_shm_ptr = nullptr;
  size_t m_shm_size = 0;

  std::atomic<bool> m_running{false};
  std::jthread m_sync_thread; // Real-time sync thread

  // Ring Buffer Pointers
  SharedMemoryHeader *m_header = nullptr;
  HotTrade *m_trades = nullptr;
  HotOrderbookSnapshot *m_books = nullptr;

  // Local tracking of read progress
  std::atomic<uint64_t> m_last_read_idx{0};
  std::atomic<uint64_t> m_last_book_read_idx{0};

  void sync_shm();
  void sync_loop(); // Real-time sync loop with high priority
};

} // namespace BTQuant