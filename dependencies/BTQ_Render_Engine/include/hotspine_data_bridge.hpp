#pragma once

#include <array>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

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

// ============================================================================
// HFT Instrument Storage (Optimized for ImPlot)
// ============================================================================

struct MarketInstrument {
  std::string symbol;
  std::string exchange;
  uint32_t symbol_id = 0;

  // Ring Buffer Storage for Time-Series
  static constexpr size_t HISTORY_CAPACITY = 10000;
  std::vector<double> timestamps;
  std::vector<double> opens, highs, lows, closes, volumes;
  size_t write_idx = 0;
  size_t size = 0;

  // Latest Snapshot for Heatmap/Orderbook
  HotOrderbookSnapshot latest_snapshot;

  // Thread-safe access for the UI thread
  mutable std::mutex data_mutex;

  MarketInstrument() {
    timestamps.resize(HISTORY_CAPACITY);
    opens.resize(HISTORY_CAPACITY);
    highs.resize(HISTORY_CAPACITY);
    lows.resize(HISTORY_CAPACITY);
    closes.resize(HISTORY_CAPACITY);
    volumes.resize(HISTORY_CAPACITY);
  }

  void push_trade(const HotTrade &trade) {
    std::lock_guard<std::mutex> lock(data_mutex);

    timestamps[write_idx] = (double)trade.ts_exchange / 1000000.0;
    opens[write_idx] = trade.price;
    highs[write_idx] = trade.price;
    lows[write_idx] = trade.price;
    closes[write_idx] = trade.price;
    volumes[write_idx] = trade.size;

    write_idx = (write_idx + 1) % HISTORY_CAPACITY;
    if (size < HISTORY_CAPACITY)
      size++;
  }

  void update_book(const HotOrderbookSnapshot &snap) {
    std::lock_guard<std::mutex> lock(data_mutex);
    latest_snapshot = snap;
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
  void poll(); // Called by the Market Data Thread

  std::map<std::string, std::shared_ptr<MarketInstrument>> GetInstruments() {
    std::lock_guard<std::mutex> lock(m_map_mutex);
    return m_instruments;
  }

  // Set MarketDataProcessor for OHLCV aggregation
  void setMarketDataProcessor(std::shared_ptr<RenderEngine::MarketDataProcessor> processor) {
    m_data_processor = processor;
  }

private:
  // MarketDataProcessor for OHLCV aggregation
  std::shared_ptr<RenderEngine::MarketDataProcessor> m_data_processor;
  std::string m_shm_path;
  int m_shm_fd = -1;
  void *m_shm_ptr = nullptr;
  size_t m_shm_size = 0;
  bool m_is_simulated = false;

  std::atomic<bool> m_running{false};

  std::map<std::string, std::shared_ptr<MarketInstrument>> m_instruments;
  std::map<uint32_t, std::shared_ptr<MarketInstrument>> m_id_map;
  std::mutex m_map_mutex;

  // Ring Buffer Pointers
  SharedMemoryHeader *m_header = nullptr;
  HotTrade *m_trades = nullptr;
  HotOrderbookSnapshot *m_books = nullptr;

  // Local tracking of read progress
  uint64_t m_last_read_idx = 0;
  uint64_t m_last_book_read_idx = 0;

  void poll_shm();
  void poll_simulated();
  void init_simulation();

  // Helper to map symbol ID to object (lazy if needed)
  std::shared_ptr<MarketInstrument> get_instrument(uint32_t symbol_id);
};

} // namespace BTQuant