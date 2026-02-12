#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <expected>
#include <format>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "../../../include/hotspine_layout_v3.hpp"  // Include the main layout definition

namespace BTQuant {
namespace RenderEngine {
class MarketDataProcessor;
}  // namespace RenderEngine

// Use the new layout from hotspine_layout_v3.hpp
using SharedMemoryHeader = HotSpine::V3::RingBufferHeader;

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
  std::map<double, double> vol_profile_;

  InstrumentStore() = default;
  ~InstrumentStore() = default;

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
    vol_profile_ = other.vol_profile_;
  }
};

// ============================================================================
// Lock-Free Shared Memory Reader
// ============================================================================

class HotSpineDataBridge {
 public:
  HotSpineDataBridge(const std::string& shm_path = "/btquant");
  ~HotSpineDataBridge();

  [[nodiscard]] std::expected<void, std::string> start();
  [[nodiscard]] std::expected<void, std::string> connect();  // New method for validation
  void stop();
  void sync();  // Performs real-time synchronization

  // Direct access to MarketDataProcessor for all data operations
  void setMarketDataProcessor(std::shared_ptr<RenderEngine::MarketDataProcessor> processor) {
    data_processor_ = processor;
  }

  // Get active symbols from MarketDataProcessor (direct)
  [[nodiscard]] std::vector<uint32_t> getActiveSymbols() const;

  // Get views into live data (C++26 optimized)
  [[nodiscard]] std::span<const HotSpine::V3::HotspineData> getTradeBuffer() const;

  // Get symbol information from registry
  std::string getSymbolName(uint32_t symbol_id) const;
  std::string getExchangeName(uint32_t symbol_id) const;

  // Public accessors for direct SHM access
  SharedMemoryHeader* getHeader() const { return header_; }
  HotSpine::V3::HotspineData* getBasePtr() const { return base_ptr_; }

 private:
  std::shared_ptr<RenderEngine::MarketDataProcessor> data_processor_;
  std::string shm_path_;
  int shm_fd_ = -1;
  void* shm_ptr_ = nullptr;
  size_t shm_size_ = 0;

  std::atomic<bool> running_{false};
  std::thread sync_thread_;  // Real-time sync thread

  // Ring Buffer Pointers
  SharedMemoryHeader* header_ = nullptr;
  HotSpine::V3::HotspineData* base_ptr_ = nullptr;  // Raw pointer to HotspineData array

  // Local tracking of read progress
  std::atomic<uint64_t> last_read_idx_{0};

  void sync_shm();
  void sync_loop();  // Real-time sync loop with high priority
};

}  // namespace BTQuant