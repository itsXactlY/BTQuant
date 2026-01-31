#include "hotspine_writer.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "hotspine_layout.hpp"

namespace HotSpine {

static uint32_t generateSymbolId(const std::string& exchange, const std::string& symbol, const std::string& market_type) {
  std::stringstream ss;
  ss << exchange << ":" << symbol << ":" << market_type;
  std::string key = ss.str();
  uint32_t hash = 5381;
  for (char c : key) {
    hash = ((hash << 5) + hash) + c;
  }
  return hash;
}

HotSpineWriter::HotSpineWriter(const std::string& shm_name) : shm_name_(shm_name) {
  std::cout << "[HotSpineWriter] Initializing with shared memory name: " << shm_name << std::endl;
  batching_enabled_ = false;
  if (!attachToSharedMemory()) {
    std::cerr << "[HotSpineWriter][ERROR] Failed to attach to shared memory: " << shm_name << std::endl;
  } else {
    std::cout << "[HotSpineWriter][INFO] Successfully initialized" << std::endl;
  }
}

bool HotSpineWriter::loadSymbolMappings(const std::string& filepath) { return BTQuant::SymbolRegistry::instance().load_from_file(filepath); }

HotSpineWriter::~HotSpineWriter() {
  std::cout << "[HotSpineWriter][INFO] Shutting down HotSpine writer" << std::endl;
  std::cout << "[HotSpineWriter][INFO] Final statistics: trades_written=" << trades_written_ << ", write_errors=" << write_errors_ << std::endl;
  detachFromSharedMemory();
}

bool HotSpineWriter::attachToSharedMemory() {
  shm_fd_ = shm_open(shm_name_.c_str(), O_RDWR | O_CREAT, 0666);
  if (shm_fd_ == -1) {
    std::cerr << "HotSpineWriter: shm_open failed: " << strerror(errno) << std::endl;
    return false;
  }
  struct stat st;
  bool needs_init = false;
  if (fstat(shm_fd_, &st) == -1 || st.st_size == 0) {
    needs_init = true;
  }
  size_t shm_size;
  if (needs_init) {
    shm_size = HotSpine::calculateSharedMemorySize(HotSpine::DEFAULT_CAPACITY, HotSpine::DEFAULT_ORDERBOOK_CAPACITY);
    if (ftruncate(shm_fd_, shm_size) == -1) {
      std::cerr << "HotSpineWriter: ftruncate failed: " << strerror(errno) << std::endl;
      close(shm_fd_);
      shm_fd_ = -1;
      return false;
    }
  } else {
    shm_size = st.st_size;
  }
  shm_ptr_ = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
  if (shm_ptr_ == MAP_FAILED) {
    std::cerr << "HotSpineWriter: mmap failed: " << strerror(errno) << std::endl;
    close(shm_fd_);
    shm_fd_ = -1;
    return false;
  }
  header_ = static_cast<HotSpine::SharedMemoryHeader*>(shm_ptr_);
  trades_buffer_ = reinterpret_cast<HotSpine::HotTrade*>(static_cast<char*>(shm_ptr_) + HotSpine::HEADER_SIZE);
  size_t trade_buffer_bytes = HotSpine::DEFAULT_CAPACITY * sizeof(HotSpine::HotTrade);
  orderbooks_buffer_ = reinterpret_cast<HotSpine::HotOrderbookSnapshot*>(static_cast<char*>(shm_ptr_) + HotSpine::HEADER_SIZE + trade_buffer_bytes);
  if (needs_init) {
    header_->magic = HotSpine::HOTSPINE_MAGIC;
    header_->version = HotSpine::HOTSPINE_VERSION;
    header_->capacity = HotSpine::DEFAULT_CAPACITY;
    header_->write_index = 0;
    header_->read_index = 0;
    header_->lost_count = 0;
    header_->orderbook_write_index = 0;
    header_->orderbook_read_index = 0;
    header_->orderbook_lost_count = 0;
    header_->orderbook_capacity = HotSpine::DEFAULT_ORDERBOOK_CAPACITY;
    std::memset(header_->padding, 0, sizeof(header_->padding));
    std::cout << "HotSpineWriter: Created and initialized shared memory: " << shm_name_ << " (capacity: " << header_->capacity << " trades, "
              << header_->orderbook_capacity << " orderbooks)" << std::endl;
  } else {
    // Validate magic number
    if (header_->magic != HotSpine::HOTSPINE_MAGIC) {
      std::cerr << "HotSpineWriter: Invalid shared memory magic number: 0x" << std::hex << header_->magic << " (expected: 0x" << HotSpine::HOTSPINE_MAGIC << ")"
                << std::endl;
      detachFromSharedMemory();
      // Force re-init by unlinking and trying again once
      shm_unlink(shm_name_.c_str());
      return attachToSharedMemory();
    }
    if (header_->version != HotSpine::HOTSPINE_VERSION) {
      std::cerr << "HotSpineWriter: Invalid shared memory version: " << header_->version << " (expected: " << HotSpine::HOTSPINE_VERSION << ")" << std::endl;
      detachFromSharedMemory();
      // Force re-init by unlinking and trying again once
      shm_unlink(shm_name_.c_str());
      return attachToSharedMemory();
    }
    std::cout << "HotSpineWriter: Successfully attached to shared memory: " << shm_name_ << " (capacity: " << header_->capacity << " trades, "
              << header_->orderbook_capacity << " orderbooks)" << std::endl;
  }
  return true;
}

bool HotSpineWriter::detachFromSharedMemory() {
  if (shm_ptr_ != nullptr && shm_ptr_ != MAP_FAILED) {
    struct stat st;
    size_t shm_size = 0;
    if (fstat(shm_fd_, &st) != -1) shm_size = st.st_size;
    if (munmap(shm_ptr_, shm_size) == -1) std::cerr << "HotSpineWriter: munmap failed: " << strerror(errno) << std::endl;
    shm_ptr_ = nullptr;
    header_ = nullptr;
    trades_buffer_ = nullptr;
    orderbooks_buffer_ = nullptr;
  }
  if (shm_fd_ != -1) {
    close(shm_fd_);
    shm_fd_ = -1;
  }
  return true;
}

uint64_t HotSpineWriter::getCurrentTimestampMicros() {
  using namespace std::chrono;
  return duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
}

uint32_t HotSpineWriter::getSymbolId(const std::string& exchange, const std::string& symbol, const std::string& market_type) const {
  auto& registry = BTQuant::SymbolRegistry::instance();
  if (auto id = registry.get_symbol_id(exchange, symbol)) {
    return *id;
  } else {
    // Auto-register new symbol dynamically
    uint32_t new_id = const_cast<BTQuant::SymbolRegistry&>(registry).register_symbol(exchange, symbol, 0);
    std::cout << "[HotSpineWriter][INFO] Dynamically registered symbol " << exchange << ":" << symbol << " with ID " << new_id << std::endl;

    // Save updated registry to shared memory location for Python reader
    const_cast<BTQuant::SymbolRegistry&>(registry).save_to_file("/dev/shm/btquant_symbols.json");

    return new_id;
  }
}

bool HotSpineWriter::isHealthy() const { return shm_ptr_ != nullptr && header_ != nullptr; }

bool HotSpineWriter::writeTrade(const MarketData::Trade& trade) {
  if (!isHealthy()) {
    write_errors_++;
    return false;
  }
  if (batching_enabled_) {
    std::lock_guard<std::mutex> lock(batch_mutex_);
    batch_buffer_.push_back(trade);
    if (batch_buffer_.size() >= batch_size_) flushBatch();
    trades_written_++;
    return true;
  }
  HotSpine::HotTrade hot_trade;
  hot_trade.ts_exchange = static_cast<uint64_t>(trade.timestamp_us);
  hot_trade.ts_local = getCurrentTimestampMicros();
  hot_trade.price = trade.price;
  hot_trade.size = trade.quantity;
  hot_trade.symbol_id = getSymbolId(trade.exchange, trade.symbol, trade.market_type);
  hot_trade.side = (trade.side == "buy") ? 0 : 1;
  uint64_t write_idx = header_->write_index;
  uint64_t next_idx = (write_idx + 1) % header_->capacity;
  if (next_idx == header_->read_index) {
    header_->lost_count++;
    write_errors_++;
    return false;
  }
  trades_buffer_[write_idx] = hot_trade;
  header_->write_index = next_idx;
  trades_written_++;
  return true;
}

bool HotSpineWriter::writeOrderbook(const MarketData::OrderbookSnapshot& ob, const std::vector<HotSpine::HotOrderbookLevel>& bids,
                                    const std::vector<HotSpine::HotOrderbookLevel>& asks) {
  if (!isHealthy() || orderbooks_buffer_ == nullptr) {
    write_errors_++;
    return false;
  }
  HotSpine::HotOrderbookSnapshot hot_ob;
  hot_ob.ts_exchange = static_cast<uint64_t>(ob.timestamp_us);
  hot_ob.ts_local = getCurrentTimestampMicros();
  hot_ob.symbol_id = getSymbolId(ob.exchange, ob.symbol, ob.market_type);

  hot_ob.bids_count = static_cast<uint8_t>(std::min<size_t>(bids.size(), 20));
  for (size_t i = 0; i < hot_ob.bids_count; ++i) {
    hot_ob.bids[i] = bids[i];
  }

  hot_ob.asks_count = static_cast<uint8_t>(std::min<size_t>(asks.size(), 20));
  for (size_t i = 0; i < hot_ob.asks_count; ++i) {
    hot_ob.asks[i] = asks[i];
  }

  uint64_t write_idx = header_->orderbook_write_index;
  if (header_->orderbook_capacity == 0) return false;
  uint64_t next_idx = (write_idx + 1) % header_->orderbook_capacity;
  if (next_idx == header_->orderbook_read_index) {
    header_->orderbook_lost_count++;
    write_errors_++;
    return false;
  }
  orderbooks_buffer_[write_idx] = hot_ob;
  header_->orderbook_write_index = next_idx;
  return true;
}

void HotSpineWriter::flushBatch() {
  if (!isHealthy() || batch_buffer_.empty()) return;
  std::vector<MarketData::Trade> batch_to_write;
  {
    std::lock_guard<std::mutex> lock(batch_mutex_);
    batch_to_write.swap(batch_buffer_);
  }
  uint64_t current_idx = header_->write_index;
  for (const auto& trade : batch_to_write) {
    HotSpine::HotTrade hot_trade;
    hot_trade.ts_exchange = static_cast<uint64_t>(trade.timestamp_us);
    hot_trade.ts_local = getCurrentTimestampMicros();
    hot_trade.price = trade.price;
    hot_trade.size = trade.quantity;
    hot_trade.symbol_id = getSymbolId(trade.exchange, trade.symbol, trade.market_type);
    hot_trade.side = (trade.side == "buy") ? 0 : 1;
    uint32_t next_idx = (current_idx + 1) % header_->capacity;
    if (next_idx == header_->read_index) {
      header_->lost_count++;
      write_errors_++;
      continue;
    }
    trades_buffer_[current_idx] = hot_trade;
    current_idx = next_idx;
  }
  header_->write_index = current_idx;
}

bool HotSpineWriter::writeTrades(const std::vector<MarketData::Trade>& trades) {
  if (!isHealthy()) return false;
  bool all_success = true;
  for (const auto& trade : trades) {
    if (!writeTrade(trade)) all_success = false;
  }
  return all_success;
}

std::string HotSpineWriter::getDetailedStats() const {
  if (!isHealthy()) return "{}";
  std::stringstream ss;
  ss << "{\"trades_written\":" << trades_written_.load() << "}";
  return ss.str();
}

}  // namespace HotSpine
