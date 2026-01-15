#include "hotspine_reader.hpp"
#include "hotspine_layout.hpp"
#include "utils/dynamic_logger.hpp"
#include <chrono>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

namespace HotSpine {

HotSpineReader::HotSpineReader(const std::string &shm_name)
    : shm_name_(shm_name), attached_(false), fd_(-1), mapped_region_(nullptr),
      mapped_size_(0) {

  attach_to_shm();
}

HotSpineReader::~HotSpineReader() { detach_from_shm(); }

void HotSpineReader::attach_to_shm() {
  // Open shared memory file
  fd_ = shm_open(shm_name_.c_str(), O_RDWR, 0666);
  if (fd_ < 0) {
    std::cerr << "[SHM] Failed to open shared memory '" << shm_name_
              << "': " << strerror(errno) << std::endl;
    attached_ = false;
    return;
  }

  // Get file size
  struct stat st;
  if (fstat(fd_, &st) < 0) {
    std::cerr << "[SHM] Failed to stat shared memory: " << strerror(errno)
              << std::endl;
    close(fd_);
    fd_ = -1;
    attached_ = false;
    return;
  }

  mapped_size_ = static_cast<size_t>(st.st_size);

  // Map into memory
  mapped_region_ =
      mmap(nullptr, mapped_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  if (mapped_region_ == MAP_FAILED) {
    std::cerr << "[SHM] Failed to mmap shared memory: " << strerror(errno)
              << std::endl;
    close(fd_);
    fd_ = -1;
    mapped_region_ = nullptr;
    mapped_size_ = 0;
    attached_ = false;
    return;
  }

  // Validate header
  if (!validate_header()) {
    detach_from_shm();
    return;
  }

  attached_ = true;
  std::cout << "[SHM] Attached to shared memory: " << shm_name_
            << " (size: " << mapped_size_ << " bytes)" << std::endl;
}

void HotSpineReader::detach_from_shm() {
  if (mapped_region_) {
    munmap(mapped_region_, mapped_size_);
    mapped_region_ = nullptr;
    mapped_size_ = 0;
  }

  if (fd_ >= 0) {
    close(fd_);
    fd_ = -1;
  }

  if (attached_) {
    std::cout << "[SHM] Detached from shared memory: " << shm_name_
              << std::endl;
  }

  attached_ = false;
}

bool HotSpineReader::validate_header() {
  if (!mapped_region_ || mapped_size_ < sizeof(SharedMemoryHeader)) {
    std::cerr << "[SHM] Shared memory too small for header" << std::endl;
    return false;
  }

  SharedMemoryHeader *header =
      static_cast<SharedMemoryHeader *>(mapped_region_);

  // Verify magic number
  if (header->magic != HOTSPINE_MAGIC) {
    std::cerr << "[SHM] Invalid magic number in shared memory: 0x" << std::hex
              << header->magic << std::endl;
    return false;
  }

  // Verify version
  if (header->version != HOTSPINE_VERSION) {
    std::cerr << "[SHM] Invalid version in shared memory: " << std::dec
              << header->version << " (expected: " << HOTSPINE_VERSION << ")"
              << std::endl;
    return false;
  }

  return true;
}

bool HotSpineReader::isAttached() const { return attached_; }

bool HotSpineReader::pollTrade(HotTrade &trade) {
  if (!attached_ || !mapped_region_) {
    return false;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  SharedMemoryHeader *header =
      static_cast<SharedMemoryHeader *>(mapped_region_);
  uint64_t write_pos = header->write_index;
  uint64_t read_pos = header->read_index;

  // Debug logging for header fields (throttled)
  static int debug_counter = 0;
  if (++debug_counter % 1000 == 0) { // Log every 1000 calls
    std::cout << "[DEBUG] Header fields: magic=0x" << std::hex << header->magic
              << ", version=" << std::dec << header->version
              << ", capacity=" << header->capacity
              << ", write_index=" << header->write_index
              << ", read_index=" << header->read_index
              << ", lost_count=" << header->lost_count
              << ", orderbook_write_index=" << header->orderbook_write_index
              << ", orderbook_read_index=" << header->orderbook_read_index
              << std::endl;
  }

  // Check if there's data available (and handle potential read_index exceeding
  // write_index)
  if (read_pos >= write_pos) {
    if (read_pos > write_pos) {
      header->read_index = write_pos; // Reset to write_pos to recover
    }
    return false; // Buffer empty or synchronized
  }

  // Calculate entry position (entries start after FIXED HEADER SIZE)
  // Note: Use HEADER_SIZE from layout, not sizeof(SharedMemoryHeader)
  size_t entry_offset =
      HEADER_SIZE + (read_pos % header->capacity) * sizeof(HotTrade);

  if (entry_offset + sizeof(HotTrade) > mapped_size_) {
    std::cerr << "[SHM] Trade entry exceeds shared memory bounds" << std::endl;
    header->read_index = write_pos; // Skip this entry
    return false;
  }

  HotTrade *entry = reinterpret_cast<HotTrade *>(
      static_cast<char *>(mapped_region_) + entry_offset);

  // Copy data to output
  trade = *entry;

  // Advance read position
  header->read_index = read_pos + 1;

  return true;
}

bool HotSpineReader::pollOrderbook(HotOrderbookSnapshot &snapshot) {
  if (!attached_ || !mapped_region_) {
    return false;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  SharedMemoryHeader *header =
      static_cast<SharedMemoryHeader *>(mapped_region_);
  uint64_t write_pos = header->orderbook_write_index;
  uint64_t read_pos = header->orderbook_read_index;

  // Check if there's data available
  if (read_pos >= write_pos) {
    if (read_pos > write_pos) {
      header->orderbook_read_index = write_pos; // Reset to recover
    }
    return false; // Buffer empty or synchronized
  }

  // Calculate trade buffer size to skip
  size_t trade_buffer_bytes = header->capacity * sizeof(HotTrade);

  // Calculate entry position (Orderbooks start after Header + Trade Buffer)
  size_t entry_offset =
      HEADER_SIZE + trade_buffer_bytes +
      (read_pos % header->orderbook_capacity) * sizeof(HotOrderbookSnapshot);

  if (entry_offset + sizeof(HotOrderbookSnapshot) > mapped_size_) {
    std::cerr << "[SHM] Orderbook entry exceeds shared memory bounds"
              << std::endl;
    header->orderbook_read_index = write_pos; // Skip this entry
    return false;
  }

  HotOrderbookSnapshot *entry = reinterpret_cast<HotOrderbookSnapshot *>(
      static_cast<char *>(mapped_region_) + entry_offset);

  // Copy data to output
  snapshot = *entry;

  // Advance read position
  header->orderbook_read_index = read_pos + 1;

  return true;
}

const std::string &HotSpineReader::getShmName() const { return shm_name_; }

bool HotSpineReader::reattach() {
  detach_from_shm();
  attach_to_shm();
  return attached_;
}

std::pair<uint64_t, uint64_t> HotSpineReader::get_buffer_status() const {
  if (!attached_ || !mapped_region_) {
    return {0, 0};
  }

  SharedMemoryHeader *header =
      static_cast<SharedMemoryHeader *>(mapped_region_);

  // Calculate used count from write_index and read_index
  // For circular buffer: used = (write_index - read_index) % capacity
  uint64_t capacity = header->capacity;
  uint64_t write_idx = header->write_index;
  uint64_t read_idx = header->read_index;

  // Logic for linear counter (reader chases writer)
  // The indices are monotonic counters, so straight subtraction works if
  // unsigned arithmetic wraps properly (which it does) But logically, used =
  // write - read.

  uint64_t used = 0;
  if (write_idx >= read_idx) {
    used = write_idx - read_idx;
  } else {
    // This theoretically shouldn't happen with monotonic counters unless
    // overflowed Or if the writer reset but reader didn't (unlikely with same
    // shm) Treat as 0 or full reset
    used = 0;
  }

  // Debug logging for buffer status
  static int status_counter = 0;
  if (++status_counter % 100 == 0) { // Log every 100 calls
    double usage_pct =
        capacity > 0 ? (static_cast<double>(used) / capacity) * 100.0 : 0.0;
    std::cout << "[DEBUG] Buffer status: used=" << used
              << ", capacity=" << capacity << ", write_index=" << write_idx
              << ", read_index=" << read_idx << ", usage=" << usage_pct << "%"
              << std::endl;
  }

  return {used, capacity};
}

bool HotSpineReader::is_healthy() const {
  return attached_ && mapped_region_ != nullptr;
}

} // namespace HotSpine
