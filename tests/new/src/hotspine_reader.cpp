#include "hotspine_reader.hpp"
#include "hotspine_layout_v3.hpp"  // Updated to use V3 layout with atomic operations
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
  if (!mapped_region_ || mapped_size_ < sizeof(HotSpine::V3::SharedMemoryLayoutV3)) {
    std::cerr << "[SHM] Shared memory too small for header" << std::endl;
    return false;
  }

  HotSpine::V3::SharedMemoryLayoutV3 *layout =
      static_cast<HotSpine::V3::SharedMemoryLayoutV3 *>(mapped_region_);
  HotSpine::V3::RingBufferHeader *header = &layout->header;

  // Verify magic number
  if (header->magic != HotSpine::V3::HOTSPINE_MAGIC) {
    std::cerr << "[SHM] Invalid magic number in shared memory: 0x" << std::hex
              << header->magic << std::endl;
    return false;
  }

  // Verify version
  if (header->version != HotSpine::V3::HOTSPINE_VERSION) {
    std::cerr << "[SHM] Invalid version in shared memory: " << std::dec
              << header->version << " (expected: " << HotSpine::V3::HOTSPINE_VERSION << ")"
              << std::endl;
    return false;
  }

  return true;
}

bool HotSpineReader::isAttached() const { return attached_; }

bool HotSpineReader::pollTrade(HotSpine::V3::HotspineData &trade) {
  if (!attached_ || !mapped_region_) {
    return false;
  }

  HotSpine::V3::SharedMemoryLayoutV3 *layout =
      static_cast<HotSpine::V3::SharedMemoryLayoutV3 *>(mapped_region_);
  HotSpine::V3::RingBufferHeader *header = &layout->header;
  
  // Atomically load the current write head
  uint64_t write_head = header->write_head.load(std::memory_order_acquire);
  uint64_t read_tail = header->read_tail.load(std::memory_order_relaxed);

  // Debug logging for header fields (throttled)
  static int debug_counter = 0;
  if (++debug_counter % 1000 == 0) { // Log every 1000 calls
    std::cout << "[DEBUG] Header fields: magic=0x" << std::hex << header->magic
              << ", version=" << std::dec << header->version
              << ", write_head=" << header->write_head.load(std::memory_order_acquire)
              << ", read_tail=" << header->read_tail.load(std::memory_order_acquire)
              << ", dropped_count=" << header->dropped_count.load()
              << std::endl;
  }

  // Check if there's data available
  if (read_tail >= write_head) {
    return false; // Buffer empty
  }

  // Calculate ring buffer index using mask
  uint64_t index = read_tail & HotSpine::V3::RING_BUFFER_MASK;
  
  // Get the ring buffer data pointer
  HotSpine::V3::HotspineData *ring_buffer = 
      reinterpret_cast<HotSpine::V3::HotspineData *>(
          layout->ring_buffer_data);

  // Copy data to output
  trade = ring_buffer[index];

  // Atomically advance read tail using compare-and-swap to avoid race conditions
  uint64_t expected = read_tail;
  while (!header->read_tail.compare_exchange_weak(expected, read_tail + 1, 
                                                 std::memory_order_release, 
                                                 std::memory_order_relaxed)) {
    // If another thread updated read_tail, check if we should continue
    if (expected > read_tail) {
      // Another thread already consumed this entry, return false
      return false;
    }
    // Retry with new expected value
    read_tail = expected;
    
    // Recalculate ring buffer index with new read_tail
    index = read_tail & HotSpine::V3::RING_BUFFER_MASK;
    
    // Update trade with new entry
    trade = ring_buffer[index];
  }

  return true;
}

bool HotSpineReader::pollOrderbook(HotOrderbookSnapshot &snapshot) {
  // For now, we'll return false since the V3 layout doesn't have a separate orderbook buffer
  // The V3 layout uses a unified ring buffer for all data types
  // Orderbook data would need to be handled differently in a unified approach
  
  // TODO: Implement orderbook polling when orderbook data is integrated into the V3 layout
  return false;
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

  HotSpine::V3::SharedMemoryLayoutV3 *layout =
      static_cast<HotSpine::V3::SharedMemoryLayoutV3 *>(mapped_region_);
  HotSpine::V3::RingBufferHeader *header = &layout->header;

  // Calculate used count from write_head and read_tail
  uint64_t write_head = header->write_head.load(std::memory_order_acquire);
  uint64_t read_tail = header->read_tail.load(std::memory_order_acquire);
  uint64_t capacity = HotSpine::V3::RING_BUFFER_SIZE;

  uint64_t used = 0;
  if (write_head >= read_tail) {
    used = write_head - read_tail;
    // Cap at capacity to handle cases where reader falls behind significantly
    if (used > capacity) {
      used = capacity;
    }
  } else {
    // This shouldn't normally happen with monotonic counters, but handle it
    used = 0;
  }

  // Debug logging for buffer status
  static int status_counter = 0;
  if (++status_counter % 100 == 0) { // Log every 100 calls
    double usage_pct =
        capacity > 0 ? (static_cast<double>(used) / capacity) * 100.0 : 0.0;
    std::cout << "[DEBUG] Buffer status: used=" << used
              << ", capacity=" << capacity << ", write_head=" << write_head
              << ", read_tail=" << read_tail << ", usage=" << usage_pct << "%"
              << std::endl;
  }

  return {used, capacity};
}

bool HotSpineReader::is_healthy() const {
  return attached_ && mapped_region_ != nullptr;
}

} // namespace HotSpine
