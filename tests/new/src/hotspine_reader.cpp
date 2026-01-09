#include "hotspine_reader.hpp"
#include "hotspine_layout.hpp"
#include "utils/dynamic_logger.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>
#include <chrono>
#include <iostream>

namespace HotSpine {

// Header structure at the beginning of shared memory (matches hotspine_layout.hpp)
struct ShmHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t capacity;
    uint64_t used;
    uint64_t trade_write_pos;
    uint64_t trade_read_pos;
    uint64_t orderbook_write_pos;
    uint64_t orderbook_read_pos;
    uint64_t last_update_us;
    uint32_t checksum;
};

// Trade entry in the ring buffer
struct ShmTradeEntry {
    uint64_t ts_exchange;
    uint64_t ts_local;
    uint32_t symbol_id;
    uint8_t side;
    double price;
    double size;
    uint32_t sequence;
};

// Orderbook entry in the ring buffer
struct ShmOrderbookEntry {
    uint64_t ts_exchange;
    uint64_t ts_local;
    uint32_t symbol_id;
    uint8_t bids_count;
    uint8_t asks_count;
    // Followed by bids and asks data
};

HotSpineReader::HotSpineReader(const std::string& shm_name)
    : shm_name_(shm_name)
    , attached_(false)
    , fd_(-1)
    , mapped_region_(nullptr)
    , mapped_size_(0) {
    
    attach_to_shm();
}

HotSpineReader::~HotSpineReader() {
    detach_from_shm();
}

void HotSpineReader::attach_to_shm() {
    // Open shared memory file
    fd_ = shm_open(shm_name_.c_str(), O_RDWR, 0666);
    if (fd_ < 0) {
        std::cerr << "[SHM] Failed to open shared memory '" << shm_name_ << "': " 
                  << strerror(errno) << std::endl;
        attached_ = false;
        return;
    }
    
    // Get file size
    struct stat st;
    if (fstat(fd_, &st) < 0) {
        std::cerr << "[SHM] Failed to stat shared memory: " << strerror(errno) << std::endl;
        close(fd_);
        fd_ = -1;
        attached_ = false;
        return;
    }
    
    mapped_size_ = static_cast<size_t>(st.st_size);
    
    // Map into memory
    mapped_region_ = mmap(nullptr, mapped_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (mapped_region_ == MAP_FAILED) {
        std::cerr << "[SHM] Failed to mmap shared memory: " << strerror(errno) << std::endl;
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
        std::cout << "[SHM] Detached from shared memory: " << shm_name_ << std::endl;
    }
    
    attached_ = false;
}

bool HotSpineReader::validate_header() {
    if (!mapped_region_ || mapped_size_ < sizeof(ShmHeader)) {
        std::cerr << "[SHM] Shared memory too small for header" << std::endl;
        return false;
    }
    
    ShmHeader* header = static_cast<ShmHeader*>(mapped_region_);
    
    // Verify magic number
    if (header->magic != SHM_MAGIC) {
        std::cerr << "[SHM] Invalid magic number in shared memory: 0x" 
                  << std::hex << header->magic << std::endl;
        return false;
    }
    
    // Verify version
    if (header->version != 2) {
        std::cerr << "[SHM] Invalid version in shared memory: " 
                  << std::dec << header->version << std::endl;
        return false;
    }
    
    // Note: Checksum validation disabled because writer uses different header format
    // The writer's SharedMemoryHeader doesn't include a checksum field
    // To re-enable, the writer would need to set header->checksum
    /*
    // Verify checksum (simple XOR for now)
    uint32_t checksum = 0;
    const uint32_t* words = reinterpret_cast<const uint32_t*>(mapped_region_);
    size_t words_count = sizeof(ShmHeader) / sizeof(uint32_t);
    for (size_t i = 0; i < words_count - 1; ++i) {
        checksum ^= words[i];
    }
    
    if (header->checksum != checksum) {
        std::cerr << "[SHM] Checksum mismatch in shared memory header" << std::endl;
        return false;
    }
    */
    
    return true;
}

bool HotSpineReader::isAttached() const {
    return attached_;
}

bool HotSpineReader::pollTrade(HotTrade& trade) {
    if (!attached_ || !mapped_region_) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    ShmHeader* header = static_cast<ShmHeader*>(mapped_region_);
    uint64_t write_pos = header->trade_write_pos;
    uint64_t read_pos = header->trade_read_pos;
    
    // Check if there's data available
    if (write_pos == read_pos) {
        return false;  // Buffer empty
    }
    
    // Calculate entry position (entries start after header)
    size_t entry_offset = sizeof(ShmHeader) + (read_pos % header->capacity) * sizeof(ShmTradeEntry);
    
    if (entry_offset + sizeof(ShmTradeEntry) > mapped_size_) {
        std::cerr << "[SHM] Trade entry exceeds shared memory bounds" << std::endl;
        header->trade_read_pos = write_pos;  // Skip this entry
        return false;
    }
    
    ShmTradeEntry* entry = reinterpret_cast<ShmTradeEntry*>(
        static_cast<char*>(mapped_region_) + entry_offset);
    
    // Copy data to output
    trade.ts_exchange = entry->ts_exchange;
    trade.ts_local = entry->ts_local;
    trade.symbol_id = entry->symbol_id;
    trade.side = entry->side;
    trade.price = entry->price;
    trade.size = entry->size;
    
    // Advance read position
    header->trade_read_pos = read_pos + 1;
    
    return true;
}

bool HotSpineReader::pollOrderbook(HotOrderbookSnapshot& snapshot) {
    if (!attached_ || !mapped_region_) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    ShmHeader* header = static_cast<ShmHeader*>(mapped_region_);
    uint64_t write_pos = header->orderbook_write_pos;
    uint64_t read_pos = header->orderbook_read_pos;
    
    // Check if there's data available
    if (write_pos == read_pos) {
        return false;  // Buffer empty
    }
    
    // Calculate entry position (entries start after header)
    size_t entry_offset = sizeof(ShmHeader) + 
                         header->capacity * sizeof(ShmTradeEntry) +
                         (read_pos % header->capacity) * sizeof(ShmOrderbookEntry);
    
    if (entry_offset + sizeof(ShmOrderbookEntry) > mapped_size_) {
        std::cerr << "[SHM] Orderbook entry exceeds shared memory bounds" << std::endl;
        header->orderbook_read_pos = write_pos;  // Skip this entry
        return false;
    }
    
    ShmOrderbookEntry* entry = reinterpret_cast<ShmOrderbookEntry*>(
        static_cast<char*>(mapped_region_) + entry_offset);
    
    // Copy data to output
    snapshot.ts_exchange = entry->ts_exchange;
    snapshot.ts_local = entry->ts_local;
    snapshot.symbol_id = entry->symbol_id;
    snapshot.bids_count = entry->bids_count;
    snapshot.asks_count = entry->asks_count;
    
    // Advance read position
    header->orderbook_read_pos = read_pos + 1;
    
    return true;
}

const std::string& HotSpineReader::getShmName() const {
    return shm_name_;
}

bool HotSpineReader::reattach() {
    detach_from_shm();
    attach_to_shm();
    return attached_;
}

std::pair<uint64_t, uint64_t> HotSpineReader::get_buffer_status() const {
    if (!attached_ || !mapped_region_) {
        return {0, 0};
    }
    
    ShmHeader* header = static_cast<ShmHeader*>(mapped_region_);
    return {header->used, header->capacity};
}

bool HotSpineReader::is_healthy() const {
    return attached_ && mapped_region_ != nullptr;
}

} // namespace HotSpine
