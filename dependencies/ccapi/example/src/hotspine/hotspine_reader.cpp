#include "hotspine_reader.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace HotSpine {

HotSpineReader::HotSpineReader(const std::string& shm_name)
    : shm_name_(shm_name) {
    if (!attachToSharedMemory()) {
        throw std::runtime_error("Failed to attach to HotSpine shared memory: " + shm_name_);
    }
}

HotSpineReader::~HotSpineReader() {
    detachFromSharedMemory();
}

bool HotSpineReader::isAttached() const {
    return shm_ptr_ != nullptr && shm_ptr_ != MAP_FAILED;
}

bool HotSpineReader::pollTrade(HotTrade& trade) {
    if (!isAttached() || !header_) {
        return false;
    }
    
    uint64_t read_index = header_->read_index;
    uint64_t write_index = header_->write_index;
    
    // Check if there's data available
    if (read_index == write_index) {
        return false; // No trades available
    }
    
    // Read the trade
    trade = trades_buffer_[read_index % header_->capacity];
    
    // Update read index (with memory ordering for thread safety)
    __atomic_store_n(&header_->read_index, read_index + 1, __ATOMIC_RELEASE);
    
    return true;
}

std::vector<HotTrade> HotSpineReader::readAllAvailableTrades() {
    std::vector<HotTrade> trades;
    
    if (!isAttached() || !header_) {
        return trades;
    }
    
    uint64_t read_index = header_->read_index;
    uint64_t write_index = header_->write_index;
    
    if (read_index == write_index) {
        return trades; // No trades available
    }
    
    // Read all available trades in bulk
    while (read_index != write_index) {
        trades.push_back(trades_buffer_[read_index % header_->capacity]);
        read_index = (read_index + 1) % header_->capacity;
    }
    
    // Update read index atomically
    __atomic_store_n(&header_->read_index, read_index, __ATOMIC_RELEASE);
    
    return trades;
}

uint64_t HotSpineReader::getLostCount() const {
    return header_ ? header_->lost_count : 0;
}

std::pair<uint64_t, uint64_t> HotSpineReader::getBufferUtilization() const {
    if (!isAttached() || !header_) {
        return {0, 0};
    }
    
    uint64_t current_size = calculateCurrentSize();
    return {current_size, header_->capacity};
}

uint64_t HotSpineReader::calculateCurrentSize() const {
    if (!header_) return 0;
    
    uint64_t read_index = header_->read_index;
    uint64_t write_index = header_->write_index;
    
    if (write_index >= read_index) {
        return write_index - read_index;
    } else {
        // Wrapped around
        return (header_->capacity - read_index) + write_index;
    }
}

bool HotSpineReader::attachToSharedMemory() {
    shm_fd_ = shm_open(shm_name_.c_str(), O_RDONLY, 0666);
    if (shm_fd_ == -1) {
        std::cerr << "HotSpineReader: shm_open failed for " << shm_name_ 
                  << ": " << strerror(errno) << std::endl;
        return false;
    }
    
    struct stat st;
    if (fstat(shm_fd_, &st) == -1) {
        std::cerr << "HotSpineReader: fstat failed: " << strerror(errno) << std::endl;
        close(shm_fd_);
        shm_fd_ = -1;
        return false;
    }
    
    shm_ptr_ = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, shm_fd_, 0);
    if (shm_ptr_ == MAP_FAILED) {
        std::cerr << "HotSpineReader: mmap failed: " << strerror(errno) << std::endl;
        close(shm_fd_);
        shm_fd_ = -1;
        return false;
    }
    
    header_ = static_cast<SharedMemoryHeader*>(shm_ptr_);
    trades_buffer_ = reinterpret_cast<HotTrade*>(static_cast<char*>(shm_ptr_) + sizeof(SharedMemoryHeader));
    
    // Validate version compatibility
    if (header_->version != HOTSPINE_VERSION) {
        std::cerr << "HotSpineReader: Version mismatch. Expected " << HOTSPINE_VERSION
                  << " but got " << header_->version << std::endl;
        detachFromSharedMemory();
        return false;
    }
    
    std::cout << "HotSpineReader: Successfully attached to shared memory: " << shm_name_
              << " (capacity: " << header_->capacity << " trades, version: " 
              << header_->version << ")" << std::endl;
    
    return true;
}

bool HotSpineReader::detachFromSharedMemory() {
    if (shm_ptr_ != nullptr && shm_ptr_ != MAP_FAILED) {
        munmap(shm_ptr_, 0);
    }
    if (shm_fd_ != -1) {
        close(shm_fd_);
    }
    
    shm_ptr_ = nullptr;
    shm_fd_ = -1;
    header_ = nullptr;
    trades_buffer_ = nullptr;
    
    return true;
}

} // namespace HotSpine