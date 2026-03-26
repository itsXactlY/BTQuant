#include "hotspine_writer.hpp"
#include "hotspine_layout.hpp"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <chrono>
#include <functional>
#include <sstream>

namespace HotSpine {

// Simple symbol ID generator (in production, use a proper symbol mapping)
static uint32_t generateSymbolId(const std::string& exchange, 
                                const std::string& symbol,
                                const std::string& market_type) {
    std::stringstream ss;
    ss << exchange << ":" << symbol << ":" << market_type;
    std::string key = ss.str();
    
    // Simple hash function for symbol ID
    uint32_t hash = 5381;
    for (char c : key) {
        hash = ((hash << 5) + hash) + c; // hash * 33 + c
    }
    return hash;
}

HotSpineWriter::HotSpineWriter(const std::string& shm_name)
    : shm_name_(shm_name) {
    if (!attachToSharedMemory()) {
        std::cerr << "HotSpineWriter: Failed to attach to shared memory: " << shm_name << std::endl;
    }
}

HotSpineWriter::~HotSpineWriter() {
    detachFromSharedMemory();
}

bool HotSpineWriter::attachToSharedMemory() {
    // Try to open existing shared memory, or create if it doesn't exist
    shm_fd_ = shm_open(shm_name_.c_str(), O_RDWR | O_CREAT, 0666);
    if (shm_fd_ == -1) {
        std::cerr << "HotSpineWriter: shm_open failed: " << strerror(errno) << std::endl;
        return false;
    }
    
    // Try to get size of existing shared memory
    struct stat st;
    bool needs_init = false;
    if (fstat(shm_fd_, &st) == -1) {
        // Shared memory doesn't exist or can't be stat'd, we'll create it
        needs_init = true;
    } else if (st.st_size == 0) {
        // Shared memory exists but is empty, we'll initialize it
        needs_init = true;
    }
    
    size_t shm_size;
    if (needs_init) {
        // Calculate required size
        shm_size = HotSpine::calculateSharedMemorySize(HotSpine::DEFAULT_CAPACITY);
        
        // Set size of shared memory
        if (ftruncate(shm_fd_, shm_size) == -1) {
            std::cerr << "HotSpineWriter: ftruncate failed: " << strerror(errno) << std::endl;
            close(shm_fd_);
            shm_fd_ = -1;
            return false;
        }
    } else {
        shm_size = st.st_size;
    }
    
    // Map shared memory
    shm_ptr_ = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
    if (shm_ptr_ == MAP_FAILED) {
        std::cerr << "HotSpineWriter: mmap failed: " << strerror(errno) << std::endl;
        close(shm_fd_);
        shm_fd_ = -1;
        return false;
    }
    
    // Initialize pointers
    header_ = static_cast<HotSpine::SharedMemoryHeader*>(shm_ptr_);
    trades_buffer_ = reinterpret_cast<HotSpine::HotTrade*>(static_cast<char*>(shm_ptr_) + sizeof(HotSpine::SharedMemoryHeader));
    
    // Initialize or validate header
    if (needs_init) {
        // Initialize header
        header_->version = HotSpine::HOTSPINE_VERSION;
        header_->capacity = HotSpine::DEFAULT_CAPACITY;
        header_->write_index = 0;
        header_->read_index = 0;
        header_->lost_count = 0;
        std::memset(header_->padding, 0, sizeof(header_->padding));
        
        std::cout << "HotSpineWriter: Created and initialized shared memory: " << shm_name_
                  << " (capacity: " << header_->capacity << " trades)" << std::endl;
    } else {
        // Validate header
        if (header_->version != HotSpine::HOTSPINE_VERSION) {
            std::cerr << "HotSpineWriter: Invalid shared memory version: " << header_->version
                      << " (expected: " << HotSpine::HOTSPINE_VERSION << ")" << std::endl;
            detachFromSharedMemory();
            return false;
        }
        
        std::cout << "HotSpineWriter: Successfully attached to shared memory: " << shm_name_
                  << " (capacity: " << header_->capacity << " trades)" << std::endl;
    }
    
    return true;
}

bool HotSpineWriter::detachFromSharedMemory() {
    if (shm_ptr_ != nullptr && shm_ptr_ != MAP_FAILED) {
        if (munmap(shm_ptr_, 0) == -1) {
            std::cerr << "HotSpineWriter: munmap failed: " << strerror(errno) << std::endl;
            return false;
        }
        shm_ptr_ = nullptr;
        header_ = nullptr;
        trades_buffer_ = nullptr;
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

uint32_t HotSpineWriter::getSymbolId(const std::string& exchange, 
                                     const std::string& symbol,
                                     const std::string& market_type) const {
    return generateSymbolId(exchange, symbol, market_type);
}

bool HotSpineWriter::isHealthy() const {
    if (shm_ptr_ == nullptr || header_ == nullptr) {
        return false;
    }
    
    // Check if we're losing too many trades
    if (header_->lost_count > 1000) {
        return false;
    }
    
    return true;
}

bool HotSpineWriter::writeTrade(const MarketData::Trade& trade) {
    if (!isHealthy()) {
        write_errors_++;
        return false;
    }
    
    // Use batching if enabled
    if (batching_enabled_) {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        batch_buffer_.push_back(trade);
        
        // Flush batch if it reaches the batch size
        if (batch_buffer_.size() >= batch_size_) {
            flushBatch();
        }
        
        trades_written_++;
        return true;
    }
    
    // Direct write (non-batched mode)
    // Convert trade to HotTrade format
    HotSpine::HotTrade hot_trade;
    hot_trade.ts_exchange = static_cast<uint64_t>(trade.timestamp_us);
    hot_trade.ts_local = getCurrentTimestampMicros();
    hot_trade.price = trade.price;
    hot_trade.size = trade.quantity;
    hot_trade.symbol_id = getSymbolId(trade.exchange, trade.symbol, trade.market_type);
    hot_trade.side = (trade.side == "buy") ? 0 : 1;
    
    // Calculate write position
    uint64_t write_index = header_->write_index;
    uint64_t next_write_index = (write_index + 1) % header_->capacity;
    
    // Check if buffer is full
    if (next_write_index == header_->read_index) {
        // Buffer is full, increment lost count
        header_->lost_count++;
        write_errors_++;
        return false;
    }
    
    // Write trade to buffer
    trades_buffer_[write_index] = hot_trade;
    
    // Update write index (memory barrier ensured by atomic operations)
    header_->write_index = next_write_index;
    
    trades_written_++;
    return true;
}

void HotSpineWriter::flushBatch() {
    if (!isHealthy() || batch_buffer_.empty()) {
        return;
    }
    
    std::vector<MarketData::Trade> batch_to_write;
    {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        batch_to_write.swap(batch_buffer_);
    }
    
    if (batch_to_write.empty()) {
        return;
    }
    
    // Reserve space in shared memory
    uint64_t current_write_index = header_->write_index;
    uint64_t required_space = batch_to_write.size();
    
    // Check if there's enough space
    uint64_t available_space = (header_->read_index > current_write_index)
        ? (header_->read_index - current_write_index - 1)
        : (header_->capacity - current_write_index + header_->read_index - 1);
    
    if (available_space < required_space) {
        // Not enough space, increment lost count
        header_->lost_count += batch_to_write.size();
        write_errors_ += batch_to_write.size();
        return;
    }
    
    // Write batch to shared memory
    for (const auto& trade : batch_to_write) {
        HotSpine::HotTrade hot_trade;
        hot_trade.ts_exchange = static_cast<uint64_t>(trade.timestamp_us);
        hot_trade.ts_local = getCurrentTimestampMicros();
        hot_trade.price = trade.price;
        hot_trade.size = trade.quantity;
        hot_trade.symbol_id = getSymbolId(trade.exchange, trade.symbol, trade.market_type);
        hot_trade.side = (trade.side == "buy") ? 0 : 1;
        
        trades_buffer_[current_write_index] = hot_trade;
        current_write_index = (current_write_index + 1) % header_->capacity;
    }
    
    // Update write index (memory barrier ensured by atomic operations)
    header_->write_index = current_write_index;
}

bool HotSpineWriter::writeTrades(const std::vector<MarketData::Trade>& trades) {
    if (!isHealthy()) {
        write_errors_++;
        return false;
    }
    
    bool all_success = true;
    for (const auto& trade : trades) {
        if (!writeTrade(trade)) {
            all_success = false;
        }
    }
    
    return all_success;
}

} // namespace HotSpine