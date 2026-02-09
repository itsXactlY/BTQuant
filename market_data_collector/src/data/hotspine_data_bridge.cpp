#include "hotspine_data_bridge.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <iostream>

namespace BTQuant {

HotSpineDataBridge::HotSpineDataBridge(const std::string& shm_path) : shm_path_(shm_path) {
    if (!connect_to_shared_memory()) {
        std::cerr << "[HotSpineDataBridge] Failed to connect to shared memory: " << shm_path << std::endl;
    } else {
        std::cout << "[HotSpineDataBridge] Successfully connected to shared memory: " << shm_path << std::endl;
    }
}

HotSpineDataBridge::~HotSpineDataBridge() {
    disconnect_from_shared_memory();
}

bool HotSpineDataBridge::connect_to_shared_memory() {
    // Open shared memory
    shm_fd_ = shm_open(shm_path_.c_str(), O_RDWR | O_CREAT, 0666);
    if (shm_fd_ == -1) {
        std::cerr << "[HotSpineDataBridge] Failed to open shared memory '" << shm_path_ 
                  << "': " << strerror(errno) << std::endl;
        return false;
    }

    // Get the size or set initial size
    struct stat sb;
    bool needs_initialization = false;
    if (fstat(shm_fd_, &sb) == -1 || sb.st_size == 0) {
        needs_initialization = true;
    }

    size_t shm_size;
    if (needs_initialization) {
        // Calculate required size: header + ring buffer data
        shm_size = sizeof(HotSpine::V3::RingBufferHeader) + 
                   (RING_BUFFER_SIZE * sizeof(HotspineData));
        
        if (ftruncate(shm_fd_, shm_size) == -1) {
            std::cerr << "[HotSpineDataBridge] Failed to set shared memory size: " 
                      << strerror(errno) << std::endl;
            close(shm_fd_);
            shm_fd_ = -1;
            return false;
        }
    } else {
        shm_size = sb.st_size;
    }

    // Map the shared memory
    shm_ptr_ = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
    if (shm_ptr_ == MAP_FAILED) {
        std::cerr << "[HotSpineDataBridge] Failed to map shared memory: " 
                  << strerror(errno) << std::endl;
        close(shm_fd_);
        shm_fd_ = -1;
        return false;
    }

    // Initialize pointers
    header_ = static_cast<HotSpine::V3::RingBufferHeader*>(shm_ptr_);
    ring_buffer_data_ = static_cast<uint8_t*>(shm_ptr_) + sizeof(HotSpine::V3::RingBufferHeader);

    // Initialize header if needed
    if (needs_initialization) {
        // Set magic number and version
        header_->magic = 0x42545155; // "BTQ3"
        header_->version = 3;
        header_->write_head.store(0, std::memory_order_relaxed);
        header_->read_tail.store(0, std::memory_order_relaxed);
        header_->dropped_count.store(0, std::memory_order_relaxed);
        
        std::cout << "[HotSpineDataBridge] Initialized shared memory with size: " 
                  << shm_size << " bytes" << std::endl;
    } else {
        // Validate header
        if (header_->magic != 0x42545155) { // "BTQ3"
            std::cerr << "[HotSpineDataBridge] Invalid magic number in shared memory: 0x" 
                      << std::hex << header_->magic << std::dec << std::endl;
            munmap(shm_ptr_, shm_size);
            shm_ptr_ = nullptr;
            close(shm_fd_);
            shm_fd_ = -1;
            return false;
        }
        
        if (header_->version != 3) {
            std::cerr << "[HotSpineDataBridge] Invalid version in shared memory: " 
                      << header_->version << std::endl;
            munmap(shm_ptr_, shm_size);
            shm_ptr_ = nullptr;
            close(shm_fd_);
            shm_fd_ = -1;
            return false;
        }
        
        std::cout << "[HotSpineDataBridge] Connected to existing shared memory (magic=0x" 
                  << std::hex << header_->magic << std::dec << ", version=" 
                  << header_->version << ")" << std::endl;
    }

    return true;
}

void HotSpineDataBridge::disconnect_from_shared_memory() {
    if (shm_ptr_ && shm_ptr_ != MAP_FAILED) {
        // Calculate the exact size that was mapped to ensure proper cleanup
        size_t mapped_size = sizeof(HotSpine::V3::RingBufferHeader) +
                            (RING_BUFFER_SIZE * sizeof(HotspineData));
        
        munmap(shm_ptr_, mapped_size);
        shm_ptr_ = nullptr;
        header_ = nullptr;
        ring_buffer_data_ = nullptr;
    }

    if (shm_fd_ != -1) {
        close(shm_fd_);
        shm_fd_ = -1;
    }
}

bool HotSpineDataBridge::write_direct(const HotspineData& event) {
    if (!header_ || !ring_buffer_data_) {
        return false;
    }

    // Load write_head (relaxed) - Step 1
    uint64_t write_head = header_->write_head.load(std::memory_order_relaxed);

    // Calculate slot index: idx = write_head & (RING_SIZE - 1) - Step 2
    // Apply mask to ensure index is within valid range [0, RING_BUFFER_SIZE-1]
    uint64_t slot_idx = write_head & RING_BUFFER_MASK;

    // Bounds check: ensure the calculated address is within the allocated buffer
    // Calculate the address where we'll write the event
    uint8_t* slot_addr = ring_buffer_data_ + (slot_idx * sizeof(HotspineData));
    uint8_t* buffer_start = ring_buffer_data_;
    uint8_t* buffer_end = buffer_start + (RING_BUFFER_SIZE * sizeof(HotspineData));

    // Verify that the slot address is within valid range
    if (slot_addr < buffer_start || slot_addr >= buffer_end) {
        std::cerr << "[HotSpineDataBridge] Buffer bounds violation in write_direct! slot_addr=" 
                  << reinterpret_cast<void*>(slot_addr) 
                  << ", buffer_start=" << reinterpret_cast<void*>(buffer_start)
                  << ", buffer_end=" << reinterpret_cast<void*>(buffer_end) << std::endl;
        return false;
    }

    // Verify that the slot address plus the event size doesn't exceed buffer bounds
    if ((slot_addr + sizeof(HotspineData)) > buffer_end) {
        std::cerr << "[HotSpineDataBridge] Buffer overflow detected in write_direct! Attempted to write past buffer end." << std::endl;
        return false;
    }

    // Additional validation: ensure we're not writing to an invalid memory region
    // by checking that the calculated offset doesn't wrap around due to integer overflow
    if ((slot_idx * sizeof(HotspineData)) / sizeof(HotspineData) != slot_idx) {
        std::cerr << "[HotSpineDataBridge] Integer overflow detected in address calculation!" << std::endl;
        return false;
    }

    // memcpy event to slot - Step 3
    std::memcpy(slot_addr, &event, sizeof(HotspineData));

    // Check for overflow condition: if write_head - read_tail > RING_SIZE
    uint64_t read_tail = header_->read_tail.load(std::memory_order_acquire);
    uint64_t available_count = write_head - read_tail;

    if (available_count >= RING_BUFFER_SIZE) {
        // Overflow: increment dropped_count atomic (diagnostic only) and overwrite (circular) - Step 4
        header_->dropped_count.fetch_add(1, std::memory_order_relaxed);
        // We still proceed with the write and update the head, as we're implementing circular buffer
    }

    // Atomic store write_head (release) - Step 4 (continued)
    header_->write_head.store(write_head + 1, std::memory_order_release);

    return true;
}

} // namespace BTQuant