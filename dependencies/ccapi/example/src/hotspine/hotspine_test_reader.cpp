#include "hotspine_layout.hpp"
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <vector>
#include <thread>
#include <chrono>

class HotSpineTestReader {
public:
    HotSpineTestReader(const std::string& shm_name) : shm_name_(shm_name) {
        attachToSharedMemory();
    }
    
    ~HotSpineTestReader() {
        detachFromSharedMemory();
    }
    
    bool isAttached() const { return shm_ptr_ != nullptr; }
    
    std::vector<HotSpine::HotTrade> readAllAvailableTrades() {
        std::vector<HotSpine::HotTrade> trades;
        
        if (!isAttached() || !header_) {
            return trades;
        }
        
        uint64_t read_index = header_->read_index;
        uint64_t write_index = header_->write_index;
        
        if (read_index == write_index) {
            return trades; // No trades available
        }
        
        // Read all available trades
        while (read_index != write_index) {
            trades.push_back(trades_buffer_[read_index]);
            read_index = (read_index + 1) % header_->capacity;
        }
        
        return trades;
    }
    
    void printTradeInfo(const HotSpine::HotTrade& trade) {
        std::cout << "Trade: " 
                  << "ts_exchange=" << trade.ts_exchange 
                  << ", ts_local=" << trade.ts_local 
                  << ", price=" << trade.price 
                  << ", size=" << trade.size 
                  << ", symbol_id=" << trade.symbol_id 
                  << ", side=" << static_cast<int>(trade.side) 
                  << std::endl;
    }
    
    uint64_t getLostCount() const {
        return header_ ? header_->lost_count : 0;
    }

private:
    std::string shm_name_;
    int shm_fd_{-1};
    void* shm_ptr_{nullptr};
    HotSpine::SharedMemoryHeader* header_{nullptr};
    HotSpine::HotTrade* trades_buffer_{nullptr};
    
    bool attachToSharedMemory() {
        shm_fd_ = shm_open(shm_name_.c_str(), O_RDONLY, 0666);
        if (shm_fd_ == -1) {
            std::cerr << "TestReader: shm_open failed: " << strerror(errno) << std::endl;
            return false;
        }
        
        struct stat st;
        if (fstat(shm_fd_, &st) == -1) {
            std::cerr << "TestReader: fstat failed: " << strerror(errno) << std::endl;
            close(shm_fd_);
            shm_fd_ = -1;
            return false;
        }
        
        shm_ptr_ = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, shm_fd_, 0);
        if (shm_ptr_ == MAP_FAILED) {
            std::cerr << "TestReader: mmap failed: " << strerror(errno) << std::endl;
            close(shm_fd_);
            shm_fd_ = -1;
            return false;
        }
        
        header_ = static_cast<HotSpine::SharedMemoryHeader*>(shm_ptr_);
        trades_buffer_ = reinterpret_cast<HotSpine::HotTrade*>(static_cast<char*>(shm_ptr_) + sizeof(HotSpine::SharedMemoryHeader));
        
        std::cout << "TestReader: Attached to shared memory: " << shm_name_ 
                  << " (capacity: " << header_->capacity << " trades)" << std::endl;
        
        return true;
    }
    
    bool detachFromSharedMemory() {
        if (shm_ptr_ != nullptr && shm_ptr_ != MAP_FAILED) {
            munmap(shm_ptr_, 0);
        }
        if (shm_fd_ != -1) {
            close(shm_fd_);
        }
        return true;
    }
};

int main(int argc, char** argv) {
    std::string shm_name = "/btquant_hotspine_test";
    if (argc > 1) {
        shm_name = argv[1];
    }
    
    std::cout << "HotSpine Test Reader - Monitoring: " << shm_name << std::endl;
    
    HotSpineTestReader reader(shm_name);
    
    if (!reader.isAttached()) {
        std::cerr << "Failed to attach to shared memory" << std::endl;
        return 1;
    }
    
    std::cout << "Waiting for trades... (Ctrl+C to exit)" << std::endl;
    
    size_t total_trades_read = 0;
    
    while (true) {
        auto trades = reader.readAllAvailableTrades();
        
        if (!trades.empty()) {
            std::cout << "\n=== Received " << trades.size() << " trades ===" << std::endl;
            
            for (const auto& trade : trades) {
                reader.printTradeInfo(trade);
            }
            
            total_trades_read += trades.size();
            std::cout << "Total trades read: " << total_trades_read << std::endl;
            std::cout << "Lost trades: " << reader.getLostCount() << std::endl;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    return 0;
}