#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace HotSpine {

struct HotTrade {
    uint32_t symbol_id;
    double price;
    double size;
    uint64_t timestamp_us;
    char side;  // 'B' for buy, 'S' for sell
};

struct HotOrderbookSnapshot {
    uint32_t symbol_id;
    uint64_t timestamp_us;
    double best_bid_price;
    double best_ask_price;
    double best_bid_size;
    double best_ask_size;
};

class HotSpineReader {
public:
    HotSpineReader(const std::string& shm_name) : shm_name_(shm_name) {}
    ~HotSpineReader() {}
    
    bool connect() { return true; }
    bool is_connected() const { return true; }
    void disconnect() {}
    
    std::vector<HotTrade> read_trades(size_t max_count = 100) {
        return {};
    }
    
    std::vector<HotOrderbookSnapshot> read_orderbooks(size_t max_count = 100) {
        return {};
    }
    
private:
    std::string shm_name_;
};

} // namespace HotSpine
