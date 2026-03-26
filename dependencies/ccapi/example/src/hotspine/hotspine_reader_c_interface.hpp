#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer to HotSpineReader instance
typedef void* HotSpineReaderHandle;

// Trade structure for C interface (must match HotTrade in hotspine_layout.hpp)
typedef struct {
    uint64_t ts_exchange;  // exchange timestamp in microseconds
    uint64_t ts_local;     // local receive timestamp in microseconds
    double price;
    double size;
    uint32_t symbol_id;    // symbol ID (hash or mapping)
    uint8_t side;         // 0=buy, 1=sell
} HotSpineTradeC;

// Create a new HotSpine reader instance
HotSpineReaderHandle hotspine_reader_create(const char* shm_name);

// Destroy a HotSpine reader instance
void hotspine_reader_destroy(HotSpineReaderHandle reader);

// Poll for a single trade (non-blocking)
// Returns 1 if trade was read, 0 if no trades available, -1 on error
int hotspine_reader_poll_trade(HotSpineReaderHandle reader, HotSpineTradeC* trade);

// Get the number of lost trades (overflow counter)
uint64_t hotspine_reader_get_lost_count(HotSpineReaderHandle reader);

// Check if reader is healthy
int hotspine_reader_is_healthy(HotSpineReaderHandle reader);

#ifdef __cplusplus
}
#endif