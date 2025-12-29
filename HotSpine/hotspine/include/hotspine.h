#pragma once
#include <cstddef>
#include <stdint.h>
#include <atomic>

struct Trade {
    uint64_t ts_ns;
    double price;
    double size;
};

constexpr size_t RING_SIZE = 1 << 20;

struct Ring {
    std::atomic<uint64_t> write_idx;
    std::atomic<uint64_t> read_idx;
    Trade buffer[RING_SIZE];
};

extern "C" {
    Ring* hotspine_init();
    void hotspine_push(Ring*, Trade);
    int hotspine_poll(Ring*, Trade*);
}
